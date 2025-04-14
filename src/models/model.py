import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration
from .adapter import AdapterPlus


class Matcha(nn.Module):
    """
    Matcha: A vision-language model that injects AdapterPlus modules into a
    pre-trained Pix2Struct encoder, with configurable freezing options.
    
    Config options expected in `cfg.model`:
        - backbone_path: path to the pre-trained backbone
        - max_length: maximum text length for the decoder
        - pad_token_id: pad token id for the decoder
        - decoder_start_token_id: start token id for the decoder
        - bos_token_id: beginning-of-sequence token id
        - len_tokenizer: vocabulary size for resizing token embeddings
        - frozen_percentage: fraction of encoder layers to freeze (0 to 1)
          (or an integer number of layers can be provided)
        - freeze_layernorm: bool, whether to freeze all LayerNorm parameters
        - freeze_pos_emb: bool, whether to freeze all positional embeddings
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        backbone_config = self._create_backbone_config(cfg)
        self.backbone = Pix2StructForConditionalGeneration.from_pretrained(
            cfg.model.backbone_path,
            config=backbone_config,
        )
        self._freeze_encoder_layers()
        self._inject_adapters()
        self._resize_token_embeddings()

    def _create_backbone_config(self, cfg):
        """Load and update the backbone configuration."""
        config = Pix2StructConfig.from_pretrained(cfg.model.backbone_path)
        config.text_config.max_length = cfg.model.max_length
        config.text_config.is_decoder = True
        config.text_config.pad_token_id = cfg.model.pad_token_id
        config.text_config.decoder_start_token_id = cfg.model.decoder_start_token_id
        config.text_config.bos_token_id = cfg.model.bos_token_id
        return config

    def _freeze_encoder_layers(self):
        """
        Freeze a portion of the encoder layers and optionally freeze additional 
        submodules, like LayerNorm and positional embeddings.
        """
        num_layers = self.backbone.config.vision_config.num_hidden_layers
        if isinstance(self.cfg.model.frozen_percentage, float):
            num_freeze = int(self.cfg.model.frozen_percentage * num_layers)
        else:
            num_freeze = int(self.cfg.model.frozen_percentage)
        print(f"Freezing first {num_freeze} out of {num_layers} encoder layers...")

        for layer in self.backbone.encoder.encoder.layer[:num_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

        if self.cfg.model.freeze_layernorm or self.cfg.model.freeze_pos_emb:
            for name, param in self.backbone.encoder.named_parameters():
                lower_name = name.lower()
                if self.cfg.model.freeze_layernorm and "layernorm" in lower_name:
                    param.requires_grad = False
                if self.cfg.model.freeze_pos_emb and "position" in lower_name:
                    param.requires_grad = False

    @staticmethod
    def _create_forward_wrapper(original_forward, adapter):
        """Creates a closure to inject adapter output after the original forward."""
        def forward_wrapper(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            layer_output = outputs[0]
            layer_output = adapter(layer_output, skip=layer_output)
            return (layer_output,) + outputs[1:]
        return forward_wrapper

    def _inject_adapters(self):
        """
        Injects an AdapterPlus into each encoder layer. The adapter is applied
        after the original forward pass.
        """
        vision_config = self.backbone.config.encoder
        adapter_config = {
            "embed_dim": vision_config.hidden_size,
            "bottleneck_dim": 8,
            "drop_path": 0.1,
            "dropout": 0.0,
        }
        for layer in self.backbone.encoder.encoder.layer:
            adapter = AdapterPlus(**adapter_config)
            layer.adapter = adapter
            layer.forward = self._create_forward_wrapper(layer.forward, adapter)

    def _resize_token_embeddings(self):
        """Resize the token embeddings for the decoder."""
        self.backbone.decoder.resize_token_embeddings(self.cfg.model.len_tokenizer)

    def forward(self, flattened_patches, attention_mask, labels=None):
        outputs = self.backbone(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        return loss, outputs
