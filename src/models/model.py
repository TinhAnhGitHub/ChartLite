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
        - frozen_layer_range: list or tuple, range of encoder layers to freeze, e.g., [0, 6]
        - adapter_layer_range: list or tuple, range of encoder layers to inject adapters, e.g., [6, 12]
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
        frozen_layer_range = self.cfg.model.get("frozen_layer_range", None)
        if frozen_layer_range is not None:
            start_freeze, end_freeze = frozen_layer_range
            print(
                f"Freezing encoder layers from {start_freeze} to {end_freeze} (exclusive) out of {num_layers} layers..."
            )
            for layer in self.backbone.encoder.encoder.layer[start_freeze:end_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            print("No encoder layers specified for freezing.")

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
        adapter_config = {
            "embed_dim": 768,
            "bottleneck_dim": 32,
            "drop_path": 0.1,
            "dropout": 0.0,
        }
        adapter_layer_range = self.cfg.model.get("adapter_layer_range", None) 
        if adapter_layer_range is not None:
            start_adapter, end_adapter = adapter_layer_range
            print(f"Injecting adapters into encoder layers from {start_adapter} to {end_adapter} (exclusive)...")

            for layer in self.backbone.encoder.encoder.layer[start_adapter:end_adapter]:
                adapter = AdapterPlus(**adapter_config)
                layer.adapter = adapter
                layer.forward = self._create_forward_wrapper(layer.forward, adapter)
        else:
            print("No encoder layers specified for adapter injection.")

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
