import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration


class Matcha(nn.Module):
    """
    The Matcha model
    """
    def __init__(self, cfg):
        super().__init__()  
        backbone_config = Pix2StructConfig.from_pretrained(cfg.model.backbone_path)
        backbone_config.text_config.max_length = cfg.model.max_length
        backbone_config.text_config.is_decoder = True
        backbone_config.text_config.pad_token_id = cfg.model.pad_token_id
        backbone_config.text_config.decoder_start_token_id = cfg.model.decoder_start_token_id
        backbone_config.text_config.bos_token_id = cfg.model.bos_token_id
        backbone_config.dropout = cfg.model.get("dropout_rate", 0.1)  # default to 0.1 if not provided
        backbone_config.attention_dropout = cfg.model.get("attention_dropout_rate", 0.1)


        self.backbone = Pix2StructForConditionalGeneration.from_pretrained(
            cfg['model']['backbone_path'],
            config=backbone_config,
        )

        print("Freezing the encoder...")
        num_hidden_layers = self.backbone.config.vision_config.num_hidden_layers
        
        to_freeze_layer = int(cfg.model.frozen_percentage * num_hidden_layers)
        print(f"Remaining: {to_freeze_layer}")
        for layer in self.backbone.encoder.encoder.layer[:to_freeze_layer]:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.backbone.decoder.resize_token_embeddings(cfg.model.len_tokenizer)
      
    def forward(self, flattened_patches, attention_mask, labels=None):
        outputs = self.backbone(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss 
        return loss, outputs 
    


