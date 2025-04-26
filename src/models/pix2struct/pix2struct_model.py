import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, Pix2StructConfig
from tokenizers import AddedToken
import numpy as np
from typing import Dict, Any

import os
import sys
root_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../..'
    )
)

sys.path.append(root_dir)
from src.models.model import BaseChartExtractionModel
from src.utils.constant import TOKEN_MAP



def get_processor(
    config: dict
):
    processor_path = config.model.backbone_path
    
    processor = Pix2StructProcessor.from_pretrained(processor_path)
    processor.image_processor.is_vqa = False
    processor.image_processor.patch_size = {
        'height': config.model.patch_size,
        'width': config.model.patch_size,
    }
    new_tokens = sorted(tok for this_tok in TOKEN_MAP.values() for tok in this_tok)
    processor.tokenizer.add_tokens([AddedToken(tok, lstrip=False, rstrip=False) for tok in new_tokens])
    return processor



class Pix2Struct(BaseChartExtractionModel, torch.nn.Module):
    def __init__(self, cfg):
        torch.nn.Module.__init__(self)
        BaseChartExtractionModel.__init__(self)
        self.cfg=cfg
        self.processor = get_processor(self.cfg)
        self.tokenizer = self.processor.tokenizer

        self.cfg.model.len_tokenizer = len(self.tokenizer)
        self.cfg.model.pad_token_id = self.tokenizer.pad_token_id
        self.cfg.model.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(TOKEN_MAP['bos_token'])[0]
        self.cfg.model.bos_token_id = self.tokenizer.convert_tokens_to_ids(TOKEN_MAP['bos_token'])[0]
        
        backbone_config = self._create_backbone_config(self.cfg)
        print(f"Using model name: {cfg.model.backbone_path}")
        self.backbone = Pix2StructForConditionalGeneration.from_pretrained(
            cfg.model.backbone_path,
            config=backbone_config,
        )
        self._freeze_encoder_layers()
        self.backbone.decoder.resize_token_embeddings(self.cfg.model.len_tokenizer)



    def _create_backbone_config(self, cfg):
        config = Pix2StructConfig.from_pretrained(cfg.model.backbone_path)
        config.text_config.max_length = cfg.model.max_length
        config.text_config.is_decoder = True
        config.text_config.pad_token_id = cfg.model.pad_token_id
        config.text_config.decoder_start_token_id = cfg.model.decoder_start_token_id
        config.text_config.bos_token_id = cfg.model.bos_token_id
        return config

    def _freeze_encoder_layers(self):
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
    

    def preprocess(self, image: np.ndarray, text: str = None):
        p_image = self.processor(
            images=image,
            max_patches=self.cfg['model']['max_patches'],
            add_special_tokens=True,
            legacy=False
        )

        p_txt = self.processor(
            text=text,
            truncation=False,
            add_special_tokens=True,
            max_length=self.cfg['model']['max_length'],
            legacy=False
        )
       

        return {
            'image': image,
            'text': text,
            'flattened_patches': p_image['flattened_patches'][0],
            'attention_mask': p_image['attention_mask'][0],
            'decoder_input_ids': p_txt.get('decoder_input_ids', p_txt.get('input_ids', [])),
            'decoder_attention_mask': p_txt.get('decoder_attention_mask', p_txt.get('attention_mask', []))
        }
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor]
    ):
        
        flattened_patches = batch['flattened_patches']
        attention_mask = batch['attention_mask']
        labels = batch['labels']



        outputs = self.backbone(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        return loss, outputs

    def generate(self, batch, **gen_kwargs):
        flattened_patches = batch['flattened_patches']
        attention_mask = batch['attention_mask']

        return self.backbone.generate(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            **gen_kwargs
        )

        


        
        