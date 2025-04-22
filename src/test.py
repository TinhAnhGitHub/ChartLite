import os
import glob
import torch
import argparse
from typing import Dict, Optional
import numpy as np
from omegaconf import OmegaConf
import yaml
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import Matcha
from data import ChartDataset, ChartCollator
from rich import print
import pprint 

import sys
import os

root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)


from src.utils.metric_utils_fast import F1ScoreEvaluator,TEDAccuracyEvaluator
from src.utils import TOKEN_MAP, post_processing


def calculate_token_length(text, tokenizer):
    tokenized = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    return len(tokenized["input_ids"][0])

def validate(config_path: str, checkpoint_path: str, output_dir: Optional[str] = None) -> Dict[str, float]:

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = OmegaConf.create(yaml_config)

    if output_dir is not None:
        config.outputs.model_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    directory = config.dataset.parquet_dict
    valid_files = glob.glob(os.path.join(directory, "valid*.parquet"))
    temp_dataset = ChartDataset(config, valid_files)
    processor = temp_dataset.processor
    tokenizer = processor.tokenizer

    config.model.len_tokenizer = len(tokenizer)  
    config.model.pad_token_id = tokenizer.pad_token_id  
    config.model.decoder_start_token_id = tokenizer.convert_tokens_to_ids(TOKEN_MAP["bos_token"])[0]  
    config.model.bos_token_id = tokenizer.convert_tokens_to_ids(TOKEN_MAP["bos_token"])[0]
    
    val_dataset = ChartDataset(config, valid_files)

    collate_fn = ChartCollator(tokenizer)
    test_dataloader = DataLoader(
        val_dataset,
        batch_size=6,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )

    model = Matcha(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = {
            k.removeprefix('model.')
                .replace('base_encoder', 'encoder')
                    .replace('.base_model.model.','.')
                        .replace('.base_layer.','.')
                            : v 
            for k, v in checkpoint['state_dict'].items()
            if k.startswith('model.') and not 'lora' in k
        }
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    #print(model.backbone.config)


    evaluator_f1 = F1ScoreEvaluator()
    evaluator_ted = F1ScoreEvaluator()
    all_label_texts = []
    
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_generated_text = []
    loss_list = []

    i = 0        
    
    with torch.no_grad():
        j= 0
        for batch in tqdm(test_dataloader, desc="Testing..."):
            # if j < 383:
            #     j += 1
            #     continue
            flattened_patches = batch["flattened_patches"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else None
            
            loss, _ = model(flattened_patches, attention_mask, labels)
            total_loss += loss.item()

            
            
            
            generated_ids = model.backbone.generate(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,  
                max_new_tokens=2100,
                do_sample=False,
                use_cache=True, 
            )
                        
            
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = batch["texts"]
            # label_token_lengths = [calculate_token_length(label_text, tokenizer) for label_text in label_texts]
            # print(f"{label_token_lengths=}")
            # if label_token_lengths[0] > 512:
            #     continue

                
            # print()
            # print('*'*50)
            # print()
            # print(f"{generated_texts=}")
            # print()
            # print(f"{label_texts=}")
            # print()
            # print(f"{loss=}")
            # print()
            
            label_dicts = [post_processing(label_text, TOKEN_MAP) for label_text in label_texts]
            preds = [post_processing(generated_text, TOKEN_MAP)
                    for item_id, generated_text in zip(batch['id'], generated_texts)]
            
            # f1_score = evaluator_f1.evaluate(preds=preds, answers=label_dicts)
            # accuracy = evaluator_ted.evaluate(preds=preds, answers=label_dicts)
            
            # print(f"{label_dicts=}")
            # print()
            # print(f"{preds=}")
            # print('*'*50)
            # print()
            # print(f"{f1_score=}")
            # print()
            # print(f"{accuracy=}")
            # print()
            # print('*'*50)
            # print()


            
            all_labels.extend(label_dicts)
            all_preds.extend(preds)
            all_generated_text.extend(generated_texts)
            all_label_texts.append(label_texts)
            loss_list.append(loss.item())
            # if len(all_generated_text) > 5:
            #     break
            

            
    
    print(f"{len(all_labels)=}")
    print(f"{len(all_preds)=}")
    lost_np = np.array(loss_list, dtype=np.float32)
    mean_loss = lost_np.mean()
    std_loss = lost_np.std()
    max_loss = lost_np.max()
    min_loss = lost_np.min()

    print(f"{mean_loss=}")
    print(f"{std_loss=}")
    print(f"{max_loss=}")
    print(f"{min_loss=}")

    
    
    f1_score = evaluator_f1.evaluate(preds=all_preds, answers=all_labels)
    accuracy = evaluator_ted.evaluate(preds=all_preds, answers=all_labels)
    detailed_results = []
    for  pred, generated_text, label in zip(all_preds,all_generated_text, all_labels):
        detailed_results.append({
            "prediction": pred,
            "ground_truth": label,
            "generated_texts":generated_text
        })
    partial_results = []
    for label_text, generated_text in zip(all_label_texts,all_generated_text):
        partial_results.append({
            'label_text': label_text,
            'generated_texts':generated_text
        })
    
    detailed_file = os.path.join(config.outputs.model_dir, "validation_detailed.json")
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=5)

    partial_file = os.path.join(config.outputs.model_dir, "validation_partial.json")
    with open(partial_file, 'w') as f:
        json.dump(partial_results, f, indent=5)

    loss_details = os.path.join(config.outputs.model_dir, "validation_partial.txt")
    loss_met = {
        'mean_loss': f"{mean_loss:.6f}",
        'std_loss': f"{std_loss:.6f}",
        'max_loss': f"{max_loss:.6f}",
        'min_loss': f"{min_loss:.6f}",
        'f1 score': f"{f1_score:.6f}",
        "Acc": f"{accuracy:.6f}"
    }
    with open(loss_details, 'w', encoding='utf-8') as f:
        for name, value in loss_met.items():
            f.write(f"{name}:  {value}")
    print(f"Done")
    
    
    


if __name__ == "__main__":
    config = "./Matcha/configs/base_config.yaml"
    checkpoint = "./run_expr/v0_ema/checkpoints/best_ckpts/best-checkpoint-v1.ckpt"
    output_dir = "./run_expr/exp_v0_ema_d09999"
    validate(config, checkpoint, output_dir)    