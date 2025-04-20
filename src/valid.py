import os
import glob
import torch
import argparse
from typing import Dict, Optional
from omegaconf import OmegaConf
import yaml
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Matcha
from data import ChartDataset, ChartCollator
from utils import TOKEN_MAP, JSONParseEvaluator, post_processing
from rich import print


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
        batch_size=13,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )

    model = Matcha(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if 'model.' in k}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()

    evaluator = JSONParseEvaluator()
    
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_generated_text = []

    i = 0        
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing..."):
            flattened_patches = batch["flattened_patches"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else None
            
            loss, _ = model(flattened_patches, attention_mask, labels)
            total_loss += loss.item()
            
            
            generated_ids = model.backbone.generate(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                max_new_tokens=4096,
                do_sample=False,
                top_k=1,
                use_cache=True,
            )
            
            
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = batch["texts"]
            print()
            print('*'*50)
            print(f"{generated_texts=}")
            print(f"{loss=}")
            
            
            label_dicts = [post_processing(label_text, TOKEN_MAP) for label_text in label_texts]
            preds = [post_processing(generated_text, TOKEN_MAP)
                    for item_id, generated_text in zip(batch['id'], generated_texts)]
            
            print(f"{label_dicts=}")
            print(f"{preds=}")
            print('*'*50)
            print()


            
            all_labels.extend(label_dicts)
            all_preds.extend(preds)
            all_generated_text.extend(generated_texts)

            
    
    print(f"{len(all_labels)=}")
    print(f"{len(all_preds)=}")
    avg_loss = total_loss / len(test_dataloader)
    f1_score = evaluator.cal_f1(preds=all_preds, answers=all_labels)
    accuracy = evaluator.cal_acc(preds=all_preds, answers=all_labels)
    # overall_sim = evaluator.compare_json_list(
    #     all_labels, all_preds,
    #     numeric_tolerance=config.metrics_tolerance.numeric_tolerance,
    #     string_tolerance=config.metrics_tolerance.string_tolerance
    # )
    detailed_results = []
    for  pred, generated_text, label in zip(all_preds,all_generated_text, all_labels):
        detailed_results.append({
            "prediction": pred,
            "ground_truth": label,
            "generated_texts":generated_text
        })
    
    detailed_file = os.path.join(config.outputs.model_dir, "validation_detailed.json")
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=5)
    
    print(f"Validation Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    #print(f"Overall Similarity: {overall_sim:.4f}")
    


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run validation on the validation set")
    # parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    # parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    # parser.add_argument("--output_dir", type=str, default=None, help="Directory to save validation results")
    
    # args = parser.parse_args()
    config = "/media/tinhanhnguyen/Data1/Projects/School/Matcha/configs/base_config.yaml"
    checkpoint = "/media/tinhanhnguyen/Data1/Projects/School/Matcha/run_expr/exp_v0_ema_d09999/checkpoints/best_ckpts/best-checkpoint-step=31387-val_loss_avg=0.04540557041764259.ckpt"
    output_dir = "/media/tinhanhnguyen/Data1/Projects/School/Matcha/run_expr/exp_v0_ema_d09999"
    
    validate(config, checkpoint, output_dir)