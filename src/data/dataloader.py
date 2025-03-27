from dataclasses import dataclass
import io
import numpy as np
import torch
from transformers import DataCollatorWithPadding
import glob
import os
from .dataset import ChartDataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# @dataclass
# class ChartCollator(DataCollatorWithPadding):
#     """
#     data collector for mga task
#     """

#     tokenizer = None
#     padding = True
#     max_length = None
#     pad_to_multiple_of = None
#     return_tensors = "pt"

#     def __call__(self, features):
#         batch = dict()

#         batch["id"] = [feature["id"] for feature in features]
#         batch["chart_type"] = [feature["chart_type"] for feature in features]
#         batch["texts"] = [feature["text"] for feature in features]
#         batch["images"] = [feature["image"] for feature in features]


#         flattened_patches = [feature["flattened_patches"] for feature in features]
#         attention_mask = [feature["attention_mask"] for feature in features]

#         flattened_patches = np.concatenate(flattened_patches, axis=0)
#         attention_mask = np.concatenate(attention_mask, axis=0)

#         batch["flattened_patches"] = flattened_patches
#         batch["attention_mask"] = attention_mask

#         decoder_features = [
#             {
#                 "input_ids": feature["decoder_input_ids"],
#                 "attention_mask": feature["decoder_attention_mask"]
#             } for feature in features
#         ]

#         decoder_batch = self.tokenizer.pad(
#             decoder_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors=None,
#         )

#         batch["decoder_input_ids"] = decoder_batch["input_ids"]
#         batch["decoder_attention_mask"] = decoder_batch["attention_mask"]

#         pad_token_id = self.tokenizer.pad_token_id
#         labels = []
#         for ex_labels in batch["decoder_input_ids"]:
#             tmp = [l if l != pad_token_id else -100 for l in ex_labels]
#             labels.append(tmp)
#         batch["labels"] = labels

#         tensor_keys = ["flattened_patches", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]
#         for key in tensor_keys:
#             if key != "flattened_patches":
#                 batch[key] = torch.tensor(batch[key], dtype=torch.int64)
#             else:
#                 batch[key] = torch.tensor(batch[key], dtype=torch.float32)

#         return batch

@dataclass
class ChartCollator:
    tokenizer:PreTrainedTokenizerBase  
    padding: bool = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"

    def __call__(self, features):
        batch = {
            "id": [f["id"] for f in features],
            "chart_type": [f["chart_type"] for f in features],
            "texts": [f["text"] for f in features],
            "images": [f["image"] for f in features],
            "flattened_patches": torch.tensor(
                np.concatenate([f["flattened_patches"] for f in features], axis=0),
                dtype=torch.float32
            ),
            "attention_mask": torch.tensor(
                np.concatenate([f["attention_mask"] for f in features], axis=0),
                dtype=torch.int64
            ),
        }

        # Pad pre-tokenized decoder inputs directly with __call__
        decoder_input_ids = [f["decoder_input_ids"] for f in features]
        decoder_attention_masks = [f["decoder_attention_mask"] for f in features]
        
        padded = self.tokenizer.pad(
            {"input_ids": decoder_input_ids, "attention_mask": decoder_attention_masks},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"  # Directly return PyTorch tensors
        )

        batch["decoder_input_ids"] = padded["input_ids"]
        batch["decoder_attention_mask"] = padded["attention_mask"]

        # Convert pad_token_id to -100 for labels
        batch["labels"] = batch["decoder_input_ids"].clone()
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

        return batch
    


def create_dataloaders(config):
    directory = config['dataset']['parquet_dict']
    train_files = glob.glob(os.path.join(directory, "train*.parquet"))  
    valid_files = glob.glob(os.path.join(directory, "validation*.parquet")) 

    valid_dataset = ChartDataset(config, valid_files)
    train_dataset = ChartDataset(config, train_files)

    tokenizer = train_dataset.processor.tokenizer
    collate_fn = ChartCollator(tokenizer=tokenizer)

    train_dl = DataLoader(
        train_dataset,
        batch_size=config['train_params']['train_bs'],
        collate_fn=collate_fn,
        num_workers=config['train_params']['num_workers'],
        pin_memory=True,
        shuffle=True
    )

    val_dl = DataLoader(
        valid_dataset,
        batch_size=config['train_params']['valid_bs'],
        collate_fn=collate_fn,
        num_workers=config['train_params']['num_workers'],
        shuffle=False,
        pin_memory=True
    )
    return train_dl, val_dl
