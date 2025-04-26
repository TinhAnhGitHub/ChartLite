import json
import pandas as pd
import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import io
from typing import List
import sys
import os
from tqdm import tqdm
import joblib

root_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../..'
    )
)
print(root_dir)
sys.path.append(root_dir)
from src.utils.constant import TOKEN_MAP



def tokenize_dict(data: dict):
    def recursive_tokenizer(d):
        if isinstance(d, dict):
            result = ""
            for key, value in d.items():
                start_token = f"<{key}>"
                end_token = f"</{key}>"
                value_string = recursive_tokenizer(value)
                result += f"{start_token}{value_string}{end_token}"
            return result
        elif isinstance(d, list):  
            try:
                if isinstance(d[0], dict) and (('x' in d[0] and 'y' in d[0]) or 'text' in d[0]):
                    return ''.join(recursive_tokenizer(item) for item in d)
                return ','.join(recursive_tokenizer(item) for item in d)
            except Exception as e:
                ...
        else:
            if type(d) == float:
                return str(round(d, 2))
            return str(d)
    token_sequence = recursive_tokenizer(data)
    return  token_sequence


class ChartDataset(Dataset):
    def __init__(self, config: dict, parquet_paths: List[str]):
        self.config = config
        self.resize_height = config['images']['rsz_height']
        self.resize_width = config['images']['rsz_width']
        self.transform = create_train_transforms(self.resize_height, self.resize_width)
        
        dfs = [pd.read_parquet(parquet_path) for parquet_path in parquet_paths]
        self.parquet_df = pd.concat(dfs, ignore_index=True)
        if 'id' in self.parquet_df.columns and self.parquet_df['id'].is_unique:
            self.data_ratio = int(len(self.parquet_df['id'].tolist())*config['dataset']['data_ratio'])
            self.graph_ids = self.parquet_df['id'].tolist()[:self.data_ratio]
        else:
            self.data_ratio = int(len(self.parquet_df.index.tolist())*config['dataset']['data_ratio'])
            self.graph_ids = self.parquet_df.index.tolist()[:self.data_ratio]
    
    def load_image(self, graph_id: str) -> Image.Image:
        image_data = self.parquet_df.loc[graph_id]["image"]["bytes"]
        return Image.open(io.BytesIO(image_data)).convert('RGB')

    def build_output(self, graph_id: str) -> tuple[str, str]:
        row = self.parquet_df.loc[graph_id]
        try:
            ground_truth = json.loads(row["annotation"])
            chart_type = ground_truth.get('chart_type', 'unknown')
            text = tokenize_dict(ground_truth)
            return f"{text}", chart_type
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for graph_id {graph_id}: {e}")
            return 'error', 'error_chart'
        except Exception as e:
            print(f"Error building output for graph_id {graph_id}: {e}")
            return 'error', 'error_chart'

    def __len__(self) -> int:
        return len(self.graph_ids)

    def __getitem__(self, index: int) -> dict:
        graph_id = self.graph_ids[index]
        image = self.load_image(graph_id)
        
        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        text, chart_type = self.build_output(graph_id)

        return {
            'id': graph_id,
            'chart_type': chart_type,
            'image': image,
            'text': text,          
        }

def create_train_transforms(resize_height: int = None, resize_width: int = None) -> A.Compose:
    """Create albumentations transforms for training."""
    transforms = [A.Resize(height=resize_height, width=resize_width)] if resize_height and resize_width else []
    return A.Compose(transforms)
