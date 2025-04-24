import json
import pandas as pd
import albumentations as A
import numpy as np
from PIL import Image
from tokenizers import AddedToken
from torch.utils.data import Dataset
from transformers import Pix2StructProcessor
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

def get_processor(config: dict) -> Pix2StructProcessor:
    """Load and configure the Pix2Struct processor."""
    processor_path = config['model']['backbone_path']
    
    processor = Pix2StructProcessor.from_pretrained(processor_path)
    processor.image_processor.is_vqa = False
    processor.image_processor.patch_size = {
        "height": config['model']['patch_size'],
        "width": config['model']['patch_size']
    }
    
    new_tokens = sorted(tok for this_tok in TOKEN_MAP.values() for tok in this_tok)
    processor.tokenizer.add_tokens([AddedToken(tok, lstrip=False, rstrip=False) for tok in new_tokens])



    
    return processor

class ChartDataset(Dataset):
    def __init__(self, config: dict, parquet_paths: List[str]):
        self.config = config
        self.resize_height = config['images']['rsz_height']
        self.resize_width = config['images']['rsz_width']
        self.transform = create_train_transforms(self.resize_height, self.resize_width)
        self.processor = get_processor(config)
        
        dfs = [pd.read_parquet(parquet_path) for parquet_path in parquet_paths]
        self.parquet_df = pd.concat(dfs, ignore_index=True)
        '''
        def parralell_get_token_length(df: pd.DataFrame, processor: Pix2StructProcessor) -> pd.DataFrame:
            results = joblib.Parallel(n_jobs=8)(
                joblib.delayed(self.calculate_token_length_from_dict)(data,processor)
                for data in tqdm(df, desc="Getting token length")
            )
            
            return results
        # print(self.parquet_df['annotation'][0])
        max_length = self.config['model']['max_length']
        print(f"Max token length: {max_length}")

        self.parquet_df['annotation'] = parralell_get_token_length(self.parquet_df['annotation'],self.processor)
        # print(self.parquet_df['annotation'][0])
        # self.parquet_df = self.parquet_df[
        #     self.parquet_df['annotation'].apply(
        #         lambda x: isinstance(x, dict) and 'token_length' in x and x['token_length'] <= max_length
        #     )
        # ].reset_index(drop=True)
        self.parquet_df = self.parquet_df.iloc[
            self.parquet_df['annotation'].apply(
                lambda x: x.get('token_length', float('-inf')) if isinstance(x, dict) else float('-inf')
            ).argsort()[::-1]  # Sort in descending order
        ].reset_index(drop=True)

        self.parquet_df['annotation'] = self.parquet_df['annotation'].apply(
            lambda x: json.dumps({k: v for k, v in x.items() if k != "token_length"}) if isinstance(x, dict) else x
        )

        print(f"Dataframe shape after filtering: {self.parquet_df.shape}")
        # print(self.parquet_df['annotation'][0])
        '''
        
        if 'id' in self.parquet_df.columns and self.parquet_df['id'].is_unique:
            self.data_ratio = int(len(self.parquet_df['id'].tolist())*config['dataset']['data_ratio'])
            self.graph_ids = self.parquet_df['id'].tolist()[:self.data_ratio]
        else:
            self.data_ratio = int(len(self.parquet_df.index.tolist())*config['dataset']['data_ratio'])
            self.graph_ids = self.parquet_df.index.tolist()[:self.data_ratio]
        
        

    @staticmethod
    def calculate_token_length_from_dict(data: dict, processor: Pix2StructProcessor) -> int:
        """
        Convert a dictionary to a string, tokenize it using the Pix2Struct tokenizer,
        and calculate the token length.

        Args:
        - data (dict): The input dictionary (e.g., JSON object).
        - processor (Pix2StructProcessor): The Pix2Struct processor with tokenizer.

        Returns:
        - token_length (int): The number of tokens in the tokenized sequence.
        """
        data = json.loads(data)
        def tokenize_dict(d):
            """Recursively convert a nested dictionary to a string."""
            if isinstance(d, dict):
                result = ""
                for key, value in d.items():
                    start_token = f"<{key}>"
                    end_token = f"</{key}>"
                    value_string = tokenize_dict(value)
                    result += f"{start_token}{value_string}{end_token}"
                return result
            elif isinstance(d, list):  
                return ' '.join(tokenize_dict(item) for item in d)
            else:
                if type(d) == float:
                    return str(round(d, 2))
                return str(d)
            # Convert the dictionary to a string
        text = tokenize_dict(data)

        # Tokenize the string using the Pix2Struct tokenizer
        token_ids = processor.tokenizer.encode(text, add_special_tokens=True)
        # print(f"Tokenized text: {text}")
        # print(f"Token IDs: {token_ids}")
        # print(f"Token length: {len(token_ids)}")
        # Return the token length
        data['token_length'] = len(token_ids)
        return data


    
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

        p_img = self.processor(
            images=image,
            max_patches=self.config['model']['max_patches'],
            add_special_tokens=True,
        )
        
        p_txt = self.processor(
            text=text,
            truncation=False,   
            add_special_tokens=True,
            max_length=self.config['model']['max_length'],
        )

        return {
            'id': graph_id,
            'chart_type': chart_type,
            'image': image,
            'text': text,
            'flattened_patches': p_img['flattened_patches'],
            'attention_mask': p_img['attention_mask'],
            'decoder_input_ids': p_txt.get('decoder_input_ids', p_txt.get('input_ids', [])),
            'decoder_attention_mask': p_txt.get('decoder_attention_mask', p_txt.get('attention_mask', [])),
        }

def create_train_transforms(resize_height: int = None, resize_width: int = None) -> A.Compose:
    """Create albumentations transforms for training."""
    transforms = [A.Resize(height=resize_height, width=resize_width)] if resize_height and resize_width else []
    return A.Compose(transforms)
