import numpy as np
import torch
from typing import Callable, Any, Dict, List, Optional

class ChartCollator:
    """
    Data collator that uses the model's own preprocess function to prepare inputs.
    Handles both encoder patches and decoder token sequences.
    """
    def __init__(
        self,
        preprocess_fn: Callable[[Any, Optional[str]], Dict[str, torch.Tensor]],
        tokenizer: Any,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt"
    ):
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        processed = [
            self.preprocess_fn(image=f["image"], text=f.get("text", None))
            for f in features
        ]

        batch: Dict[str, torch.Tensor] = {}
        for key in processed[0].keys():
            vals = [p[key] for p in processed]
            if isinstance(vals[0], torch.Tensor):
                batch[key] = torch.stack(vals, dim=0)
            elif isinstance(vals[0], np.ndarray):
                batch[key] = torch.tensor(np.stack(vals, axis=0))
            else:
                if key !='image' and key !="text":
                    batch[key]=torch.tensor(vals, dtype=torch.int64)

        if "decoder_input_ids" in batch:
            pad_inputs = {
                "input_ids": batch["decoder_input_ids"],
                "attention_mask": batch.get("decoder_attention_mask")
            }
            decoder_batch = self.tokenizer.pad(
                pad_inputs,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
            batch["decoder_input_ids"] = decoder_batch["input_ids"]
            batch["decoder_attention_mask"] = decoder_batch["attention_mask"]
            labels = decoder_batch["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels


        return batch
