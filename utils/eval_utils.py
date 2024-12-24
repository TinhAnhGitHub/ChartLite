
from omegaconf import OmegaConf
import torch
from transformers import GenerationConfig
from tqdm import tqdm

from .data_util import post_processing
from .constant import TOKEN_MAP
from .metric_utils import JSONParseEvaluator

def run_evaluation(
        cfg: OmegaConf, 
        model, 
        valid_dl, 
        tokenizer, 
):
    conf_g = {
        "max_new_tokens": cfg.model.max_length_generation,  
        "do_sample": False,
        "top_k": 1,
        "use_cache": True,
    }
    generation_config = GenerationConfig(**conf_g)

    all_ids = []
    all_texts = []
    label_dict = []
    progress_bar = tqdm(range(len(valid_dl)), desc='Running evaluation...')


    for batch in valid_dl:
        
        with torch.no_grad():
            batch_ids = batch["id"]
            
            generated_ids = model.backbone.generate(
                flattened_patches=batch['flattened_patches'],
                attention_mask=batch['attention_mask'],
                generation_config=generation_config,
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


            all_ids.extend(batch_ids)
            all_texts.extend(generated_texts)
            print(all_texts)
            label_dict.extend(batch['texts'])
        progress_bar.update(1)
    progress_bar.close()

    label_dicts = [
        post_processing(
            label_str,
            TOKEN_MAP,
        ) for label_str in label_dict
    ]

    # prepare output dataframe ---
    preds_dict = []
    for this_id, this_text in zip(all_ids, all_texts):
        pred_dictionary = post_processing(this_text, TOKEN_MAP)
        preds_dict.append((this_id,pred_dictionary))
        

    eval_JSON = JSONParseEvaluator()
    print(preds_dict)
    f1_score = eval_JSON.cal_f1(
        preds=preds_dict,
        answers = label_dicts
    )

    accuracy = sum([eval_JSON.cal_acc(
        pred=pred,
        answer=label
    ) for pred, label in zip(preds_dict, label_dicts)]) / len(preds_dict)

    return {
        'f1_score': f1_score,
        'accuracy': accuracy
    }
