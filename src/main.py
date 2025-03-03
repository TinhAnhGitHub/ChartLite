import os
import glob
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, GenerationConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Timer, StochasticWeightAveraging, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import yaml
import argparse
import OmegaConf

from utils import TOKEN_MAP, JSONParseEvaluator, post_processing, AverageMeter
from data import ChartCollator, ChartDataset
from models import Matcha

BOS_TOKEN = TOKEN_MAP["bos_token"]

class ChartDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        directory = self.config.dataset.parquet_dict
        train_files = glob.glob(os.path.join(directory, "train*.parquet"))
        valid_files = glob.glob(os.path.join(directory, "validation*.parquet"))

        temp_dataset = ChartDataset(self.config, train_files)
        self.processor = temp_dataset.processor
        self.tokenizer = self.processor.tokenizer

        if valid_files:
            self.train_dataset = ChartDataset(self.config, train_files)
            self.val_dataset = ChartDataset(self.config, valid_files)
        else:
            full_dataset = ChartDataset(self.config, train_files)
            val_size = int(len(full_dataset) * (1 - self.config.dataset.percent_to_take_in_train))
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )  
    
    def train_dataloader(self):
        collate_fn = ChartCollator(tokenizer=self.tokenizer)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_params.train_bs,
            collate_fn=collate_fn,
            num_workers=self.config.train_params.num_workers,
            pin_memory=True,
            shuffle=True 
        )

    def val_dataloader(self):
        collate_fn = ChartCollator(tokenizer=self.tokenizer)
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train_params.val_bs,
            collate_fn=collate_fn,
            num_workers=self.config.train_params.num_workers,
            pin_memory=True,
            shuffle=False 
        )

class MatchaLightningModule(LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = Matcha(config)
        
        self.generation_config = GenerationConfig(
            max_new_tokens=config.model.max_length_generation,
            do_sample=False,
            top_k=1,
            use_cache=True
        )


        self.train_metrics = AverageMeter()
       

        

        self.save_hyperparameters()
    
    def forward(self, flattened_patches, attention_mask, labels=None):
        return self.model(flattened_patches, attention_mask, labels)
    
    def training_step(self, batch, batch_idx):
        loss = self(
            flattened_patches=batch["flattened_patches"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        self.train_metrics.update(loss,1)
        loss_avg = self.train_metrics.avg

        self.log('train/loss_step', loss, on_step=True, prog_bar=True)
        self.log('train/loss_avg', loss_avg, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()
        
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            generated_ids = self.model.backbone.generate(
                flattened_patches=batch["flattened_patches"],
                attention_mask=batch["attention_mask"],
                generation_config=self.generation_config,
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = batch["texts"]

        return {
            "ids": batch["id"],
            "generated_texts": generated_texts,
            "label_texts": label_texts,
        }
    
    def validation_epoch_end(self, outputs):
        all_ids = []
        all_generated_texts = []
        all_label_texts = []

        for output in outputs:
            all_ids.extend(output["ids"])
            all_generated_texts.extend(output["generated_texts"])
            all_label_texts.extend(output["label_texts"])
        
        label_dicts = [post_processing(label_str, TOKEN_MAP) for label_str in all_label_texts]
        preds_dict = [(this_id, post_processing(this_text, TOKEN_MAP)) 
                      for this_id, this_text in zip(all_ids, all_generated_texts)]

    
        
        eval_json = JSONParseEvaluator()

        f1_score = eval_json.cal_f1(preds=preds_dict, answers=label_dicts)
        accuracy = sum([eval_json.cal_acc(pred=pred, answer=label) 
                       for pred, label in zip(preds_dict, label_dicts)]) / len(preds_dict)
    
        overall_sim_average = eval_json.compare_json_list(
            label_dicts, 
            preds_dict,
            numeric_tolerance=self.config.metrics_tolerance.numeric_tolerance,
            string_tolerance=self.config.metrics_tolerance.string_tolerance
        )
        
        self.log("val_f1", f1_score, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        self.log("val_overall_sim", overall_sim_average["mean_overall_metric"], prog_bar=True)
        self.log("val_average_sim", overall_sim_average["mean_average_metric"], prog_bar=True)


    

    def configure_optimizers(self):
        optimizer = torch.optim.Adafactor(
            self.model.parameters(),
            scale_parameter=False,
            relative_step=False,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )

        estimated_steps_per_epoch = self.config.train_params.estimated_steps_per_epoch

        num_update_steps_per_epoch = estimated_steps_per_epoch // self.config.train_params.grad_accumulation

        num_training_steps = self.config.train_params.num_epochs * num_update_steps_per_epoch

        num_warmup_steps = int(self.config.learning_rate_scheduler.warmup_pct * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def run_training(cfg):
    seed_everything(cfg.general.seed)

    

    data_module = ChartDataModule(cfg)
    data_module.setup()
    tokenizer = data_module.tokenizer
    model = MatchaLightningModule(cfg, tokenizer)

    logger = WandbLogger(project=cfg.wandb.project) if cfg.wandb.use else None

    swa_callback = StochasticWeightAveraging(
        swa_epoch_start=cfg.training.get("swa_epoch_start", 0.5),   
        swa_lrs=cfg.training.get("swa_lrs", 0.05),
        annealing_epochs=cfg.training.get("swa_annealing_epochs", 10),
        annealing_strategy=cfg.training.get("swa_annealing_strategy", "cos")
    )

    checkpoint_dir = os.path.join(cfg.outputs.model_dir, "checkpoints")

    callbacks = [
        ModelCheckpoint(
            monitor=cfg.best_ckpt.monitor,
            mode=cfg.best_ckpt.mode,
            save_top_k=cfg.best_ckpt.save_top_k,
            filename="best-checkpoint-{epoch:02d}-{val_f1:.4f}",
            dirpath=os.path.join(cfg.checkpoint_dir, 'best_ckpts')
        ),
        ModelCheckpoint(
            save_last=True,
            filename="last-checkpoint",
            dirpath=os.path.join(cfg.checkpoint_dir, 'last_ckpts')
        ),  
        EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode = cfg.early_stopping.mode
        ),
        LearningRateMonitor(logging_interval='step'),
        Timer(),
        swa_callback,
        TQDMProgressBar(refresh_rate=20)

    ]
    
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.optimizer.grad_clip_value,
        accumulate_grad_batches=cfg.training.grad_accumulation,
        precision=16 if cfg.training.use_fp16 else 32,
        check_val_every_n_epoch=1,
        fast_dev_run=cfg.general.fast_dev_run
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    config = OmegaConf.create(yaml_config)

    run_training(cfg=config)