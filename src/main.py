import os
import glob
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup, GenerationConfig
from torch.optim import AdamW

from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Timer, TQDMProgressBar

import warnings
import wandb
import yaml

import argparse
from  omegaconf import OmegaConf

from utils import TOKEN_MAP, JSONParseEvaluator, AverageMeter, EMA, AWPCallback
from data import ChartCollator, ChartDataset
from models import Matcha
warnings.filterwarnings("ignore")
BOS_TOKEN = TOKEN_MAP["bos_token"]
torch.set_float32_matmul_precision('high')

class ChartDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.val_size = None
    
    def setup(self, stage=None):
        directory = self.config.dataset.parquet_dict
        train_files = glob.glob(os.path.join(directory, "train*.parquet"))
        valid_files = glob.glob(os.path.join(directory, "validation*.parquet"))


        temp_dataset = ChartDataset(self.config, train_files)
        self.processor = temp_dataset.processor
        self.tokenizer = self.processor.tokenizer
        self.config.model.len_tokenizer = len(self.tokenizer)  
        self.config.model.pad_token_id = self.tokenizer.pad_token_id  
        self.config.model.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]  
        self.config.model.bos_token_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]  

        

        self.train_dataset = ChartDataset(self.config, train_files)
        self.val_dataset = ChartDataset(self.config, valid_files)
        train_size = int(len(self.train_dataset) * self.config.dataset.data_ratio )
        val_size = int(len(self.val_dataset) * self.config.dataset.data_ratio)
        if train_size != len(self.train_dataset):
            self.train_dataset, _ = torch.utils.data.random_split(self.train_dataset, [train_size, len(self.train_dataset) - train_size])
            self.val_dataset, _ = torch.utils.data.random_split(self.val_dataset, [val_size, len(self.val_dataset) - val_size])
            self.val_size = val_size



    def train_dataloader(self):
        collate_fn = ChartCollator(self.tokenizer)

        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.trainer.world_size,  
            rank=self.trainer.global_rank,         
            shuffle=True                           
        ) if self.trainer.num_devices > 1 or self.trainer.num_nodes > 1 else None

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_params.train_bs,
            collate_fn=collate_fn,
            num_workers=self.config.train_params.num_workers,
            pin_memory=True,
            shuffle=(train_sampler is None),       
            sampler=train_sampler,                 
            persistent_workers=True
        )

    def val_dataloader(self):
        collate_fn = ChartCollator(self.tokenizer)
        val_sampler = DistributedSampler(
            self.val_dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False                          
        ) if self.trainer.num_devices > 1 or self.trainer.num_nodes > 1 else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train_params.val_bs,
            collate_fn=collate_fn,
            num_workers=self.config.train_params.num_workers,
            pin_memory=True,
            shuffle=False,                         
            sampler=val_sampler,                  
            persistent_workers=True
        )

class MatchaLightningModule(LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        
        self.config = config
        self.tokenizer = tokenizer
        self.model = Matcha(config)

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if eos_token_id is None or pad_token_id is None:
            raise ValueError("EOS or PAD token ID is None. Ensure tokenizer has them defined.")


        self.train_metrics = {
            'loss': AverageMeter(),
        }
        self.val_metrics = {
            "loss": AverageMeter(),
        }
        self.validation_outputs = []
        self.use_wandb = config.wandb.enabled
        self.eval_json = JSONParseEvaluator()
        self.save_hyperparameters()
    
    def forward(self, flattened_patches, attention_mask, labels=None):
        loss, outputs = self.model(flattened_patches, attention_mask, labels)
        return loss, outputs
    
    def training_step(self, batch, batch_idx):
        loss, _ = self(
            flattened_patches=batch["flattened_patches"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss_item = loss.item()
        self.train_metrics['loss'].update(loss_item, 1)
        
        self.log('train/loss_step', loss_item, on_step=True, prog_bar=True)
   

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/learning_rate', current_lr, on_step=True, prog_bar=True)
        if self.use_wandb:
            wandb.log({
                'train/loss_step': loss_item,
                'learning_rate': current_lr,
            }, step=self.global_step)
        
        return loss
        
    def on_train_epoch_end(self):
        epoch_loss = self.train_metrics['loss'].avg
        self.log('train/epoch_loss', epoch_loss, on_epoch=True)

        if self.use_wandb:
            wandb.log({
                'train/epoch_loss': epoch_loss,
                'epoch': self.current_epoch
            })

        
        if self.global_rank == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.print(f"[Train End Rank {self.global_rank}] Epoch {self.current_epoch} - Avg Loss: {epoch_loss:.4f} - Learning rate: {current_lr:.6f}")
        self.train_metrics['loss'].reset() 

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, _ = self(
                flattened_patches=batch["flattened_patches"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            self.val_metrics["loss"].update(loss.item(), 1)
        self.log('val/loss_step', loss.item(), on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        for metric in self.val_metrics.values():
            metric.reset()
        self.validation_outputs = []


    def on_validation_epoch_end(self):
        current_step = self.global_step
        val_loss_avg = self.val_metrics["loss"].avg

        self.log('val/loss_avg', val_loss_avg, on_epoch=True, prog_bar=True)
        
        if self.use_wandb:
            wandb.log({
                "val/loss_avg": val_loss_avg,
            }, step=current_step)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr = float(self.config.optimizer.lr),
            weight_decay= float(self.config.optimizer.weight_decay)
        )
        max_steps = self.config.train_params.max_steps
        num_warmup_steps = int(self.config.learning_rate_scheduler.warmup_pct * max_steps)
        num_cycles = self.config.learning_rate_scheduler.num_cycles

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_steps,
            num_cycles=num_cycles
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def run_training(cfg, ckpt_path=None):
    seed_everything(cfg.general.seed)
    checkpoint_dir = os.path.join(cfg.outputs.model_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'best_ckpts'), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'last_ckpts'), exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.best_ckpt.monitor,
            mode=cfg.best_ckpt.mode,
            save_top_k=cfg.best_ckpt.save_top_k,
            filename="best-checkpoint-{step}-{val/loss_avg}",
            dirpath=os.path.join(checkpoint_dir, 'best_ckpts'),
        ),
        ModelCheckpoint(
            save_last=True,
            filename="last-checkpoint-{step}-{val/loss_avg}",
            every_n_train_steps=cfg.train_params.save_every_n_train_steps,
            every_n_epochs=None,
            dirpath=os.path.join(checkpoint_dir, 'last_ckpts')
        ),
        EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            check_on_train_epoch_end=False
        ),
        LearningRateMonitor(logging_interval='step'),
        Timer(),
        TQDMProgressBar(refresh_rate=cfg.train_params.train_bs * cfg.train_params.grad_accumulation),
        
    ]

    if cfg.train_params.ema_enable:
        print("Use EMA")
        callbacks.append(
            EMA(
                decay=cfg.train_params.ema_decay,
                validate_original_weights=cfg.train_params.ema_validate_original_weights,
                every_n_steps=cfg.train_params.ema_every_n_steps,
                cpu_offload=cfg.train_params.ema_cpu_offload
            )
        )

    if cfg.train_params.awp_enable:
        print("Use AWP")
        callbacks.append(
            AWPCallback(
                adv_param='weight',
                adv_lr=float(cfg.train_params.awp_adv_lr),
                adv_eps=float(cfg.train_params.adv_eps),
                apply_every=int(cfg.train_params.apply_every)
            )
        )
    print("Callbacks being passed to Trainer:", [type(cb).__name__ for cb in callbacks])
    data_module = ChartDataModule(cfg)
    data_module.setup()
    tokenizer = data_module.tokenizer
    cfg.val_size = data_module.val_size
    model = MatchaLightningModule(cfg, tokenizer)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity='matcha_hehe'
        )
    
    trainer = Trainer(
        max_steps=cfg.train_params.max_steps,
        max_epochs=None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        gradient_clip_val=cfg.optimizer.grad_clip_value,
        accumulate_grad_batches=cfg.train_params.grad_accumulation,
        precision= '16-mixed' if cfg.train_params.use_fp16_mixed else '32',
        val_check_interval=cfg.train_params.val_check_interval,
        fast_dev_run=cfg.general.fast_dev_run,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        logger=True,
        
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    config = OmegaConf.create(yaml_config)

    run_training(cfg=config, ckpt_path=args.checkpoint)
