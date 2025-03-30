import os
import glob
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup, GenerationConfig
from torch.optim import AdamW
# from transformers.optimization import Adafactor

from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Timer, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import warnings
import wandb
import yaml

import argparse
from  omegaconf import OmegaConf

from utils import TOKEN_MAP, JSONParseEvaluator, post_processing, AverageMeter, EMA, AWPCallback
from data import ChartCollator, ChartDataset
from models import Matcha
warnings.filterwarnings("ignore")
BOS_TOKEN = TOKEN_MAP["bos_token"]
# torch.set_float32_matmul_precision('medium')

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

        
        self.generation_config = GenerationConfig(
            max_new_tokens=config.model.max_length_generation,
            num_beams=4,
            do_sample=False,
            top_k=1,
            use_cache=True,
            early_stopping=True,
            eos_token_id=eos_token_id, 
            pad_token_id=pad_token_id
        )


        self.train_metrics = {
            'loss': AverageMeter(),
            'f1': AverageMeter(),
            'accuracy': AverageMeter(),
            'overall_sim': AverageMeter()
        }
        self.val_metrics = {
            "loss": AverageMeter(),
            "f1": AverageMeter(),
            "accuracy": AverageMeter(),
            "overall_sim": AverageMeter()
        }
        self.validation_outputs = []
        
        self.use_wandb = config.wandb.enabled
        self.eval_json = JSONParseEvaluator()
        self.save_hyperparameters()
    
    def forward(self, flattened_patches, attention_mask, labels=None):
        loss, outputs = self.model(flattened_patches, attention_mask, labels)
        return loss, outputs
    
    def training_step(self, batch, batch_idx):
        self.current_batch = batch
        loss, _ = self(
            flattened_patches=batch["flattened_patches"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        self.train_metrics['loss'].update(loss,1)
        
        current_step = self.global_step

        self.log('train/loss_step', loss, on_step=True, prog_bar=True)
        self.log('train/learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)

        if self.use_wandb:
            wandb.log({
                'train/loss_step': loss.item(),
                'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
            }, step=current_step)
        
        if batch_idx % 50 == 0 and self.global_rank == 0:
            self.print(f"[Train Rank {self.global_rank}] Step {current_step} - Loss: {loss:.4f}")
        
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
        self.train_metrics['f1'].reset()
        self.train_metrics['accuracy'].reset()
        self.train_metrics['overall_sim'].reset()
        
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, _ = self(
                flattened_patches=batch["flattened_patches"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            self.val_metrics["loss"].update(loss.item(), 1)

            generated_ids = self.model.backbone.generate(
                flattened_patches=batch["flattened_patches"],
                attention_mask=batch["attention_mask"],
                generation_config=self.generation_config,
            )

            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            label_texts = batch["texts"][0]

            label_dicts = [post_processing(label_texts, TOKEN_MAP)]
            preds = [(batch['id'], post_processing(generated_texts, TOKEN_MAP))]

            output_str = (
                f"{'*'*50}\n"
                f"During validation step {batch_idx}\n"
                f"Label dict: {label_dicts}\n"
                f"Generated text: {generated_texts}\n"
                f"Predictions: {preds}\n"
                f"{'*'*50}\n"
            )
            self.validation_outputs.append(output_str)

            f1_score = self.eval_json.cal_f1(preds=preds, answers=label_dicts)
            accuracy = self.eval_json.cal_acc(pred=preds[0], answer=label_dicts[0])

            overall_sim = self.eval_json.compare_json_list(
                label_dicts, preds,
                numeric_tolerance=self.config.metrics_tolerance.numeric_tolerance,
                string_tolerance=self.config.metrics_tolerance.string_tolerance
            )

            self.val_metrics["f1"].update(f1_score, 1)
            self.val_metrics["accuracy"].update(accuracy, 1)
            self.val_metrics["overall_sim"].update(overall_sim, 1)

        self.log('val/loss_step', loss.item(), on_step=True, prog_bar=True)
        self.log('val/f1_step', f1_score, on_step=True)
        self.log('val/accuracy_step', accuracy, on_step=True)
        self.log('val/overall_sim_step', overall_sim, on_step=True)
        return loss

    def on_validation_epoch_start(self):
        for metric in self.val_metrics.values():
            metric.reset()
        self.validation_outputs = []


    def on_validation_epoch_end(self):
        current_step = self.global_step
        val_loss_avg = self.val_metrics["loss"].avg
        f1_avg = self.val_metrics["f1"].avg
        accuracy_avg = self.val_metrics["accuracy"].avg
        overall_sim_avg = self.val_metrics["overall_sim"].avg

        self.log('val/loss_avg', val_loss_avg, on_epoch=True, prog_bar=True)
        self.log('val/f1_avg', f1_avg, on_epoch=True, prog_bar=True)
        self.log('val/accuracy_avg', accuracy_avg, on_epoch=True)
        self.log('val/overall_sim_avg', overall_sim_avg, on_epoch=True) 
        
        if self.use_wandb:
            wandb.log({
                "val/loss_avg": val_loss_avg,
                "val/f1_avg": f1_avg,
                "val/accuracy_avg": accuracy_avg,
                "val/overall_sim_avg": overall_sim_avg,
            }, step=current_step)

        if self.global_rank == 0:
            self.print(f"[Val End Rank {self.global_rank}] Step {current_step} - Loss: {val_loss_avg:.4f}, F1: {f1_avg:.4f}, Acc: {accuracy_avg:.4f}")

            log_file_path = os.path.join(os.getcwd(), f"validation_step_{current_step}_outputs.txt")
            try:
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"--- Validation outputs for Step {current_step} ---\n")
                    f.write(f"Avg Loss: {val_loss_avg:.4f}, Avg F1: {f1_avg:.4f}, Avg Acc: {accuracy_avg:.4f}, Avg Sim: {overall_sim_avg:.4f}\n")
                    f.write("="*50 + "\n\n")
                    for output in self.validation_outputs:
                        f.write(output + "\n")
                self.print(f"Validation outputs written to {log_file_path}")
            except Exception as e:
                self.print(f"Error writing validation outputs to file: {e}")
                    

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


def run_training(cfg):

    # os.environ["MASTER_ADDR"] = "11.84.11.29"
    # os.environ["MASTER_PORT"] = "53154"
    # os.environ['NODE_RANK'] = "0"
    # os.environ['LOCAL_RANK'] = "0"
    # os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET"
    
    # vpn_interface = cfg.vpn.name  # name for WireGuard interfaces
    # os.environ["NCCL_SOCKET_IFNAME"] = vpn_interface


    seed_everything(cfg.general.seed)
    checkpoint_dir = os.path.join(cfg.outputs.model_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.best_ckpt.monitor,
            mode=cfg.best_ckpt.mode,
            save_top_k=cfg.best_ckpt.save_top_k,
            filename="best-checkpoint-{step}-{val_f1_avg:.4f}",
            dirpath=os.path.join(checkpoint_dir, 'best_ckpts'),
        ),
        ModelCheckpoint(
            save_last=True,
            filename="last-checkpoint-{step}",
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
        # num_nodes=cfg.vpn.nodes,
        gradient_clip_val=cfg.optimizer.grad_clip_value,
        accumulate_grad_batches=cfg.train_params.grad_accumulation,
        precision= '16' if cfg.train_params.use_fp16_mixed else '32',
        check_val_every_n_epoch=None,
        val_check_interval=cfg.train_params.val_check_interval,
        fast_dev_run=cfg.general.fast_dev_run,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        logger=True,
        
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