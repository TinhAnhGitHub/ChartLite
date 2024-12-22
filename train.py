import os
import json
import time
import glob
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Tuple, Any
import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import  get_cosine_schedule_with_warmup
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random

# Import custom modules
from utils import TOKEN_MAP, JSONParseEvaluator, EMA, AverageMeter, init_wandb, setup, cleanup_processes, save_checkpoint, seed_everything, print_gpu_utilization, run_evaluation, as_minutes, print_line
from data import ChartCollator, ChartDataset
from model import Matcha, AWP

BOS_TOKEN = TOKEN_MAP["bos_token"]


class Logger:
    """
    Handles Logging setup and operations
    """
    def __init__(self, log_dir: str, config: OmegaConf, str='logs'):
        self.logger = self.setup_logging(log_dir)
        if config.use_wandb:  
            cfg_dict = OmegaConf.to_container(config, resolve=True)  
            init_wandb(cfg_dict)  
    
    def setup_logging(self, log_dir: str) -> logging.Logger:
        """Sets up the logging."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        logger = logging.getLogger("training_logger")
        logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def log(self, message: str, level:int = logging.DEBUG):
        print(message)
        self.logger.log(level, message)



class Trainer:  
    """Handles the training process, including setup, training, evaluation, and checkpointing."""  

    def __init__(self, config: OmegaConf, rank: int, world_size: int) -> None:  
        self.config = config  
        self.rank = rank  
        self.world_size = world_size  
        self.logger = Logger(log_dir="logs", config=config).log  
        self.accelerator = Accelerator(mixed_precision="fp16", device_placement=True)  
        self.tokenizer = None  
        self.awp = None  
        self.awp_flag = False  
        self.model = None  
        self.optimizer = None  
        self.scheduler = None  
        self.ema = None  
        self.train_dl = None  
        self.valid_dl = None  
        self.train_sampler = None  
        self.best_f1 = 0  
        self.best_accuracy = 0  
        self.patience_tracker = 0  
        self.current_iteration = 0  

    def setup(self):  
        """Initial setup: seeds, environment, and logging."""  
        setup(self.rank, self.world_size)  
        seed_everything(self.config.seed)  
        self.logger("Environment setup complete.", logging.DEBUG)  

    def load_data(self):  
        """Loads training and validation datasets."""  
        directory = self.config.dataset.parquet_dict  
        train_files = glob.glob(os.path.join(directory, "train*.parquet"))  
        valid_files = glob.glob(os.path.join(directory, "validation*.parquet"))  

        train_dataset = ChartDataset(self.config, train_files)  
        valid_dataset = ChartDataset(self.config, valid_files)  

        if len(valid_dataset) == 0:
            valid_dataset = ChartDataset(self.config, train_files[random.randint(0,len(train_files))],self.config.dataset.percent_to_take_in_train)
        self.logger(f"Train dataset size: {len(train_dataset)}, Valid dataset size: {len(valid_dataset)}")  

        self.tokenizer = train_dataset.processor.tokenizer  
        self.config.model.len_tokenizer = len(self.tokenizer)  
        self.config.model.pad_token_id = self.tokenizer.pad_token_id  
        self.config.model.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]  
        self.config.model.bos_token_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]  

        collate_fn = ChartCollator(tokenizer=self.tokenizer)  

        if self.world_size > 1:  
            self.train_sampler = DistributedSampler(  
                train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True  
            )  
            self.train_dl = DataLoader(  
                train_dataset,  
                batch_size=self.config.train_params.train_bs,  
                collate_fn=collate_fn,  
                sampler=self.train_sampler,  
                num_workers=self.config.train_params.num_workers,  
                pin_memory=True,  
            )  
        else:  
            self.train_dl = DataLoader(  
                train_dataset,  
                batch_size=self.config.train_params.train_bs,  
                collate_fn=collate_fn,  
                num_workers=self.config.train_params.num_workers,  
                pin_memory=True,  
            )  

        self.valid_dl = DataLoader(  
            valid_dataset,  
            batch_size=self.config.train_params.valid_bs,  
            collate_fn=collate_fn,  
            shuffle=False,  
            num_workers=self.config.train_params.num_workers,  
            pin_memory=True,  
        )  

    def initialize_model(self):  
        """Initializes the model, optimizer, scheduler, AWP, and EMA."""  
        self.logger("Creating Matcha Model")  
        self.model = Matcha(self.config)  
        self.model = self.model.cuda(self.rank)  

        if self.world_size > 1:  
            self.model = DDP(self.model, device_ids=[self.rank])  

        self.logger("Setting up Optimizer", level=logging.DEBUG)  
        self.optimizer = torch.optim.AdamW(  
            self.model.parameters(),  
            lr=self.config.optimizer.lr,  
            weight_decay=self.config.optimizer.weight_decay,  
        )  

        num_update_steps_per_epoch = len(self.train_dl) // self.config.train_params.grad_accumulation  
        num_training_steps = self.config.train_params.num_epochs * num_update_steps_per_epoch  
        num_warmup_steps = int(self.config.train_params.warmup_pct * num_training_steps)  

        self.scheduler = get_cosine_schedule_with_warmup(  
            self.optimizer, num_warmup_steps, num_training_steps  
        )  

        if self.config.awp.use_awp:  
            self.awp = AWP(  
                self.model,  
                self.optimizer,  
                adv_lr=self.config.awp.adv_lr,  
                adv_eps=self.config.awp.adv_eps,  
            )  
            self.awp_flag = True  
            self.logger(f"Setting up AWP with adv_lr: {self.config.awp.adv_lr} and adv_eps: {self.config.awp.adv_eps}")  

        if self.config.train_params.use_ema:  
            self.ema = EMA(self.model, decay=self.config.train_params.decay_rate)  
            self.logger(f"Setting up EMA with decay: {self.config.train_params.decay_rate}")  
            self.ema.register()  

        self.model, self.optimizer, self.train_dl, self.valid_dl = self.accelerator.prepare(  
            self.model, self.optimizer, self.train_dl, self.valid_dl  
        )  

    def evaluate(self) -> Dict[str, float]:  
        """Runs evaluation on the validation dataset."""  
        self.model.eval()  
        if self.ema:  
            self.ema.apply_shadow()  

        f1_and_acc = run_evaluation(self.config, self.model, self.valid_dl, self.tokenizer)  

        if self.ema:  
            self.ema.restore()  

        return f1_and_acc  

    def train_one_epoch(self, epoch: int):  
        """Trains the model for one epoch."""  
        if self.awp_flag and epoch >= self.config.awp.awp_trigger_epoch:  
            self.logger("AWP is triggered...", logging.INFO)  

        if self.world_size > 1:  
            self.train_sampler.set_epoch(epoch)  

        progress_bar = tqdm(range(len(self.train_dl)), desc=f"Epoch {epoch + 1}/{self.config.train_params.num_epochs}")  
        loss_meter = AverageMeter()  

        self.model.train()  

        for step, batch in enumerate(self.train_dl):  
            loss, _ = self.model(  
                flattened_patches=batch["flattened_patches"],  
                attention_mask=batch["attention_mask"],  
                labels=batch["labels"],  
            )  
            self.accelerator.backward(loss)  

            if self.awp_flag:  
                self.awp.attack_backward(batch, self.accelerator)  
            if (step + 1) % self.config.train_params.validation_per_step ==0:
                self.logger("Running evaluation...", logging.INFO)  
                f1_and_acc = self.evaluate()  



                f1 = f1_and_acc["f1_score"]  
                acc = f1_and_acc["accuracy"]  
                self.logger(f"Evaluation - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}", logging.INFO)                  

            if (step + 1) % self.config.train_params.save_checkpoint_per_step ==0 :
                self.save_checkpoint_eval_step(step, f1, acc)  
    
            if (step + 1) % self.config.train_params.print_gpu_stats_each_steps ==0 :
                print_gpu_utilization()

            if (step + 1) % self.config.train_params.grad_accumulation == 0:  
                self.accelerator.clip_grad_norm_(  
                    self.model.parameters(), self.config.optimizer.grad_clip_value  
                )  
                self.optimizer.step()  
                self.scheduler.step()  
                self.optimizer.zero_grad()  
                loss_meter.update(loss.item())  

                if self.ema:  
                    self.ema.update()  

                progress_bar.set_description(  
                    f"Loss: {loss_meter.avg:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}, Time Duration: {as_minutes(time.time() - self.start_time)}"  
                )  
                progress_bar.update(1)  

                if self.config.use_wandb:  
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=self.current_iteration)  
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]}, step=self.current_iteration)  

        progress_bar.close()  
        self.logger(f"End of epoch {epoch + 1}: Average Loss: {loss_meter.avg:.4f} | Time Duration: {time.time() - self.start_time}", logging.INFO)  

    def save_checkpoint_eval(self, epoch: int, f1: float, acc: float):  
        """Saves a checkpoint if performance improves."""  
        checkpoint_name = f"checkpoint_epoch{epoch + 1}_f1{f1:.4f}_acc{acc:.4f}.pt"  
        save_checkpoint(  
            self.config,  
            {  
                "step": self.current_iteration,  
                "epoch": epoch + 1,  
                "state_dict": self.model.state_dict(),  
            },  
            checkpoint_name,  
        )  
        self.logger(f"Checkpoint saved: {checkpoint_name}", logging.INFO)  
    def save_checkpoint_eval_step(self, step: int, f1: float, acc: float):  
        """Saves a checkpoint if performance improves."""  
        checkpoint_name = f"checkpoint_epoch{step + 1}_f1{f1:.4f}_acc{acc:.4f}.pt"  
        save_checkpoint(  
            self.config,  
            {  
                "step": self.current_iteration,  
                "step": step + 1,  
                "state_dict": self.model.state_dict(),  
            },  
            checkpoint_name,  
        )  
        self.logger(f"Checkpoint saved: {checkpoint_name}", logging.INFO)  

    def train(self):  
        """Main training loop."""  
        self.setup()  
        self.load_data()  
        self.initialize_model() 

        self.start_time = time.time() 

        for epoch in range(self.config.train_params.num_epochs):  
            self.train_one_epoch(epoch)  

            if self.config.train_params.validation_per_epoch:
                if (epoch + 1) % self.config.train_params.validation_per_epoch == 0:  
                    self.logger("Running evaluation...", logging.INFO)  
                    f1_and_acc = self.evaluate()  



                    f1 = f1_and_acc["f1_score"]  
                    acc = f1_and_acc["accuracy"]  
                    self.logger(f"Evaluation - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}", logging.INFO)                  

                    if self.config.early_stopping_enable:
                        if f1 > self.best_f1 + 0.001 or acc > self.best_accuracy + 0.001:  
                            self.best_f1, self.best_accuracy = max(self.best_f1, f1), max(self.best_accuracy, acc)  
                            self.patience_tracker = 0  
                            self.save_checkpoint_eval(epoch, f1, acc)  
                        else:  
                            self.patience_tracker += 1  

                        if self.patience_tracker >= self.config.train_params.patience:  
                            self.logger("Early stopping triggered.", logging.INFO)  
                            break  
            if self.config.train_params.save_checkpoint_per_epoch:
                if (epoch + 1) % self.config.train_params.save_checkpoint_per_epoch ==0:
                    self.save_checkpoint_eval(epoch, f1, acc)  

        self.logger("Training complete.", logging.INFO)  
            

def train_process(rank: int, world_size: int, config: OmegaConf):  
    """  
    Function to be executed by each process in the DDP setup.  
    Initializes the Trainer class and starts training.  
    """  
    try:  
        trainer = Trainer(config=config, rank=rank, world_size=world_size)  
        trainer.train()  
    except Exception as e:  
        print(f"Error in process {rank}: {e}")  
        cleanup_processes()  
        raise e  
def main_ddp(world_size: int, config: OmegaConf):  
    """  
    Main function for distributed data parallel (DDP) training.  
    Spawns processes for each GPU and initializes the Trainer class.  
    """  


    if world_size > 1:  
        mp.spawn(  
            train_process,  
            args=(world_size, config),  
            nprocs=world_size,  
            join=True  
        )  
    else:  
        train_process(rank=0, world_size=1, config=config)  
    cleanup_processes()  
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run_training(cfg):
    world_size = torch.cuda.device_count()  #
    main_ddp(world_size, cfg)

if __name__ == "__main__":
    run_training()      


# def run_train_ddp(rank, world_size, config):  
#     # ---------- Setting up ------------  
#     setup(rank, world_size)  
#     seed_everything(config['seed'])  
#     if config.use_wandb:  
#         cfg_dict = OmegaConf.to_container(config, resolve=True)  
#         init_wandb(cfg_dict)  
#     global logger  
#     logger = setup_logging()  
#     print_and_log("Starting training process")  

#     # ------- Data Collators --------------------------------------------------------------#  
#     directory = config.dataset_config  
#     train_files = glob.glob(os.path.join(directory, "train*.parquet"))  
#     train_dataset = ChartDataset(config, train_files)  

#     valid_files = glob.glob(os.path.join(directory, "validation*.parquet"))  
#     valid_dataset = ChartDataset(config, valid_files)  

#     print_and_log(f"Train dataset size: {len(train_dataset)}, Valid dataset size: {len(valid_dataset)}", logging.DEBUG)  

#     tokenizer = train_dataset.processor.tokenizer  
#     config.model.len_tokenizer = len(tokenizer)  
#     config.model.pad_token_id = tokenizer.pad_token_id  
#     config.model.decoder_start_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]  
#     config.model.bos_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]  

#     collate_fn = ChartCollator(tokenizer=tokenizer)  

#     if world_size > 1:  
#         train_sampler = DistributedSampler(  
#             dataset=train_dataset,  
#             num_replicas=world_size,  
#             rank=rank,  
#             shuffle=True,  
#         )  

#         train_dl = DataLoader(  
#             train_dataset,  
#             batch_size=config.train_params.train_bs, collate_fn=collate_fn,  
#             num_workers=config.train_params.num_workers, pin_memory=True,  
#             sampler=train_sampler  
#         )  

#     else:  
#         train_dl = DataLoader(  
#             train_dataset,  
#             batch_size=config.train_params.train_bs, collate_fn=collate_fn,  
#             num_workers=config.train_params.num_workers, pin_memory=True,  
#         )  

#     valid_dl = DataLoader(  
#         valid_dataset,  
#         batch_size=config.train_params.valid_bs, collate_fn=collate_fn,  
#         shuffle=False, pin_memory=True,  
#         num_workers=config.train_params.num_workers,  
#     )  

#     print_and_log("Setting up dataloader successfully!", level=logging.DEBUG)  

#     # ------- Model Setup --------------------------------------------------------------#  
#     print_and_log("Creating Matcha Model", level=logging.DEBUG)  
#     model = Matcha(config)  
#     model = model.cuda(rank)  

#     if world_size > 1:  
#         model = DDP(model, device_ids=[rank])  

#     print_and_log("Setting up Optimizer", level=logging.DEBUG)  
#     optimizer = torch.optim.AdamW(  
#         model.parameters(),  
#         lr=config.optimizer.lr,  
#         weight_decay=config.optimizer.weight_decay  
#     )  

#     # Scheduler setup  
#     num_epochs = config.train_params.num_epochs  
#     grad_accumulation_steps = config.train_params.grad_accumulation  
#     warmup_pct = config.train_params.warmup_pct  

#     num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps  
#     num_training_steps = num_epochs * num_update_steps_per_epoch  
#     num_warmup_steps = int(warmup_pct * num_training_steps)  

#     scheduler = get_cosine_schedule_with_warmup(  
#         optimizer=optimizer,  
#         num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps  
#     )  

#     # AWP setup  
#     AWP_FLAG = False  
#     if config.awp.use_awp:  
#         awp = AWP(model, optimizer, adv_lr=config.awp.adv_lr, adv_eps=config.awp.adv_eps)  

#     # Accelerator setup  
#     accelerator = Accelerator(mixed_precision='bf16', device_placement=True)  
#     model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)  

#     # EMA setup  
#     if config.train_params.use_ema:  
#         ema = EMA(model, decay=config.train_params.decay_rate)  
#         ema.register()  

#     # ------- Training Loop --------------------------------------------------------------#  
#     best_f1 = 0  
#     best_accuracy = 0  
#     patience_tracker = 0  
#     min_delta = 0.001  
#     current_iteration = 0  

#     start_time = time.time()  

#     for epoch in range(num_epochs):  
#         if (config.awp.use_awp) & (epoch >= config.awp.awp_trigger_epoch):  
#             print("AWP is triggered...")  

#         if world_size > 1:  
#             train_sampler.set_epoch(epoch)  

#         progress_bar = tqdm(range(num_update_steps_per_epoch), desc=f"Epoch {epoch + 1}/{num_epochs}")  
#         loss_meter = AverageMeter()  

#         model.train()  

#         for step, batch in enumerate(train_dl):  
#             loss, loss_dict = model(  
#                 flattened_patches=batch["flattened_patches"],  
#                 attention_mask=batch["attention_mask"],  
#                 labels=batch["labels"],  
#             )  
#             accelerator.backward(loss)  

#             if AWP_FLAG:  
#                 awp.attack_backward(batch, accelerator)  

#             if (step + 1) % grad_accumulation_steps == 0:  
#                 accelerator.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_value)  
#                 optimizer.step()  
#                 scheduler.step()  
#                 optimizer.zero_grad()  
#                 loss_meter.update(loss.item())  

#                 if config.train_params.use_ema:  
#                     ema.update()  

#                 progress_bar.set_description(  
#                     f"Loss: {loss_meter.avg:.4f}, LR: {get_lr(optimizer):.6f}"  
#                 )  
#                 progress_bar.update(1)  
#                 current_iteration += 1  

#                 if config.use_wandb:  
#                     wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)  
#                     wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)  

#         print_and_log(f"End of epoch {epoch + 1}: Average Loss: {loss_meter.avg:.4f}", logging.INFO)  

#         # ------- Evaluation --------------------------------------------------------------#  
#         if (epoch + 1) % config.train_params.epoch_frequency == 0:  
#             print_and_log("Running evaluation...", logging.INFO)  
#             model.eval()  

#             if config.train_params.use_ema:  
#                 ema.apply_shadow()  

#             f1_and_acc = run_evaluation(  
#                 config,  
#                 model=model,  
#                 valid_dl=valid_dl,  
#                 tokenizer=tokenizer,  
#                 token_map=TOKEN_MAP  
#             )  

#             f1 = f1_and_acc['f1_score']  
#             acc = f1_and_acc['accuracy']  
#             print_and_log(f"Evaluation - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}", logging.INFO)  

#             # Save checkpoint if performance improves  
#             if f1 > best_f1 + min_delta or acc > best_accuracy + min_delta:  
#                 best_f1, best_accuracy = max(best_f1, f1), max(best_accuracy, acc)  
#                 patience_tracker = 0  

#                 checkpoint_name = f"checkpoint_epoch{epoch + 1}_f1{f1:.4f}_acc{acc:.4f}.pt"  
#                 save_checkpoint(config, {  
#                     'step': current_iteration,  
#                     'epoch': epoch + 1,  
#                     'state_dict': model.state_dict(),  
#                 }, checkpoint_name)  
#                 print_and_log(f"Checkpoint saved: {checkpoint_name}", logging.INFO)  
#             else:  
#                 patience_tracker += 1  

#             if config.train_params.use_ema:  
#                 ema.restore()  

#             if patience_tracker >= config.train_params.patience:  
#                 print_and_log("Early stopping triggered.", logging.INFO)  
#                 break  

#         progress_bar.close()  

#     # Final checkpoint  
#     final_checkpoint_name = f"final_checkpoint_epoch{num_epochs}_f1{best_f1:.4f}_acc{best_accuracy:.4f}.pt"  
#     save_checkpoint(config, {  
#         'step': current_iteration,  
#         'epoch': num_epochs,  
#         'state_dict': model.state_dict(),  
#     }, final_checkpoint_name)  
#     print_and_log(f"Final checkpoint saved: {final_checkpoint_name}", logging.INFO)  




