import math
import os
import random
import shutil
from datetime import timedelta


import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)

def print_line() -> None:
    """
    Print a decorative line separator for console output.
    
    This line consists of a prefix and suffix '#' with 50 '--' in between.
    """
    prefix, unit, suffix = "#", "--", "#"
    print(prefix + unit * 50 + suffix)

def as_minutes(seconds: float) -> str:
    """
    Convert seconds to a human-readable minutes and seconds format.
    
    Args:
        seconds (float): Total number of seconds to convert.
    
    Returns:
        str: Formatted string representing time in minutes and seconds 
             (e.g., '5m30s' for 5 minutes and 30 seconds).
    """
    minutes = math.floor(seconds / 60)
    remaining_seconds = seconds - minutes * 60
    return f'{minutes}m{remaining_seconds}s'

def seed_everything(seed: int) -> None:
    """
    Set a consistent random seed across multiple libraries for reproducibility.
    
    This function ensures that random number generation is deterministic 
    for Python's random module, NumPy, and PyTorch (including CUDA).
    
    Args:
        seed (int): The seed value to be used for random number generation.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def init_wandb(config: dict) :
    """
    Initialize a Weights & Biases (wandb) run with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing wandb settings.
    
    Returns:
        wandb.Run: Initialized wandb run object.
    
    Notes:
        - Supports both full dataset and fold-specific run naming
        - Uses anonymous mode for logging
        - Sets project, tags, and run name from config
    """
    project = config["wandb"]["project"]
    tags = config["tags"]
    
    
    run_id = f"{config['wandb']['run_name']}-all-data"
    
    return wandb.init(
        project=project,
        config=config,
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
    )

def print_gpu_utilization() -> None:
    """
    Print the current GPU memory utilization.
    
    This function uses NVML to retrieve GPU memory information 
    for the first GPU device (index 0).
    
    Notes:
        - Requires NVML to be initialized
        - Prints memory usage in megabytes (MB)
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Retrieve the current learning rate from an optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to get the learning rate from.
    
    Returns:
        float: Current learning rate multiplied by 1e6 for easier reading.
    """
    return optimizer.param_groups[0]['lr'] * 1e6

class AverageMeter:
    """
    A utility class for tracking and computing running averages.
    
    Useful for tracking metrics during training, such as loss or accuracy.
    Computes and stores the current value, sum, count, and running average.
    
    Attributes:
        val (float): Current value
        avg (float): Running average
        sum (float): Cumulative sum of values
        count (int): Number of updates
    """
    
    def __init__(self) -> None:
        """Initialize the AverageMeter with default values."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracking metrics to their initial state."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update the running average with a new value.
        
        Args:
            val (float): New value to incorporate
            n (int, optional): Number of samples. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


  

class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    Maintains a shadow copy of model parameters with exponential decay,
    which can help stabilize training and potentially improve generalization.
    
    Attributes:
        model (torch.nn.Module): The model to apply EMA to
        decay (float): Decay rate for moving average
        shadow (dict): Shadow copy of model parameters
        backup (dict): Temporary backup of original parameters
    """
    
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        """
        Initialize EMA wrapper for a model.
        
        Args:
            model (torch.nn.Module): Model to apply EMA to
            decay (float): Decay rate for moving average (between 0 and 1)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    
    def register(self) -> None:
        """
        Create initial shadow copies of model parameters.
        
        Only registers parameters that require gradient.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """
        Update shadow parameters using exponential moving average.
        
        Applies decay to blend current and previous parameter values.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """
        Replace model parameters with their shadow copies.
        
        Stores original parameters in backup before replacement.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """
        Restore original model parameters from backup.
        
        Clears the backup dictionary after restoration.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



def setup(rank: int, world_size: int, timeout_seconds: float = 3600000000.0):
    """
    Set up the process group for DDP on Kaggle with NCCL backend and use a timedelta for timeout.
    
    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes in DDP.
        timeout_seconds (float, optional): Timeout for the GPU setup and process group initialization (in seconds).
                                           Defaults to 30.0 seconds.
    """
    timeout = timedelta(seconds=timeout_seconds)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5553'  

    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(rank) 




def cleanup_processes():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    
    for p in mp.active_children():
        p.terminate()
        p.join()
    
    print("Cleanup completed")
