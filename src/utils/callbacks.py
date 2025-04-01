# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable
import torch
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        every_n_steps: Apply EMA every N steps.
        cpu_offload: Offload weights to CPU.
    """

    def __init__(
        self,
        decay: float,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ):
        super().__init__()
        if not (0 <= decay <= 1):
            raise MisconfigurationException(
                "EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device if not self.cpu_offload else torch.device(
            'cpu')
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]
        print("[EMA] on_fit_start: EMA optimizers have been initialized.")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        ema_path = trainer.checkpoint_callback.dirpath
        if ema_path is not None:
            ema_filename = os.path.join(ema_path, f"{trainer.checkpoint_callback.filename}-EMA.ckpt")
            with self.save_ema_model(trainer):
                torch.save(checkpoint, ema_filename)
                print(f"[EMA] Saved EMA checkpoint at {ema_filename}")
    
    

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            print("[EMA] on_validation_start: Swapping to EMA weights for validation.")
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            print(
                "[EMA] on_validation_end: Swapping back to original weights after validation.")
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            print("[EMA] on_test_start: Swapping to EMA weights for testing.")
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            print("[EMA] on_test_end: Swapping back to original weights after testing.")
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: "pl.Trainer") -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: "pl.Trainer") -> bool:
        return any(isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers)

    def swap_model_weights(self, trainer: "pl.Trainer", saving_ema_model: bool = False):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            print(
                f"[EMA] swap_model_weights: Swapping weights. saving_ema_model={saving_ema_model}")
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer"):
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            print("[EMA] save_ema_model: swapping to EMA weights and saving")
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: "pl.Trainer"):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        ckpt_path = trainer.ckpt_path

        if ckpt_path and checkpoint_callback is not None:
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f'-EMA{ext}'):
                rank_zero_info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f'-EMA{ext}')
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(
                    ema_path, map_location=torch.device('cpu'), weights_only=False)

                checkpoint['optimizer_states'] = ema_state_dict['optimizer_states']
                del ema_state_dict
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    "Unable to find the associated EMA weights when re-loading, "
                    f"training will start with new EMA weights. Expected them to be at: {ema_path}",
                )


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        return (param for group in self.param_groups for param in group['params'])

    def step(self, closure=None, grad_scaler=None, **kwargs):
        self.join()
        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()
            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())
            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device) for param in opt_params[len(self.ema_params):]
            )
            self.rebuild_ema_params = False

        if getattr(self.optimizer, "_step_supports_amp_scaling", False) and grad_scaler is not None:
            loss = self.optimizer.step(
                closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True) for param in self.all_parameters()
            )
            if self.device.type == 'cuda':
                ema_update(self.ema_params, current_model_state, self.decay)
        if self.device.type == 'cpu':
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        print(
            f"[EMAOptimizer] switch_main_parameter_weights: Swapping weights. saving_ema_model={saving_ema_model}")
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()
        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()
        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()
        ema_params = self.ema_params if not self.in_saving_ema_model_context else list(
            self.all_parameters())
        state_dict = {
            'opt': self.optimizer.state_dict(),
            'ema': ema_params,
            'current_step': self.current_step,
            'decay': self.decay,
            'every_n_steps': self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()
        self.optimizer.load_state_dict(state_dict['opt'])
        self.ema_params = tuple(param.to(self.device)
                                for param in copy.deepcopy(state_dict['ema']))
        self.current_step = state_dict['current_step']
        self.decay = state_dict['decay']
        self.every_n_steps = state_dict['every_n_steps']
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True


class AWPCallback(Callback):
    """Callback for Adversarial Weight Perturbation (AWP) in PyTorch Lightning.

    This callback implements AWP to improve model robustness by perturbing the model's weights
    adversarially during training. It hooks into the training loop after the regular backward pass,
    perturbs the weights, computes an adversarial loss, performs an additional backward pass,
    and restores the original weights.
    """

    def __init__(self, adv_param="weight", adv_lr=1.0, adv_eps=0.0001, apply_every=1):
        super().__init__()
        self.adv_param = adv_param
        self.adv_lr = float(adv_lr)
        self.adv_eps = float(adv_eps)
        if isinstance(apply_every, int) and apply_every < 1:
            raise ValueError(
                "apply_every must be a positive integer or 'epoch'")
        if not isinstance(apply_every, (int, str)) or (isinstance(apply_every, str) and apply_every != "epoch"):
            raise ValueError(
                "apply_every must be a positive integer or 'epoch'")
        self.apply_every = apply_every
        self.backup = {}
        self.backup_eps = {}

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if isinstance(self.apply_every, int) and trainer.global_step == 0 and trainer.global_step % self.apply_every != 0:
            return
        if self.adv_lr == 0:
            return
        print(
            f"[AWP] on_after_backward: Applying AWP at global step {trainer.global_step}.")
        self._apply_awp(trainer, pl_module)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.apply_every == "epoch" and self.adv_lr != 0:
            print("[AWP] on_train_epoch_end: Applying AWP at epoch end.")
            self._apply_awp(trainer, pl_module)

    def _apply_awp(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._save(pl_module)
        self._attack_step(pl_module)
        batch = pl_module.current_batch
        print("[AWP] _apply_awp: Computing adversarial loss.")
        adv_loss, _ = pl_module(
            flattened_patches=batch["flattened_patches"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        adv_loss.backward()
        self._restore(pl_module)

    def _attack_step(self, pl_module: "pl.LightningModule") -> None:
        e = 1e-6
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad_norm = torch.norm(param.grad)
                data_norm = torch.norm(param.data.detach())
                if grad_norm != 0 and not torch.isnan(grad_norm):
                    perturbation = self.adv_lr * param.grad / \
                        (grad_norm + e) * (data_norm + e)
                    param.data.add_(perturbation)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1]
                    )
        print("[AWP] _attack_step: Model weights have been perturbed adversarially.")

    def _save(self, pl_module: "pl.LightningModule") -> None:
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                self.backup[name] = param.data.clone()
                grad_eps = self.adv_eps * param.abs().detach()
                self.backup_eps[name] = (
                    self.backup[name] - grad_eps,
                    self.backup[name] + grad_eps
                )
        print("[AWP] _save: Original model weights have been backed up.")

    def _restore(self, pl_module: "pl.LightningModule") -> None:
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
        self.backup_eps.clear()
        print("[AWP] _restore: Original model weights have been restored.")
