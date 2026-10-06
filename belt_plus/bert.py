# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

from __future__ import annotations
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Optional, Union, Any, List, Callable, Dict
import logging
import inspect
import os

import torch
from torch import Tensor, nn
from torch.nn import DataParallel
from torch.optim import AdamW, Optimizer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel
from .neural_networks.pooling_nn import PoolingNeuralNetwork
import numpy as np
import random
from transformers import logging as transformers_logging

from belt_plus.section_aware import register_tag_tokens

transformers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)

from torch.amp import GradScaler
AMP_AVAILABLE = True

_ADAMW_SUPPORTS_FUSED = "fused" in inspect.signature(AdamW).parameters

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int):
    """Deterministic worker initialization for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


class BertBase(ABC):

    @staticmethod
    def _parse_worker_env(key: str) -> Optional[int]:
        value = os.environ.get(key)
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return max(0, parsed)

    def _build_loader_kwargs(self, workers: int) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "pin_memory": True,
            "num_workers": workers,
            "worker_init_fn": _worker_init_fn,  # Deterministic worker seeding
        }
        if workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = True
        return kwargs

    @abstractmethod
    def __init__(
        self,
        num_labels: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        neural_network: Optional[nn.Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda",
        many_gpus: bool = False,
        use_amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "auto",
        weight_decay: float = 0.0,
        class_weights: Optional[List[float]] = None,
        problem_type: Optional[str] = "binary_classification",
        flash_attention: bool = True,
        fused_adamw: Optional[bool] = None,
        use_torch_compile: bool = False,
        torch_compile_backend: Optional[str] = None,
        torch_compile_mode: Optional[str] = None,
        gradient_checkpointing: bool = False,
        # Dataloader and scheduler config
        warmup_ratio: float = 0.05,
        prefetch_factor: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.pretrained_model_name_or_path = pretrained_model_name_or_path  
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.neural_network = neural_network
        self.device = device
        self.many_gpus = many_gpus
        self.flash_attention_enabled = bool(flash_attention)
        use_cuda = isinstance(device, str) and device.lower().startswith("cuda")
        self.use_amp = use_amp and AMP_AVAILABLE and use_cuda
        self.amp_dtype: Optional[torch.dtype] = None
        if self.use_amp:
            self.amp_dtype = self._resolve_amp_dtype(amp_dtype, device)
        self.weight_decay = weight_decay
        self.class_weights = class_weights 

        amp_dtype_str = None
        if self.amp_dtype is not None:
            if self.amp_dtype == torch.float16:
                amp_dtype_str = "float16"
            elif self.amp_dtype == torch.bfloat16:
                amp_dtype_str = "bfloat16"
            else:
                amp_dtype_str = str(self.amp_dtype)

        self._params = {
            "num_labels": self.num_labels,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "accumulation_steps": self.accumulation_steps,
            "use_amp": self.use_amp,
            "amp_dtype": amp_dtype_str,
            "weight_decay": self.weight_decay,
            "class_weights": self.class_weights,
            "warmup_ratio": warmup_ratio,
        }
        self._config_max_workers = max_workers
        self._config_prefetch_factor = prefetch_factor

        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if fused_adamw is None:
            self.use_fused_adamw = bool(
                _ADAMW_SUPPORTS_FUSED
                and torch.cuda.is_available()
                and isinstance(device, str)
                and device.lower().startswith("cuda")
            )
        else:
            self.use_fused_adamw = bool(fused_adamw) and _ADAMW_SUPPORTS_FUSED and torch.cuda.is_available()

        self.use_torch_compile = bool(use_torch_compile)
        self.torch_compile_backend = torch_compile_backend or "inductor"
        self.torch_compile_mode = torch_compile_mode or "default"
        self._torch_compile_applied = False
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self._configure_cudnn()

        self.problem_type = (problem_type or "binary_classification").lower()
        if self.problem_type not in ["binary_classification", "multi_class_classification", "regression"]:
            raise ValueError(
                f"Invalid problem_type: {self.problem_type}. "
                "Must be one of 'binary_classification', 'multi_class_classification', or 'regression'."
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=True)
        except Exception as e:
            logger.info(f"[Tokenizer Warning] Falling back to slow tokenizer for {pretrained_model_name_or_path}: {e}")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)
        self.tokenizer = tokenizer

        # Worker count: config > env var > auto-detect
        auto_workers = self._parse_worker_env("SLURM_CPUS_PER_TASK")
        if not auto_workers:
            cpu_total = os.cpu_count() or 8
            half = cpu_total // 2
            if half <= 0:
                half = 1
            auto_workers = max(1, min(32, half))
        # Config max_workers overrides auto-detection
        if self._config_max_workers is not None:
            auto_workers = min(auto_workers, self._config_max_workers)
        base_workers = self._parse_worker_env("BELT_NUM_WORKERS") or auto_workers
        self.train_num_workers = self._parse_worker_env("BELT_TRAIN_WORKERS") or base_workers
        self.eval_num_workers = self._parse_worker_env("BELT_VAL_WORKERS") or self.train_num_workers
        self.predict_num_workers = self._parse_worker_env("BELT_PRED_WORKERS") or self.eval_num_workers
        self.train_num_workers = max(1, self.train_num_workers)
        self.eval_num_workers = max(1, self.eval_num_workers)
        self.predict_num_workers = max(1, self.predict_num_workers)
        # Prefetch factor: config > env var > default (2)
        self.prefetch_factor = self._config_prefetch_factor or self._parse_worker_env("BELT_PREFETCH_FACTOR") or 2
        if self.prefetch_factor <= 0:
            self.prefetch_factor = 2
        logger.info(
            "[BeltBase] dataloader workers=train:%d val:%d pred:%d",
            self.train_num_workers,
            self.eval_num_workers,
            self.predict_num_workers,
        )
        
        if self.neural_network is None:
            raise ValueError("Neural network must be provided.")

        self._hf_core_module, self._hf_core_config = self._resolve_hf_module()
        if not self.flash_attention_enabled:
            # Force SDPA attention implementation when flash_attention flag is disabled
            self._set_attention_impl("sdpa")
            self._replace_rotary_embeddings()
        # Report chosen attention implementation for debugging
        try:
            impl = getattr(self._hf_core_config, "_attn_implementation", None)
            logger.info("[BeltBase] attention implementation: %s", impl)
        except Exception:
            pass

        # Some pretrained ModernBERT checkpoints ship in bf16, which our CUDA stack cannot host.
        # Force-cast everything to fp32 while still on CPU before transferring to the target device.
        try:
            self.neural_network.to(torch.device("cpu"))
            self.neural_network.to(dtype=torch.float32)
            for param in self.neural_network.parameters():
                if param.is_floating_point() and param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float32)
            for buf in self.neural_network.buffers():
                if buf.is_floating_point() and buf.dtype == torch.bfloat16:
                    buf.data = buf.data.to(torch.float32)
        except Exception as exc:
            logger.warning("Failed to sanitize bf16 parameters before CUDA transfer (%r).", exc)

        target_device = torch.device(device)
        sample_param = next(self.neural_network.parameters(), None)
        logger.info(
            "[BeltBase] moving encoder to %s (dtype %s)",
            target_device,
            getattr(sample_param, "dtype", "<none>"),
        )

        self.neural_network.to(device=target_device, dtype=torch.float32)

        self.device = str(target_device)
        if target_device.type != "cuda":
            self._set_attention_impl("sdpa")
            self._replace_rotary_embeddings(target_device)

        self._maybe_enable_gradient_checkpointing()
        self._maybe_compile_model()

        if device.startswith("cuda") and many_gpus and torch.cuda.device_count() > 1:
            self.neural_network = DataParallel(self.neural_network)

        self.collate_fn: Optional[Callable] = None
        self.scaler = (
            GradScaler(device='cuda')
            if (self.use_amp and self.amp_dtype == torch.float16)
            else None
        )
        self.current_epoch = 0

        self._init_loss_function()
        self._params.update(
            {
                "use_fused_adamw": self.use_fused_adamw,
                "use_torch_compile": self._torch_compile_applied,
                "torch_compile_backend": self.torch_compile_backend if self._torch_compile_applied else None,
                "torch_compile_mode": self.torch_compile_mode if self._torch_compile_applied else None,
                "gradient_checkpointing": self._gradient_checkpointing_applied,
            }
        )

    @staticmethod
    def _resolve_amp_dtype(
        amp_dtype: Optional[Union[str, torch.dtype]],
        device: str,
    ) -> torch.dtype:
        if isinstance(amp_dtype, torch.dtype):
            return amp_dtype

        choice = "auto" if amp_dtype is None else str(amp_dtype).lower()
        if choice in {"fp16", "float16", "half"}:
            return torch.float16
        if choice in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if choice not in {"auto", "auto_select"}:
            raise ValueError(f"Unsupported amp_dtype value: {amp_dtype}")

        resolved = torch.float16
        try:
            dev = torch.device(device)
        except Exception:
            dev = torch.device("cuda")

        if dev.type == "cuda" and torch.cuda.is_available():
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            try:
                major, _ = torch.cuda.get_device_capability(idx)
            except Exception:
                major = None
            name = ""
            try:
                name = torch.cuda.get_device_name(idx).lower()
            except Exception:
                pass

            if major is not None and major >= 9:
                resolved = torch.bfloat16
            elif "h100" in name or "hopper" in name:
                resolved = torch.bfloat16

        return resolved

    def _autocast_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"device_type": "cuda"}
        if self.amp_dtype is not None:
            kwargs["dtype"] = self.amp_dtype
        return kwargs

    def _init_loss_function(self) -> None:
        if self.problem_type == "binary_classification":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif self.problem_type == "multi_class_classification":
            if self.class_weights is not None:
                cw_tensor = torch.tensor(self.class_weights).to(self.device)
                self.loss_function = nn.CrossEntropyLoss(weight=cw_tensor)
            else:
                self.loss_function = nn.CrossEntropyLoss()
        elif self.problem_type == "regression":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported problem_type: {self.problem_type}")
        
    def _configure_cudnn(self) -> None:
        """Configure cuDNN settings for all model types.

        benchmark=False (PyTorch default) for all configs.
        Testing showed <1% difference either way on H100/cuDNN 9.1.
        Using consistent settings across all models for simplicity.
        """
        if not torch.cuda.is_available():
            return
        torch.backends.cudnn.benchmark = False


    @abstractmethod
    def _tokenize(self, texts: list[str]): ...
    @abstractmethod
    def _evaluate_single_batch(self, batch: tuple[Tensor, ...]) -> Tensor: ...
    @abstractmethod
    def _create_dataset(self, tokens, labels=None) -> Dataset: ...

    def _maybe_enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing_applied = False
        if not self.gradient_checkpointing:
            return
        module = getattr(self, "_hf_core_module", None)
        if module is not None and hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
            self._gradient_checkpointing_applied = True
            logger.info("Gradient checkpointing enabled on %s.", type(module).__name__)
        else:
            logger.warning("Gradient checkpointing requested but model does not support it.")

    def _maybe_compile_model(self) -> None:
        if not self.use_torch_compile:
            return
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available; skipping requested compilation.")
            self.use_torch_compile = False
            return
        if self.many_gpus and torch.cuda.device_count() > 1:
            logger.warning("torch.compile is incompatible with DataParallel; skipping.")
            self.use_torch_compile = False
            return
        try:
            self.neural_network = torch.compile(
                self.neural_network,
                backend=self.torch_compile_backend,
                mode=self.torch_compile_mode,
            )
            self._torch_compile_applied = True
            logger.info("torch.compile enabled (backend=%s mode=%s).", self.torch_compile_backend, self.torch_compile_mode)
        except Exception as exc:
            logger.warning("torch.compile failed (%r); continuing without compilation.", exc)
            self.use_torch_compile = False
            self._torch_compile_applied = False

    def _create_optimizer(self, parameters) -> Optimizer:
        kwargs = {"lr": self.learning_rate, "weight_decay": self.weight_decay}
        optimizer_name = "adamw"
        if self.use_fused_adamw and _ADAMW_SUPPORTS_FUSED:
            kwargs["fused"] = True
            optimizer_name = "adamw_torch_fused"
        try:
            optimizer = AdamW(parameters, **kwargs)
        except TypeError as exc:
            if "fused" in kwargs:
                logger.warning("AdamW(fused=True) unavailable (%s); falling back to standard AdamW.", exc)
                kwargs.pop("fused", None)
                optimizer_name = "adamw"
                optimizer = AdamW(parameters, **kwargs)
                self.use_fused_adamw = False
            else:
                raise
        self._params["optimizer"] = optimizer_name
        return optimizer

    def _compute_loss(self, logits_or_tuple, labels):
        """Compute loss, handling auxiliary/contrastive modes."""
        if isinstance(logits_or_tuple, tuple):
            logits_pooled, second = logits_or_tuple
            nn = self.neural_network
            if hasattr(nn, 'module'):
                nn = nn.module

            # Contrastive mode: second is a raw loss (MSE), not logits
            c_alpha = getattr(nn, 'contrastive_alpha', 0.0)
            if c_alpha > 0:
                loss_pooled = self._single_loss(logits_pooled, labels)
                return loss_pooled + c_alpha * second

            # Aux mode: second is abstract logits
            alpha = getattr(nn, 'aux_abstract_alpha', 0.0)
            loss_pooled = self._single_loss(logits_pooled, labels)
            loss_abstract = self._single_loss(second, labels)
            return loss_pooled + alpha * loss_abstract
        return self._single_loss(logits_or_tuple, labels)

    def _single_loss(self, logits, labels):
        if self.problem_type == "binary_classification":
            return self.loss_function(logits.squeeze(dim=1), labels.float())
        elif self.problem_type == "multi_class_classification":
            return self.loss_function(logits, labels)
        elif self.problem_type == "regression":
            return self.loss_function(torch.flatten(logits), labels.float())
        raise ValueError(f"Unsupported problem_type: {self.problem_type}")

    def _train_single_epoch(self, dataloader: DataLoader, optimizer: Optimizer, scheduler: Optional[Any] = None) -> float:
        self.neural_network.train()
        scaler = self.scaler
        epoch_loss = 0.0
        # Optional KD: set self._kd_targets (tensor [N,1]) and self._kd_alpha before calling fit
        kd_targets = getattr(self, '_kd_targets', None)
        kd_alpha = getattr(self, '_kd_alpha', 0.1)
        kd_temp = getattr(self, '_kd_temperature', 4.0)
        kd_cursor = 0

        for step, batch in enumerate(dataloader):
            if self.use_amp:
                with torch.amp.autocast(**self._autocast_kwargs()):
                    logits = self._evaluate_single_batch(batch)
                    labels = batch[-1].to(self.device, non_blocking=True)
                    loss = self._compute_loss(logits, labels)
                    if kd_targets is not None:
                        B = logits.size(0)
                        t_batch = kd_targets[kd_cursor:kd_cursor+B].to(self.device)
                        kd_cursor += B
                        s = torch.sigmoid(logits / kd_temp).squeeze(-1)
                        t = torch.sigmoid(t_batch / kd_temp).squeeze(-1)
                        eps = 1e-6
                        kd_loss = -(t * torch.log(s+eps) + (1-t) * torch.log(1-s+eps)).mean()
                        loss = (1 - kd_alpha) * loss + kd_alpha * kd_temp * kd_temp * kd_loss
                    loss /= self.accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                logits = self._evaluate_single_batch(batch)
                labels = batch[-1].to(self.device, non_blocking=True)
                loss = self._compute_loss(logits, labels)
                if kd_targets is not None:
                    B = logits.size(0)
                    t_batch = kd_targets[kd_cursor:kd_cursor+B].to(self.device)
                    kd_cursor += B
                    s = torch.sigmoid(logits / kd_temp).squeeze(-1)
                    t = torch.sigmoid(t_batch / kd_temp).squeeze(-1)
                    eps = 1e-6
                    kd_loss = -(t * torch.log(s+eps) + (1-t) * torch.log(1-s+eps)).mean()
                    loss = (1 - kd_alpha) * loss + kd_alpha * kd_temp * kd_temp * kd_loss
                loss /= self.accumulation_steps
                loss.backward()

            epoch_loss += loss.item()

            if ((step + 1) % 100 == 0) or (step + 1 == len(dataloader)):
                logger.info(f"[Train] Epoch {self.current_epoch} | Step {step + 1}/{len(dataloader)} | Loss: {loss.item():.5f}")

            if ((step + 1) % self.accumulation_steps == 0) or (step + 1 == len(dataloader)):
                if self.use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), self.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), self.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None and not (self.current_epoch == 0 and step == 0):
                    scheduler.step()

        return epoch_loss / max(1, len(dataloader))
    
    def _validate(self, dataloader: DataLoader, return_logits: bool = False):
        self.neural_network.eval()
        epoch_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                labels = batch[-1].to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast(**self._autocast_kwargs()):
                        logits = self._evaluate_single_batch(batch)
                else:
                    logits = self._evaluate_single_batch(batch)

                if self.problem_type == "binary_classification":
                    loss = self.loss_function(logits.squeeze(dim=1), labels.float())
                elif self.problem_type == "multi_class_classification":
                    loss = self.loss_function(logits, labels)
                elif self.problem_type == "regression":
                    scores = torch.flatten(logits)
                    loss = self.loss_function(scores, labels.float())
                else:
                    raise ValueError(f"Unsupported problem_type: {self.problem_type}")

                epoch_loss += loss.item()
                if return_logits:
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())

        avg_val_loss = epoch_loss / max(1, len(dataloader))
        if return_logits:
            return avg_val_loss, torch.cat(all_logits), torch.cat(all_labels)
        return avg_val_loss

    def fit(
        self,
        x_train: list[str],
        y_train: list[int],
        x_val: Optional[list[str]] = None,
        y_val: Optional[list[int]] = None,
        epochs: Optional[int] = None,
        patience: Optional[int] = 2,
        min_delta: float = 0.0,
        save_best_model_path: Optional[str] = None,
        on_epoch_end: Optional[Callable[[int, dict], None]] = None,
        early_stopping: bool = True
    ):
        if epochs is None:
            epochs = self.epochs

        # Tokenize and datasets
        train_tokens = self._tokenize(x_train)
        train_dataset = self._create_dataset(train_tokens, y_train)
        # Use sequential sampler when KD targets are set (must align with target order)
        _use_kd = getattr(self, '_kd_targets', None) is not None
        train_dataloader = DataLoader(
            train_dataset,
            sampler=SequentialSampler(train_dataset) if _use_kd else RandomSampler(train_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            **self._build_loader_kwargs(self.train_num_workers),
        )

        if x_val is not None and y_val is not None:
            val_tokens = self._tokenize(x_val)
            val_dataset = self._create_dataset(val_tokens, y_val)
            val_dataloader = DataLoader(
                val_dataset,
                sampler=SequentialSampler(val_dataset),
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                **self._build_loader_kwargs(self.eval_num_workers),
            )
        else:
            val_dataloader = None

        optimizer = self._create_optimizer(self.neural_network.parameters())
        # Compute total_steps in OPTIMIZER steps (not micro-batches).
        # scheduler.step() is called per optimizer step (every accumulation_steps),
        # so total_steps must count optimizer steps to get correct warmup/decay.
        micro_batches_per_epoch = len(train_dataloader)
        optimizer_steps_per_epoch = micro_batches_per_epoch // self.accumulation_steps
        if micro_batches_per_epoch % self.accumulation_steps != 0:
            optimizer_steps_per_epoch += 1  # partial accumulation at end of epoch
        total_steps = max(1, optimizer_steps_per_epoch * epochs)
        # Allow overriding warmup via config param `warmup_ratio` (default 5%).
        try:
            warmup_ratio = float(self._params.get("warmup_ratio", 0.05))
        except Exception:
            warmup_ratio = 0.05
        warmup_ratio = max(0.0, min(warmup_ratio, 0.5))  # clamp to sane range
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        self.scheduler = scheduler

        best_val_loss = float('inf')
        wait = 0
        best_model_state = None
        self.training_history = []  # Track per-epoch metrics for analysis
        self.best_epoch = None

        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss = self._train_single_epoch(train_dataloader, optimizer, scheduler)

            val_loss = None; val_logits = None; val_labels = None
            if val_dataloader is not None:
                if on_epoch_end:
                    val_loss, val_logits, val_labels = self._validate(val_dataloader, return_logits=True)
                else:
                    val_loss = self._validate(val_dataloader)

                if early_stopping:
                    if val_loss < (best_val_loss - min_delta):
                        best_val_loss = val_loss; wait = 0
                        best_model_state = self.neural_network.state_dict()
                        self.best_epoch = epoch
                        if save_best_model_path:
                            torch.save(best_model_state, save_best_model_path)
                    else:
                        wait += 1
                        if wait >= patience:
                            break
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.neural_network.state_dict()
                        self.best_epoch = epoch
                        if save_best_model_path:
                            torch.save(best_model_state, save_best_model_path)

            # Track training history for downstream analysis
            self.training_history.append({
                "epoch": epoch,
                "train_loss": float(train_loss) if train_loss is not None else None,
                "val_loss": float(val_loss) if val_loss is not None else None,
                "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
            })

            if on_epoch_end:
                epoch_data = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_logits": val_logits,
                    "val_labels": val_labels,
                    "model": self,
                    "best_val_loss": best_val_loss,
                }
                on_epoch_end(epoch, epoch_data)

        if best_model_state:
            self.neural_network.load_state_dict(best_model_state)

    def _predict_logits(self, x: list[str], batch_size: Optional[int] = None) -> Tensor:
        if not batch_size: batch_size = self.batch_size
        tokens = self._tokenize(x)
        dataset = self._create_dataset(tokens)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=self.collate_fn)
        self.neural_network.eval()
        total_logits = []
        with torch.no_grad():
            for batch in dataloader:
                if self.use_amp:
                    with torch.amp.autocast(**self._autocast_kwargs()):
                        logits = self._evaluate_single_batch(batch)
                else:
                    logits = self._evaluate_single_batch(batch)
                total_logits.append(logits.cpu())
        return torch.cat(total_logits)

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(self._params, f)
        self.tokenizer.save_pretrained(model_dir)
        torch.save(self.neural_network.state_dict(), model_dir / "model.bin")
        if hasattr(self, "scheduler") and self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), model_dir / "scheduler.pth")

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda", many_gpus: bool = False) -> "BertBase":
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        with open(model_dir / "params.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        token_pooling_strategy = params.get("token_pooling_strategy", params.get("pooling_strategy", "mean"))

        neural_network = PoolingNeuralNetwork(
            encoder=AutoModel.from_pretrained(
                params.get("pretrained_model_name_or_path", "bert-base-uncased"),
                add_pooling_layer=True
            ),
            chunk_size=params.get("chunk_size", 512),
            stride=params.get("stride", params.get("chunk_stride", 512)),
            num_labels=params["num_labels"],
            max_chunks=params.get("max_chunks", 10),
            use_chunk_transform=params.get("use_chunk_transform", False),
            chunk_transform_layers=params.get("chunk_transform_layers", 1),
            chunk_transform_ffn_mult=params.get("chunk_transform_ffn_mult", 4),
            chunk_transform_dropout=params.get("chunk_transform_dropout", 0.1),
            chunk_transform_nhead=params.get("chunk_transform_nhead", None),
            chunk_transform_stride=params.get("chunk_transform_stride", params.get("chunk_transform_stride", 128)),
            chunk_transform_arch=params.get("chunk_transform_arch", "transformer"),
            chunk_transform_mode=params.get("chunk_transform_mode", "residual_mean"),
            token_pooling_strategy=token_pooling_strategy,
            dropout_rate=params.get("dropout_rate", 0.1),
            chunk_pooling_strategy=params.get("chunk_pooling_strategy", "mean"),
            use_section_chunk_embeddings=params.get("use_section_chunk_embeddings", False),
            section_chunk_embedding_dim=params.get("section_chunk_embedding_dim", 32),
            section_chunk_vocab_size=params.get("section_chunk_vocab_size", 128),
            section_chunk_position_size=params.get("section_chunk_position_size", 32),
            use_chunk_relation_attention=params.get("use_chunk_relation_attention", False),
        )

        register_tag_tokens(tokenizer, getattr(neural_network, "encoder", None))

        state_dict = torch.load(model_dir / "model.bin", map_location=device)
        new_state_dict = { (k[len('module.'):] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        neural_network.load_state_dict(new_state_dict)

        section_aware_tokenization = params.get("section_aware_tokenization", False)
        section_aware_chunking = params.get("section_aware_chunking", False)

        model = cls(
            num_labels=params["num_labels"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            accumulation_steps=params.get("accumulation_steps", 1),
            max_grad_norm=params.get("max_grad_norm", 1.0),
            neural_network=neural_network,
            pretrained_model_name_or_path=params.get("pretrained_model_name_or_path", "bert-base-uncased"),
            device=device,
            many_gpus=many_gpus,
            use_amp=params.get("use_amp", False),
            amp_dtype=params.get("amp_dtype", "auto"),
            weight_decay=params.get("weight_decay", 0.0),
            class_weights=params.get("class_weights", None),
            problem_type=params.get("problem_type", "binary_classification"),
            section_aware_tokenization=section_aware_tokenization,
            section_aware_chunking=section_aware_chunking,
            use_section_chunk_embeddings=params.get("use_section_chunk_embeddings", False),
            section_chunk_embedding_dim=params.get("section_chunk_embedding_dim", 32),
            section_chunk_vocab_size=params.get("section_chunk_vocab_size", 128),
            section_chunk_position_size=params.get("section_chunk_position_size", 32),
            use_chunk_relation_attention=params.get("use_chunk_relation_attention", False),
        )

        model.tokenizer = tokenizer
        model.neural_network.to(device)

        if many_gpus and torch.cuda.device_count() > 1:
            model.neural_network = nn.DataParallel(model.neural_network)
        sched_path = model_dir / "scheduler.pth"
        if sched_path.exists():
            try:
                model._params["scheduler_state_dict"] = torch.load(sched_path, map_location=device)
            except Exception as e:
                logger.info(f"[load] Warning: could not restore scheduler state: {e}")

        return model

    def _resolve_hf_module(self) -> tuple[Optional[nn.Module], Optional[Any]]:
        candidates: List[nn.Module] = []
        if hasattr(self.neural_network, "encoder") and isinstance(self.neural_network.encoder, nn.Module):
            candidates.append(self.neural_network.encoder)
        if hasattr(self.neural_network, "model") and isinstance(self.neural_network.model, nn.Module):
            candidates.append(self.neural_network.model)
        candidates.append(self.neural_network)
        for module in candidates:
            cfg = getattr(module, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                return module, cfg
        return None, None

    def _set_attention_impl(self, impl: str) -> None:
        cfg = getattr(self, "_hf_core_config", None)
        if cfg is None:
            return
        try:
            cfg._attn_implementation = impl
            if hasattr(cfg, "_attn_implementation_autoset"):
                cfg._attn_implementation_autoset = True
            if impl != "flash_attention_2" and hasattr(cfg, "reference_compile"):
                cfg.reference_compile = False
        except Exception as exc:
            logger.warning("Could not set attention implementation to %s (%r)", impl, exc)

    def _replace_rotary_embeddings(self, device: Optional[torch.device] = None) -> None:
        module = getattr(self, "_hf_core_module", None)
        cfg = getattr(self, "_hf_core_config", None)
        if module is None or cfg is None:
            return
        if "modernbert" not in cfg.__class__.__name__.lower():
            return
        try:
            from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
        except Exception:
            return
        encoder = getattr(module, "encoder", None)
        layers = None
        if encoder is not None:
            layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
        if layers is None:
            layers = getattr(module, "layers", None)
        if layers is None:
            return
        target_device = device or torch.device("cpu")
        for layer in layers:
            attn = getattr(layer, "attn", None)
            if attn is None or not hasattr(attn, "rotary_emb"):
                continue
            if attn.rotary_emb.__class__.__name__ == "ModernBertUnpaddedRotaryEmbedding":
                rope_theta = getattr(cfg, "global_rope_theta", 10000.0)
                max_position_embeddings = getattr(cfg, "max_position_embeddings", 8192)
                local_attention = getattr(attn, "local_attention", (-1, -1))
                if local_attention != (-1, -1):
                    local_theta = getattr(cfg, "local_rope_theta", None)
                    if local_theta is not None:
                        rope_theta = local_theta
                    max_position_embeddings = local_attention[0] + local_attention[1]
                attn.rotary_emb = ModernBertRotaryEmbedding(
                    config=cfg,
                    dim=getattr(attn, "head_dim", getattr(cfg, "hidden_size", 0) // max(1, getattr(cfg, "num_attention_heads", 1))),
                    base=rope_theta,
                    device=target_device,
                )
                attn.rotary_emb.max_seqlen = max_position_embeddings
