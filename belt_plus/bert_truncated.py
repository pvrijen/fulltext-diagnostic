# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

from __future__ import annotations

from typing import Optional, List, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
    AutoModel,
    AutoConfig,
)

from belt_plus.bert import BertBase
from belt_plus.neural_networks.truncation_nn import TruncationNeuralNetwork
from belt_plus.section_aware import tokenize_section_aware_truncated, register_tag_tokens
import logging

logger = logging.getLogger(__name__)


def collate_fn_truncated(batch):
    first = batch[0]
    if len(first) == 3:
        input_ids, attention_mask, labels = zip(*batch)
        input_ids = torch.stack(list(input_ids), dim=0).long()
        attention_mask = torch.stack(list(attention_mask), dim=0).long()
        labels = torch.tensor([int(l.item() if isinstance(l, torch.Tensor) else l) for l in labels], dtype=torch.long)
        return input_ids, attention_mask, labels

    if len(first) == 2:
        input_ids, attention_mask = zip(*batch)
        input_ids = torch.stack(list(input_ids), dim=0).long()
        attention_mask = torch.stack(list(attention_mask), dim=0).long()
        return input_ids, attention_mask

    raise ValueError("Unexpected batch structure for truncated collate")


class TruncatedDataset(Dataset):
    def __init__(self, tokens: BatchEncoding, labels: Optional[List[int]] = None):
        def _ensure_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.long()
            return torch.tensor(value, dtype=torch.long)

        self.input_ids = _ensure_tensor(tokens["input_ids"])
        self.attention_mask = _ensure_tensor(tokens["attention_mask"])

        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        if self.labels is not None and self.labels.size(0) != self.input_ids.size(0):
            raise ValueError("Labels length does not match number of inputs.")

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (
                self.input_ids[idx],
                self.attention_mask[idx],
                self.labels[idx],
            )
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
        )


class BertTruncated(BertBase):
    """
    Truncated (student) model:
      encoder -> token pooling -> dropout -> FeatureHead(H→H) -> classifier

    Supports optional KD:
      - kd_type="logit": BCE on temperature-soft logits
      - kd_type="feature": cosine (default) or normalized-MSE between student
        post-FeatureHead features and teacher features. If teacher feature dim
        != student H, a projector is created (H -> H_t).
    """

    def __init__(
        self,
        num_labels: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        neural_network: Optional[TruncationNeuralNetwork] = None,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        preloaded_state_dict: Optional[dict] = None,
        device: str = "cuda",
        many_gpus: bool = False,
        dropout_rate: float = 0.1,
        use_amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "auto",
        weight_decay: float = 0.0,
        class_weights: Optional[List[float]] = None,
        problem_type: str = "binary_classification",
        token_pooling_strategy: str = "auto",
        log_signature: bool = False,
        section_aware_tokenization: bool = False,
        max_length: Optional[int] = None,
        fused_adamw: Optional[bool] = None,
        use_torch_compile: bool = False,
        torch_compile_backend: Optional[str] = None,
        torch_compile_mode: Optional[str] = None,
        flash_attention: bool = True,
        gradient_checkpointing: bool = False,
        warmup_ratio: float = 0.05,
        prefetch_factor: Optional[int] = None,
        max_workers: Optional[int] = None,
        reference_compile: Optional[bool] = None,
    ):
        pending_state_dict = None
        if neural_network is None:
            # Load config and optionally disable ModernBERT's internal torch.compile
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if reference_compile is not None and hasattr(config, "reference_compile"):
                config.reference_compile = reference_compile

            if preloaded_state_dict is not None:
                backbone = AutoModel.from_config(config)
                pending_state_dict = preloaded_state_dict
            else:
                backbone = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)

            neural_network = TruncationNeuralNetwork(
                encoder=backbone,
                num_labels=num_labels,
                dropout_rate=dropout_rate,
                token_pooling_strategy=token_pooling_strategy,
                log_signature=log_signature
            )

        super().__init__(
            num_labels=num_labels,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            neural_network=neural_network,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            many_gpus=many_gpus,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            weight_decay=weight_decay,
            class_weights=class_weights,
            problem_type=problem_type,
            fused_adamw=fused_adamw,
            use_torch_compile=use_torch_compile,
            torch_compile_backend=torch_compile_backend,
            torch_compile_mode=torch_compile_mode,
            flash_attention=flash_attention,
            gradient_checkpointing=gradient_checkpointing,
            warmup_ratio=warmup_ratio,
            prefetch_factor=prefetch_factor,
            max_workers=max_workers,
        )
        self.collate_fn = collate_fn_truncated
        self._params["token_pooling_strategy"] = token_pooling_strategy
        self.section_aware_tokenization = bool(section_aware_tokenization)
        self._params["section_aware_tokenization"] = self.section_aware_tokenization

        # Register section tag special tokens early so embeddings are resized
        # before any save/checkpoint, not deferred to first _tokenize() call.
        if self.section_aware_tokenization:
            encoder = getattr(self.neural_network, "encoder", None)
            if isinstance(self.neural_network, nn.DataParallel):
                encoder = getattr(self.neural_network.module, "encoder", None)
            register_tag_tokens(self.tokenizer, encoder)

        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_max is None or tokenizer_max > 1e6:
            tokenizer_max = None
        self._max_length = int(max_length or tokenizer_max or 512)
        self._params["max_length"] = self._max_length

        # KD projector (created lazily to match teacher feature dim)
        self._kd_project: Optional[torch.nn.Linear] = None

        self._params["log_signature"] = log_signature

        if pending_state_dict is not None:
            encoder_module = self.neural_network
            if isinstance(encoder_module, nn.DataParallel):
                encoder_module = encoder_module.module
            encoder = getattr(encoder_module, "encoder", None)
            if self.section_aware_tokenization:
                register_tag_tokens(self.tokenizer, encoder)
            missing, unexpected = encoder.load_state_dict(pending_state_dict, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                logger.info(
                    f"[BertTruncated] Warning: load_state_dict non-strict. "
                    f"Missing: {len(missing)} | Unexpected: {len(unexpected)}"
                )

    # ----------------------------
    # Tokenization / batching
    # ----------------------------
    def _tokenize(self, texts: List[str]) -> BatchEncoding:
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise AttributeError("BertTruncated expects a PreTrainedTokenizerBase at self.tokenizer.")
        if self.section_aware_tokenization:
            encoder = getattr(self.neural_network, "encoder", None)
            return tokenize_section_aware_truncated(
                texts,
                self.tokenizer,
                max_length=self._max_length,
                encoder=encoder,
            )

        return self.tokenizer(
            texts,
            max_length=self._max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )

    def _evaluate_single_batch(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        input_ids = batch[0].to(self.device, non_blocking=True)
        attention_mask = batch[1].to(self.device, non_blocking=True)

        logits = self.neural_network(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return logits

    def _create_dataset(self, tokens: BatchEncoding, labels: Optional[List[int]] = None) -> Dataset:
        return TruncatedDataset(tokens=tokens, labels=labels)

    # ----------------------------
    # KD helpers
    # ----------------------------
    def _student_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Uses the student's public feature path (post FeatureHead), shape [B, H]
        return self.neural_network.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
