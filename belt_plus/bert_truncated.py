# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# src/belt_plus/bert_truncated.py
from __future__ import annotations

from typing import Optional, List, Tuple, Union, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from belt_plus.bert import BertBase
from belt_plus.neural_networks.truncation_nn import TruncationNeuralNetwork
from belt_plus.section_aware import tokenize_section_aware_truncated, register_tag_tokens


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
        device: str = "cuda:0",
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
                print(
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

    def _ensure_kd_projector(self, teacher_feat_dim: int, optimizer: AdamW):
        """
        Create a linear projector H -> teacher_feat_dim iff dims mismatch.
        Register its params with the optimizer so it learns jointly.
        """
        student_hidden = getattr(self.neural_network, "hidden_size", None)
        if student_hidden is None:
            # Fallback to encoder config
            student_hidden = getattr(self.neural_network.encoder.config, "hidden_size", 768)

        if teacher_feat_dim == student_hidden:
            self._kd_project = None
            return

        # Recreate only if needed or different
        if self._kd_project is not None:
            if (self._kd_project.in_features, self._kd_project.out_features) == (student_hidden, teacher_feat_dim):
                return

        self._kd_project = torch.nn.Linear(student_hidden, teacher_feat_dim).to(self.device)
        optimizer.add_param_group({"params": self._kd_project.parameters(), "lr": self.learning_rate, "weight_decay": self.weight_decay})

    # ----------------------------
    # Training (with optional KD)
    # ----------------------------
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
        early_stopping: bool = True,
        # --- KD additions ---
        kd_type: str = "none",                              # "none" | "logit" | "feature" | "combined"
        kd_targets_train: Optional[torch.Tensor] = None,    # logits or features (teacher)
        kd_targets_val: Optional[torch.Tensor] = None,      # optional for val monitoring
        kd_alpha: float = 0.5,                              # weight for KD loss vs supervised
        kd_temperature: float = 2.0,                        # temperature for logit KD
        kd_feature_weight: float = 1.0,                     # scale for feature KD
        # Combined KD: logit + feature simultaneously
        kd_logit_targets_train: Optional[torch.Tensor] = None,
        kd_logit_targets_val: Optional[torch.Tensor] = None,
        kd_feat_targets_train: Optional[torch.Tensor] = None,
        kd_feat_targets_val: Optional[torch.Tensor] = None,
        # NEW (optional): init student tails from teacher and choose cosine vs MSE KD
        teacher_feature_head_state: Optional[dict] = None,
        teacher_classifier_state: Optional[dict] = None,
        use_cosine_kd: bool = True,
    ):
        if epochs is None:
            epochs = self.epochs

        # Tokenize + datasets. For KD we prefer deterministic order → SequentialSampler.
        use_kd = (kd_type != "none") and (kd_targets_train is not None or kd_type == "combined")
        train_tokens = self._tokenize(x_train)
        train_dataset = self._create_dataset(train_tokens, y_train)
        train_loader = DataLoader(
            train_dataset,
            sampler=SequentialSampler(train_dataset) if use_kd else RandomSampler(train_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            **self._build_loader_kwargs(self.train_num_workers),
        )

        if x_val is not None and y_val is not None:
            val_tokens = self._tokenize(x_val)
            val_dataset = self._create_dataset(val_tokens, y_val)
            val_loader = DataLoader(
                val_dataset,
                sampler=SequentialSampler(val_dataset),
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                **self._build_loader_kwargs(self.eval_num_workers),
            )
        else:
            val_loader = None

        optimizer = self._create_optimizer(self.neural_network.parameters())
        total_steps = max(1, len(train_loader) * epochs)
        # Use warmup_ratio from params (default 0.05)
        warmup_ratio = float(self._params.get("warmup_ratio", 0.05))
        warmup_ratio = max(0.0, min(warmup_ratio, 0.5))  # clamp to sane range
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        self.scheduler = scheduler

        # If feature KD, possibly add projector to teacher dim and load teacher tails
        if kd_type == "feature" and kd_targets_train is not None:
            teacher_dim = int(kd_targets_train.size(-1))
            self._ensure_kd_projector(teacher_dim, optimizer)

            # Initialize student tails from teacher to place both in same feature space
            if teacher_feature_head_state:
                try:
                    self.neural_network.feature_head.load_state_dict(teacher_feature_head_state, strict=False)
                    print("[KD] Loaded teacher feature_head into student.")
                except Exception as e:
                    print(f"[KD] warn: couldn't load teacher feature_head: {e}")

            if teacher_classifier_state:
                try:
                    self.neural_network.classifier.load_state_dict(teacher_classifier_state, strict=False)
                    print("[KD] Loaded teacher classifier into student.")
                except Exception as e:
                    print(f"[KD] warn: couldn't load teacher classifier: {e}")

        best_val_loss = float('inf'); wait = 0; best_state = None

        mse = torch.nn.MSELoss()

        def _kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
            t = kd_temperature
            if self.problem_type == "multi_class_classification":
                # KL divergence for multiclass
                s = F.log_softmax(student_logits / t, dim=-1)
                y = F.softmax(teacher_logits / t, dim=-1)
                loss = F.kl_div(s, y, reduction='batchmean')
            else:
                # temperature-soft BCE for binary
                s = torch.sigmoid(student_logits / t).squeeze(-1)
                y = torch.sigmoid(teacher_logits / t).squeeze(-1)
                eps = 1e-6
                loss = -(y * torch.log(s + eps) + (1 - y) * torch.log(1 - s + eps)).mean()
            return (t * t) * loss

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.neural_network.train()
            epoch_loss = 0.0

            kd_cursor = 0  # rolling cursor into kd_targets
            kd_logit_cursor = 0  # for combined mode
            kd_feat_cursor = 0   # for combined mode

            for step, batch in enumerate(train_loader):
                if len(batch) == 3:
                    input_ids, attention_mask, labels = batch
                elif len(batch) == 2:
                    input_ids, attention_mask = batch
                    labels = None
                else:
                    raise ValueError("Unexpected batch format for truncated training")

                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                if labels is not None:
                    labels = labels.to(self.device, non_blocking=True)

                logits = self.neural_network(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # supervised loss (binary / multi-class)
                if labels is not None:
                    if self.problem_type == "binary_classification":
                        sup_loss = F.binary_cross_entropy_with_logits(
                            logits.squeeze(1), labels.to(self.device, non_blocking=True).float()
                        )
                    elif self.problem_type == "multi_class_classification":
                        sup_loss = F.cross_entropy(logits, labels.to(self.device, non_blocking=True).long())
                    else:
                        sup_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    sup_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # KD loss
                kd_loss = torch.tensor(0.0, device=self.device)
                if use_kd:
                    B = input_ids.size(0)

                    if kd_type == "combined":
                        # Combined uses separate target tensors; skip shared kd_batch
                        pass
                    else:
                        kd_batch = kd_targets_train[kd_cursor: kd_cursor + B].to(self.device, non_blocking=True)
                        kd_cursor += B

                    if kd_type == "logit":
                        kd_loss = _kd_logit_loss(logits, kd_batch)

                    elif kd_type == "feature":
                        # Student post-FeatureHead features
                        stu_feat = self._student_features(
                            input_ids,
                            attention_mask,
                        )  # [B, H]
                        if self._kd_project is not None:
                            stu_feat = self._kd_project(stu_feat)  # [B, H_t]

                        # Cosine distance (default) or normalized MSE
                        stu   = F.normalize(stu_feat, dim=-1, eps=1e-6)
                        teach = F.normalize(kd_batch, dim=-1, eps=1e-6)

                        if use_cosine_kd:
                            kd_loss = kd_feature_weight * (1.0 - F.cosine_similarity(stu, teach, dim=-1).mean())
                        else:
                            kd_loss = kd_feature_weight * mse(stu, teach)

                    elif kd_type == "combined":
                        # Simultaneous logit + feature KD
                        B = input_ids.size(0)
                        logit_batch = kd_logit_targets_train[kd_logit_cursor: kd_logit_cursor + B].to(self.device, non_blocking=True)
                        feat_batch = kd_feat_targets_train[kd_feat_cursor: kd_feat_cursor + B].to(self.device, non_blocking=True)
                        kd_logit_cursor += B
                        kd_feat_cursor += B

                        # Logit component
                        logit_loss = _kd_logit_loss(logits, logit_batch)

                        # Feature component
                        stu_feat = self._student_features(input_ids, attention_mask)
                        if self._kd_project is not None:
                            stu_feat = self._kd_project(stu_feat)
                        stu = F.normalize(stu_feat, dim=-1, eps=1e-6)
                        teach = F.normalize(feat_batch, dim=-1, eps=1e-6)
                        if use_cosine_kd:
                            feat_loss = kd_feature_weight * (1.0 - F.cosine_similarity(stu, teach, dim=-1).mean())
                        else:
                            feat_loss = kd_feature_weight * mse(stu, teach)

                        kd_loss = 0.5 * logit_loss + 0.5 * feat_loss
                    else:
                        raise ValueError(f"Unknown kd_type: {kd_type}")

                loss = (1.0 - kd_alpha) * sup_loss + kd_alpha * kd_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and not (epoch == 0 and step == 0):
                    scheduler.step()

                epoch_loss += float(loss.detach().cpu())

            # === validation ===
            val_loss = None
            if val_loader is not None:
                self.neural_network.eval()
                with torch.no_grad():
                    epoch_v = 0.0
                    kd_cursor = 0
                    kd_logit_cursor = 0
                    kd_feat_cursor = 0
                    for batch in val_loader:
                        if len(batch) == 3:
                            input_ids, attention_mask, labels = batch
                        elif len(batch) == 2:
                            input_ids, attention_mask = batch
                            labels = None
                        else:
                            raise ValueError("Unexpected batch format for truncated validation")

                        input_ids = input_ids.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        if labels is not None:
                            labels = labels.to(self.device)

                        logits = self.neural_network(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                        if self.problem_type == "binary_classification" and labels is not None:
                            sup_v = F.binary_cross_entropy_with_logits(
                                logits.squeeze(1), labels.to(self.device, non_blocking=True).float()
                            )
                        else:
                            sup_v = torch.tensor(0.0, device=self.device)

                        kd_v = torch.tensor(0.0, device=self.device)
                        if use_kd and kd_targets_val is not None:
                            B = input_ids.size(0)
                            kd_batch = kd_targets_val[kd_cursor: kd_cursor + B].to(self.device, non_blocking=True)
                            kd_cursor += B

                            if kd_type == "logit":
                                kd_v = _kd_logit_loss(logits, kd_batch)

                            elif kd_type == "feature":
                                stu_feat = self._student_features(
                                    input_ids,
                                    attention_mask,
                                )
                                if self._kd_project is not None:
                                    stu_feat = self._kd_project(stu_feat)

                                stu   = F.normalize(stu_feat, dim=-1, eps=1e-6)
                                teach = F.normalize(kd_batch, dim=-1, eps=1e-6)

                                if use_cosine_kd:
                                    kd_v = kd_feature_weight * (1.0 - F.cosine_similarity(stu, teach, dim=-1).mean())
                                else:
                                    kd_v = kd_feature_weight * mse(stu, teach)

                        epoch_v += float(((1.0 - kd_alpha) * sup_v + kd_alpha * kd_v).detach().cpu())

                    val_loss = epoch_v / max(1, len(val_loader))

                # early stopping bookkeeping
                if val_loss < (best_val_loss - min_delta):
                    best_val_loss, wait = val_loss, 0
                    best_state = self.neural_network.state_dict()
                    if save_best_model_path:
                        torch.save(best_state, save_best_model_path)
                else:
                    wait += 1
                    if wait >= (patience or 0) and early_stopping:
                        break  # stop next epoch

            if on_epoch_end:
                on_epoch_end(
                    epoch,
                    {
                        "epoch": epoch,
                        "train_loss": epoch_loss / max(1, len(train_loader)),
                        "val_loss": val_loss,
                        "model": self,
                        "best_val_loss": best_val_loss,
                    },
                )

        if best_state:
            self.neural_network.load_state_dict(best_state)
