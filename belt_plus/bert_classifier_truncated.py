# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# src/belt_plus/bert_classifier_truncated.py
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, List, Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from belt_plus.bert_truncated import BertTruncated, TruncatedDataset


class BertClassifierTruncated(BertTruncated):
    """
    Truncated (student) classifier with optional knowledge distillation (KD).

    Feature-KD compares student post-FeatureHead features to teacher features.
    If the teacher feature dimension != student hidden size, we lazily create
    a projector H -> H_t and register it with the optimizer.
    """

    def __init__(
        self,
        num_labels: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        neural_network: Optional[torch.nn.Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        preloaded_state_dict: Optional[dict] = None,
        device: str = "cuda:0",
        many_gpus: bool = False,
        dropout_rate: float = 0.3,
        use_amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "auto",
        weight_decay: float = 0.0,
        class_weights: Optional[List[float]] = None,
        problem_type: str = "binary_classification",
        log_signature: bool = False,
        token_pooling_strategy: str = "auto",
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
        super().__init__(
            num_labels=num_labels,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            neural_network=neural_network,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            preloaded_state_dict=preloaded_state_dict,
            device=device,
            many_gpus=many_gpus,
            dropout_rate=dropout_rate,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            weight_decay=weight_decay,
            class_weights=class_weights,
            problem_type=problem_type,
            token_pooling_strategy=token_pooling_strategy,
            log_signature=log_signature,
            section_aware_tokenization=section_aware_tokenization,
            max_length=max_length,
            fused_adamw=fused_adamw,
            use_torch_compile=use_torch_compile,
            torch_compile_backend=torch_compile_backend,
            torch_compile_mode=torch_compile_mode,
            flash_attention=flash_attention,
            gradient_checkpointing=gradient_checkpointing,
            warmup_ratio=warmup_ratio,
            prefetch_factor=prefetch_factor,
            max_workers=max_workers,
            reference_compile=reference_compile,
        )

        # keep params snapshot consistent with the base class
        self._params.update({"num_labels": self.num_labels})

        # KD projector (lazy-created when needed to match teacher feature dim)
        self._kd_project: Optional[torch.nn.Linear] = None

    def _create_dataset(self, tokens, labels=None):
        return TruncatedDataset(tokens, labels)

    # --------- Inference helpers ----------
    def predict(self, x: List[str], batch_size: Optional[int] = None) -> Tensor:
        logits = self._predict_logits(x, batch_size=batch_size)

        if self.problem_type == "binary_classification":
            probs = torch.sigmoid(logits).squeeze(dim=1)
            preds = (probs >= 0.5).long()
            return preds
        elif self.problem_type == "multi_class_classification":
            return torch.argmax(logits, dim=1)

        raise ValueError(f"predict() not defined for {self.problem_type}.")

    def predict_scores(self, x: List[str], batch_size: Optional[int] = None) -> Tensor:
        logits = self._predict_logits(x, batch_size=batch_size)

        if self.problem_type == "binary_classification":
            return torch.sigmoid(logits).squeeze(dim=1)
        elif self.problem_type == "multi_class_classification":
            return torch.softmax(logits, dim=1)

        raise ValueError(f"predict_scores() not defined for {self.problem_type}.")

    # --------- KD plumbing ----------
    def _student_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Student features at the KD anchor (post FeatureHead), shape [B, H].
        """
        # TruncationNeuralNetwork exposes forward_features
        return self.neural_network.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _ensure_kd_projector(self, teacher_feat_dim: int, optimizer: AdamW):
        """
        Lazily create a linear projector H -> teacher_feat_dim and register its
        parameters with the existing optimizer.
        """
        student_hidden = getattr(self.neural_network, "hidden_size", None)
        if student_hidden is None:
            student_hidden = getattr(self.neural_network.encoder.config, "hidden_size", None)
        if student_hidden is None:
            raise RuntimeError("Cannot detect student hidden size for KD projector.")

        # If dims already match, we don't need a projector
        if teacher_feat_dim == int(student_hidden):
            self._kd_project = None
            return

        if self._kd_project is not None:
            in_f, out_f = self._kd_project.in_features, self._kd_project.out_features
            if in_f == int(student_hidden) and out_f == int(teacher_feat_dim):
                return  # already correct

        self._kd_project = torch.nn.Linear(int(student_hidden), int(teacher_feat_dim)).to(self.device)
        optimizer.add_param_group(
            {"params": self._kd_project.parameters(), "lr": self.learning_rate, "weight_decay": self.weight_decay}
        )

    # --------- Training with optional KD ----------
    def fit(
        self,
        x_train: List[str],
        y_train: List[int],
        x_val: Optional[List[str]] = None,
        y_val: Optional[List[int]] = None,
        epochs: Optional[int] = None,
        patience: Optional[int] = 2,
        min_delta: float = 0.0,
        save_best_model_path: Optional[str] = None,
        on_epoch_end: Optional[Callable[[int, dict], None]] = None,
        early_stopping: bool = True,
        # --- KD additions ---
        kd_type: str = "none",                              # "none" | "logit" | "feature" | "relational" | "ranking"
        kd_targets_train: Optional[torch.Tensor] = None,    # logits (B,1) or features (B,Ht)
        kd_targets_val: Optional[torch.Tensor] = None,      # optional for val monitoring
        kd_alpha: float = 0.5,                              # KD weight vs supervised
        kd_temperature: float = 2.0,                        # temperature for logit KD
        kd_feature_weight: float = 1.0,                     # scale for feature KD (MSE or cosine)
        use_cosine_kd: bool = True,                         # if True, use cosine distance; else normalized MSE
        teacher_feature_head_state: Optional[dict] = None,  # optional warm-start of tails
        teacher_classifier_state: Optional[dict] = None,
        kd_sample_weights: Optional[torch.Tensor] = None,  # per-sample KD weight [N], overrides kd_alpha per sample
    ):
        if epochs is None:
            epochs = self.epochs

        # Tokenize + datasets
        train_tokens = self._tokenize(x_train)
        train_dataset = self._create_dataset(train_tokens, y_train)
        use_kd = (kd_type != "none") and (kd_targets_train is not None)

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
        # Compute total_steps in OPTIMIZER steps (not micro-batches).
        # scheduler.step() is called per optimizer step (every accumulation_steps),
        # so total_steps must count optimizer steps to get correct warmup/decay.
        micro_batches_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = micro_batches_per_epoch // self.accumulation_steps
        if micro_batches_per_epoch % self.accumulation_steps != 0:
            optimizer_steps_per_epoch += 1  # partial accumulation at end of epoch
        total_steps = max(1, optimizer_steps_per_epoch * epochs)
        # Read warmup_ratio from config (matching BertBase.fit), default 5%
        try:
            warmup_ratio = float(self._params.get("warmup_ratio", 0.05))
        except Exception:
            warmup_ratio = 0.05
        warmup_ratio = max(0.0, min(warmup_ratio, 0.5))
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        self.scheduler = scheduler

        # If feature KD: prepare projector to teacher dim (and register with optimizer)
        if kd_type == "feature" and kd_targets_train is not None:
            teacher_dim = int(kd_targets_train.size(-1))
            self._ensure_kd_projector(teacher_dim, optimizer)

            # (Optional) initialize student head/classifier from teacher to align spaces
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

        best_val_loss = float("inf")
        wait = 0
        best_state = None
        self.training_history = []
        self.best_epoch = None

        mse = torch.nn.MSELoss()

        def kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                          sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Logit KD with temperature. Supports both binary and multiclass.
            Optional per-sample weights for confidence-aware KD (Shi et al. 2024).
            """
            t = kd_temperature
            if self.problem_type == "multi_class_classification":
                s = F.log_softmax(student_logits / t, dim=-1)
                y = F.softmax(teacher_logits / t, dim=-1)
                loss = F.kl_div(s, y, reduction='batchmean')
            else:
                s = torch.sigmoid(student_logits / t).squeeze(-1)   # [B]
                y = torch.sigmoid(teacher_logits / t).squeeze(-1)   # [B]
                eps = 1e-6
                per_sample = -(y * torch.log(s + eps) + (1 - y) * torch.log(1 - s + eps))
                if sample_weights is not None:
                    loss = (per_sample * sample_weights).mean()
                else:
                    loss = per_sample.mean()
            return (t * t) * loss

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.neural_network.train()
            epoch_loss = 0.0

            # keep a rolling cursor into kd_targets per epoch
            kd_cursor = 0
            kd_weight_cursor = 0

            for step, batch in enumerate(train_loader):
                if len(batch) == 3:
                    input_ids, attention_mask, labels = batch
                elif len(batch) == 2:
                    input_ids, attention_mask = batch
                    labels = None
                else:
                    raise ValueError("Unexpected batch format for truncated training")

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)

                # AMP autocast (matching BertBase._train_single_epoch)
                amp_ctx = torch.amp.autocast(**self._autocast_kwargs()) if self.use_amp else nullcontext()
                with amp_ctx:
                    logits = self.neural_network(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    # supervised loss
                    if labels is not None:
                        if self.problem_type == "binary_classification":
                            sup_loss = F.binary_cross_entropy_with_logits(
                                logits.squeeze(1), labels.float()
                            )
                        elif self.problem_type == "multi_class_classification":
                            sup_loss = F.cross_entropy(logits, labels.long())
                        else:
                            sup_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        sup_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # KD loss
                kd_loss = torch.tensor(0.0, device=self.device)
                if kd_type != "none" and kd_targets_train is not None:
                    B = input_ids.size(0)
                    kd_batch = kd_targets_train[kd_cursor: kd_cursor + B].to(self.device)
                    # Per-sample KD weights (confidence-aware)
                    sw_batch = None
                    if kd_sample_weights is not None:
                        sw_batch = kd_sample_weights[kd_weight_cursor: kd_weight_cursor + B].to(self.device)
                    kd_weight_cursor += B
                    kd_cursor += B

                    if kd_type == "logit":
                        kd_loss = kd_logit_loss(logits, kd_batch, sample_weights=sw_batch)

                    elif kd_type == "feature":
                        stu_feat = self._student_features(
                            input_ids,
                            attention_mask,
                        )  # [B,H]
                        if self._kd_project is not None:
                            stu_feat = self._kd_project(stu_feat)  # [B,Ht]

                        # Cosine (default) or normalized MSE
                        stu   = F.normalize(stu_feat, dim=-1, eps=1e-6)
                        teach = F.normalize(kd_batch,   dim=-1, eps=1e-6)

                        if use_cosine_kd:
                            kd_loss = kd_feature_weight * (1.0 - F.cosine_similarity(stu, teach, dim=-1).mean())
                        else:
                            kd_loss = kd_feature_weight * mse(stu, teach)

                    elif kd_type == "relational":
                        # Relational KD: match pairwise similarity structure, not absolute features
                        stu_feat = self._student_features(
                            input_ids,
                            attention_mask,
                        )  # [B,H]
                        if self._kd_project is not None:
                            stu_feat = self._kd_project(stu_feat)  # [B,Ht]

                        # Normalize features
                        stu_norm = F.normalize(stu_feat, dim=-1, eps=1e-6)   # [B,H]
                        teach_norm = F.normalize(kd_batch, dim=-1, eps=1e-6)  # [B,Ht]

                        # Compute pairwise similarity matrices
                        stu_sim = torch.mm(stu_norm, stu_norm.t())      # [B,B]
                        teach_sim = torch.mm(teach_norm, teach_norm.t())  # [B,B]

                        # Apply temperature and convert to distributions (row-wise softmax)
                        stu_dist = F.softmax(stu_sim / kd_temperature, dim=-1)
                        teach_dist = F.softmax(teach_sim / kd_temperature, dim=-1)

                        # KL divergence between similarity distributions
                        kd_loss = kd_feature_weight * F.kl_div(
                            stu_dist.log(), teach_dist, reduction='batchmean'
                        )

                    elif kd_type == "ranking":
                        # Ranking KD: match relative orderings based on teacher logits
                        # kd_batch contains teacher logits [B,1]
                        # Student should learn to produce same ranking

                        # Student logits already computed above
                        stu_logits = logits.squeeze(-1)  # [B]
                        teach_logits = kd_batch.squeeze(-1)  # [B]

                        # ListNet-style ranking loss (Cao et al. 2007)
                        # Use log_softmax for numerical stability (avoids log(0) = -inf)
                        # NOTE: reduction='sum' makes loss scale with batch size; results
                        # reported in thesis use this setting (ranking KD is neutral at +0.003).
                        stu_log_dist = F.log_softmax(stu_logits / kd_temperature, dim=0)
                        teach_dist = F.softmax(teach_logits / kd_temperature, dim=0)

                        # KL divergence between ranking distributions
                        kd_loss = kd_feature_weight * F.kl_div(
                            stu_log_dist, teach_dist, reduction='sum'
                        )

                    else:
                        raise ValueError(f"Unknown kd_type: {kd_type}")

                loss = (1.0 - kd_alpha) * sup_loss + kd_alpha * kd_loss
                loss /= self.accumulation_steps

                loss.backward()
                if ((step + 1) % self.accumulation_steps == 0) or (step + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), self.max_grad_norm)
                    # If projector exists, its params are already in the optimizer param groups.
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None and not (epoch == 0 and step == 0):
                        scheduler.step()

                epoch_loss += loss.item()

                if ((step + 1) % 100 == 0) or (step + 1 == len(train_loader)):
                    print(f"[Train] Epoch {epoch} | Step {step + 1}/{len(train_loader)} | Loss: {loss.item():.5f}")

            # === validation ===
            val_loss = None
            if val_loader is not None:
                self.neural_network.eval()
                with torch.no_grad():
                    epoch_v = 0.0
                    kd_cursor = 0
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

                        amp_ctx = torch.amp.autocast(**self._autocast_kwargs()) if self.use_amp else nullcontext()
                        with amp_ctx:
                            logits = self.neural_network(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )

                        if self.problem_type == "binary_classification" and labels is not None:
                            sup_v = F.binary_cross_entropy_with_logits(
                                logits.squeeze(1), labels.to(self.device).float()
                            )
                        else:
                            sup_v = torch.tensor(0.0, device=self.device)

                        kd_v = torch.tensor(0.0, device=self.device)
                        if kd_type != "none" and kd_targets_val is not None:
                            B = input_ids.size(0)
                            kd_batch = kd_targets_val[kd_cursor: kd_cursor + B].to(self.device)
                            kd_cursor += B

                            if kd_type == "logit":
                                kd_v = kd_logit_loss(logits, kd_batch)

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
                    self.best_epoch = epoch
                    if save_best_model_path:
                        torch.save(best_state, save_best_model_path)
                else:
                    wait += 1

            # Track training history
            train_loss_avg = epoch_loss / max(1, len(train_loader))
            self.training_history.append({
                "epoch": epoch,
                "train_loss": float(train_loss_avg),
                "val_loss": float(val_loss) if val_loss is not None else None,
                "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
            })

            # Check early stopping after recording history
            if val_loader is not None and wait >= (patience or 0) and early_stopping:
                break

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
