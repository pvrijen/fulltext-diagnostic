# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# belt_plus/neural_networks/truncation_nn.py
from __future__ import annotations

import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from belt_plus.neural_networks.base_nn import BaseNeuralNetwork
from belt_plus.neural_networks.pooling_nn import FeatureHead  # shared tail


class TruncationNeuralNetwork(nn.Module):
    """
    Student (truncated) network:
      encoder -> token pooling -> dropout -> FeatureHead(H→H) -> classifier
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        dropout_rate: float = 0.1,
        token_pooling_strategy: str = "auto",
        *,
        log_signature: bool = False,   # <- external flag to toggle prints
    ):
        super().__init__()
        self.encoder = encoder
        self.token_pooling_strategy = str(token_pooling_strategy).lower()

        # ---- signature / debug flag (set attribute BEFORE any use) ----
        env_on = os.environ.get("BELT_DEBUG", "0").lower() in {"1", "true", "yes", "y"}
        self._log_signature = bool(log_signature or env_on)
        self._sig_fwd_once = False

        # Hidden size H
        try:
            self.hidden_size = getattr(self.encoder.config, "hidden_size")
        except Exception:
            self.hidden_size = (
                getattr(self.encoder, "hidden_size", None)
                or getattr(getattr(self.encoder, "config", None), "hidden_size", 768)
            )

        # Shared tail
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_head = FeatureHead(self.hidden_size, dropout=dropout_rate)
        out_dim = 1 if int(num_labels) == 1 else int(num_labels)
        self.classifier = nn.Linear(self.hidden_size, out_dim)

        # ---- init-time signature ----
        if self._log_signature:
            enc_name = getattr(getattr(self.encoder, "config", None), "name_or_path", None) \
                       or self.encoder.__class__.__name__
            print(
                "[TruncNN:init] "
                f"encoder={enc_name} | H={self.hidden_size} | num_labels={num_labels} | "
                f"token_pooling={self.token_pooling_strategy} | dropout={dropout_rate}",
                flush=True,
            )

    # ---- forward-time signature ----
    def _dump_forward_signature(self, input_ids: Tensor, attention_mask: Tensor) -> None:
        if self._log_signature and not self._sig_fwd_once:
            print(
                "[TruncNN:fwd] shapes: "
                f"ids={tuple(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None} "
                f"mask={tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None} | "
                f"pool(tok)={self.token_pooling_strategy}",
                flush=True,
            )
            self._sig_fwd_once = True

    # ----------------------------
    # Public feature path (KD anchor)
    # ----------------------------
    def forward_features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        self._dump_forward_signature(input_ids, attention_mask)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = BaseNeuralNetwork.pool_tokens(
            outputs.last_hidden_state,
            attention_mask,
            pooling=self.token_pooling_strategy,
            model_outputs=outputs,
        )  # [B, H]
        x = self.dropout(pooled)
        feats = self.feature_head(x)  # [B, H]
        return feats

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        feats = self.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(feats)  # [B, out_dim]
        return logits

    @torch.no_grad()
    def extract_features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        return self.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def forward_minimal_basic(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
