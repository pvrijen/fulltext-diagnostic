# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

from abc import ABC, abstractmethod
from torch import nn, Tensor

class BaseNeuralNetwork(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        pass

    @staticmethod
    def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Compute mean-pooling over valid tokens only.
        hidden_states: [B, T, H]
        attention_mask: [B, T] or [B, T, 1]
        """
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(-1)

        mask = attention_mask.float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    @staticmethod
    def cls_pool(hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """
        Extract the [CLS] token embedding as a global summary.
        hidden_states: [B, T, H]
        attention_mask: (unused) included for signature compatibility.
        """
        # BERT convention: CLS token is at position 0
        return hidden_states[:, 0, :]

    @staticmethod
    def pool_tokens(last_hidden_state, attention_mask, pooling: str, model_outputs=None, model_type: str|None=None):
        p = pooling.lower()
        if p == "auto":
            if model_outputs is not None and getattr(model_outputs, "pooler_output", None) is not None:
                return model_outputs.pooler_output
            # Prefer mean if there is no pooler (common for RoBERTa/DistilBERT)
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        if p == "cls":
            return last_hidden_state[:, 0, :]
        if p == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        raise ValueError(f"Unknown pooling: {pooling}")
