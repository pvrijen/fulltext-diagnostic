# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# src/belt_plus/neural_networks/pooling_nn.py
from __future__ import annotations

from typing import Optional
import os
import math
import logging
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base_nn import BaseNeuralNetwork


class _GradScale(torch.autograd.Function):
    """Scale gradient in backward pass without changing the forward value."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor) -> Tensor:
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (scale,) = ctx.saved_tensors
        return grad_output * scale, None


class FeatureHead(nn.Module):
    """
    Small shared tail used by both teacher (pooled) and student (truncated).
    Keeps H constant so the classifier sees an identical feature space.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, H]


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max_len))
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        if positions is None:
            C = x.size(1)
            pe = self.pe[:C].unsqueeze(0).to(device=x.device, dtype=x.dtype)
            return x + pe
        max_index = self.pe.size(0) - 1
        idx = positions.clamp(min=0, max=max_index).to(torch.long)
        pe = self.pe.index_select(0, idx.view(-1)).view_as(x).to(device=x.device, dtype=x.dtype)
        return x + pe


class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    """Transformer encoder layer that caches attention weights for logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights: Optional[Tensor] = None

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        y, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        # Cache detached weights for external inspection
        self.last_attn_weights = attn_weights.detach()
        return self.dropout1(y)


class LightweightGlobalAttention(nn.Module):
    """LayerNorm + multi-head attention mixer over global tokens."""

    def __init__(
        self,
        *,
        hidden_size: int,
        nhead: Optional[int] = None,
        dropout: float = 0.1,
        max_tokens: int = 4096,
        pos_encoding: str = "sin",
        log_stats: bool = False,
    ) -> None:
        super().__init__()
        if nhead is None or (hidden_size % (nhead or 1)) != 0:
            for cand in (8, 6, 4, 3, 2):
                if hidden_size % cand == 0:
                    nhead = cand
                    break
            else:
                nhead = 1
        self.nhead = int(nhead)
        self.hidden_size = hidden_size
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.post_ln = nn.LayerNorm(hidden_size)
        self.pos_encoding = pos_encoding if pos_encoding in {"sin", "alibi", "none"} else "sin"
        self.pos = (
            SinusoidalPositionalEncoding(hidden_size, max_len=max_tokens)
            if self.pos_encoding == "sin"
            else None
        )
        self.log_stats = bool(log_stats)
        self.last_attn_weights: Optional[Tensor] = None

    def forward(self, tokens: Tensor, mask: Optional[Tensor], positions: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            key_padding = ~mask.bool()
        else:
            key_padding = None

        x = tokens
        if self.pos_encoding == "sin" and self.pos is not None:
            x = self.pos(x, positions)
        x = self.pre_ln(x)

        attn_bias = None
        if self.pos_encoding == "alibi":
            attn_bias = self._build_alibi_bias(mask, positions, x)

        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding,
            attn_mask=attn_bias,
            need_weights=self.log_stats,
            average_attn_weights=False,
        )
        attn_out = self.dropout(attn_out)
        x = x + attn_out
        x = self.post_ln(x)

        if self.log_stats and attn_weights is not None:
            self.last_attn_weights = attn_weights.detach().cpu()
        else:
            self.last_attn_weights = None
        return x

    def _build_alibi_bias(self, mask: Optional[Tensor], positions: Optional[Tensor], tokens: Tensor) -> Optional[Tensor]:
        B, L, _ = tokens.shape
        if L == 0:
            return None
        device = tokens.device
        dtype = tokens.dtype
        if positions is not None:
            base_pos = positions.to(device=device, dtype=dtype)
        else:
            base_pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        rel = base_pos.unsqueeze(-1) - base_pos.unsqueeze(-2)
        rel = rel.abs()
        slopes = self._get_alibi_slopes().to(device=device, dtype=dtype)
        bias = -rel.unsqueeze(1) * slopes.view(1, self.nhead, 1, 1)
        bias = bias.view(B * self.nhead, L, L)
        if mask is not None:
            pad = ~mask.bool()
            pad = pad.unsqueeze(1).expand(B, self.nhead, L).reshape(B * self.nhead, L)
            bias = bias.masked_fill(pad.unsqueeze(-1), float("-inf"))
        return bias

    def _get_alibi_slopes(self) -> Tensor:
        n = self.nhead

        def get_slopes_power_of_2(n_heads: int) -> list[float]:
            start = 2 ** (-2 ** -(math.log2(n_heads) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n_heads)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes += get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        return torch.tensor(slopes, dtype=torch.float32)


class GlobalTokensTransformer(nn.Module):
    """Transformer running over global sparse tokens."""

    def __init__(
        self,
        *,
        hidden_size: int,
        nhead: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        max_tokens: int = 4096,
        pos_encoding: str = "sin",
        log_stats: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if nhead is None or (hidden_size % (nhead or 1) != 0):
            for h in (12, 8, 6, 4, 3, 2):
                if hidden_size % h == 0:
                    nhead = h
                    break
            else:
                nhead = 1
        self.nhead = nhead

        d_ff = ffn_mult * hidden_size
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithWeights(
                    d_model=hidden_size,
                    nhead=self.nhead,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size) if num_layers > 0 else nn.Identity()
        self.pos = SinusoidalPositionalEncoding(hidden_size, max_len=max_tokens)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.pos_encoding = pos_encoding if pos_encoding in {"sin", "alibi", "none"} else "sin"
        self.log_stats = bool(log_stats)
        self.latest_attn_weights: Optional[list[Tensor]] = None

    def forward(self, tokens: Tensor, mask: Optional[Tensor], positions: Optional[Tensor] = None) -> Tensor:
        # tokens: [B, L, H], mask: [B, L] where True = valid
        if mask is not None:
            key_pad = ~(mask.bool())
        else:
            key_pad = None

        if self.pos_encoding == "sin":
            x = self.pos(tokens, positions)
        elif self.pos_encoding == "none":
            x = tokens
        else:
            x = tokens
        x = self.pre_ln(x)

        attn_bias = None
        if self.pos_encoding == "alibi":
            attn_bias = self._build_alibi_bias(mask, positions, x)

        latest_weights: list[Tensor] = []
        for layer in self.layers:
            x = layer(x, src_mask=attn_bias, src_key_padding_mask=key_pad)
            attn = getattr(layer, "last_attn_weights", None)
            if attn is not None:
                latest_weights.append(attn.detach().cpu())
        x = self.final_norm(x)
        if self.log_stats:
            self.latest_attn_weights = latest_weights if latest_weights else None
        return x

    def clear_latest_stats(self) -> None:
        self.latest_attn_weights = None

    def _build_alibi_bias(self, mask: Optional[Tensor], positions: Optional[Tensor], tokens: Tensor) -> Optional[Tensor]:
        B, L, _ = tokens.shape
        if L == 0:
            return None
        device = tokens.device
        dtype = tokens.dtype
        if positions is not None:
            base_pos = positions.to(device=device, dtype=dtype)
        else:
            base_pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        rel = base_pos.unsqueeze(-1) - base_pos.unsqueeze(-2)
        rel = rel.abs()
        slopes = self._get_alibi_slopes().to(device=device, dtype=dtype)
        bias = -rel.unsqueeze(1) * slopes.view(1, self.nhead, 1, 1)
        bias = bias.view(B * self.nhead, L, L)
        return bias

    def _get_alibi_slopes(self) -> Tensor:
        # Based on GPT-NeoX/LLAMA rope conversion
        n = self.nhead
        def get_slopes(n_heads: int) -> list[float]:
            # Implementation from Deepspeed
            def get_slopes_power_of_2(n):
                start = 2 ** (-2 ** -(math.log2(n) - 3))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]

        slopes = torch.tensor(get_slopes(n), dtype=torch.float32)
        return slopes


class PoolingNeuralNetwork(BaseNeuralNetwork):
    """
    Encoder + token pooling (inside chunk) + OPTIONAL between-chunk Transformer
    + chunk pooling (across chunks) + shared FeatureHead + classifier.

    Post-feature extraction ALWAYS yields a vector of size H (encoder hidden_size),
    regardless of the chunk_pooling_strategy. For strategies that produce 2H
    (e.g., meanmax), we project back to H with a small learnable projector.
    After that, a shared FeatureHead(H→H) is applied so that teacher/student
    tails are identical for feature-KD.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        num_labels: int,
        token_pooling_strategy: str = "auto",
        chunk_pooling_strategy: str = "mean",
        chunk_size: int = 512,
        stride: int = 512,
        max_chunks: int = 10,
        dropout_rate: float = 0.1,
        use_chunk_transform: bool = False,
        first_chunk_only: bool = False,
        normalize_chunks: bool = True,
        # Transformer knobs (used only if use_chunk_transform=True)
        chunk_transform_layers: int = 1,
        chunk_transform_ffn_mult: int = 4,
        chunk_transform_dropout: float = 0.1,
        chunk_transform_nhead: Optional[int] = None,
        global_tokens_per_chunk: int = 64,  # Number of evenly-spaced global tokens per chunk
        chunk_transform_arch: str = "transformer",
        chunk_transform_mode: str = "residual_mean",
        chunk_transform_pos_encoding: str = "sin",
        # Debug
        debug: bool = False,
        log_signature: bool = False,
        use_section_chunk_embeddings: bool = False,
        section_chunk_embedding_dim: int = 32,
        section_chunk_vocab_size: int = 128,
        section_chunk_position_size: int = 32,
        use_chunk_relation_attention: bool = False,
    ):
        super().__init__()
        self._debug_enabled = debug or (os.environ.get("BELT_DEBUG", "0") == "1")
        self._debug_forward_once = False
        self._global_transform_logged = False
        self._global_transform_debug_once = False
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.token_pooling_strategy = str(token_pooling_strategy).lower()
        self.chunk_pooling_strategy = str(chunk_pooling_strategy).lower()
        self.chunk_size = int(chunk_size)
        self.stride = int(stride)
        self.max_chunks = int(max_chunks)
        self.first_chunk_only = bool(first_chunk_only)
        self.normalize_chunks = bool(normalize_chunks)
        raw_use_chunk_transform = bool(use_chunk_transform)
        self.use_section_chunk_embeddings = bool(use_section_chunk_embeddings)
        self.section_chunk_embedding_dim = int(section_chunk_embedding_dim)
        self.section_chunk_vocab_size = int(section_chunk_vocab_size)
        self.section_chunk_position_size = int(section_chunk_position_size)
        self.use_chunk_relation_attention = bool(use_chunk_relation_attention)

        # Chunk-attention interpretability buffer (attn-k pooling only)
        self._last_chunk_attn: Optional[Tensor] = None
        self._collect_chunk_attn: bool = False
        self._chunk_attn_buffer: list[Tensor] = []

        # Hidden size H of the encoder
        try:
            self.hidden_size = getattr(self.encoder.config, "hidden_size")
        except Exception:
            self.hidden_size = (
                getattr(self.encoder, "hidden_size", None)
                or getattr(getattr(self.encoder, "config", None), "hidden_size", 768)
            )

        # --- Optional cross-chunk transform ---
        mode_normalized = (chunk_transform_mode or "").strip().lower()
        if mode_normalized not in {"residual_mean", "cross_attn", "global_only", "chunk_self_attn", "none", ""}:
            mode_normalized = "residual_mean"
        if not mode_normalized:
            # Empty or None mode: use_chunk_transform flag decides
            mode_normalized = "residual_mean" if raw_use_chunk_transform else "none"
        elif not raw_use_chunk_transform and mode_normalized == "residual_mean":
            # BUG FIX: The default parameter value chunk_transform_mode="residual_mean"
            # was silently overriding use_chunk_transform=False. When the caller
            # explicitly sets use_chunk_transform=False (or doesn't set it, defaulting
            # to False), and the mode is still at its default "residual_mean", disable.
            # Non-default modes (chunk_self_attn, cross_attn, global_only) from Hydra
            # configs still correctly enable the transform.
            mode_normalized = "none"
        if mode_normalized == "none":
            self.use_chunk_transform = False
        else:
            self.use_chunk_transform = True
        self.global_transform_mode = mode_normalized
        arch_normalized = (chunk_transform_arch or "transformer").strip().lower()
        if arch_normalized not in {"transformer", "light_attn"}:
            arch_normalized = "transformer"
        self.global_transform_arch = arch_normalized
        self.global_pos_encoding = str(chunk_transform_pos_encoding or "sin").lower()
        if self.global_pos_encoding not in {"sin", "alibi", "none"}:
            self.global_pos_encoding = "sin"
        # Number of evenly-spaced global tokens per chunk (first and last always included)
        self._global_tokens_per_chunk = max(2, int(global_tokens_per_chunk or 64))
        max_global_tokens = max(1, self.max_chunks * self._global_tokens_per_chunk)
        max_position = max(1, (self.max_chunks - 1) * max(self.stride, 1) + self.chunk_size)

        self.chunk_self_attn_encoder = None  # overridden below for chunk_self_attn
        self.chunk_pos_embed = None  # learnable position embeddings for chunk_self_attn

        if self.use_chunk_transform:
            if mode_normalized == "chunk_self_attn":
                # Standard HAT-style: TransformerEncoder over chunk vectors [B, C, H].
                # Each chunk is a single token; the encoder captures inter-chunk dependencies.
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=chunk_transform_nhead,
                    dim_feedforward=self.hidden_size * chunk_transform_ffn_mult,
                    dropout=chunk_transform_dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.chunk_self_attn_encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=chunk_transform_layers,
                )
                # Learnable position embeddings for chunk ordering
                if self.global_pos_encoding != "none":
                    self.chunk_pos_embed = nn.Embedding(self.max_chunks, self.hidden_size)
                self.global_transform = None
                self.chunk_transform = None
                self.chunk_transform_prep = None
                self.global_cross_attn = None
                self.cross_attn_norm_q = None
                self.cross_attn_norm_kv = None
                self.cross_attn_norm_out = None
                self.cross_attn_ffn = None
                self.cross_attn_ffn_norm = None
            elif self.global_transform_arch == "light_attn":
                self.global_transform = LightweightGlobalAttention(
                    hidden_size=self.hidden_size,
                    nhead=chunk_transform_nhead,
                    dropout=chunk_transform_dropout,
                    max_tokens=max_position,
                    pos_encoding=self.global_pos_encoding,
                    log_stats=self._debug_enabled,
                )
            else:
                self.global_transform = GlobalTokensTransformer(
                    hidden_size=self.hidden_size,
                    nhead=chunk_transform_nhead,
                    num_layers=chunk_transform_layers,
                    dropout=chunk_transform_dropout,
                    ffn_mult=chunk_transform_ffn_mult,
                    max_tokens=max_position,
                    pos_encoding=self.global_pos_encoding,
                    log_stats=self._debug_enabled,
                )
            self.chunk_transform = self.global_transform  # backwards compatibility for legacy hooks
            if self.normalize_chunks:
                self.chunk_transform_prep = nn.LayerNorm(self.hidden_size)
            else:
                self.chunk_transform_prep = None
            if self.global_transform_mode == "cross_attn":
                nhead_cross = self.global_transform.nhead
                self.global_cross_attn = nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=nhead_cross,
                    dropout=chunk_transform_dropout,
                    batch_first=True,
                )
                # Add LayerNorms for proper transformer-style cross-attention
                self.cross_attn_norm_q = nn.LayerNorm(self.hidden_size)
                self.cross_attn_norm_kv = nn.LayerNorm(self.hidden_size)
                self.cross_attn_norm_out = nn.LayerNorm(self.hidden_size)
                # Small FFN after cross-attention (standard transformer block)
                self.cross_attn_ffn = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size * 2),
                    nn.GELU(),
                    nn.Dropout(chunk_transform_dropout),
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.Dropout(chunk_transform_dropout),
                )
                self.cross_attn_ffn_norm = nn.LayerNorm(self.hidden_size)
            else:
                self.global_cross_attn = None
                self.cross_attn_norm_q = None
                self.cross_attn_norm_kv = None
                self.cross_attn_norm_out = None
                self.cross_attn_ffn = None
                self.cross_attn_ffn_norm = None
        else:
            self.global_transform = None
            self.chunk_transform = None
            self.chunk_transform_prep = None
            self.global_cross_attn = None
            self.chunk_self_attn_encoder = None
            self.global_transform_mode = "none"

        # --- Chunk-level pooling projector ---
        self.chunk_project = nn.Identity()
        self._concat_before_project = False

        # Optional simple additive attention across chunks (legacy option)
        if self.chunk_pooling_strategy == "attn":
            self.attn_query = nn.Parameter(torch.randn(self.hidden_size))
            self.attn_w = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        else:
            self.attn_query = None
            self.attn_w = None

        # Abstract-grounded cross-attention: chunk 0 attends to body chunks
        if self.chunk_pooling_strategy in ("abstract_cross_attn", "abstract_cross_attn_sg"):
            # Two gates: one for cross-attn context, one for mean pool
            self.abs_cross_gate = nn.Parameter(torch.tensor(-5.0))    # cross-attn gate, starts ~0
            self.abs_mean_gate = nn.Parameter(torch.tensor(-5.0))     # mean pool gate, starts ~0
            # Learnable positional embeddings per chunk position
            self.abs_cross_pos_embed = nn.Embedding(self.max_chunks, self.hidden_size)
            with torch.no_grad():
                self.abs_cross_pos_embed.weight.mul_(0.01)

        if self.use_section_chunk_embeddings:
            embed_dim = self.section_chunk_embedding_dim
            self.section_embed = nn.Embedding(self.section_chunk_vocab_size, embed_dim)
            self.section_pos_embed = nn.Embedding(self.section_chunk_position_size, embed_dim)
            self.section_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.hidden_size),
            )
            self.section_mix_vector = nn.Parameter(torch.zeros(self.hidden_size))
        else:
            self.section_embed = None
            self.section_pos_embed = None
            self.section_proj = None
            self.section_mix_vector = None

        if self.use_chunk_relation_attention:
            self.chunk_rel_attn = nn.Linear(self.hidden_size, 1)
            self.chunk_attn_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.chunk_attn_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.chunk_rel_attn = None
            self.chunk_attn_proj = None
            self.chunk_attn_norm = None

        # --- Shared tail & classifier (identical to student) ---
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_head = FeatureHead(self.hidden_size, dropout=dropout_rate)
        out_dim = 1 if self.num_labels == 1 else self.num_labels
        self.classifier = nn.Linear(self.hidden_size, out_dim)

        # --- Auxiliary abstract-only loss ---
        # When aux_abstract_alpha > 0, a second forward path pools only chunk 0
        # through the SAME feature_head + classifier. No extra parameters.
        # loss_total = loss_pooled + alpha * loss_abstract
        self.aux_abstract_alpha = 0.0  # set via set_aux_abstract_alpha()

        # --- Chunk dropout ---
        # When chunk_dropout_prob > 0, randomly mask all body chunks (keep only
        # chunk 0) with this probability during training. Forces the model to
        # classify from abstract alone on a fraction of training steps.
        self.chunk_dropout_prob = 0.0  # set via set_chunk_dropout()

        # --- Contrastive abstract alignment ---
        # When contrastive_alpha > 0, align chunk 0 representation with the
        # pooled full-text representation via MSE loss. The full-text side is
        # detached so body-text gradients do NOT flow through the contrastive
        # loss. The abstract encoder learns to produce "full-text-like"
        # representations without being corrupted by body-text gradients.
        self.contrastive_alpha = 0.0  # set via set_contrastive_alpha()

        # --- Debug ---
        if self._debug_enabled:
            self._debug_dump_config(prefix="[PoolingNN:init] ")
        
        # --- signature logging ---
        self._log_signature = bool(log_signature)
        if self._log_signature:
            print("[PoolingNN:init] "
                  f"token_pool={self.token_pooling_strategy} "
                  f"chunk_pool={self.chunk_pooling_strategy} "
                  f"max_chunks={self.max_chunks} "
                  f"normalize_chunks={self.normalize_chunks} "
                  f"use_chunk_transform={self.use_chunk_transform} "
                  f"first_chunk_only={self.first_chunk_only}")

    # ----------------------------
    # Debug helpers
    # ----------------------------
    def _debug_dump_config(self, prefix: str = ""):
        try:
            enc_name = getattr(getattr(self.encoder, "config", None), "name_or_path", None) or \
                       getattr(self.encoder, "name", None) or self.encoder.__class__.__name__
        except Exception:
            enc_name = self.encoder.__class__.__name__
        msg = (
            f"{prefix}"
            f"encoder={enc_name} | H={self.hidden_size} | num_labels={self.num_labels} | "
            f"token_pooling={self.token_pooling_strategy} | chunk_pooling={self.chunk_pooling_strategy} | "
            f"max_chunks={self.max_chunks} | first_chunk_only={self.first_chunk_only} | "
            f"normalize_chunks={self.normalize_chunks} | use_chunk_transform={self.use_chunk_transform}"
        )
        if self.use_chunk_transform and self.global_transform is not None:
            ct = self.global_transform
            num_layers = len(getattr(ct, "encoder", nn.Module()).layers)
            msg += (
                f" | global_mode={self.global_transform_mode}"
                f" tokens_per_chunk={self._global_tokens_per_chunk}"
                f" layers={num_layers} nhead={getattr(ct, 'nhead', '?')}"
            )
        print(msg, flush=True)

    def _debug_dump_forward(self, input_ids: Tensor, attention_mask: Tensor, chunks_mask: Optional[Tensor]):
        B, C, T = input_ids.shape
        print(
            f"[PoolingNN:fwd] shapes: ids={tuple(input_ids.shape)} mask={tuple(attention_mask.shape)} "
            f"chunks_mask={tuple(chunks_mask.shape) if chunks_mask is not None else None} | "
            f"pool(tok)={self.token_pooling_strategy} pool(chunk)={self.chunk_pooling_strategy} "
            f"first_chunk_only={self.first_chunk_only} normalize={self.normalize_chunks} "
            f"transform={self.use_chunk_transform} mode={self.global_transform_mode if self.use_chunk_transform else 'none'}",
            flush=True
        )

    def _log_transform_stats(
        self,
        pre_norms: Tensor,
        post_norms: Tensor,
        chunks_mask: Optional[Tensor],
    ) -> None:
        try:
            logger = logging.getLogger(__name__)
            if chunks_mask is not None:
                mask_cpu = chunks_mask.detach().to(torch.bool).cpu()
                B, C = mask_cpu.shape
            else:
                total = pre_norms.numel()
                B, C = 1, total
                mask_cpu = torch.ones((B, C), dtype=torch.bool)

            pre_view = pre_norms.view(B, C)
            post_view = post_norms.view(B, C)

            valid_mask = mask_cpu
            pre_valid = pre_view[valid_mask]
            post_valid = post_view[valid_mask]

            chunk_counts = valid_mask.sum(dim=1).to(torch.float32)

            def _summary(vec: Tensor) -> tuple[float, float, float, float]:
                if vec.numel() == 0:
                    return (0.0, 0.0, 0.0, 0.0)
                return (
                    float(vec.mean().item()),
                    float(vec.std(unbiased=False).item()),
                    float(vec.min().item()),
                    float(vec.max().item()),
                )

            pre_stats = _summary(pre_valid)
            post_stats = _summary(post_valid)
            chunk_stats = _summary(chunk_counts)

            attn_layers = getattr(self.global_transform, "latest_attn_weights", None)
            attn_descriptions: list[str] = []
            if attn_layers:
                for layer_idx, attn in enumerate(attn_layers):
                    if attn is None:
                        continue
                    attn = attn.float()
                    entropy_vals: list[Tensor] = []
                    for b in range(attn.size(0)):
                        probs = attn[b].reshape(-1, attn.size(-1))
                        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
                        entropy_vals.append(entropy)
                    if entropy_vals:
                        entropy_tensor = torch.cat(entropy_vals, dim=0)
                        ent_stats = _summary(entropy_tensor)
                        attn_descriptions.append(
                            f"layer{layer_idx}: entropy=mean {ent_stats[0]:.3f}±{ent_stats[1]:.3f} "
                            f"(min {ent_stats[2]:.3f} max {ent_stats[3]:.3f})"
                        )

            logger.info(
                "[GlobalTransform] chunks=%.1f±%.1f (min=%.0f max=%.0f) | norms pre=%.4f±%.4f (min=%.4f max=%.4f) "
                "| post=%.4f±%.4f (min=%.4f max=%.4f)%s",
                chunk_stats[0], chunk_stats[1], chunk_stats[2], chunk_stats[3],
                pre_stats[0], pre_stats[1], pre_stats[2], pre_stats[3],
                post_stats[0], post_stats[1], post_stats[2], post_stats[3],
                " | attn=" + "; ".join(attn_descriptions) if attn_descriptions else "",
            )
        finally:
            self._global_transform_logged = True
            if self.global_transform is not None:
                self.global_transform.clear_latest_stats()

    def _extract_global_tokens(self, hidden_bc: Tensor, mask_bc: Tensor) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Select evenly-spaced global tokens per chunk (including start and end)."""
        T = hidden_bc.size(2)
        device = hidden_bc.device
        # Use linspace for even spacing including both endpoints
        n_tokens = min(self._global_tokens_per_chunk, T)  # Can't have more tokens than sequence length
        idx = torch.linspace(0, T - 1, n_tokens, device=device).long()
        tokens = hidden_bc[:, :, idx, :]
        mask = mask_bc[:, :, idx] > 0
        chunk_indices = torch.arange(hidden_bc.size(1), device=device).view(1, -1, 1)
        B = hidden_bc.size(0)
        C = hidden_bc.size(1)
        chunk_indices = torch.arange(C, device=device, dtype=torch.long).view(1, C, 1)
        chunk_indices = chunk_indices.expand(B, C, 1)
        positions = chunk_indices * self.stride + idx.view(1, 1, -1)
        positions = positions.contiguous()
        return tokens, mask, positions

    # ----------------------------
    # Token-level pooling per chunk
    # ----------------------------
    def _pool_tokens_per_chunk(
        self, last_hidden_state: Tensor, attention_mask: Tensor,
        model_outputs=None,
    ) -> Tensor:
        """
        last_hidden_state: [B*C, T, H]
        attention_mask:    [B*C, T]
        model_outputs:     encoder outputs (for pooler_output access in CLS pooling)
        returns:           [B*C, H]
        """
        return BaseNeuralNetwork.pool_tokens(
            last_hidden_state,
            attention_mask,
            pooling=self.token_pooling_strategy,
            model_outputs=model_outputs,
        )

    # ----------------------------
    # Chunk-level pooling per doc
    # ----------------------------
    def _pool_chunks(self, chunk_feats: Tensor, chunks_mask: Tensor) -> Tensor:
        """
        chunk_feats: [B, C, H]   (optionally normalized or transformed)
        chunks_mask: [B, C]      booleans or 0/1
        returns:     [B, H]  (or [B, 2H] before projection in meanmax)
        """
        B, C, H = chunk_feats.shape
        mask = chunks_mask.float().clamp(min=0.0, max=1.0)  # [B, C]

        if self.first_chunk_only:
            # select the first valid chunk; if none marked, fall back to index 0
            first_idx = torch.zeros(B, dtype=torch.long, device=chunk_feats.device)
            valid = mask > 0.5
            for b in range(B):
                idx = torch.nonzero(valid[b], as_tuple=False)
                if idx.numel() > 0:
                    first_idx[b] = idx[0, 0]
            return chunk_feats[torch.arange(B, device=chunk_feats.device), first_idx]  # [B,H]

        strat = self.chunk_pooling_strategy

        if strat in {"mean", "avg"}:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (chunk_feats * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,H]

        if strat == "max":
            neg_inf = torch.finfo(chunk_feats.dtype).min
            cm = mask.unsqueeze(-1)  # [B,C,1]
            masked = chunk_feats.masked_fill(cm < 0.5, neg_inf)
            pooled, _ = masked.max(dim=1)  # [B,H]
            return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

        if strat in {"meanmax", "meanmax_avg", "mean+max", "concat_meanmax", "concat"}:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            mean_pool = (chunk_feats * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,H]
            neg_inf = torch.finfo(chunk_feats.dtype).min
            cm = mask.unsqueeze(-1)
            masked = chunk_feats.masked_fill(cm < 0.5, neg_inf)
            max_pool, _ = masked.max(dim=1)  # [B,H]
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
            return (mean_pool + max_pool) / 2  # [B, H] — parameter-free

        if strat.startswith("residual_"):
            # Abstract-anchored: chunk_0 + pool(all_chunks)
            chunk_0 = chunk_feats[:, 0, :]  # [B, H] — always the abstract chunk
            sub = strat[len("residual_"):]  # mean, max, or meanmax

            if sub == "mean":
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                pool = (chunk_feats * mask.unsqueeze(-1)).sum(dim=1) / denom
            elif sub == "max":
                neg_inf = torch.finfo(chunk_feats.dtype).min
                masked = chunk_feats.masked_fill(mask.unsqueeze(-1) < 0.5, neg_inf)
                pool, _ = masked.max(dim=1)
                pool = torch.where(torch.isfinite(pool), pool, torch.zeros_like(pool))
            elif sub == "meanmax":
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                mean_pool = (chunk_feats * mask.unsqueeze(-1)).sum(dim=1) / denom
                neg_inf = torch.finfo(chunk_feats.dtype).min
                masked = chunk_feats.masked_fill(mask.unsqueeze(-1) < 0.5, neg_inf)
                max_pool, _ = masked.max(dim=1)
                max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
                pool = (mean_pool + max_pool) / 2
            else:
                raise ValueError(f"Unknown residual sub-strategy: {sub}")

            return chunk_0 + pool  # [B, H] — parameter-free

        if strat == "first":
            first_idx = torch.zeros(B, dtype=torch.long, device=chunk_feats.device)
            valid = mask > 0.5
            for b in range(B):
                idx = torch.nonzero(valid[b], as_tuple=False)
                first_idx[b] = idx[0, 0] if idx.numel() > 0 else 0
            return chunk_feats[torch.arange(B, device=chunk_feats.device), first_idx]

        if strat == "attn":
            # legacy light attention option (not the Transformer)
            Wh = torch.tanh(self.attn_w(chunk_feats))  # [B,C,H]
            q = self.attn_query.view(1, 1, H)          # [1,1,H]
            scores = (Wh * q).sum(dim=-1)              # [B,C]
            scores = scores.masked_fill(mask < 0.5, -1e9)
            alpha = F.softmax(scores, dim=1)            # [B,C]
            # Store attention weights for interpretability
            self._last_chunk_attn = alpha.detach().cpu()
            if self._collect_chunk_attn:
                self._chunk_attn_buffer.append(alpha.detach().cpu())
            return (chunk_feats * alpha.unsqueeze(-1)).sum(dim=1)  # [B,H]

        if strat in ("abstract_cross_attn", "abstract_cross_attn_sg"):
            # Abstract (chunk 0) cross-attends to body chunks via multi-head attention
            abstract = chunk_feats[:, 0, :]  # [B, H]

            if C == 1:
                return abstract  # no body chunks

            body = chunk_feats[:, 1:, :]      # [B, C-1, H]
            body_mask = mask[:, 1:]            # [B, C-1]

            # Stop-gradient variant: body chunks don't send gradient to encoder
            if strat == "abstract_cross_attn_sg":
                body = body.detach()

            # Add positional embeddings so model knows chunk order
            body_positions = torch.arange(1, C, device=body.device)
            body = body + self.abs_cross_pos_embed(body_positions).unsqueeze(0)

            # Simple scaled dot-product attention: abstract queries body
            scale = float(H) ** 0.5
            scores = torch.bmm(abstract.unsqueeze(1), body.transpose(1, 2)).squeeze(1) / scale  # [B, C-1]
            scores = scores.masked_fill(body_mask < 0.5, -1e4)
            weights = F.softmax(scores, dim=-1)        # [B, C-1]
            cross_context = (body * weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

            # Mean pool of ALL chunks (including abstract)
            all_mask = mask.float()
            denom = all_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            mean_context = (chunk_feats * all_mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, H]

            # Two-channel gated residual:
            #   abstract + gate1 * cross_attn + gate2 * mean_pool
            gate_cross = torch.sigmoid(self.abs_cross_gate)
            gate_mean = torch.sigmoid(self.abs_mean_gate)
            return abstract + gate_cross * cross_context + gate_mean * mean_context  # [B, H]

        raise ValueError(f"Unknown chunk_pooling_strategy: {self.chunk_pooling_strategy}")

    # ----------------------------
    # Public forward APIs
    # ----------------------------
    def forward_features(
        self,
        input_ids: Tensor,          # [B, C, T]
        attention_mask: Tensor,     # [B, C, T]
        chunks_mask: Optional[Tensor] = None,  # [B, C]
        section_ids: Optional[Tensor] = None,
        section_positions: Optional[Tensor] = None,
        section_mix_flags: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns post-FeatureHead features of size H for each document: [B, H].
        This is the KD feature point used by the student.
        """
        B, C, T = input_ids.shape
        device = input_ids.device

        if chunks_mask is None:
            chunks_mask = torch.ones(B, C, device=device, dtype=torch.bool)

        if self._debug_enabled and not self._debug_forward_once:
            self._debug_dump_forward(input_ids, attention_mask, chunks_mask)
            self._debug_forward_once = True

        # Flatten chunks → run encoder on [B*C, T]
        ids_flat = input_ids.view(B * C, T)
        am_flat = attention_mask.view(B * C, T)

        # Skip padding chunks: use chunks_mask (set by the chunking layer)
        # rather than attention_mask.  "Empty" padding chunks still contain
        # [CLS] and [SEP] tokens (attention_mask > 0), so an attention_mask
        # check would encode them — wasting compute and producing noisy
        # gradients from near-empty sequences.  chunks_mask is False for
        # padding slots, correctly skipping them.
        chunks_mask_flat = chunks_mask.view(B * C)  # [B*C] bool
        valid_flat = chunks_mask_flat
        n_valid = int(valid_flat.sum().item())

        if n_valid < B * C and n_valid > 0:
            # Compact: encode only valid chunks
            ids_valid = ids_flat[valid_flat]       # [n_valid, T]
            am_valid = am_flat[valid_flat]         # [n_valid, T]

            outputs = self.encoder(input_ids=ids_valid, attention_mask=am_valid)
            hidden_valid = outputs.last_hidden_state  # [n_valid, T, H]
            vecs_valid = self._pool_tokens_per_chunk(hidden_valid, am_valid, model_outputs=outputs)  # [n_valid, H]

            # Scatter back into full [B*C, H] with zeros for padding chunks
            chunk_vecs = torch.zeros(B * C, vecs_valid.size(-1),
                                     device=device, dtype=vecs_valid.dtype)
            chunk_vecs[valid_flat] = vecs_valid

            # Reconstruct full hidden and am_flat for downstream use
            # (global transform needs hidden_bc).  Padding chunks get zeros.
            H_enc = hidden_valid.size(-1)
            hidden = torch.zeros(B * C, T, H_enc, device=device, dtype=hidden_valid.dtype)
            hidden[valid_flat] = hidden_valid
        else:
            # All chunks valid (or zero — unlikely), run the standard path
            outputs = self.encoder(input_ids=ids_flat, attention_mask=am_flat)
            hidden = outputs.last_hidden_state  # [B*C, T, H]
            chunk_vecs = self._pool_tokens_per_chunk(hidden, am_flat, model_outputs=outputs)  # [B*C, H]
        log_transform = (
            self.use_chunk_transform
            and self.global_transform is not None
            and self.global_transform_mode != "none"
            and self._debug_enabled
            and not self._global_transform_logged
        )
        pre_norms = None
        if self.normalize_chunks:
            if self.global_transform is None:
                chunk_vecs = F.normalize(chunk_vecs, dim=-1, eps=1e-6)
            elif self.chunk_transform_prep is not None:
                chunk_vecs = self.chunk_transform_prep(chunk_vecs)
        if log_transform:
            pre_norms = chunk_vecs.detach().norm(dim=-1).to("cpu")

        # Unflatten chunks to [B, C, H]
        chunk_vecs = chunk_vecs.view(B, C, self.hidden_size)

        # --- Body gradient scaling (optional) ---
        # Scale body-chunk gradients during training. Two modes:
        #   BELT_BODY_GRAD_MODE=fixed (default): uniform λ via BELT_BODY_GRAD_SCALE
        #   BELT_BODY_GRAD_MODE=cosine: per-chunk λ_i = max(0, cos(h_abstract, h_i))
        # Forward pass unchanged; only backward gradient magnitudes are scaled.
        if self.training and not self.first_chunk_only:
            _body_grad_mode = os.environ.get("BELT_BODY_GRAD_MODE", "fixed")
            if _body_grad_mode == "cosine":
                abstract_rep = chunk_vecs[:, 0:1, :]   # [B, 1, H]
                body_reps = chunk_vecs[:, 1:, :]       # [B, C-1, H]
                cos_sim = F.cosine_similarity(abstract_rep, body_reps, dim=-1)  # [B, C-1]
                scale = cos_sim.detach().clamp(min=0).nan_to_num(0.0).unsqueeze(-1)  # [B, C-1, 1]
                chunk_vecs = torch.cat([
                    chunk_vecs[:, :1, :],
                    _GradScale.apply(body_reps, scale),
                ], dim=1)
            else:
                _body_scale = float(os.environ.get("BELT_BODY_GRAD_SCALE", "1.0"))
                if _body_scale < 1.0:
                    scale_t = torch.tensor(_body_scale, device=chunk_vecs.device, dtype=chunk_vecs.dtype)
                    chunk_vecs = torch.cat([
                        chunk_vecs[:, :1, :],  # abstract: full gradient
                        _GradScale.apply(chunk_vecs[:, 1:, :], scale_t),  # body: scaled
                    ], dim=1)

        # Cache pre-attention chunk 0 for contrastive alignment
        self._pre_attn_chunk0 = chunk_vecs[:, 0, :].clone()

        # --- chunk_self_attn: standard HAT-style transformer over chunk vectors ---
        if self.global_transform_mode == "chunk_self_attn":
            if self.chunk_pos_embed is not None:
                positions = torch.arange(chunk_vecs.size(1), device=chunk_vecs.device)
                chunk_vecs = chunk_vecs + self.chunk_pos_embed(positions).unsqueeze(0)
            src_key_padding_mask = ~chunks_mask.bool()  # True = ignore
            chunk_vecs = self.chunk_self_attn_encoder(
                chunk_vecs, src_key_padding_mask=src_key_padding_mask,
            )

        global_tokens = None
        global_mask = None
        doc_vecs_override: Optional[Tensor] = None
        if self.global_transform is not None and self.global_transform_mode != "none":
            hidden_bc = hidden.view(B, C, T, self.hidden_size)
            mask_bc = am_flat.view(B, C, T)
            global_tokens, global_mask, global_positions = self._extract_global_tokens(hidden_bc, mask_bc)
            if global_tokens is not None:
                B_, C_, G, H = global_tokens.shape
                tokens_flat = global_tokens.view(B_, C_ * G, H)
                mask_flat_bool = global_mask.view(B_, C_ * G)
                positions_flat = global_positions.view(B_, C_ * G) if global_positions is not None else None
                transformed = self.global_transform(tokens_flat, mask_flat_bool, positions_flat)  # [B, C*G, H]
                transformed = transformed.view(B_, C_, G, H)

                transformed_flat = transformed.view(B_, C_ * G, H)
                mask_doc_float = mask_flat_bool.to(transformed.dtype)
                sum_doc = (transformed_flat * mask_doc_float.unsqueeze(-1)).sum(dim=1)
                counts_doc = mask_doc_float.sum(dim=1)
                counts_safe = counts_doc.clamp_min(1.0)
                doc_vecs_global = sum_doc / counts_safe.unsqueeze(-1)
                fallback_mask = counts_doc <= 0
                if not self._global_transform_debug_once:
                    logger = logging.getLogger(__name__)
                    tokens_mean = float(counts_doc.mean().item())
                    tokens_min = int(counts_doc.min().item())
                    tokens_max = int(counts_doc.max().item())
                    zero_docs = int((counts_doc <= 0).sum().item())
                    logger.info(
                        "[GlobalTransform] mode=%s arch=%s tokens_per_chunk=%d tokens/doc: mean=%.2f min=%d max=%d zero_docs=%d",
                        self.global_transform_mode,
                        self.global_transform_arch,
                        self._global_tokens_per_chunk,
                        tokens_mean,
                        tokens_min,
                        tokens_max,
                        zero_docs,
                    )
                    self._global_transform_debug_once = True
                if fallback_mask.any():
                    fallback = chunk_vecs.mean(dim=1)
                    doc_vecs_global = torch.where(
                        (~fallback_mask).unsqueeze(-1),
                        doc_vecs_global,
                        fallback,
                    )
                doc_vecs_global = doc_vecs_global.to(chunk_vecs.dtype)

                if self.global_transform_mode == "cross_attn" and self.global_cross_attn is not None:
                    attn_out = torch.zeros_like(chunk_vecs)
                    valid_batches = mask_flat_bool.any(dim=1)
                    if valid_batches.any():
                        # Pre-norm Q and K/V (proper transformer-style)
                        q_normed = self.cross_attn_norm_q(chunk_vecs[valid_batches])
                        kv_normed = self.cross_attn_norm_kv(transformed_flat[valid_batches])
                        attn_local, _ = self.global_cross_attn(
                            q_normed,
                            kv_normed,
                            kv_normed,
                            key_padding_mask=(~mask_flat_bool[valid_batches]),
                        )
                        attn_local = attn_local.to(attn_out.dtype)
                        attn_out[valid_batches] = attn_local
                    # Residual connection + post-norm
                    chunk_vecs = self.cross_attn_norm_out(chunk_vecs + attn_out)
                    # FFN block with residual + post-norm
                    chunk_vecs = self.cross_attn_ffn_norm(chunk_vecs + self.cross_attn_ffn(chunk_vecs))
                elif self.global_transform_mode == "global_only":
                    doc_vecs_override = doc_vecs_global
                else:
                    # masked mean per chunk
                    mask_f = mask_flat_bool.view(B_, C_, G).to(chunk_vecs.dtype)
                    sum_tokens = (transformed * mask_f.unsqueeze(-1)).sum(dim=2)
                    counts = mask_f.sum(dim=2).clamp_min(1.0)
                    mean_tokens = sum_tokens / counts.unsqueeze(-1)
                    if self.global_transform_mode == "residual_mean":
                        chunk_vecs = chunk_vecs + mean_tokens
                    elif self.global_transform_mode == "none":
                        pass
                    else:
                        chunk_vecs = chunk_vecs + mean_tokens

                if log_transform and pre_norms is not None:
                    post_norms = chunk_vecs.detach().norm(dim=-1).to("cpu")
                    self._log_transform_stats(pre_norms, post_norms, chunks_mask)
                if self.global_transform_mode == "global_only" and fallback_mask.any():
                    logger = logging.getLogger(__name__)
                    logger.info(
                        "[GlobalTransform] global_only fallback on %d docs (using chunk mean)",
                        int(fallback_mask.sum().item()),
                    )

        if self.global_transform_mode == "global_only":
            if doc_vecs_override is None:
                doc_vecs_override = chunk_vecs.mean(dim=1)
            doc_vecs = doc_vecs_override
            doc_vecs = self.dropout(doc_vecs)
            feats = self.feature_head(doc_vecs)
            return feats

        if self.use_section_chunk_embeddings:
            if section_ids is None:
                section_ids = torch.zeros((B, C), dtype=torch.long, device=device)
            if section_positions is None:
                section_positions = torch.zeros((B, C), dtype=torch.long, device=device)
            sec_ids = torch.clamp(section_ids, min=0, max=self.section_chunk_vocab_size - 1)
            pos_ids = torch.clamp(section_positions, min=0, max=self.section_chunk_position_size - 1)
            sec_emb = self.section_embed(sec_ids)
            pos_emb = self.section_pos_embed(pos_ids)
            extra = self.section_proj(sec_emb + pos_emb)
            chunk_vecs = chunk_vecs + extra
            if section_mix_flags is not None and self.section_mix_vector is not None:
                mix = section_mix_flags.to(device=device, dtype=chunk_vecs.dtype).unsqueeze(-1)
                chunk_vecs = chunk_vecs + mix * self.section_mix_vector

        if self.use_chunk_relation_attention:
            attn_mask = chunks_mask.bool()
            logits = self.chunk_rel_attn(chunk_vecs).squeeze(-1)
            logits = logits.masked_fill(~attn_mask, float("-inf"))
            weights = torch.softmax(logits, dim=1).unsqueeze(-1)
            context = torch.sum(weights * chunk_vecs, dim=1, keepdim=True)
            chunk_vecs = self.chunk_attn_norm(chunk_vecs + self.chunk_attn_proj(context))

        # --- Abstract-only training loss (optional) ---
        # When enabled, pool only the first (abstract) chunk during training
        # so the encoder is optimised purely for abstract reading.  At
        # inference, pool all chunks so body text contributes information.
        # Controlled via BELT_ABSTRACT_ONLY_LOSS=1 environment variable.
        if (self.training and not self.first_chunk_only
                and os.environ.get("BELT_ABSTRACT_ONLY_LOSS", "0") == "1"):
            train_mask = torch.zeros_like(chunks_mask)
            train_mask[:, 0] = chunks_mask[:, 0]
            pool_mask = train_mask
        else:
            pool_mask = chunks_mask

        # --- Chunk dropout: randomly use only chunk 0 ---
        if self.training and self.chunk_dropout_prob > 0 and not self.first_chunk_only:
            if torch.rand(1).item() < self.chunk_dropout_prob:
                pool_mask = torch.zeros_like(pool_mask)
                pool_mask[:, 0] = chunks_mask[:, 0]

        # --- Adaptive pooling bypass (optional) ---
        # For documents with only 1 valid chunk (abstract-only), bypass the
        # pooling head entirely and use the single chunk vector directly.
        # This prevents padding-dilution when training on mixed PMC corpora
        # where most documents lack body text.
        # Controlled via BELT_ADAPTIVE_POOL=1 environment variable.
        if os.environ.get("BELT_ADAPTIVE_POOL", "0") == "1":
            n_valid_per_doc = pool_mask.sum(dim=1)  # [B]
            single_chunk = (n_valid_per_doc <= 1)   # [B] bool
            if single_chunk.any() and not single_chunk.all():
                # Mixed batch: split into single-chunk and multi-chunk paths
                multi_chunk = ~single_chunk
                # Multi-chunk docs: normal pooling
                doc_vecs_multi = self._pool_chunks(
                    chunk_vecs[multi_chunk], pool_mask[multi_chunk])
                # Single-chunk docs: take chunk 0 directly (no pooling)
                doc_vecs_single = chunk_vecs[single_chunk, 0, :]
                # Reassemble in original order
                H_out = doc_vecs_multi.size(-1)
                # Handle concat strategies (2H → need project for multi)
                if H_out == 2 * self.hidden_size and hasattr(self, 'chunk_project'):
                    doc_vecs_multi = self.chunk_project(doc_vecs_multi)
                    H_out = self.hidden_size
                doc_vecs = torch.zeros(B, H_out, device=device, dtype=chunk_vecs.dtype)
                doc_vecs[multi_chunk] = doc_vecs_multi
                doc_vecs[single_chunk] = doc_vecs_single
                doc_vecs = self.dropout(doc_vecs)
                feats = self.feature_head(doc_vecs)
                return feats
            elif single_chunk.all():
                # All single-chunk: skip pooling entirely
                doc_vecs = chunk_vecs[:, 0, :]
                doc_vecs = self.dropout(doc_vecs)
                feats = self.feature_head(doc_vecs)
                return feats
            # else: all multi-chunk, fall through to normal pooling

        # Cache chunk_vecs for aux abstract loss (used in forward())
        self._last_chunk_vecs = chunk_vecs

        # Pool across chunks
        doc_vecs = self._pool_chunks(chunk_vecs, pool_mask)  # [B, H] or [B, 2H]

        # If concat strategy produced 2H, project back to H
        if doc_vecs.size(-1) == 2 * self.hidden_size:
            doc_vecs = self.chunk_project(doc_vecs)  # [B, H]

        # Shared tail (identical to student) + dropout
        doc_vecs = self.dropout(doc_vecs)
        feats = self.feature_head(doc_vecs)  # [B, H]  <-- KD feature point
        return feats

    def forward(
        self,
        input_ids: Tensor,          # [B, C, T]
        attention_mask: Tensor,     # [B, C, T]
        chunks_mask: Optional[Tensor] = None,
        section_ids: Optional[Tensor] = None,
        section_positions: Optional[Tensor] = None,
        section_mix_flags: Optional[Tensor] = None,
    ) -> Tensor:
        feats = self.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunks_mask=chunks_mask,
            section_ids=section_ids,
            section_positions=section_positions,
            section_mix_flags=section_mix_flags,
        )  # [B, H]
        logits_pooled = self.classifier(feats)  # [B, num_labels] or [B,1]

        # Auxiliary abstract-only loss: pool chunk 0 only through the SAME
        # feature_head + classifier. No extra parameters, no re-encoding.
        if self.training and self.aux_abstract_alpha > 0:
            c0_vec = self._last_chunk_vecs[:, 0, :]  # [B, H]
            c0_vec = self.dropout(c0_vec)
            c0_feats = self.feature_head(c0_vec)      # same feature_head
            logits_abstract = self.classifier(c0_feats)  # same classifier
            return logits_pooled, logits_abstract

        # Contrastive: align pre-attention chunk 0 toward post-attention chunk 0.
        # With self-attention enabled, post-attention chunk 0 is "abstract
        # enriched by body context." Pre-attention chunk 0 is the raw abstract.
        # The encoder learns to produce abstract representations that approximate
        # what the contextualized version would look like.
        # Without self-attention, falls back to aligning toward pooled features.
        if self.training and self.contrastive_alpha > 0:
            c0_pre = self._pre_attn_chunk0  # [B, H] raw abstract
            if self.global_transform_mode == "chunk_self_attn":
                c0_post = self._last_chunk_vecs[:, 0, :]  # [B, H] contextualized
            else:
                c0_post = feats  # fallback: pooled features
            # Cosine with margin: loss = 0 when similarity > margin (0.8)
            cos_sim = F.cosine_similarity(c0_pre, c0_post.detach())
            contrastive_loss = torch.clamp(0.8 - cos_sim, min=0.0).mean()
            return logits_pooled, contrastive_loss

        return logits_pooled
