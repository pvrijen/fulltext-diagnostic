# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

"""Utilities to split long documents into overlapping fixed-length chunks that are
compatible with Transformer encoders.

Empty chunks are formatted as `[CLS] [SEP] [PAD] ...` instead of all-`[PAD]`.
  This guarantees that every chunk sent to the encoder contains a valid sentence
  boundary and therefore produces a stable, non-zero embedding even when the
  chunk has no real content tokens.

This is implemented inside `add_special_tokens_and_pad()`: all
chunk slots, whether they correspond to real text or are just padding, receive a
minimal valid sequence consisting of `[CLS]` and `[SEP]` followed by padding.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerBase





# -----------------------------------------------------------------------------
# Helper: detect whether a chunk has *any* real (non-special) tokens
# -----------------------------------------------------------------------------

def get_valid_chunks_mask(input_ids: torch.Tensor, special_token_ids: List[int]) -> torch.Tensor:
    """Return a boolean mask of shape *(chunks,)* where *True* means the chunk has
    at least one non-special token.
    """
    special_token_ids_tensor = torch.tensor(
        [i for i in special_token_ids if i is not None], device=input_ids.device
    )
    special_mask = torch.isin(input_ids, special_token_ids_tensor)
    if special_mask.ndim == 1:
        return torch.tensor([~special_mask.all()], device=input_ids.device)
    return ~special_mask.all(dim=1)

# -----------------------------------------------------------------------------
# Public splitting API
# -----------------------------------------------------------------------------

def transform_list_of_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int = 10,
    maximal_text_length: Optional[int] = None,
    max_chunks: int = 10,
) -> BatchEncoding:
    """Tokenize, split and pad a *list* of texts so they can be stacked into a
    single tensor batch.  Returns the trio *(input_ids, attention_mask,
    chunks_mask)* wrapped into a `BatchEncoding`.
    """
    tokens = tokenizer(
        texts,
        truncation=False,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_id_list = tokens["input_ids"]
    attention_mask_list = tokens["attention_mask"]

    model_inputs: List[Tuple[Tensor, Tensor, Tensor]] = [
        transform_single_text_batch(
            input_ids=iids,
            attention_mask=amask,
            chunk_size=chunk_size,
            stride=stride,
            minimal_chunk_length=minimal_chunk_length,
            max_chunks=max_chunks,
            tokenizer=tokenizer,
        )
        for iids, amask in zip(input_id_list, attention_mask_list)
    ]

    input_ids_tensor      = torch.stack([m[0] for m in model_inputs])
    attention_mask_tensor = torch.stack([m[1] for m in model_inputs])
    chunks_mask_tensor    = torch.stack([m[2] for m in model_inputs])

    return BatchEncoding(
        {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "chunks_mask": chunks_mask_tensor,
        }
    )


def transform_single_text_batch(
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    max_chunks: int,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Trim right-hand padding so we only operate on real tokens
    real_len = attention_mask.sum().item()
    real_tokens = input_ids[:real_len]
    real_mask   = attention_mask[:real_len]


    # Fake batch-dim required by the older helper
    tokens = {
        "input_ids": real_tokens.unsqueeze(0),
        "attention_mask": real_mask.unsqueeze(0),
    }

    id_chunks, mask_chunks, chunks_mask = split_tokens_into_smaller_chunks_jointly(
        tokens,
        chunk_size,
        stride,
        minimal_chunk_length,
        max_chunks,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    )

    final_chunk_len = chunk_size + 2  # +[CLS] +[SEP]

    input_ids_tensor, attention_mask_tensor = add_special_tokens_and_pad(
        id_chunks.unbind(0),
        mask_chunks.unbind(0),
        chunk_size=final_chunk_len,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )

    return input_ids_tensor, attention_mask_tensor, chunks_mask


def transform_single_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    maximal_text_length: Optional[int],
    max_chunks: int = 10,
) -> Tuple[Tensor, Tensor, Tensor]:
    tokens = (
        tokenize_text_with_truncation(text, tokenizer, maximal_text_length)
        if maximal_text_length
        else tokenize_whole_text(text, tokenizer)
    )

    id_chunks, mask_chunks, chunks_mask = split_tokens_into_smaller_chunks_jointly(
        tokens,
        chunk_size,
        stride,
        minimal_chunk_length,
        max_chunks,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    )

    final_chunk_len = chunk_size + 2
    input_ids, attention_mask = add_special_tokens_and_pad(
        id_chunks,
        mask_chunks,
        chunk_size=final_chunk_len,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )

    return input_ids, attention_mask, chunks_mask

# -----------------------------------------------------------------------------
# Core splitting helpers
# -----------------------------------------------------------------------------

def split_tokens_into_smaller_chunks_jointly(
    tokens: BatchEncoding,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    max_chunks: int,
    cls_token_id: int,
    sep_token_id: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Slice a **single** sequence into overlapping windows.

    Returns three tensors all with first dimension `max_chunks`:
      * `id_chunks`        – token ids (no special tokens yet)
      * `mask_chunks`      – attention masks (1 for real, 0 for pad)
      * `chunks_mask`      – boolean mask indicating *which* rows actually
                             contain content longer than `minimal_chunk_length`.
    """
    input_ids      = tokens["input_ids"][0]
    attention_mask = tokens["attention_mask"][0]

    device = input_ids.device
    dtype  = input_ids.dtype

    # Pre-allocate
    id_chunks   = torch.zeros((max_chunks, chunk_size), dtype=dtype, device=device)
    mask_chunks = torch.zeros((max_chunks, chunk_size), dtype=attention_mask.dtype, device=device)
    chunks_mask = torch.zeros(max_chunks, dtype=torch.bool, device=device)

    real_chunks = 0
    for i in range(0, len(input_ids), stride):
        if real_chunks >= max_chunks:
            break
        id_chunk   = input_ids[i : i + chunk_size]
        mask_chunk = attention_mask[i : i + chunk_size]

        # Always keep the *first* window, even if it is shorter than the
        # user-defined `minimal_chunk_length`.  This prevents us from
        # discarding the entire document when it is shorter than that
        # threshold (common for abstracts, titles, etc.)
        if len(id_chunk) < minimal_chunk_length and real_chunks > 0:
            continue

        id_chunks[real_chunks, : len(id_chunk)]   = id_chunk
        mask_chunks[real_chunks, : len(mask_chunk)] = mask_chunk
        chunks_mask[real_chunks] = True
        real_chunks += 1

    # Guarantee at least **one** chunk so downstream code never sees an empty doc
    if real_chunks == 0:
        chunks_mask[0] = True  # dummy but *valid* chunk (will become CLS+SEP)

    return id_chunks, mask_chunks, chunks_mask

# -----------------------------------------------------------------------------
# Tokenization wrappers
# -----------------------------------------------------------------------------

def tokenize_whole_text(text: str, tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )


def tokenize_text_with_truncation(
    text: str, tokenizer: PreTrainedTokenizerBase, maximal_text_length: int
) -> BatchEncoding:
    return tokenizer(
        text,
        add_special_tokens=False,
        max_length=maximal_text_length,
        truncation=True,
        return_tensors="pt",
    )

# -----------------------------------------------------------------------------
# Pad-and-special-token routine
# -----------------------------------------------------------------------------

def add_special_tokens_and_pad(
    input_id_chunks: List[Tensor] | Tensor,
    mask_chunks: List[Tensor] | Tensor,
    *,
    chunk_size: int = 512,
    pad_token_id: int | None = 0,
    cls_token_id: int | None = 101,
    sep_token_id: int | None = 102,
) -> Tuple[Tensor, Tensor]:
    """Convert raw token windows into fixed-length model inputs.

    Every row ends up as `[CLS] + tokens[:N] + [SEP] + [PAD]*(chunk_size-N-2)`.
    Even windows that have *zero* real tokens (all-pad) still get `[CLS] [SEP]`
    so that the encoder sees a valid sentence boundary.
    """

    # Accept both list[Tensor] and pre-stacked Tensor for convenience
    if isinstance(input_id_chunks, (list, tuple)):
        input_id_chunks = torch.stack(input_id_chunks)
        mask_chunks     = torch.stack(mask_chunks)

    num_chunks = input_id_chunks.size(0)
    device     = input_id_chunks.device
    dtype_ids  = input_id_chunks.dtype

    pad_token_id = pad_token_id or 0
    cls_token_id = cls_token_id or 0
    sep_token_id = sep_token_id or 0

    # Pre-allocate output tensors full of PAD
    padded_ids   = torch.full((num_chunks, chunk_size), pad_token_id,
                              dtype=dtype_ids, device=device)
    # padded_masks = torch.zeros((num_chunks, chunk_size), dtype=dtype_mask, device=device)
    padded_masks = torch.zeros((num_chunks, chunk_size), dtype=torch.long, device=device)

    for idx in range(num_chunks):
        ids  = input_id_chunks[idx]
        mask = mask_chunks[idx]

        # Determine how many *real* tokens are present in this window
        content_len = int(mask.sum().item())
        if content_len > 0:
            ids = ids[:content_len]
        else:
            # Explicitly clear ids to guarantee empty slice below
            ids = ids[:0]

        # Cap to available space (-2 for CLS+SEP)
        content_len = min(content_len, chunk_size - 2)

        # [CLS]
        padded_ids[idx, 0] = cls_token_id

        # tokens[ :content_len ]
        if content_len > 0:
            padded_ids[idx, 1 : 1 + content_len] = ids[:content_len]

        # [SEP]
        padded_ids[idx, 1 + content_len] = sep_token_id

        # Attention mask (CLS + content + SEP)
        # padded_masks[idx, : content_len + 2] = 1
        padded_masks[idx, : content_len + 2] = 1

    return padded_ids, padded_masks

# -----------------------------------------------------------------------------
# (Optional) legacy helper – keep for backward compatibility
# -----------------------------------------------------------------------------

def stack_tokens_from_all_chunks(input_id_chunks: List[Tensor], mask_chunks: List[Tensor]) -> Tuple[Tensor, Tensor]:
    input_ids = torch.stack(input_id_chunks).long()
    attention_mask = torch.stack(mask_chunks).int()
    return input_ids, attention_mask
