# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# src/belt_plus/tensor_utils.py
import torch
from torch import equal, Tensor
from belt_plus.exceptions import InconsistentSplittingParamsException

def list_of_tensors_deep_equal(list_1: list[Tensor], list_2: list[Tensor]) -> bool:
    if len(list_1) != len(list_2):
        return False
    for x, y in zip(list_1, list_2):
        if not equal(x, y):
            return False
    return True

def split_overlapping(
    tensor: Tensor,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    max_chunks: int
) -> tuple[Tensor, Tensor]:
    """
    Returns a single tensor of shape (max_chunks, chunk_size) with left-padded zeros
    and a boolean mask (max_chunks,) where True indicates a 'real' chunk.
    """
    # (Optional) parameter checks
    # check_split_parameters_consistency(chunk_size, stride, minimal_chunk_length)

    device = tensor.device
    dtype = tensor.dtype
    chunks = torch.zeros((max_chunks, chunk_size), dtype=dtype, device=device)
    chunks_mask = torch.zeros(max_chunks, dtype=torch.bool, device=device)

    real = 0
    for i in range(0, len(tensor), stride):
        if real >= max_chunks:
            break
        window = tensor[i: i + chunk_size]
        if len(window) < minimal_chunk_length and real > 0:
            continue
        chunks[real, :len(window)] = window
        chunks_mask[real] = True
        real += 1

    if real == 0:
        # ensure at least one chunk exists (empty, will become CLS+SEP later)
        chunks_mask[0] = True

    return chunks, chunks_mask

def check_split_parameters_consistency(chunk_size: int, stride: int, minimal_chunk_length: int) -> None:
    if chunk_size > 510:
        raise InconsistentSplittingParamsException("Size of each chunk cannot be bigger than 510!")
    if minimal_chunk_length > chunk_size:
        raise InconsistentSplittingParamsException("Minimal length cannot be bigger than size!")
    if stride > chunk_size:
        raise InconsistentSplittingParamsException(
            "Stride cannot be bigger than size! Chunks must overlap or be near each other!"
        )