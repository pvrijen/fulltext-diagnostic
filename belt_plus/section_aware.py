# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# src/belt_plus/section_aware.py
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Set

import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase


_SPECIAL_TAGS: Tuple[str, ...] = ("<s>", "</s>", "<s:Unlabeled/>", "<s:unnamed/>")
_OPEN_TAG = re.compile(r"<s>(?P<name>[^<]*)</s>", flags=re.IGNORECASE)
_SELF_TAG = re.compile(r"<s:(?P<name>[^/>]+)\s*/>", flags=re.IGNORECASE)

_MAX_SECTION_VOCAB = 256
_SECTION_VOCAB: Dict[str, int] = {"unlabeled": 0}


def register_tag_tokens(
    tokenizer: PreTrainedTokenizerBase,
    maybe_encoder: Optional[PreTrainedModel] = None,
) -> int:
    missing = [
        token
        for token in _SPECIAL_TAGS
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id
    ]
    if not missing:
        return 0

    added = tokenizer.add_special_tokens({"additional_special_tokens": missing})
    if maybe_encoder is not None:
        try:
            maybe_encoder.resize_token_embeddings(len(tokenizer))
        except AttributeError:
            pass
    return added


def parse_tagged_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    cursor = 0
    current = "Unlabeled"

    while cursor < len(text):
        open_match = _OPEN_TAG.search(text, cursor)
        self_match = _SELF_TAG.search(text, cursor)
        next_match = _earliest(open_match, self_match)

        if next_match is None:
            tail = text[cursor:]
            if tail.strip():
                sections.append((current, tail.strip()))
            break

        body = text[cursor:next_match.start()]
        if body.strip():
            sections.append((current, body.strip()))

        if self_match is not None and self_match.start() == next_match.start():
            current = _normalise(self_match.group("name"))
            cursor = self_match.end()
            continue

        current = _normalise(next_match.group("name"))
        cursor = next_match.end()

    if not sections:
        stripped = text.strip()
        if stripped:
            sections.append(("Unlabeled", stripped))

    return sections


def tokenize_section_aware_truncated(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 512,
    encoder: Optional[PreTrainedModel] = None,
) -> BatchEncoding:
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    register_tag_tokens(tokenizer, encoder)

    input_batches: List[List[int]] = []

    for text in texts:
        sections = parse_tagged_sections(text)
        tokens = _concatenate_sections(tokenizer, sections)
        encoded = tokenizer.prepare_for_model(
            tokens,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
        )
        input_batches.append(encoded["input_ids"])

    # Dynamic padding: pad to longest sequence in batch (not max_length)
    longest = max(len(ids) for ids in input_batches)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids = []
    attn_masks = []
    for ids in input_batches:
        pad_len = longest - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        attn_masks.append([1] * len(ids) + [0] * pad_len)

    return BatchEncoding(
        {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
        }
    )


def tokenize_section_aware_pooled(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    max_chunks: int,
    stride: int = 0,
    minimal_chunk_len: int = 0,
    pack_small_sections: bool = True,
    encoder: Optional[PreTrainedModel] = None,
) -> BatchEncoding:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    register_tag_tokens(tokenizer, encoder)

    empty_ids, empty_attn = _empty_chunk(tokenizer, max_length)

    batch_ids: List[List[List[int]]] = []
    batch_attn: List[List[List[int]]] = []
    batch_section_ids: List[List[int]] = []
    batch_section_pos: List[List[int]] = []
    batch_section_mix: List[List[bool]] = []
    batch_mask: List[List[bool]] = []

    for text in texts:
        sections = parse_tagged_sections(text)
        chunk_tokens, chunk_sec_ids, chunk_sec_pos, chunk_sec_mix = _build_chunks(
            tokenizer,
            sections,
            max_length=max_length,
            pack_small_sections=pack_small_sections,
        )

        chunk_encodings: List[List[int]] = []
        chunk_attn: List[List[int]] = []
        chunk_ids_meta: List[int] = []
        chunk_pos_meta: List[int] = []
        chunk_mix_meta: List[bool] = []

        usable_chunks = min(len(chunk_tokens), max_chunks)
        for idx in range(usable_chunks):
            tokens_piece = chunk_tokens[idx]
            encoded = tokenizer.prepare_for_model(
                tokens_piece,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            chunk_encodings.append(encoded["input_ids"])
            chunk_attn.append(encoded["attention_mask"])
            chunk_ids_meta.append(chunk_sec_ids[idx])
            chunk_pos_meta.append(chunk_sec_pos[idx])
            chunk_mix_meta.append(chunk_sec_mix[idx])

        while len(chunk_encodings) < max_chunks:
            chunk_encodings.append(empty_ids)
            chunk_attn.append(empty_attn)
            chunk_ids_meta.append(0)
            chunk_pos_meta.append(0)
            chunk_mix_meta.append(False)

        batch_ids.append(chunk_encodings)
        batch_attn.append(chunk_attn)
        batch_section_ids.append(chunk_ids_meta)
        batch_section_pos.append(chunk_pos_meta)
        batch_section_mix.append(chunk_mix_meta)
        batch_mask.append([True] * usable_chunks + [False] * (max_chunks - usable_chunks))

    return BatchEncoding(
        {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "chunks_mask": torch.tensor(batch_mask, dtype=torch.bool),
            "section_ids": torch.tensor(batch_section_ids, dtype=torch.long),
            "section_positions": torch.tensor(batch_section_pos, dtype=torch.long),
            "section_mix_flags": torch.tensor(batch_section_mix, dtype=torch.bool),
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(name: Optional[str]) -> str:
    cleaned = (name or "").strip()
    return cleaned if cleaned else "Unlabeled"


def _earliest(*matches: Optional[re.Match]) -> Optional[re.Match]:
    candidates = [m for m in matches if m is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda m: m.start())


def _section_header(name: str) -> str:
    clean = _normalise(name)
    return f"<s>{clean}</s> "


def _section_header_tokens(tokenizer: PreTrainedTokenizerBase, name: str) -> List[int]:
    header = _section_header(name)
    return tokenizer.encode(header, add_special_tokens=False)


def _section_body_tokens(tokenizer: PreTrainedTokenizerBase, body: str) -> List[int]:
    return tokenizer.encode(body, add_special_tokens=False)


def _concatenate_sections(
    tokenizer: PreTrainedTokenizerBase,
    sections: Sequence[Tuple[str, str]],
) -> List[int]:
    tokens: List[int] = []
    for name, body in sections:
        header_ids = _section_header_tokens(tokenizer, name)
        body_ids = _section_body_tokens(tokenizer, body)
        tokens.extend(header_ids + body_ids)
    return tokens


def _build_chunks(
    tokenizer: PreTrainedTokenizerBase,
    sections: Sequence[Tuple[str, str]],
    *,
    max_length: int,
    pack_small_sections: bool,
) -> Tuple[List[List[int]], List[int], List[int], List[bool]]:
    special_skeleton = tokenizer.build_inputs_with_special_tokens([])
    usable_len = max_length - len(special_skeleton)
    usable_len = max(1, usable_len)

    pieces: List[Tuple[List[int], str]] = []
    for name, body in sections:
        header_ids = _section_header_tokens(tokenizer, name)
        body_ids = _section_body_tokens(tokenizer, body)
        section_key = _normalise(name).casefold()
        for tokens_piece in _section_pieces(
            header_ids=header_ids,
            body_ids=body_ids,
            usable_len=usable_len,
        ):
            pieces.append((tokens_piece, section_key))

    chunks: List[List[int]] = []
    chunk_section_ids: List[int] = []
    chunk_section_pos: List[int] = []
    chunk_section_mix: List[bool] = []

    section_chunk_index = defaultdict(int)

    current_tokens: List[int] = []
    current_main_key: Optional[str] = None
    current_main_id: int = 0
    current_main_pos: int = 0
    current_multi: bool = False
    current_sections: Set[str] = set()

    def reset_chunk() -> None:
        nonlocal current_tokens, current_main_key, current_main_id, current_main_pos, current_multi, current_sections
        current_tokens = []
        current_main_key = None
        current_main_id = 0
        current_main_pos = 0
        current_multi = False
        current_sections = set()

    def flush_chunk() -> None:
        if not current_tokens:
            return
        chunks.append(list(current_tokens))
        chunk_section_ids.append(current_main_id)
        chunk_section_pos.append(current_main_pos)
        chunk_section_mix.append(current_multi)
        for key in current_sections:
            section_chunk_index[key] += 1
        reset_chunk()

    for piece_tokens, section_key in pieces:
        tokens_list = list(piece_tokens)
        section_id = _section_global_id(section_key)

        if not current_tokens:
            current_tokens = tokens_list
            current_main_key = section_key
            current_main_id = section_id
            current_main_pos = section_chunk_index[section_key]
            current_sections = {section_key}
            current_multi = False
            continue

        can_pack = (
            pack_small_sections
            and len(current_tokens) + len(tokens_list) <= usable_len
        )

        if can_pack:
            current_tokens.extend(tokens_list)
            if section_key != current_main_key:
                current_multi = True
            current_sections.add(section_key)
            continue

        flush_chunk()
        current_tokens = tokens_list
        current_main_key = section_key
        current_main_id = section_id
        current_main_pos = section_chunk_index[section_key]
        current_sections = {section_key}
        current_multi = False

    flush_chunk()

    if not chunks:
        chunks.append([])
        chunk_section_ids.append(0)
        chunk_section_pos.append(0)
        chunk_section_mix.append(False)

    return chunks, chunk_section_ids, chunk_section_pos, chunk_section_mix


def _empty_chunk(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    base = tokenizer.build_inputs_with_special_tokens([])
    attn = [1] * len(base)
    if len(base) < max_length:
        pad_len = max_length - len(base)
        base = base + [tokenizer.pad_token_id] * pad_len
        attn = attn + [0] * pad_len
    return base, attn


def _section_pieces(
    *,
    header_ids: List[int],
    body_ids: List[int],
    usable_len: int,
) -> List[List[int]]:
    if usable_len <= 0:
        return [header_ids]

    if len(header_ids) >= usable_len:
        return [header_ids[:usable_len]]

    if not body_ids:
        return [header_ids]

    capacity = max(1, usable_len - len(header_ids))
    pieces: List[List[int]] = []
    idx = 0
    first_take = min(len(body_ids), capacity)
    first_tokens = header_ids + body_ids[idx : idx + first_take]
    idx += first_take
    pieces.append(first_tokens)

    while idx < len(body_ids):
        slice_ids = body_ids[idx : idx + capacity]
        idx += len(slice_ids)
        pieces.append(header_ids + slice_ids)

    return pieces


def _section_global_id(name: str) -> int:
    key = name.lower()
    if key not in _SECTION_VOCAB:
        if len(_SECTION_VOCAB) >= _MAX_SECTION_VOCAB:
            return _SECTION_VOCAB.get("unlabeled", 0)
        _SECTION_VOCAB[key] = len(_SECTION_VOCAB)
    return _SECTION_VOCAB[key]
