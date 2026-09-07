# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# src/belt_plus/bert_with_pooling.py
from __future__ import annotations

from typing import Optional, List, Union
import warnings
import json, os
import torch
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding, PreTrainedTokenizerBase, AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from pathlib import Path

from belt_plus.bert import BertBase
from belt_plus.splitting import transform_list_of_texts
from belt_plus.exceptions import InconsistentSplittingParamsException
from belt_plus.neural_networks.pooling_nn import PoolingNeuralNetwork
from belt_plus.section_aware import tokenize_section_aware_pooled, register_tag_tokens

from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW


def collate_fn_pooled_tokens(data):
    input_ids = torch.stack([item[0] for item in data]).long()
    attention_mask = torch.stack([item[1] for item in data]).long()
    chunks_mask = torch.stack([item[2] for item in data]).bool()
    section_ids = torch.stack([item[3] for item in data]).long()
    section_positions = torch.stack([item[4] for item in data]).long()
    section_mix_flags = torch.stack([item[5] for item in data]).bool()

    if len(data[0]) == 7:
        labels = torch.tensor([item[6] for item in data], dtype=torch.long)
        return (
            input_ids,
            attention_mask,
            chunks_mask,
            section_ids,
            section_positions,
            section_mix_flags,
            labels,
        )

    return (
        input_ids,
        attention_mask,
        chunks_mask,
        section_ids,
        section_positions,
        section_mix_flags,
    )


class ChunkedDataset(Dataset):
    """Dataset for chunked inputs with optional labels."""
    def __init__(self, tokens: BatchEncoding, labels: Optional[List[int]] = None):
        def _ensure_tensor(value, *, dtype=torch.long):
            if isinstance(value, torch.Tensor):
                return value.to(dtype=dtype)
            return torch.tensor(value, dtype=dtype)

        self.input_ids = _ensure_tensor(tokens["input_ids"], dtype=torch.long)
        self.attention_mask = _ensure_tensor(tokens["attention_mask"], dtype=torch.long)
        raw_chunks_mask = tokens["chunks_mask"]
        if isinstance(raw_chunks_mask, torch.Tensor):
            self.chunks_mask = raw_chunks_mask.bool()
        else:
            self.chunks_mask = torch.tensor(raw_chunks_mask, dtype=torch.bool)

        self.section_ids = _ensure_tensor(tokens.get("section_ids", torch.zeros_like(self.chunks_mask, dtype=torch.long)), dtype=torch.long)
        self.section_positions = _ensure_tensor(tokens.get("section_positions", torch.zeros_like(self.chunks_mask, dtype=torch.long)), dtype=torch.long)
        mix_flags = tokens.get("section_mix_flags", torch.zeros_like(self.chunks_mask, dtype=torch.bool))
        if isinstance(mix_flags, torch.Tensor):
            self.section_mix_flags = mix_flags.bool()
        else:
            self.section_mix_flags = torch.tensor(mix_flags, dtype=torch.bool)

        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (
                self.input_ids[idx],
                self.attention_mask[idx],
                self.chunks_mask[idx],
                self.section_ids[idx],
                self.section_positions[idx],
                self.section_mix_flags[idx],
                self.labels[idx],
            )
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.chunks_mask[idx],
            self.section_ids[idx],
            self.section_positions[idx],
            self.section_mix_flags[idx],
        )


class BertWithPooling(BertBase):
    """
    Handles tokenizing, chunking, and pooling setups.
    Implements the abstract methods from BertBase.
    """

    def __init__(
        self,
        num_labels: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
        chunk_pooling_strategy: str = "meanmax",
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        maximal_text_length: Optional[int] = None,
        max_chunks: int = 10,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        preloaded_state_dict: Optional[dict] = None,
        device: str = "cuda:0",
        many_gpus: bool = False,
        use_chunk_transform: bool = False,
        chunk_transform_layers: int = 1,
        chunk_transform_ffn_mult: int = 4,
        chunk_transform_dropout: float = 0.1,
        chunk_transform_nhead: Optional[int] = None,
        chunk_transform_stride: int = 128,
        global_tokens_per_chunk: int = 64,
        chunk_transform_arch: str = "transformer",
        chunk_transform_mode: str = "residual_mean",
        chunk_transform_pos_encoding: str = "sin",
        dropout_rate: float = 0.1,
        use_amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "auto",
        weight_decay: float = 0.0,
        class_weights: Optional[List[float]] = None,
        problem_type: str = "binary_classification",
        token_pooling_strategy: str = "auto",
        first_chunk_only: bool = False,
        normalize_chunks: bool = True,
        log_signature : bool = False,
        section_aware_tokenization: bool = False,
        section_aware_chunking: bool = False,
        section_pack_small_sections: bool = True,
        use_section_chunk_embeddings: bool = False,
        section_chunk_embedding_dim: int = 32,
        section_chunk_vocab_size: int = 128,
        section_chunk_position_size: int = 32,
        use_chunk_relation_attention: bool = False,
        fused_adamw: Optional[bool] = None,
        use_torch_compile: bool = False,
        torch_compile_backend: Optional[str] = None,
        torch_compile_mode: Optional[str] = None,
        flash_attention: bool = True,
        gradient_checkpointing: bool = False,
        warmup_ratio: float = 0.05,
        prefetch_factor: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        # Build default neural network if none provided
        if neural_network is None:
            if preloaded_state_dict:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
                encoder = AutoModel.from_config(config)  # avoids download
                encoder.load_state_dict(preloaded_state_dict)
            else:
                encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)

            neural_network = PoolingNeuralNetwork(
                encoder=encoder,
                chunk_size=chunk_size,
                stride=stride,
                max_chunks=max_chunks,
                use_chunk_transform=use_chunk_transform,
                chunk_transform_layers=chunk_transform_layers,
                chunk_transform_ffn_mult=chunk_transform_ffn_mult,
                chunk_transform_dropout=chunk_transform_dropout,
                chunk_transform_nhead=chunk_transform_nhead,
                global_tokens_per_chunk=global_tokens_per_chunk,
                chunk_transform_arch=chunk_transform_arch,
                chunk_transform_mode=chunk_transform_mode,
                chunk_transform_pos_encoding=chunk_transform_pos_encoding,
                token_pooling_strategy=token_pooling_strategy,
                num_labels=num_labels,
                chunk_pooling_strategy=chunk_pooling_strategy,
                dropout_rate=dropout_rate,
                first_chunk_only=first_chunk_only,
                normalize_chunks=normalize_chunks,
                log_signature=log_signature,
                use_section_chunk_embeddings=use_section_chunk_embeddings,
                section_chunk_embedding_dim=section_chunk_embedding_dim,
                section_chunk_vocab_size=section_chunk_vocab_size,
                section_chunk_position_size=section_chunk_position_size,
                use_chunk_relation_attention=use_chunk_relation_attention,
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

        # --- runtime config ----
        self.chunk_pooling_strategy = chunk_pooling_strategy
        self.use_chunk_transform = use_chunk_transform
        self.chunk_size = chunk_size
        self.stride = stride
        self.minimal_chunk_length = minimal_chunk_length
        self.maximal_text_length = maximal_text_length
        self.max_chunks = max_chunks
        self.chunk_transform_layers = int(chunk_transform_layers)
        self.chunk_transform_ffn_mult = int(chunk_transform_ffn_mult)
        self.chunk_transform_dropout = float(chunk_transform_dropout)
        self.chunk_transform_nhead = chunk_transform_nhead if chunk_transform_nhead is None else int(chunk_transform_nhead)
        self.chunk_transform_arch = (chunk_transform_arch or "transformer").strip()
        self.chunk_transform_mode = chunk_transform_mode
        self.global_tokens_per_chunk = int(global_tokens_per_chunk or 64)
        self.chunk_transform_pos_encoding = chunk_transform_pos_encoding
        self.section_aware_tokenization = bool(section_aware_tokenization)
        self.section_aware_chunking = bool(section_aware_chunking)
        self.section_pack_small_sections = bool(section_pack_small_sections)
        self.use_section_chunk_embeddings = bool(use_section_chunk_embeddings)
        self.section_chunk_embedding_dim = int(section_chunk_embedding_dim)
        self.section_chunk_vocab_size = int(section_chunk_vocab_size)
        self.section_chunk_position_size = int(section_chunk_position_size)
        self.use_chunk_relation_attention = bool(use_chunk_relation_attention)

        # Register section tag special tokens so <s>, </s> etc. are single
        # tokens (not subword noise) even when section-aware chunking is off.
        if self.section_aware_tokenization:
            encoder = getattr(self.neural_network, "encoder", None)
            register_tag_tokens(self.tokenizer, encoder)

        # --- persist alignment-critical params for save/load + KD parity ---
        self._params.update({
            "token_pooling_strategy": token_pooling_strategy,
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "minimal_chunk_length": self.minimal_chunk_length,
            "use_chunk_transform": self.use_chunk_transform,
            "chunk_pooling_strategy": self.chunk_pooling_strategy,
            "maximal_text_length": self.maximal_text_length,
            "max_chunks": self.max_chunks,
            "chunk_transform_layers": self.chunk_transform_layers,
            "chunk_transform_ffn_mult": self.chunk_transform_ffn_mult,
            "chunk_transform_dropout": self.chunk_transform_dropout,
            "chunk_transform_nhead": self.chunk_transform_nhead,
            "global_tokens_per_chunk": self.global_tokens_per_chunk,
            "chunk_transform_mode": self.chunk_transform_mode,
            "chunk_transform_arch": self.chunk_transform_arch,
            "chunk_transform_pos_encoding": self.chunk_transform_pos_encoding,
            "first_chunk_only": first_chunk_only,
            "normalize_chunks": normalize_chunks,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "dropout_rate": dropout_rate,
            "log_signature": log_signature,
            "section_aware_tokenization": self.section_aware_tokenization,
            "section_aware_chunking": self.section_aware_chunking,
            "section_pack_small_sections": self.section_pack_small_sections,
            "use_section_chunk_embeddings": self.use_section_chunk_embeddings,
            "section_chunk_embedding_dim": self.section_chunk_embedding_dim,
            "section_chunk_vocab_size": self.section_chunk_vocab_size,
            "section_chunk_position_size": self.section_chunk_position_size,
            "use_chunk_relation_attention": self.use_chunk_relation_attention,
        })

        self.collate_fn = collate_fn_pooled_tokens

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        if self.section_aware_chunking:
            if not self.section_aware_tokenization:
                raise ValueError(
                    "section_aware_chunking requires section_aware_tokenization"
                )
            encoder = self.get_core_encoder()
            tokens = tokenize_section_aware_pooled(
                texts,
                self.tokenizer,
                max_length=self.chunk_size + 2,
                max_chunks=self.max_chunks,
                stride=self.stride,
                minimal_chunk_len=self.minimal_chunk_length,
                pack_small_sections=self.section_pack_small_sections,
                encoder=encoder,
            )
        else:
            tokens = transform_list_of_texts(
                texts,
                self.tokenizer,
                self.chunk_size,
                self.stride,
                self.minimal_chunk_length,
                self.maximal_text_length,
                self.max_chunks,
            )
            if "section_ids" not in tokens:
                zeros_long = torch.zeros_like(tokens["chunks_mask"], dtype=torch.long)
                zeros_bool = torch.zeros_like(tokens["chunks_mask"], dtype=torch.bool)
                tokens["section_ids"] = zeros_long
                tokens["section_positions"] = zeros_long.clone()
                tokens["section_mix_flags"] = zeros_bool
        return tokens

    def _evaluate_single_batch(self, batch: tuple) -> Tensor:
        input_ids, attention_mask, chunks_mask, section_ids, section_positions, section_mix = batch[:6]

        # Move tensors to device
        input_ids = input_ids.to(self.device, non_blocking=True).long()
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        chunks_mask = chunks_mask.to(self.device, non_blocking=True)
        section_ids = section_ids.to(self.device, non_blocking=True)
        section_positions = section_positions.to(self.device, non_blocking=True)
        section_mix = section_mix.to(self.device, non_blocking=True)

        # Forward pass through the neural network
        logits = self.neural_network(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunks_mask=chunks_mask,
            section_ids=section_ids,
            section_positions=section_positions,
            section_mix_flags=section_mix,
        )
        return logits

    def _create_dataset(self, tokens, labels=None):
        return ChunkedDataset(tokens, labels)

    def get_core_encoder(self):
        return getattr(self.neural_network, "encoder", None)

    @torch.no_grad()
    def predict_doc_features(self, x: list[str], batch_size: Optional[int] = None) -> torch.Tensor:
        if not batch_size:
            batch_size = self.batch_size
        tokens = self._tokenize(x)
        dataset = ChunkedDataset(tokens, labels=None)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.collate_fn,
        )
        self.neural_network.eval()
        feats = []
        for batch in loader:
            if len(batch) == 7:
                input_ids, attention_mask, chunks_mask, section_ids, section_positions, section_mix, _ = batch
            elif len(batch) == 6:
                input_ids, attention_mask, chunks_mask, section_ids, section_positions, section_mix = batch
            elif len(batch) == 3:
                input_ids, attention_mask, chunks_mask = batch
                section_ids = torch.zeros_like(chunks_mask, dtype=torch.long)
                section_positions = torch.zeros_like(chunks_mask, dtype=torch.long)
                section_mix = torch.zeros_like(chunks_mask, dtype=torch.bool)
            else:
                raise ValueError("Unexpected batch format while predicting features")

            input_ids = input_ids.to(self.device).long()
            attention_mask = attention_mask.to(self.device)
            chunks_mask = chunks_mask.to(self.device)
            section_ids = section_ids.to(self.device)
            section_positions = section_positions.to(self.device)
            section_mix = section_mix.to(self.device)

            f = self.neural_network.forward_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunks_mask=chunks_mask,
                section_ids=section_ids,
                section_positions=section_positions,
                section_mix_flags=section_mix,
            )
            feats.append(f.cpu())
        return torch.cat(feats)

    # Optional xcore helpers left commented for future use
    # def save_xcore(self, save_path: str, config: dict = None):
    #     encoder = self.get_core_encoder()
    #     if hasattr(encoder, "save_pretrained"):
    #         save_path = Path(save_path)
    #         save_dir = save_path if save_path.suffix == "" else save_path.parent
    #         encoder.save_pretrained(save_dir)
    #         if config:
    #             with open(save_dir / "xcore_config.json", "w") as f:
    #                 json.dump(config, f, indent=2)
    #         print(f"[xcore] Saved encoder to {save_dir}")
    #     else:
    #         raise NotImplementedError("Encoder doesn't support save_pretrained")
