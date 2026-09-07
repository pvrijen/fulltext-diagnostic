# Based on BELT (https://github.com/mim-solutions/bert_for_longer_texts)
# by MIM Solutions, licensed under MIT.
# Extended with knowledge distillation, section-aware processing,
# and pooling enhancements for this work.

# src/belt_plus/bert_classifier_with_pooling.py
from __future__ import annotations

from typing import Optional, List, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, SequentialSampler

from belt_plus.bert_with_pooling import BertWithPooling, ChunkedDataset


class BertClassifierWithPooling(BertWithPooling):
    """
    Classification wrapper over BertWithPooling with consistent param surface.

    Key fix: expose/forward alignment-critical params (token_pooling_strategy,
    first_chunk_only, normalize_chunks) to avoid silent mismatches between teacher/student
    and during save/load.
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
        chunk_pooling_strategy: str = "mean",
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
        global_tokens_per_chunk: Optional[int] = None,
        chunk_transform_arch: str = "transformer",
        chunk_transform_mode: str = "residual_mean",
        chunk_transform_pos_encoding: str = "sin",
        dropout_rate: float = 0.1,
        use_amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "auto",
        weight_decay: float = 0.0,
        class_weights: Optional[List[float]] = None,
        problem_type: str = "binary_classification",
        # --- alignment-critical params now exposed ---
        token_pooling_strategy: str = "auto",
        first_chunk_only: bool = False,
        normalize_chunks: bool = True,
        log_signature: bool = False,
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
        super().__init__(
            num_labels=num_labels,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            chunk_size=chunk_size,
            stride=stride,
            minimal_chunk_length=minimal_chunk_length,
            chunk_pooling_strategy=chunk_pooling_strategy,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            maximal_text_length=maximal_text_length,
            max_chunks=max_chunks,
            neural_network=neural_network,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            preloaded_state_dict=preloaded_state_dict,
            device=device,
            many_gpus=many_gpus,
            use_chunk_transform=use_chunk_transform,
            chunk_transform_layers=chunk_transform_layers,
            chunk_transform_ffn_mult=chunk_transform_ffn_mult,
            chunk_transform_dropout=chunk_transform_dropout,
            chunk_transform_nhead=chunk_transform_nhead,
            chunk_transform_stride=chunk_transform_stride,
            global_tokens_per_chunk=global_tokens_per_chunk,
            chunk_transform_arch=chunk_transform_arch,
            chunk_transform_mode=chunk_transform_mode,
            chunk_transform_pos_encoding=chunk_transform_pos_encoding,
            dropout_rate=dropout_rate,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            weight_decay=weight_decay,
            class_weights=class_weights,
            problem_type=problem_type,
            token_pooling_strategy=token_pooling_strategy,  # ✅ forward
            first_chunk_only=first_chunk_only,              # ✅ forward
            normalize_chunks=normalize_chunks,              # ✅ forward
            log_signature=log_signature,
            section_aware_tokenization=section_aware_tokenization,
            section_aware_chunking=section_aware_chunking,
            section_pack_small_sections=section_pack_small_sections,
            use_section_chunk_embeddings=use_section_chunk_embeddings,
            section_chunk_embedding_dim=section_chunk_embedding_dim,
            section_chunk_vocab_size=section_chunk_vocab_size,
            section_chunk_position_size=section_chunk_position_size,
            use_chunk_relation_attention=use_chunk_relation_attention,
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
        # Persist classifier-specific surface
        self._params.update({"num_labels": self.num_labels})

    def set_aux_abstract_alpha(self, alpha: float):
        """Enable auxiliary abstract-only loss with weight alpha."""
        nn = self.neural_network
        if hasattr(nn, "module"):
            nn = nn.module
        nn.aux_abstract_alpha = alpha

    def set_chunk_dropout(self, prob: float):
        """Enable chunk dropout: randomly use only chunk 0 with this probability."""
        nn = self.neural_network
        if hasattr(nn, "module"):
            nn = nn.module
        nn.chunk_dropout_prob = prob

    def set_contrastive_alpha(self, alpha: float):
        """Enable contrastive abstract alignment with weight alpha."""
        nn = self.neural_network
        if hasattr(nn, "module"):
            nn = nn.module
        nn.contrastive_alpha = alpha

    def _create_dataset(self, tokens, labels=None):
        return ChunkedDataset(tokens, labels)

    def predict(self, x: List[str], batch_size: Optional[int] = None) -> Tensor:
        """
        Returns integer predictions:
            - Binary => {0,1}
            - Multi-class => argmax(...).
        """
        self.neural_network.eval()
        with torch.no_grad():
            logits = self._predict_logits(x, batch_size=batch_size)

            if self.problem_type == "binary_classification":
                probs = torch.sigmoid(logits).squeeze(dim=1)
                return (probs >= 0.5).long()

            if self.problem_type == "multi_class_classification":
                return torch.argmax(logits, dim=1)

            raise ValueError(f"predict() not defined for {self.problem_type}.")

    def predict_scores(self, x: List[str], batch_size: Optional[int] = None) -> Tensor:
        """
        Returns probabilities:
            - Binary => shape (batch_size,).
            - Multi-class => shape (batch_size, num_classes).
        """
        self.neural_network.eval()
        with torch.no_grad():
            logits = self._predict_logits(x, batch_size=batch_size)

            if self.problem_type == "binary_classification":
                return torch.sigmoid(logits).squeeze(dim=1)

            if self.problem_type == "multi_class_classification":
                return torch.softmax(logits, dim=1)

            raise ValueError(f"predict_scores() not defined for {self.problem_type}.")

    @torch.no_grad()
    def predict_chunk_attention(
        self, x: List[str], batch_size: Optional[int] = None
    ) -> Optional[List[List[float]]]:
        """Per-chunk attention weights [N, C] for attn-k pooling. None if not attn."""
        nn = self.neural_network
        if getattr(nn, 'chunk_pooling_strategy', None) != "attn":
            return None
        nn.eval()
        nn._collect_chunk_attn = True
        nn._chunk_attn_buffer = []
        try:
            _ = self._predict_logits(x, batch_size=batch_size)
        finally:
            nn._collect_chunk_attn = False
        if not nn._chunk_attn_buffer:
            return None
        weights = torch.cat(nn._chunk_attn_buffer, dim=0)  # [N, C]
        return weights.tolist()

    @torch.no_grad()
    def predict_features(self, texts: List[str], batch_size: Optional[int] = None) -> Tensor:
        """
        Returns pooled document embeddings (pre-classifier), shape [N, H].
        Uses the same tokenization/chunking as training.
        """
        self.neural_network.eval()
        if batch_size is None:
            batch_size = self.batch_size

        tokens = self._tokenize(texts)
        dataset = self._create_dataset(tokens, labels=None)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.collate_fn,
            **self._build_loader_kwargs(self.predict_num_workers),
        )

        feats = []
        with torch.no_grad():
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
                    raise ValueError("Unexpected batch format for predict_features")

                input_ids = input_ids.to(self.device)
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
                feats.append(f.detach().cpu())

        return torch.cat(feats, dim=0)
