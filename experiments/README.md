# Reproducing Paper Results

This directory contains the scripts to reproduce every table and figure in:

> van Rijen, P. and Ruch, P. (2026). "Beyond abstracts: diagnosing full-text
> effects in biomedical screening." *Database* (Oxford University Press).

## Prerequisites

1. Install the package and experiment dependencies:

```bash
pip install -e .
pip install -r requirements-experiments.txt
```

2. Place the six datasets as CSVs under `data/` (see the main README for format).

3. Fold indices are provided in `fold_indices.json` for exact replication of the
   5-fold stratified splits (seed = 42).

## Script-to-artifact mapping

| Artifact | Script | Description |
|----------|--------|-------------|
| Table 2 | `run_encoders.sh` | Encoder comparison (8 encoders, 6 datasets, 5 folds) |
| Table 3, 10--11 | `run_decomposition.sh` | TE/IE decomposition (C, P@0%, P@100%) |
| Table 4, 8 | `run_crossfit.py` | Cross-fitted PFD vs in-sample PFD |
| Table 6 | `run_llm.py` | Zero-shot LLM screening (7 models) |
| Table 9 | `run_multi_encoder.sh` | Multi-encoder generalisation (TF-IDF, SVM teachers) |
| Table 12 | `run_alpha_sweep.py` | Hyperparameter sensitivity (alpha sweep) |
| Table 13 | `run_silhouette.py` | Silhouette scores for CLS embeddings |
| Figure 2 | `gen_figures.py` (waterfall) | TE/IE waterfall plot |
| Figure 3 | `gen_figures.py` (UMAP) | UMAP visualisation of CLS embeddings |

## Running experiments

### Table 2: Encoder comparison

```bash
bash run_encoders.sh          # launches one GPU job per encoder
```

### Tables 3, 10--11: TE/IE decomposition

```bash
bash run_decomposition.sh     # trains C and P models, computes TE/IE
```

### Tables 4, 8: Cross-fitted PFD

```bash
CUDA_VISIBLE_DEVICES=0 python run_crossfit.py
```

### Table 6: LLM baselines

```bash
CUDA_VISIBLE_DEVICES=0 python run_llm.py --model llama70b
CUDA_VISIBLE_DEVICES=1 python run_llm.py --model qwen72b
CUDA_VISIBLE_DEVICES=2 python run_llm.py --model gemma27b
# ... see script for all 7 models
```

### Table 12: Alpha sweep

```bash
CUDA_VISIBLE_DEVICES=0 python run_alpha_sweep.py --alphas 0.05,0.1,0.2,0.3,0.5,1.0 --gpu 0
```

### Table 13: Silhouette scores

```bash
CUDA_VISIBLE_DEVICES=0 python run_silhouette.py --gpu 0 --datasets pfas,bpa,cd012661,cd012069,cd008122,cd011431
```

### Figures 2--3: Waterfall and UMAP

```bash
python gen_figures.py         # generates all figures from cached embeddings
```

## Fold indices

`fold_indices.json` contains the exact train/test splits used in the paper.
Structure:

```json
{
  "pfas": [
    {"train": [0, 1, 2, ...], "test": [3, 4, 5, ...]},
    ...
  ],
  ...
}
```

Six datasets, five folds each. Indices refer to row positions in the
corresponding dataset CSV after sorting by ID.

## Hardware

All experiments were run on 4x NVIDIA H100 (48 GB each). Encoder comparison
and LLM experiments benefit from multi-GPU parallelism. A single decomposition
run (one dataset, one encoder, 5 folds) takes approximately 2 hours on one H100.
