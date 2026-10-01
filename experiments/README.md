# Reproducing Paper Results

This directory contains the scripts to reproduce every table and figure in:

> van Rijen, P. et al. (2026). "Beyond the abstract: diagnosing full-text
> effects in biomedical screening." *Journal of the American Medical Informatics
> Association* (under review).

## Prerequisites

1. Install the package and experiment dependencies:

```bash
pip install -e .
pip install -r requirements-experiments.txt
```

2. Obtain the six datasets (see Data Availability below) and place them as CSVs
   under `data/` matching the format in the main README.

**Note on experiment scripts:** The scripts in this directory were used to
produce the paper's results within the full thesis codebase. They import
helper modules (`run_rq2_unified`, `run_pooling_chapter`) that are not
included in this repository. For standalone reproduction, use the
`fulltext-diagnostic` CLI commands documented in the main README. The scripts
are provided for transparency and reference only.

3. Fold indices are provided in `fold_indices.json` for exact replication of the
   5-fold stratified splits (seed = 42).

## Script-to-artifact mapping

| Artifact | Script | Description |
|----------|--------|-------------|
| Table 2 (encoders) | `run_crossfit.py` (C condition) | BioMedBERT baseline; other encoders via `--encoder` flag |
| Table 3 (decomposition) | `run_crossfit.py` (C + P conditions) | TE/IE decomposition (C, P@0%, P@100%) on PR-AUC |
| Table 4 (KD strategies) | `run_crossfit.py` | Six standard KD variants (all fail) |
| Table 5 (cross-fitted PFD) | `run_crossfit.py` | Cross-fitted PFD vs in-sample PFD |
| Table 6 (operating points) | Computed from Table 3 + Table 5 outputs | PR-AUC and WSS@95 summary |
| Table 7 (LLM baselines) | `run_llm.py` | Zero-shot/few-shot screening (Qwen2.5-32B, Mistral-7B) |
| Table 9 (per-fold) | `run_crossfit.py` | Per-fold PR-AUC for cross-fitting |
| Table 10 (multi-encoder decomp) | `run_crossfit.py --encoder <name>` | TE/IE for Bio-Longformer, BiocModernBERT |
| Table 11 (WSS@95 decomp) | `run_crossfit.py` | TE/IE decomposition on WSS@95 |
| Table 12 (alpha sweep) | `run_alpha_sweep.py` | Distillation weight sensitivity |
| Table 13 (silhouette) | `run_silhouette.py` | Cosine silhouette scores on CLS embeddings |
| Figure 2 (waterfall) | `gen_figures.py` | TE/IE waterfall plot |
| Figure 3 (UMAP) | `gen_figures.py` | UMAP visualisation of CLS embeddings |

## Running experiments

### Core experiment (Tables 2--5, 8--11)

The main script handles C training, P training, decomposition, and cross-fitted
PFD in a single run per dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python run_crossfit.py --datasets pfas bpa
CUDA_VISIBLE_DEVICES=1 python run_crossfit.py --datasets cd008122 cd011431
CUDA_VISIBLE_DEVICES=2 python run_crossfit.py --datasets cd012069 cd012661
```

### Table 6: LLM baselines

```bash
CUDA_VISIBLE_DEVICES=0 python run_llm.py --model qwen2.5-32b --mode zeroshot,fewshot
CUDA_VISIBLE_DEVICES=1 python run_llm.py --model mistral-7b --mode fewshot
```

### Table 12: Alpha sweep

```bash
CUDA_VISIBLE_DEVICES=0 python run_alpha_sweep.py --alphas 0.05,0.1,0.2,0.3,0.5,1.0
```

### Table 13: Silhouette scores

```bash
CUDA_VISIBLE_DEVICES=0 python run_silhouette.py --datasets pfas,bpa,cd012661,cd012069,cd008122,cd011431
```

### Figures 2--3: Waterfall and UMAP

```bash
python gen_figures.py  # generates all figures from cached results
```

## Fold indices

`fold_indices.json` contains the exact train/test splits used in the paper.
Six datasets, five folds each. Indices refer to row positions in the
corresponding dataset CSV after sorting by ID. Seed = 42.

## Data availability

Datasets are not distributed with this repository. To obtain them:

- **PFAS, BPA** (NIEHS/NTP OHAT): Howard et al. via https://hawcproject.org/
  (SWIFT-Review export format)
- **CD008122, CD011431, CD012069, CD012661** (Cochrane/CLEF TAR):
  CLEF eHealth shared task repositories
  (https://ceur-ws.org/Vol-1866/, https://ceur-ws.org/Vol-2125/, https://ceur-ws.org/Vol-2380/)

See the paper's Data Availability section for full details.

## Hardware

All experiments were run on NVIDIA H100 GPUs (48 GB each). A single
decomposition + cross-fitting run (one dataset, 5 folds) takes approximately
4 hours on one H100.
