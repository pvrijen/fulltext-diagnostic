# Reproducing the paper's experiments

Scripts in this directory reproduce the core results of:

> van Rijen P, Knafou J, Mottaz A, Gobeill J, Gaillac E, Ruch P. Beyond the abstract: diagnosing full-text effects in biomedical screening. *JAMIA*. 2026 (under review).

They are standalone: they depend only on this package (`belt_plus/`) and on `_helpers.py`, which holds every hyperparameter from the paper's hyperparameter table.

## Setup

```bash
pip install -e ".[experiments]"
export FTD_DATA_DIR=/path/to/data      # default: ./data
export FTD_OUT_DIR=/path/to/output     # default: ./output
```

Place one CSV per dataset under `$FTD_DATA_DIR` named `pfas.csv`, `bpa.csv`, `cd008122.csv`, `cd011431.csv`, `cd012069.csv`, `cd012661.csv`, in the format described in the main README (`id, text_abstract, text_fulltext, label`). The original curation export format (`PMID, MEDLINE, PMC, Status`) is also accepted. Run all scripts from the repository root.

## Coverage

| Paper artifact | Script | Notes |
|----------------|--------|-------|
| Encoder comparison on abstracts (Table "Model comparison") | `run_crossfit.py --encoder <key>` | the C row of each run; keys: `biomedbert-abs`, `biobert`, `bert`, `roberta` |
| TE/IE decomposition, PR-AUC (Table "Training vs. inference decomposition") | `run_crossfit.py` | C, P@0%, P@100% per dataset-fold |
| TE/IE decomposition, WSS@95 (appendix) | `run_crossfit.py` | `wss95` block of each result |
| Decomposition across encoder families (appendix) | `run_crossfit.py --encoder bio-longformer`, `--encoder biocmodernbert` | long-context encoders use the same chunking |
| Cross-fitted PFD vs in-sample PFD (Table "Cross-fitting validation") | `run_crossfit.py` | `IS_PFD`, `CF_PFD` |
| Alpha sensitivity (appendix) | `run_alpha_sweep.py` | needs `run_crossfit.py` outputs |
| Silhouette scores (appendix) | `run_silhouette.py` | needs `run_crossfit.py` outputs |
| LLM zero-shot and few-shot screening (appendix) | `run_llm.py` | Qwen2.5-32B-Instruct, Mistral-7B-Instruct-v0.3, 4-bit |
| Waterfall figure | `gen_figures.py` | reads `run_crossfit.py` summary; falls back to paper values |
| UMAP figure | `gen_figures.py` | needs `run_silhouette.py` cache |
| Standard KD strategies (appendix) | not included | produced with the thesis codebase; the six variants map to `kd_type` options of `belt_plus.BertClassifierTruncated.fit` |
| Context expansion 512 / 2,048 / 8,192 tokens | not included | run with the thesis Hydra configs; same `belt_plus` models with `max_chunks` 1, 4, 16 |
| SVM decomposition and SVM-PFD (appendix) | not included | scikit-learn TF-IDF + LinearSVC pipeline, same folds |
| Recall-at-effort curves (appendix) | not included | computed from the cached fold-level probabilities |
| Dataset characteristics, hyperparameter table | `_helpers.py` | constants only |

The `fulltext-diagnostic` CLI in the main README implements the same two-model decomposition and PFD protocol for a single dataset and is the recommended entry point for new data.

## Running

```bash
# Core experiment: decomposition + cross-fitted PFD (about 4 h per dataset on one H100)
python experiments/run_crossfit.py --datasets pfas,bpa --device cuda:0
python experiments/run_crossfit.py --datasets cd008122,cd011431 --device cuda:1
python experiments/run_crossfit.py --datasets cd012069,cd012661 --device cuda:2

# Other encoders
python experiments/run_crossfit.py --encoder biobert
python experiments/run_crossfit.py --encoder bio-longformer

# LLM baselines (one model per GPU)
CUDA_VISIBLE_DEVICES=0 python experiments/run_llm.py --model qwen2.5-32b --mode zeroshot,fewshot
CUDA_VISIBLE_DEVICES=1 python experiments/run_llm.py --model mistral-7b --mode zeroshot,fewshot

# Alpha sweep (after run_crossfit.py)
python experiments/run_alpha_sweep.py --alphas 0.05,0.1,0.2,0.3,0.5,1.0 --gpu 0

# Silhouette scores (after run_crossfit.py)
python experiments/run_silhouette.py --gpu 0 --datasets pfas,bpa,cd008122,cd011431,cd012069,cd012661

# Figures
python experiments/gen_figures.py
```

Outputs land under `$FTD_OUT_DIR/{crossfit,llm,alpha_sweep,silhouette,figures}/`. Every dataset-fold result is cached as JSON, so interrupted runs resume.

## Fold indices

`fold_indices.json` holds the exact outer train/test row indices used in the paper: six datasets, five folds, `{"train": [...], "test": [...]}` per fold. Indices refer to row positions in the per-dataset CSV after filtering to articles with full text and sorting by `id`. The scripts regenerate identical splits with `StratifiedKFold(5, shuffle=True, random_state=42)`; the file is provided for verification and for use outside this codebase. The inner validation split (20% of each training fold) uses seed `42 + fold`.

## Statistics

Per-fold metrics are PR-AUC (`sklearn.metrics.average_precision_score`) and WSS@95. Paired comparisons use two-sided Wilcoxon signed-rank tests over the 30 dataset-fold pairs; the scripts print these. The bootstrap confidence intervals in the waterfall figure resample the 30 dataset-fold effects 10,000 times (percentile method). Standard deviations use ddof = 1.

## Hardware

Experiments were run on NVIDIA H100 GPUs. One dataset (5 folds, C + P + 5 cross-fitted teachers + 2 PFD students) takes about 4 hours on one GPU.
