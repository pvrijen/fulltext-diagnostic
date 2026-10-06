# fulltext-diagnostic

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: under review](https://img.shields.io/badge/status-under%20review-orange)

> **Note**: This repository accompanies a manuscript under peer review. The API may change before the final release.

**Diagnose whether full-text processing helps biomedical literature screening, and transfer full-text knowledge to an abstract-only model via Privileged Features Distillation (PFD).**

Companion code for:

> van Rijen P, Knafou J, Mottaz A, Gobeill J, Gaillac E, Ruch P. Beyond the abstract: diagnosing full-text effects in biomedical screening. *Journal of the American Medical Informatics Association*. 2026 (under review).

## Installation

From source (the package is not yet on PyPI):

```bash
git clone https://github.com/pvrijen/fulltext-diagnostic.git
cd fulltext-diagnostic
pip install -e .            # core tool
pip install -e ".[dev]"     # + pytest, ruff
pip install -e ".[experiments]"   # + matplotlib, umap-learn, bitsandbytes, accelerate
```

Tested with Python 3.12, PyTorch 2.x and Transformers 4.x and 5.x. A GPU is strongly recommended for real datasets; the test suite runs on CPU with a 2-layer BERT.

```bash
pytest tests/
```

## Quick start

### 1. Prepare your data

One CSV with the columns:

| Column | Content |
|--------|---------|
| `id` | article identifier (for example PMID) |
| `text_abstract` | title and abstract |
| `text_fulltext` | abstract followed by the body sections, concatenated |
| `label` | 1 = included, 0 = excluded |

Articles without full text should be dropped before running the diagnostic; the decomposition requires both views of every article.

The six datasets used in the paper are not redistributed here (see [Data availability](#data-availability)).

### 2. Run the diagnostic (two training runs)

```bash
fulltext-diagnostic train \
    --data my_dataset.csv \
    --encoder microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract \
    --output-dir results/ \
    --folds 5 \
    --device cuda
```

This trains, per fold:

- **C**: abstract-only baseline, trained and evaluated on abstracts.
- **P**: full-text model (chunk-mean pooling over up to 12 chunks of 510 tokens), trained on full text and evaluated twice: on abstracts (**P@0%**) and on full text (**P@100%**).

Defaults follow the paper's hyperparameter table (C: learning rate 3e-5, weight decay 0.10, dropout 0.5, batch 32; P: learning rate 1e-5, weight decay 0.01, dropout 0.1, effective batch 64; both: AdamW, 5% warmup, up to 20 epochs, early stopping on validation loss with patience 3). The default encoder is the abstracts-only BioMedBERT checkpoint used in the paper. Run `fulltext-diagnostic train --help` for all options.

The pipeline is resumable: every job leaves a `.done_*` marker and is skipped on re-run. The command exits with status 1 if any job failed.

### 3. View the decomposition

```bash
fulltext-diagnostic diagnose --results-dir results/
```

Prints C, P@0%, P@100% (mean +/- SD across folds), the three effects, and a recommendation:

- **Training effect** TE = P@0% - C. Negative means training on full text degrades abstract predictions.
- **Inference effect** IE = P@100% - P@0%. Positive means the body text carries signal at prediction time.
- **Total** = P@100% - C = TE + IE.

Decision rule (as in the paper's decision framework):

| Pattern | Recommendation |
|---------|----------------|
| IE > 0 and TE < 0 (cancellation) | Run PFD, deploy the student C_PFD on abstracts |
| IE > 0 and TE >= 0 | Deploy P on full text |
| IE <= 0 | Deploy C; full text is not worth the cost |

Wilcoxon signed-rank P values are reported only when at least six paired folds are available. With the default five folds they are marked n/a: pool folds across datasets (the paper uses n = 30) before testing.

### 4. Apply PFD (if recommended)

```bash
fulltext-diagnostic pfd --results-dir results/ --data my_dataset.csv --device cuda
```

Warm-starts the student from each fold's trained C, then fine-tunes it on abstracts for up to 3 epochs at learning rate 5e-6 with the loss `(1 - alpha) * BCE(labels) + alpha * T^2 * BCE(soft teacher targets)`, alpha = 0.1, T = 4. Teacher logits come from the cached P predictions, so no teacher inference is needed.

### 5. Full report

```bash
fulltext-diagnostic report --results-dir results/
```

### 6. Pipeline monitoring (interactive TUI)

```bash
ftd-monitor --output-dir results/ --folds 5
```

Four tabs: job status table, dependency tree, decomposition and PFD tables, and the log of failed jobs.

### 7. HPC deployment (SLURM)

```bash
fulltext-diagnostic slurm --data my_dataset.csv --output-dir results/ --partition gpu --time 04:00:00
bash results/slurm/submit_all.sh
```

Generates one `.sbatch` per step (split, train C, train P, evaluate, diagnose, PFD, report) chained with `--dependency`. Individual steps can also be run with `fulltext-diagnostic run-job <step> --fold N ...`; `status`, `dag` and `reset` inspect and repair pipeline state.

## Output files

| Path | Content |
|------|---------|
| `config.json` | run configuration (encoder, folds, seed, hyperparameters) |
| `folds.json` | per-fold `train`, `val`, `test`, `train_all` row indices |
| `models/C_f{k}/` | trained C (HuggingFace format) |
| `models/P_f{k}/encoder.pt`, `config.json` | trained P encoder weights and chunking config |
| `cache/C_f{k}.npz`, `cache/P_f{k}.npz` | test-fold logits, probabilities, labels, teacher logits |
| `cache/PFD_f{k}.npz` | student predictions |
| `diagnostic_results.csv` | columns `fold, C, P@0%, P@100%, TE, IE, Total` |
| `diagnostic_report.json` | means, SDs, P values (or null), recommendation |
| `pfd_results.csv` | columns `fold, C, C_PFD, delta` |
| `.done_*`, `.failed_*` | job state markers (`.failed_*` holds the traceback) |

## Runtime

On one NVIDIA H100, a dataset of 1,000 to 6,000 articles takes about 4 hours for the full diagnostic (C about 45 min, P about 3 h, PFD about 1 h, 5 folds). CPU execution works but is only practical for the test suite.

## Reproducing the paper

Scripts, fold indices and the mapping from paper artifacts to scripts are in [`experiments/README.md`](experiments/README.md). The `belt_plus/` directory contains the extended BELT model code used by all experiments; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

## Data availability

- **PFAS, BPA** (NIEHS/NTP OHAT systematic reviews): obtain the SWIFT-Review exports via the [HAWC project](https://hawcproject.org/) and the OHAT protocols for PFAS immunotoxicity and BPA metabolic effects. PFAS: Howard BE et al., *Environ Int* 2016.
- **CD008122, CD011431, CD012069, CD012661**: CLEF eHealth Technology Assisted Review (TAR) tasks 2017 to 2019. Topic files and PMID lists are in the [CLEF TAR GitHub repository](https://github.com/CLEF-TAR/tar); the Cochrane review IDs are the dataset names.

Full texts were retrieved from PubMed Central (open access subset) by PMID and concatenated as title, abstract and body sections. Articles without full text were dropped. Row indices in `experiments/fold_indices.json` refer to the resulting per-dataset CSV after filtering to articles with full text and sorting by `id`.

## Citation

```bibtex
@article{vanrijen2026fulltext,
  title   = {Beyond the abstract: diagnosing full-text effects in biomedical screening},
  author  = {van Rijen, Paul and Knafou, Julien and Mottaz, Ana\"{i}s and Gobeill, Julien and Gaillac, Esteban and Ruch, Patrick},
  journal = {Journal of the American Medical Informatics Association},
  year    = {2026},
  note    = {Under review. Code: \url{https://github.com/pvrijen/fulltext-diagnostic}}
}
```

## License

MIT. See [LICENSE](LICENSE). Vendored BELT code is MIT-licensed by MIM Solutions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
