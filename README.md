# fulltext-diagnostic

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Diagnose whether full-text processing helps biomedical literature screening, and transfer knowledge via Privileged Features Distillation (PFD).**

Companion code for:

> van Rijen, P. and Ruch, P. (2026). "Beyond abstracts: diagnosing full-text effects in biomedical screening." *Database* (Oxford University Press).

## Installation

```bash
pip install fulltext-diagnostic
```

Or install from source:

```bash
git clone https://github.com/pvr-biocuration/fulltext-diagnostic.git
cd fulltext-diagnostic
pip install -e .
```

## Quick Start

### 1. Prepare your data

Create a CSV with columns:
- `id`: article identifier (e.g., PMID)
- `text_abstract`: abstract text
- `text_fulltext`: full-text (body sections concatenated)
- `label`: binary label (1 = included, 0 = excluded)

### 2. Run the diagnostic (two training runs)

```bash
fulltext-diagnostic train \
    --data my_dataset.csv \
    --encoder microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --output-dir results/ \
    --folds 5 \
    --device cuda
```

This trains:
- **C**: abstract-only baseline
- **P**: full-text model, evaluated at 0% and 100% coverage

### 3. View the decomposition

```bash
fulltext-diagnostic diagnose --results-dir results/
```

Output:
```
+-----------------------+------------+-----------------+
|        Full-Text Diagnostic Decomposition            |
+-----------------------+------------+-----------------+
| Metric                | Value      | Interpretation  |
+-----------------------+------------+-----------------+
| Training Effect (TE)  | -0.058     | FT training hurts|
| Inference Effect (IE) | +0.074     | Body text helps  |
| Total Effect          | +0.017     | Nearly cancel    |
+-----------------------+------------+-----------------+

Recommendation: PFD is recommended.
```

### 4. Apply PFD (if recommended)

```bash
fulltext-diagnostic pfd \
    --results-dir results/ \
    --data my_dataset.csv \
    --alpha 0.1 \
    --temperature 4.0 \
    --device cuda
```

### 5. Full report

```bash
fulltext-diagnostic report --results-dir results/
```

### 6. Pipeline monitoring (interactive TUI)

```bash
ftd-monitor --output-dir results/ --folds 5
```

### 7. HPC deployment (SLURM)

```bash
fulltext-diagnostic slurm \
    --data my_dataset.csv \
    --output-dir results/ \
    --partition gpu \
    --time 04:00:00

bash results/slurm/submit_all.sh
```

## The Protocol

The diagnostic answers three questions with just two training runs:

1. **Does body text carry signal?** (IE > 0)
2. **Does full-text training hurt abstract predictions?** (TE < 0)
3. **How large is the trade-off?** (|TE| vs IE)

If IE > 0 and TE < 0, the dataset exhibits the *cancellation pattern*: body text is informative but training on it degrades abstract-only performance. PFD resolves this by having a full-text teacher generate soft labels for an abstract-only student.

## How it works

```
                    +------------------+
                    |    Your Data     |
                    |    (CSV)         |
                    +--------+---------+
                             |
                +------------+------------+
                |                         |
       +--------v--------+    +-----------v-----------+
       |  Model C         |    |  Model P              |
       |  Train: abstract  |    |  Train: full-text      |
       |  Eval:  abstract  |    |  Eval:  abstract (0%)  |
       |                   |    |         full-text(100%) |
       +--------+----------+    +-----------+------------+
                |                           |
                |    TE = P@0% - C          |
                |    IE = P@100% - P@0%     |
                |                           |
                +------------+--------------+
                             |
                    +--------v--------+
                    |  If TE<0, IE>0: |
                    |  Apply PFD      |
                    |                 |
                    |  Teacher: P@100%|
                    |  Student: C     |
                    |  -> C_PFD       |
                    +-----------------+
```

## Reproducing the Paper

Full reproduction instructions, including the script-to-table/figure mapping
and exact fold indices, are in [`experiments/README.md`](experiments/README.md).

Install experiment dependencies:

```bash
pip install -e .
pip install -r requirements-experiments.txt
```

Quick summary of the mapping:

| Artifact | Script |
|----------|--------|
| Table 2 (encoders) | `experiments/run_encoders.sh` |
| Table 3, 10--11 (decomposition) | `experiments/run_decomposition.sh` |
| Table 4, 8 (cross-fitted PFD) | `experiments/run_crossfit.py` |
| Table 6 (LLM baselines) | `experiments/run_llm.py` |
| Table 9 (multi-encoder) | `experiments/run_multi_encoder.sh` |
| Table 12 (alpha sweep) | `experiments/run_alpha_sweep.py` |
| Table 13 (silhouette) | `experiments/run_silhouette.py` |
| Figure 2 (waterfall) | `experiments/gen_figures.py` |
| Figure 3 (UMAP) | `experiments/gen_figures.py` |

The `belt_plus/` directory contains the extended BELT model code used by all
experiments. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
attribution.

## Citation

```bibtex
@article{vanrijen2026fulltext,
    title   = {Beyond abstracts: diagnosing full-text effects in
               biomedical screening},
    author  = {van Rijen, Paul and Ruch, Patrick},
    journal = {Database},
    year    = {2026},
    publisher = {Oxford University Press},
    url     = {https://github.com/pvr-biocuration/fulltext-diagnostic}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
