"""Shared helpers for the paper's experiment scripts.

Hyperparameters, data loading, cross-validation splits and evaluation
utilities. All hyperparameters match the paper's hyperparameter table.

Paths
-----
Scripts read datasets from ``$FTD_DATA_DIR`` (default ``data/``) and write
results under ``$FTD_OUT_DIR`` (default ``output/``). Both are resolved
relative to the current working directory, so run the scripts from the
repository root or set the variables to absolute paths.

Dataset format
--------------
``load_data`` accepts the CSV format described in the main README
(``id, text_abstract, text_fulltext, label``). It also accepts the original
curation export used for the paper (``PMID, MEDLINE, PMC, Status``) and
converts it on the fly: ``Status == "Included"`` becomes label 1, articles
without a PMC full text are dropped, and the abstract is prepended to the
body when the export did not already include it.
"""
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("FTD_DATA_DIR", "data"))
OUT_ROOT = Path(os.environ.get("FTD_OUT_DIR", "output"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODERS = {
    "biomedbert-abs": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "bio-longformer": "kiddothe2b/biomedical-longformer-base",
    "biocmodernbert": "answerdotai/ModernBERT-base",
}

DATASET_NAMES = ["pfas", "bpa", "cd008122", "cd011431", "cd012069", "cd012661"]
DATASETS = {name: DATA_DIR / f"{name}.csv" for name in DATASET_NAMES}

SEED = 42

# Student (abstract baseline) hyperparameters
HP_LR_C = 3e-05
HP_WD_C = 0.10
HP_DR_C = 0.5
HP_WARMUP = 0.05
HP_GRADNORM = 1.0
HP_TRUNC_BS = 32

# Teacher (full-text pooled) hyperparameters
HP_LR = 1e-05
HP_WD = 0.01
HP_DR = 0.1
T_BS = 8
T_ACCUM = 8

# Early stopping
PATIENCE_C = 3
PATIENCE_P = 3

# PFD nudge hyperparameters
NUDGE_LR = 5e-6
NUDGE_WD = 0.01
NUDGE_DR = 0.1
NUDGE_EPOCHS = 3
NUDGE_ALPHA = 0.1
NUDGE_TEMP = 4.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[se]:[^>]+>")


def _strip_tags(texts):
    """Remove section tags (<s:...> / <e:...>) from text."""
    return [_TAG_RE.sub("", str(t)).strip() for t in texts]


def _fix_pmc_text(pmc_text, medline_text):
    """Ensure the abstract is prepended to the PMC full text."""
    if not pmc_text or not isinstance(pmc_text, str):
        return str(medline_text or "")
    if medline_text and str(medline_text) not in pmc_text[:len(str(medline_text)) + 200]:
        return str(medline_text) + " " + pmc_text
    return pmc_text


def load_data(dataset_name):
    """Load a dataset.

    Returns ``(df, texts_full, texts_abstract, labels)``.
    """
    csv_path = DATASETS.get(dataset_name, DATA_DIR / f"{dataset_name}.csv")
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Set FTD_DATA_DIR or place the CSV under "
            f"{DATA_DIR}/. See README.md for the expected columns."
        )
    df = pd.read_csv(csv_path)

    if {"text_abstract", "text_fulltext", "label"} <= set(df.columns):
        df = df.copy()
        df["text_abstract"] = df["text_abstract"].fillna("").astype(str)
        df["text_fulltext"] = df["text_fulltext"].fillna("").astype(str)
        has_ft = df["text_fulltext"].str.len() > 50
        df = df[has_ft].copy()
        texts_full = _strip_tags(df["text_fulltext"])
        texts_abs = _strip_tags(df["text_abstract"])
        labels = df["label"].astype(int).tolist()
        return df, texts_full, texts_abs, labels

    if {"MEDLINE", "PMC", "Status"} <= set(df.columns):
        df = df.copy()
        df["label"] = df["Status"].apply(lambda x: 1 if x == "Included" else 0)
        has_pmc = df["PMC"].notna() & (df["PMC"].astype(str).str.len() > 50)
        df = df[has_pmc].copy()
        fixed = [_fix_pmc_text(row["PMC"], row["MEDLINE"]) for _, row in df.iterrows()]
        texts_full = _strip_tags(fixed)
        texts_abs = _strip_tags([str(row["MEDLINE"] or "") for _, row in df.iterrows()])
        labels = df["label"].tolist()
        return df, texts_full, texts_abs, labels

    raise ValueError(
        f"{csv_path}: expected columns id,text_abstract,text_fulltext,label "
        f"(or the legacy PMID,MEDLINE,PMC,Status export)."
    )


# ---------------------------------------------------------------------------
# Cross-validation utilities
# ---------------------------------------------------------------------------

def make_cv_splits(labels, seed, n_splits=5):
    """Deterministic stratified k-fold splits (seed 42 reproduces the paper)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(range(len(labels)), labels))


def make_val_split(train_idx, labels, seed, fold, val_size=0.2):
    """Split train indices into train_sub + val_sub (random, seed + fold)."""
    rng = np.random.RandomState(seed + fold)
    n_val = max(1, int(len(train_idx) * val_size))
    idx = rng.permutation(len(train_idx))
    val_mask = np.zeros(len(train_idx), dtype=bool)
    val_mask[idx[:n_val]] = True
    train_sub = [train_idx[i] for i in range(len(train_idx)) if not val_mask[i]]
    val_sub = [train_idx[i] for i in range(len(train_idx)) if val_mask[i]]
    return train_sub, val_sub


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def timed_eval(model, texts, y_true):
    """Evaluate a model and return PR-AUC, WSS@95, timing and scores."""
    t0 = time.time()
    with torch.no_grad():
        scores = model.predict_scores(texts)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy().flatten()
    elapsed = time.time() - t0
    prauc = float(average_precision_score(y_true, scores))
    return {"pr_auc": prauc, "wss95": wss95(scores, y_true),
            "time": elapsed, "probs": np.asarray(scores).tolist()}


def wss95(probs, labels):
    """Work saved over sampling at 95% recall."""
    probs = np.array(probs, dtype=float)
    labels = np.array(labels, dtype=float)
    n, n_pos = len(labels), int(labels.sum())
    if n_pos == 0:
        return 0.0
    thr = int(np.ceil(0.95 * n_pos))
    order = np.argsort(-probs)
    cumsum = np.cumsum(labels[order])
    return float((n - (int(np.searchsorted(cumsum, thr)) + 1)) / n - 0.05)
