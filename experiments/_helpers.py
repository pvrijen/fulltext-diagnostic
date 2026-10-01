"""Standalone helpers for paper reproduction experiments.

Extracts the functions and constants needed from the thesis codebase
(run_rq2_unified.py, run_pooling_chapter.py) so experiments can run
without the full thesis infrastructure.

All hyperparameters match the paper's reported values exactly.
"""
import os, random, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# ═══════════════════════════════════════════════════════════
# Constants (from run_rq2_unified.py + run_pooling_chapter.py)
# ═══════════════════════════════════════════════════════════

ENCODERS = {
    "biomedbert-abs": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

DATASETS = {
    "pfas": "data/dataset/pfas.csv",
    "bpa": "data/dataset/bpa.csv",
    "cd008122": "data/dataset/cd008122.csv",
    "cd011431": "data/dataset/cd011431.csv",
    "cd012069": "data/dataset/cd012069.csv",
    "cd012661": "data/dataset/cd012661.csv",
}

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


# ═══════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════

def _strip_tags(texts):
    """Remove section tags (<s:...> / <e:...>) from text."""
    import re
    pattern = re.compile(r"<[se]:[^>]+>")
    return [pattern.sub("", t).strip() for t in texts]


def _fix_pmc_text(pmc_text, medline_text):
    """Ensure abstract is prepended to PMC full text."""
    if not pmc_text or not isinstance(pmc_text, str):
        return str(medline_text or "")
    if medline_text and str(medline_text) not in pmc_text[:len(str(medline_text)) + 200]:
        return str(medline_text) + " " + pmc_text
    return pmc_text


def load_data(dataset_name):
    """Load dataset (PMC-only subset).

    Returns: (df, texts_full, texts_medline, labels)
    """
    csv_path = DATASETS[dataset_name]
    df = pd.read_csv(csv_path)
    df["label"] = df["Status"].apply(lambda x: 1 if x == "Included" else 0)
    has_pmc = df["PMC"].notna() & (df["PMC"].astype(str).str.len() > 50)
    df = df[has_pmc].copy()
    df["has_pmc"] = True

    fixed = [_fix_pmc_text(row["PMC"], row["MEDLINE"]) for _, row in df.iterrows()]
    texts_full = _strip_tags(fixed)
    texts_medline = _strip_tags([str(row["MEDLINE"] or "") for _, row in df.iterrows()])
    labels = df["label"].tolist()
    return df, texts_full, texts_medline, labels


# ═══════════════════════════════════════════════════════════
# Cross-validation utilities
# ═══════════════════════════════════════════════════════════

def make_cv_splits(labels, seed, n_splits=5):
    """Deterministic stratified k-fold splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(range(len(labels)), labels))


def make_val_split(train_idx, labels, seed, fold, val_size=0.2):
    """Split train indices into train_sub + val_sub."""
    rng = np.random.RandomState(seed + fold)
    y_train = np.array([labels[i] for i in train_idx])
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


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

def timed_eval(model, texts, y_true):
    """Evaluate model and return PR-AUC + timing."""
    t0 = time.time()
    with torch.no_grad():
        scores = model.predict_scores(texts)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy().flatten()
    elapsed = time.time() - t0
    prauc = float(average_precision_score(y_true, scores))
    return {"pr_auc": prauc, "time": elapsed, "probs": scores.tolist()}


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
