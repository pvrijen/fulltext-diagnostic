"""EXP12: fourth cell of the training x inference factorial (Paper B v2.6).

The decomposition pipeline (exp8) has three cells: L_abs->abs (truncated
model), L_ft->abs and L_ft->ft (chunk-mean pooled model). This script adds
the missing cell with the SAME pooled architecture and hyperparameters as
the full-text model, trained on abstracts only:

    A_abs->abs : pooled model trained on abstracts, evaluated on abstracts
    A_abs->ft  : the same model evaluated on full text

Together with exp8 this gives a 2x2 (training input x inference input)
with a single architecture, so training and inference main effects and
their interaction can be estimated without the architecture confound.

Outer folds and the validation split are those of exp8 (seed 42), so the
test sets are identical. GPU required. Split by dataset:
  CUDA_VISIBLE_DEVICES=0 python exp12_factorial_cell.py --datasets pfas bpa
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.chdir("/home/paul/fulltext_biocuration")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+",
                    default=["pfas", "bpa", "cd008122", "cd011431", "cd012069", "cd012661"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="fold order; cached folds are skipped, so a second worker can run the list reversed")
parser.add_argument("--out", default="papers/paper_b_biocuration_guide/versions/2.6_jamia/results")
args = parser.parse_args()

from belt_plus.bert_classifier_with_pooling import BertClassifierWithPooling  # noqa: E402
from run_rq2_unified import (load_data, make_cv_splits, make_val_split, set_seeds,  # noqa: E402
                             ENCODERS, HP_LR, HP_WD, HP_DR, HP_WARMUP, HP_GRADNORM, timed_eval)
from run_pooling_chapter import PATIENCE_P, T_BS, T_ACCUM  # noqa: E402

MODEL = ENCODERS["biomedbert-abs"]
SPLIT_SEED = 42
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda:0"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wss95(probs, labels):
    probs, labels = np.array(probs, dtype=float), np.array(labels, dtype=float)
    n, n_pos = len(labels), int(labels.sum())
    if n_pos == 0:
        return 0.0
    thr = int(np.ceil(0.95 * n_pos))
    order = np.argsort(-probs)
    cumsum = np.cumsum(labels[order])
    return float((n - (int(np.searchsorted(cumsum, thr)) + 1)) / n - 0.05)


def eval_full(model, X_te, y_te):
    ev = timed_eval(model, X_te, y_te)
    probs = ev.get("probs")
    if isinstance(probs, np.ndarray):
        probs = probs.tolist()
    return {"prauc": float(ev["pr_auc"]), "wss95": wss95(probs, [int(y) for y in y_te])}


def build_pooled():
    # identical to the exp8 full-text teacher
    return BertClassifierWithPooling(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=HP_LR, weight_decay=HP_WD, dropout_rate=HP_DR,
        batch_size=T_BS, accumulation_steps=T_ACCUM, epochs=20,
        chunk_size=510, stride=510, minimal_chunk_length=50, max_chunks=12,
        chunk_pooling_strategy="mean", use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


for ds in args.datasets:
    set_seeds(SPLIT_SEED)
    df, ft, med, labels = load_data(ds)
    splits = make_cv_splits(labels, SPLIT_SEED)
    for fold in args.folds:
        tag = f"{ds}_f{fold}"
        outpath = OUT / f"exp12_{tag}.json"
        if outpath.exists():
            log(f"SKIP {tag} (cached)")
            continue
        log(f"=== {tag} ===")
        train_idx, test_idx = splits[fold]
        train_sub, val_sub = make_val_split(train_idx, labels, SPLIT_SEED, fold)
        X_tr = [med[i] for i in train_sub]
        X_vl = [med[i] for i in val_sub]
        y_tr = [labels[i] for i in train_sub]
        y_vl = [labels[i] for i in val_sub]
        X_te_abs = [med[i] for i in test_idx]
        X_te_ft = [ft[i] for i in test_idx]
        y_te = [labels[i] for i in test_idx]

        t0 = time.time()
        set_seeds(args.seed)
        model = build_pooled()
        model.fit(X_tr, y_tr, X_vl, y_vl, patience=PATIENCE_P, early_stopping=True)
        aa = eval_full(model, X_te_abs, y_te)
        af = eval_full(model, X_te_ft, y_te)
        log(f"  A_abs->abs prauc={aa['prauc']:.4f}  A_abs->ft prauc={af['prauc']:.4f}  ({time.time()-t0:.0f}s)")
        del model
        clear_gpu()
        row = {"seed": args.seed, "dataset": ds, "fold": fold,
               "Aabs_abs_prauc": aa["prauc"], "Aabs_abs_wss95": aa["wss95"],
               "Aabs_ft_prauc": af["prauc"], "Aabs_ft_wss95": af["wss95"]}
        with open(outpath, "w") as f:
            json.dump(row, f, indent=2)

log("done")
