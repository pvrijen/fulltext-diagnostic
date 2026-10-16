"""EXP11: alpha = 0 control arm and student-side seed replicate (Paper B v2.6).

Per (seed, dataset, fold):
  1. Train the abstract baseline C from scratch (seed controls init/shuffling;
     outer folds and the inner validation split stay those of seed 42, so the
     evaluation sets are identical to exp8).
  2. From the same converged C encoder state, run two warm-start students with
     the cached cross-fitted teacher logits (teacher side fixed, from exp8):
       alpha = 0.0  -> hard labels only (continued fine-tuning control)
       alpha = 0.1  -> CF-PFD as in the paper
  3. Evaluate C and both students on the test fold (PR-AUC, WSS@95).

Seed 42 reproduces the exp8 configuration (gives a run-to-run replicate of C
and CF-PFD); seeds 123 and 456 are the student-side seed replicate.

GPU required. Split across GPUs by dataset:
  CUDA_VISIBLE_DEVICES=0 python exp11_alpha0_seeds.py --datasets pfas bpa
  CUDA_VISIBLE_DEVICES=1 python exp11_alpha0_seeds.py --datasets cd008122 cd011431
  CUDA_VISIBLE_DEVICES=2 python exp11_alpha0_seeds.py --datasets cd012069 cd012661
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
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1])
parser.add_argument("--out", default="papers/paper_b_biocuration_guide/versions/2.6_jamia/results")
args = parser.parse_args()

from belt_plus.bert_classifier_truncated import BertClassifierTruncated  # noqa: E402
from run_rq2_unified import (load_data, make_cv_splits, make_val_split, set_seeds,  # noqa: E402
                             ENCODERS, HP_LR_C, HP_WD_C, HP_DR_C, HP_WARMUP, HP_GRADNORM,
                             HP_TRUNC_BS, timed_eval)
from run_pooling_chapter import PATIENCE_C  # noqa: E402

MODEL = ENCODERS["biomedbert-abs"]
SPLIT_SEED = 42            # outer folds and validation split as in exp8
NUDGE_LR, NUDGE_WD, NUDGE_DR, NUDGE_EPOCHS, NUDGE_TEMP = 5e-6, 0.01, 0.1, 3, 4.0
LOGITS_DIR = Path("papers/output_warmstart_pfd_crossfit/cached_logits")
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


def build_student():
    return BertClassifierTruncated(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=NUDGE_LR, weight_decay=NUDGE_WD, dropout_rate=NUDGE_DR,
        batch_size=32, epochs=NUDGE_EPOCHS, max_length=512, use_amp=False)


for seed in args.seeds:
    for ds in args.datasets:
        set_seeds(SPLIT_SEED)
        df, ft, med, labels = load_data(ds)
        splits = make_cv_splits(labels, SPLIT_SEED)
        for fold in range(5):
            tag = f"s{seed}_{ds}_f{fold}"
            outpath = OUT / f"exp11_{tag}.json"
            if outpath.exists():
                log(f"SKIP {tag} (cached)")
                continue
            cf_logits_path = LOGITS_DIR / ds / f"CF5_{ds}_f{fold}.pt"
            if not cf_logits_path.exists():
                log(f"WARNING: no CF logits for {ds} f{fold}, skipping")
                continue
            cf_logits = torch.load(str(cf_logits_path), map_location="cpu", weights_only=True)

            log(f"=== {tag} ===")
            train_idx, test_idx = splits[fold]
            train_sub, val_sub = make_val_split(train_idx, labels, SPLIT_SEED, fold)
            X_tr = [med[i] for i in train_sub]
            X_vl = [med[i] for i in val_sub]
            X_te = [med[i] for i in test_idx]
            y_tr = [labels[i] for i in train_sub]
            y_vl = [labels[i] for i in val_sub]
            y_te = [labels[i] for i in test_idx]

            # 1. baseline C, seed-controlled
            t0 = time.time()
            set_seeds(seed)
            c_bs = HP_TRUNC_BS // 2 if len(y_tr) > 2000 else HP_TRUNC_BS
            c_acc = 2 if len(y_tr) > 2000 else 1
            c = BertClassifierTruncated(
                pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
                learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
                batch_size=c_bs, accumulation_steps=c_acc, epochs=20,
                max_length=512, use_amp=False,
                warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)
            c.fit(X_tr, y_tr, X_vl, y_vl, patience=PATIENCE_C, early_stopping=True)
            c_ev = eval_full(c, X_te, y_te)
            enc = c.neural_network.encoder
            if hasattr(enc, "module"):
                enc = enc.module
            encoder_state = {k: v.cpu().clone() for k, v in enc.state_dict().items()}
            del c
            clear_gpu()
            log(f"  C: prauc={c_ev['prauc']:.4f} wss95={c_ev['wss95']:.4f} ({time.time()-t0:.0f}s)")

            # 2. students from the same encoder state
            row = {"seed": seed, "dataset": ds, "fold": fold,
                   "C_prauc": c_ev["prauc"], "C_wss95": c_ev["wss95"]}
            for alpha in args.alphas:
                set_seeds(seed)
                student = build_student()
                student.neural_network.encoder.load_state_dict(
                    {k: v.to(device) for k, v in encoder_state.items()}, strict=False)
                student.fit(X_tr, y_tr, X_vl, y_vl, patience=2, early_stopping=True,
                            kd_type="logit", kd_targets_train=cf_logits["tr"],
                            kd_targets_val=cf_logits["vl"],
                            kd_alpha=alpha, kd_temperature=NUDGE_TEMP)
                ev = eval_full(student, X_te, y_te)
                key = f"a{alpha:g}"
                row[f"{key}_prauc"] = ev["prauc"]
                row[f"{key}_wss95"] = ev["wss95"]
                row[f"{key}_delta"] = ev["prauc"] - c_ev["prauc"]
                log(f"  alpha={alpha:g}: prauc={ev['prauc']:.4f} (delta={ev['prauc']-c_ev['prauc']:+.4f})")
                del student
                clear_gpu()
            del encoder_state
            clear_gpu()
            with open(outpath, "w") as f:
                json.dump(row, f, indent=2)

log("done")
