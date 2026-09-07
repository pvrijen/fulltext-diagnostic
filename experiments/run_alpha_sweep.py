#!/usr/bin/env python3
"""HP sensitivity for warm-start PFD under cross-fitting.

Sweep alpha in {0.05, 0.1, 0.2, 0.3, 0.5, 1.0} at epochs=3, T=4.0.
Uses cached C checkpoints and CF teacher logits from the warmstart experiment.
Runs on cd011431 + cd012661 (fastest datasets with most tagged docs).

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python papers/hp_sensitivity_sweep.py --alphas 0.05,0.1 --gpu 0
"""
import argparse, gc, json, os, sys, time
import numpy as np, torch
from pathlib import Path
from scipy.stats import wilcoxon

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from belt_plus.bert_classifier_truncated import BertClassifierTruncated
from run_rq2_unified import load_data, make_cv_splits, make_val_split, set_seeds, ENCODERS, timed_eval

MODEL = ENCODERS["biomedbert-abs"]
DATASETS = ["cd011431", "cd012661"]
SEED = 42
NUDGE_T = 4.0
NUDGE_LR = 5e-6
NUDGE_WD = 0.01
NUDGE_DR = 0.1
NUDGE_EPOCHS = 3

C_CKPT = Path("papers/output_warmstart_pfd_crossfit/checkpoints")
CF_LOGITS = Path("papers/output_warmstart_pfd_crossfit/cached_logits")
OUT = Path("papers/output_hp_sensitivity")
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--alphas", type=str, required=True)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
alphas = [float(a) for a in args.alphas.split(",")]
device = f"cuda:{args.gpu}"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()

rows = []
for ds in DATASETS:
    set_seeds(SEED)
    df, ft, med, labels = load_data(ds)
    splits = make_cv_splits(labels, SEED)

    for fold in range(5):
        train_idx, test_idx = splits[fold]
        train_sub, val_sub = make_val_split(train_idx, labels, SEED, fold)
        X_tr = [med[i] for i in train_sub]
        X_vl = [med[i] for i in val_sub]
        y_tr = [labels[i] for i in train_sub]
        y_vl = [labels[i] for i in val_sub]
        X_te = [med[i] for i in test_idx]
        y_te = [labels[i] for i in test_idx]

        c_ckpt = C_CKPT / ds / f"C_{ds}_s42_f{fold}.pt"
        cf_path = CF_LOGITS / ds / f"CF5_{ds}_f{fold}.pt"

        if not c_ckpt.exists() or not cf_path.exists():
            log(f"SKIP {ds}_f{fold} — missing checkpoints")
            continue

        cf_data = torch.load(str(cf_path), map_location='cpu', weights_only=True)
        cf_logits_tr = cf_data['tr']
        cf_logits_vl = cf_data['vl']

        for alpha in alphas:
            tag = f"{ds}_f{fold}_a{alpha}"
            result_path = OUT / f"{tag}.json"
            if result_path.exists():
                rows.append(json.load(open(result_path)))
                log(f"SKIP {tag}")
                continue

            log(f"{tag}")
            set_seeds(SEED)
            c = BertClassifierTruncated(
                pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
                learning_rate=NUDGE_LR, weight_decay=NUDGE_WD, dropout_rate=NUDGE_DR,
                batch_size=32, epochs=NUDGE_EPOCHS, max_length=512, use_amp=False)
            c.neural_network.encoder.load_state_dict(
                torch.load(str(c_ckpt), map_location=device, weights_only=True), strict=False)
            c.fit(X_tr, y_tr, X_vl, y_vl, patience=2, early_stopping=True,
                  kd_type="logit", kd_targets_train=cf_logits_tr, kd_targets_val=cf_logits_vl,
                  kd_alpha=alpha, kd_temperature=NUDGE_T)
            ev = timed_eval(c, X_te, y_te)
            del c; clear_gpu()

            r = {"dataset": ds, "fold": fold, "alpha": alpha, "prauc": ev['pr_auc']}
            rows.append(r)
            result_path.write_text(json.dumps(r, indent=2))
            log(f"  PR-AUC: {ev['pr_auc']:.4f}")

# Also get ND baseline for comparison
log("\nND baselines:")
nd_rows = []
for ds in DATASETS:
    for fold in range(5):
        nd_path = OUT / f"{ds}_f{fold}_ND.json"
        if nd_path.exists():
            nd_rows.append(json.load(open(nd_path)))
            continue
        set_seeds(SEED)
        df, ft, med, labels = load_data(ds)
        splits = make_cv_splits(labels, SEED)
        train_idx, test_idx = splits[fold]
        train_sub, val_sub = make_val_split(train_idx, labels, SEED, fold)
        c_ckpt = C_CKPT / ds / f"C_{ds}_s42_f{fold}.pt"
        if not c_ckpt.exists(): continue
        from run_rq2_unified import HP_LR_C, HP_WD_C, HP_DR_C, HP_WARMUP, HP_GRADNORM, HP_TRUNC_BS
        from run_pooling_chapter import PATIENCE_C
        c_bs = HP_TRUNC_BS // 2 if len([labels[i] for i in train_sub]) > 2000 else HP_TRUNC_BS
        c = BertClassifierTruncated(
            pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
            learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
            batch_size=c_bs, epochs=20, max_length=512, use_amp=False,
            warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)
        c.neural_network.encoder.load_state_dict(
            torch.load(str(c_ckpt), map_location=device, weights_only=True), strict=False)
        X_tr = [med[i] for i in train_sub]; X_vl = [med[i] for i in val_sub]
        y_tr = [labels[i] for i in train_sub]; y_vl = [labels[i] for i in val_sub]
        X_te = [med[i] for i in test_idx]; y_te = [labels[i] for i in test_idx]
        c.fit(X_tr, y_tr, X_vl, y_vl, patience=PATIENCE_C, early_stopping=True)
        ev = timed_eval(c, X_te, y_te)
        del c; clear_gpu()
        r = {"dataset": ds, "fold": fold, "alpha": 0, "prauc": ev['pr_auc']}
        nd_rows.append(r)
        nd_path.write_text(json.dumps(r, indent=2))

# Aggregate
log(f"\n{'='*60}")
log("HP SENSITIVITY (CF-PFD, warm-start)")
log(f"{'='*60}")
all_rows = nd_rows + rows
for alpha in [0] + sorted(set(r['alpha'] for r in rows)):
    ar = [r for r in all_rows if r['alpha'] == alpha]
    if len(ar) >= 5:
        label = "ND" if alpha == 0 else f"alpha={alpha}"
        mean = np.mean([r['prauc'] for r in ar])
        if alpha > 0:
            deltas = []
            for r in ar:
                nd = [n for n in nd_rows if n['dataset']==r['dataset'] and n['fold']==r['fold']]
                if nd: deltas.append(r['prauc'] - nd[0]['prauc'])
            if len(deltas) >= 5:
                _, p = wilcoxon(deltas)
                log(f"  {label:12s}: {mean:.4f} (delta={np.mean(deltas):+.4f}, p={p:.4f})")
            else:
                log(f"  {label:12s}: {mean:.4f}")
        else:
            log(f"  {label:12s}: {mean:.4f}")

with open(OUT / "summary.json", "w") as f:
    json.dump(all_rows, f, indent=2)
