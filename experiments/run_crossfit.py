#!/usr/bin/env python3
"""THE definitive experiment: warm-start PFD (thesis protocol) under cross-fitting.

Protocol:
  1. Train C (abstract student) to convergence, save checkpoint
  2. Train IS teacher on 100% FT → cache logits (in-sample PFD, reproduces Paper B)
  3. Train B=5 cross-fitted teachers (each on 80% FT) → cache logits (cross-fitted PFD)
  4. Load C checkpoint, nudge 3 epochs at LR=5e-6 with IS logits → IS-PFD result
  5. Load C checkpoint, nudge 3 epochs at LR=5e-6 with CF logits → CF-PFD result

If IS-PFD ≈ +0.016 (reproduces Paper B) and CF-PFD ≈ +0.016 → PFD is REAL
If IS-PFD ≈ +0.016 but CF-PFD ≈ 0 → PFD is MEMORISATION

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/run_crossfit.py
"""
import gc, json, os, sys, time
import numpy as np, torch
from pathlib import Path


from belt_plus.bert_classifier_truncated import BertClassifierTruncated
from belt_plus.bert_classifier_with_pooling import BertClassifierWithPooling
from _helpers import *
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon

MODEL = ENCODERS["biomedbert-abs"]
DATASETS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
SEED = 42
B_CF = 5  # cross-fitting folds

# Thesis PFD nudge hyperparameters (from run_teacher_ablation_5fold.py)
NUDGE_LR = 5e-6
NUDGE_WD = 0.01
NUDGE_DR = 0.1
NUDGE_EPOCHS = 3
NUDGE_ALPHA = 0.1
NUDGE_TEMP = 4.0

OUT = Path("papers/output_warmstart_pfd_crossfit")
CKPT = OUT / "checkpoints"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def build_teacher(device):
    return BertClassifierWithPooling(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=HP_LR, weight_decay=HP_WD, dropout_rate=HP_DR,
        batch_size=T_BS, accumulation_steps=T_ACCUM, epochs=20,
        chunk_size=510, stride=510, minimal_chunk_length=50, max_chunks=12,
        chunk_pooling_strategy="mean", use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


def get_logits(model, X):
    scores = model.predict_scores(X)
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    probs = scores.detach().cpu().float().clamp(1e-6, 1 - 1e-6)
    return torch.logit(probs).unsqueeze(-1)


def train_c_and_save(X_tr, y_tr, X_vl, y_vl, X_te, y_te, ckpt_path, device):
    """Train C to convergence, save checkpoint, return test PR-AUC."""
    set_seeds(SEED)
    c_bs = HP_TRUNC_BS // 2 if len(y_tr) > 2000 else HP_TRUNC_BS
    c_acc = 2 if len(y_tr) > 2000 else 1
    c = BertClassifierTruncated(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
        batch_size=c_bs, accumulation_steps=c_acc, epochs=20,
        max_length=512, use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)
    c.fit(X_tr, y_tr, X_vl, y_vl, patience=PATIENCE_C, early_stopping=True)
    ev = timed_eval(c, X_te, y_te)

    # Save encoder state
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    nn = c.neural_network
    if hasattr(nn, "module"):
        nn = nn.module
    torch.save(nn.encoder.state_dict(), ckpt_path)
    log(f"  C checkpoint saved to {ckpt_path}")

    del c; clear_gpu()
    return ev['pr_auc']


def nudge_c(ckpt_path, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
            logits_tr, logits_vl, device):
    """Load C checkpoint, fine-tune 3 epochs with teacher soft labels."""
    set_seeds(SEED)
    c = BertClassifierTruncated(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=NUDGE_LR, weight_decay=NUDGE_WD, dropout_rate=NUDGE_DR,
        batch_size=32, epochs=NUDGE_EPOCHS, max_length=512, use_amp=False)
    c.neural_network.encoder.load_state_dict(
        torch.load(str(ckpt_path), map_location=device, weights_only=True), strict=False)
    c.fit(X_tr, y_tr, X_vl, y_vl, patience=2, early_stopping=True,
          kd_type="logit", kd_targets_train=logits_tr, kd_targets_val=logits_vl,
          kd_alpha=NUDGE_ALPHA, kd_temperature=NUDGE_TEMP)
    ev = timed_eval(c, X_te, y_te)
    del c; clear_gpu()
    return ev['pr_auc']


def run_fold(ds, fold, device):
    tag = f"{ds}_f{fold}"
    result_path = OUT / f"{tag}.json"
    if result_path.exists():
        log(f"SKIP {tag} (cached)")
        return json.loads(result_path.read_text())

    log(f"\n{'='*50}")
    log(f"{tag}")
    log(f"{'='*50}")

    set_seeds(SEED)
    df, ft, med, labels = load_data(ds)
    splits = make_cv_splits(labels, SEED)
    train_idx, test_idx = splits[fold]
    train_sub, val_sub = make_val_split(train_idx, labels, SEED, fold)

    X_tr_ft = [ft[i] for i in train_sub]
    X_vl_ft = [ft[i] for i in val_sub]
    X_tr_med = [med[i] for i in train_sub]
    X_vl_med = [med[i] for i in val_sub]
    y_tr = [labels[i] for i in train_sub]
    y_vl = [labels[i] for i in val_sub]
    X_te = [med[i] for i in test_idx]
    y_te = [labels[i] for i in test_idx]

    # ── Step 1: Train C, save checkpoint ──
    c_ckpt = CKPT / ds / f"C_{ds}_s42_f{fold}.pt"
    if c_ckpt.exists():
        log(f"  C checkpoint exists, training ND for PR-AUC...")
        nd = train_c_and_save.__wrapped__(X_tr_med, y_tr, X_vl_med, y_vl, X_te, y_te, c_ckpt, device) if False else None
        # Just re-evaluate from checkpoint
        set_seeds(SEED)
        c_bs = HP_TRUNC_BS // 2 if len(y_tr) > 2000 else HP_TRUNC_BS
        c_acc = 2 if len(y_tr) > 2000 else 1
        c = BertClassifierTruncated(
            pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
            learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
            batch_size=c_bs, accumulation_steps=c_acc, epochs=20,
            max_length=512, use_amp=False,
            warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)
        c.neural_network.encoder.load_state_dict(
            torch.load(str(c_ckpt), map_location=device, weights_only=True), strict=False)
        c.fit(X_tr_med, y_tr, X_vl_med, y_vl, patience=PATIENCE_C, early_stopping=True)
        ev = timed_eval(c, X_te, y_te)
        nd = ev['pr_auc']
        del c; clear_gpu()
    else:
        log(f"  Training C from scratch...")
        nd = train_c_and_save(X_tr_med, y_tr, X_vl_med, y_vl, X_te, y_te, c_ckpt, device)
    log(f"  ND (C baseline): {nd:.4f}")

    # ── Step 2: IS teacher (full data, in-sample logits) ──
    is_logits_path = OUT / "cached_logits" / ds / f"IS_{tag}.pt"
    is_logits_path.parent.mkdir(parents=True, exist_ok=True)
    if is_logits_path.exists():
        log(f"  Loading cached IS logits...")
        is_logits = torch.load(str(is_logits_path), map_location='cpu', weights_only=True)
    else:
        log(f"  Training IS teacher (100% FT)...")
        teacher = build_teacher(device)
        teacher.fit(X_tr_ft, y_tr, X_vl_ft, y_vl, patience=PATIENCE_P, early_stopping=True)
        is_logits = {'tr': get_logits(teacher, X_tr_ft), 'vl': get_logits(teacher, X_vl_ft)}
        torch.save(is_logits, str(is_logits_path))
        del teacher; clear_gpu()

    # ── Step 3: IS PFD nudge ──
    log(f"  Nudging C with IS logits...")
    is_pfd = nudge_c(c_ckpt, X_tr_med, y_tr, X_vl_med, y_vl, X_te, y_te,
                     is_logits['tr'], is_logits['vl'], device)
    log(f"  IS-PFD: {is_pfd:.4f} (delta={is_pfd - nd:+.4f})")

    # ── Step 4: Cross-fitted teachers (B=5) ──
    cf_logits_path = OUT / "cached_logits" / ds / f"CF{B_CF}_{tag}.pt"
    if cf_logits_path.exists():
        log(f"  Loading cached CF logits...")
        cf_data = torch.load(str(cf_logits_path), map_location='cpu', weights_only=True)
        cf_logits_tr = cf_data['tr']
        cf_logits_vl = cf_data['vl']
    else:
        log(f"  Training {B_CF} cross-fitted teachers...")
        n = len(y_tr)
        cf_logits_tr = torch.zeros(n, 1)
        skf = StratifiedKFold(n_splits=B_CF, shuffle=True, random_state=SEED)
        for b, (inner_tr, inner_pred) in enumerate(skf.split(np.zeros(n), y_tr)):
            log(f"    CF fold {b+1}/{B_CF} (train {len(inner_tr)}, predict {len(inner_pred)})")
            X_i_ft = [X_tr_ft[i] for i in inner_tr]
            y_i = [y_tr[i] for i in inner_tr]
            n_v = max(1, len(inner_tr) // 10)

            teacher = build_teacher(device)
            teacher.fit(X_i_ft[:-n_v], y_i[:-n_v], X_i_ft[-n_v:], y_i[-n_v:],
                       patience=PATIENCE_P, early_stopping=True)

            X_pred = [X_tr_ft[i] for i in inner_pred]
            pred_logits = get_logits(teacher, X_pred)
            for j, idx in enumerate(inner_pred):
                cf_logits_tr[idx] = pred_logits[j]
            del teacher; clear_gpu()

        # Validation logits: train one teacher on all training data
        log(f"    Training val teacher for CF logits...")
        vt = build_teacher(device)
        vt.fit(X_tr_ft, y_tr, X_vl_ft, y_vl, patience=PATIENCE_P, early_stopping=True)
        cf_logits_vl = get_logits(vt, X_vl_ft)
        del vt; clear_gpu()

        torch.save({'tr': cf_logits_tr, 'vl': cf_logits_vl}, str(cf_logits_path))

    # ── Step 5: CF PFD nudge ──
    log(f"  Nudging C with CF logits...")
    cf_pfd = nudge_c(c_ckpt, X_tr_med, y_tr, X_vl_med, y_vl, X_te, y_te,
                     cf_logits_tr, cf_logits_vl, device)
    log(f"  CF-PFD: {cf_pfd:.4f} (delta={cf_pfd - nd:+.4f})")

    log(f"  SUMMARY: ND={nd:.4f}  IS-PFD={is_pfd:.4f}({is_pfd-nd:+.4f})  CF-PFD={cf_pfd:.4f}({cf_pfd-nd:+.4f})")

    r = {"dataset": ds, "fold": fold, "ND": nd, "IS_PFD": is_pfd, "CF_PFD": cf_pfd}
    result_path.write_text(json.dumps(r, indent=2))
    return r


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated dataset list (default: all 6)")
    args = p.parse_args()

    ds_list = args.datasets.split(",") if args.datasets else DATASETS

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    rows = []

    for ds in ds_list:
        for fold in range(5):
            r = run_fold(ds, fold, device)
            rows.append(r)

    # Aggregate
    log(f"\n{'='*60}")
    log(f"AGGREGATE (n={len(rows)})")
    log(f"{'='*60}")
    nd_m = np.mean([r['ND'] for r in rows])
    is_m = np.mean([r['IS_PFD'] for r in rows])
    cf_m = np.mean([r['CF_PFD'] for r in rows])
    log(f"  ND:      {nd_m:.4f}")
    log(f"  IS-PFD:  {is_m:.4f}")
    log(f"  CF-PFD:  {cf_m:.4f}")

    d_is = [r['IS_PFD'] - r['ND'] for r in rows]
    d_cf = [r['CF_PFD'] - r['ND'] for r in rows]
    _, p_is = wilcoxon(d_is)
    _, p_cf = wilcoxon(d_cf)
    log(f"  IS-PFD - ND: {np.mean(d_is):+.4f} (p={p_is:.4f})")
    log(f"  CF-PFD - ND: {np.mean(d_cf):+.4f} (p={p_cf:.4f})")
    log(f"  IS-CF gap:   {np.mean(d_is) - np.mean(d_cf):+.4f}")

    print("\nPer-dataset:")
    for ds in DATASETS:
        dr = [r for r in rows if r['dataset'] == ds]
        d_is = np.mean([r['IS_PFD'] - r['ND'] for r in dr])
        d_cf = np.mean([r['CF_PFD'] - r['ND'] for r in dr])
        print(f"  {ds:10s}  IS:{d_is:+.4f}  CF:{d_cf:+.4f}  gap:{d_is-d_cf:+.4f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    log(f"Saved to {OUT / 'summary.json'}")
