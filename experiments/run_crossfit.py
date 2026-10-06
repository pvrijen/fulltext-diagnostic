#!/usr/bin/env python3
"""Core experiment: TE/IE decomposition and warm-start PFD under cross-fitting.

Per dataset and outer fold:
  1. Train C (abstract baseline) to convergence, save encoder checkpoint.
  2. Train the in-sample (IS) teacher P on 100% full text. Evaluate it on
     test abstracts (P@0%) and on test full text (P@100%), cache its logits
     on the training set, save its encoder checkpoint.
  3. Train B=5 cross-fitted (CF) teachers, each on 80% of the training set,
     and cache out-of-sample logits for the remaining 20%.
  4. Warm-start C, fine-tune 3 epochs at LR 5e-6 with IS logits -> IS-PFD.
  5. Warm-start C, fine-tune 3 epochs at LR 5e-6 with CF logits -> CF-PFD.

Outputs (under $FTD_OUT_DIR/crossfit/<encoder>/): one JSON per
dataset-fold with C, P@0%, P@100%, TE, IE, Total, IS_PFD, CF_PFD (PR-AUC)
and the same quantities on WSS@95, plus summary.json.

Usage (from the repository root):
  python experiments/run_crossfit.py --datasets pfas,bpa
  python experiments/run_crossfit.py --encoder bio-longformer --datasets pfas
"""
import argparse
import gc
import json
import time

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold

from belt_plus.bert_classifier_truncated import BertClassifierTruncated
from belt_plus.bert_classifier_with_pooling import BertClassifierWithPooling
from _helpers import (
    DATASET_NAMES, ENCODERS, HP_DR, HP_DR_C, HP_GRADNORM, HP_LR, HP_LR_C,
    HP_TRUNC_BS, HP_WARMUP, HP_WD, HP_WD_C, NUDGE_ALPHA, NUDGE_DR,
    NUDGE_EPOCHS, NUDGE_LR, NUDGE_TEMP, NUDGE_WD, OUT_ROOT, PATIENCE_C,
    PATIENCE_P, SEED, T_ACCUM, T_BS, load_data, make_cv_splits,
    make_val_split, set_seeds, timed_eval,
)

B_CF = 5  # cross-fitting folds


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def build_teacher(model_id, device):
    return BertClassifierWithPooling(
        pretrained_model_name_or_path=model_id, num_labels=1, device=device,
        learning_rate=HP_LR, weight_decay=HP_WD, dropout_rate=HP_DR,
        batch_size=T_BS, accumulation_steps=T_ACCUM, epochs=20,
        chunk_size=510, stride=510, minimal_chunk_length=50, max_chunks=12,
        chunk_pooling_strategy="mean", use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


def build_c(model_id, n_train, device):
    c_bs = HP_TRUNC_BS // 2 if n_train > 2000 else HP_TRUNC_BS
    c_acc = 2 if n_train > 2000 else 1
    return BertClassifierTruncated(
        pretrained_model_name_or_path=model_id, num_labels=1, device=device,
        learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
        batch_size=c_bs, accumulation_steps=c_acc, epochs=20,
        max_length=512, use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


def get_logits(model, X):
    scores = model.predict_scores(X)
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    probs = scores.detach().cpu().float().clamp(1e-6, 1 - 1e-6)
    return torch.logit(probs).unsqueeze(-1)


def save_encoder(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nn = model.neural_network
    if hasattr(nn, "module"):
        nn = nn.module
    torch.save(nn.encoder.state_dict(), path)


def train_c(model_id, X_tr, y_tr, X_vl, y_vl, X_te, y_te, ckpt_path, device):
    """Train C to convergence, save checkpoint, return test metrics."""
    set_seeds(SEED)
    c = build_c(model_id, len(y_tr), device)
    c.fit(X_tr, y_tr, X_vl, y_vl, patience=PATIENCE_C, early_stopping=True)
    ev = timed_eval(c, X_te, y_te)
    save_encoder(c, ckpt_path)
    log(f"  C checkpoint saved to {ckpt_path}")
    del c
    clear_gpu()
    return ev


def nudge_c(model_id, ckpt_path, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
            logits_tr, logits_vl, device):
    """Warm-start from the C checkpoint, fine-tune with teacher soft labels."""
    set_seeds(SEED)
    c = BertClassifierTruncated(
        pretrained_model_name_or_path=model_id, num_labels=1, device=device,
        learning_rate=NUDGE_LR, weight_decay=NUDGE_WD, dropout_rate=NUDGE_DR,
        batch_size=32, epochs=NUDGE_EPOCHS, max_length=512, use_amp=False)
    c.neural_network.encoder.load_state_dict(
        torch.load(str(ckpt_path), map_location=device, weights_only=True),
        strict=False)
    c.fit(X_tr, y_tr, X_vl, y_vl, patience=2, early_stopping=True,
          kd_type="logit", kd_targets_train=logits_tr, kd_targets_val=logits_vl,
          kd_alpha=NUDGE_ALPHA, kd_temperature=NUDGE_TEMP)
    ev = timed_eval(c, X_te, y_te)
    del c
    clear_gpu()
    return ev


def run_fold(ds, fold, model_id, out, device):
    tag = f"{ds}_f{fold}"
    result_path = out / f"{tag}.json"
    if result_path.exists():
        log(f"SKIP {tag} (cached)")
        return json.loads(result_path.read_text())

    log(f"\n{'=' * 50}\n{tag}\n{'=' * 50}")
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
    X_te_med = [med[i] for i in test_idx]
    X_te_ft = [ft[i] for i in test_idx]
    y_te = [labels[i] for i in test_idx]

    ckpt = out / "checkpoints" / ds
    cache = out / "cached_logits" / ds
    cache.mkdir(parents=True, exist_ok=True)

    # Step 1: C
    c_ckpt = ckpt / f"C_{ds}_s{SEED}_f{fold}.pt"
    c_eval_path = cache / f"C_eval_{tag}.json"
    if c_ckpt.exists() and c_eval_path.exists():
        log("  C checkpoint and evaluation cached")
        c_ev = json.loads(c_eval_path.read_text())
    else:
        log("  Training C...")
        c_ev = train_c(model_id, X_tr_med, y_tr, X_vl_med, y_vl, X_te_med, y_te,
                       c_ckpt, device)
        c_eval_path.write_text(json.dumps(
            {k: v for k, v in c_ev.items() if k != "probs"}))
    nd = c_ev["pr_auc"]
    log(f"  C (abstract baseline): PR-AUC={nd:.4f} WSS@95={c_ev['wss95']:.4f}")

    # Step 2: IS teacher, decomposition, logits
    is_logits_path = cache / f"IS_{tag}.pt"
    p_eval_path = cache / f"P_eval_{tag}.json"
    p_ckpt = ckpt / f"P_{ds}_s{SEED}_f{fold}.pt"
    if is_logits_path.exists() and p_eval_path.exists():
        log("  Loading cached IS logits and P evaluation...")
        is_logits = torch.load(str(is_logits_path), map_location="cpu",
                               weights_only=True)
        p_ev = json.loads(p_eval_path.read_text())
    else:
        log("  Training IS teacher (100% full text)...")
        set_seeds(SEED)
        teacher = build_teacher(model_id, device)
        teacher.fit(X_tr_ft, y_tr, X_vl_ft, y_vl, patience=PATIENCE_P,
                    early_stopping=True)
        p0 = timed_eval(teacher, X_te_med, y_te)
        p100 = timed_eval(teacher, X_te_ft, y_te)
        p_ev = {"p0_pr_auc": p0["pr_auc"], "p0_wss95": p0["wss95"],
                "p100_pr_auc": p100["pr_auc"], "p100_wss95": p100["wss95"]}
        p_eval_path.write_text(json.dumps(p_ev))
        is_logits = {"tr": get_logits(teacher, X_tr_ft),
                     "vl": get_logits(teacher, X_vl_ft)}
        torch.save(is_logits, str(is_logits_path))
        save_encoder(teacher, p_ckpt)
        del teacher
        clear_gpu()
    log(f"  P@0%={p_ev['p0_pr_auc']:.4f}  P@100%={p_ev['p100_pr_auc']:.4f}  "
        f"TE={p_ev['p0_pr_auc'] - nd:+.4f}  "
        f"IE={p_ev['p100_pr_auc'] - p_ev['p0_pr_auc']:+.4f}")

    # Step 3: IS PFD
    log("  Nudging C with IS logits...")
    is_pfd = nudge_c(model_id, c_ckpt, X_tr_med, y_tr, X_vl_med, y_vl,
                     X_te_med, y_te, is_logits["tr"], is_logits["vl"], device)
    log(f"  IS-PFD: {is_pfd['pr_auc']:.4f} (delta={is_pfd['pr_auc'] - nd:+.4f})")

    # Step 4: cross-fitted teachers
    cf_logits_path = cache / f"CF{B_CF}_{tag}.pt"
    if cf_logits_path.exists():
        log("  Loading cached CF logits...")
        cf_data = torch.load(str(cf_logits_path), map_location="cpu",
                             weights_only=True)
        cf_logits_tr, cf_logits_vl = cf_data["tr"], cf_data["vl"]
    else:
        log(f"  Training {B_CF} cross-fitted teachers...")
        n = len(y_tr)
        cf_logits_tr = torch.zeros(n, 1)
        skf = StratifiedKFold(n_splits=B_CF, shuffle=True, random_state=SEED)
        for b, (inner_tr, inner_pred) in enumerate(skf.split(np.zeros(n), y_tr)):
            log(f"    CF fold {b + 1}/{B_CF} "
                f"(train {len(inner_tr)}, predict {len(inner_pred)})")
            X_i_ft = [X_tr_ft[i] for i in inner_tr]
            y_i = [y_tr[i] for i in inner_tr]
            n_v = max(1, len(inner_tr) // 10)
            set_seeds(SEED + b)
            teacher = build_teacher(model_id, device)
            teacher.fit(X_i_ft[:-n_v], y_i[:-n_v], X_i_ft[-n_v:], y_i[-n_v:],
                        patience=PATIENCE_P, early_stopping=True)
            pred_logits = get_logits(teacher, [X_tr_ft[i] for i in inner_pred])
            for j, idx in enumerate(inner_pred):
                cf_logits_tr[idx] = pred_logits[j]
            del teacher
            clear_gpu()
        # Validation logits come from the IS teacher (trained on all training data)
        cf_logits_vl = is_logits["vl"]
        torch.save({"tr": cf_logits_tr, "vl": cf_logits_vl}, str(cf_logits_path))

    # Step 5: CF PFD
    log("  Nudging C with CF logits...")
    cf_pfd = nudge_c(model_id, c_ckpt, X_tr_med, y_tr, X_vl_med, y_vl,
                     X_te_med, y_te, cf_logits_tr, cf_logits_vl, device)
    log(f"  CF-PFD: {cf_pfd['pr_auc']:.4f} (delta={cf_pfd['pr_auc'] - nd:+.4f})")

    r = {
        "dataset": ds, "fold": fold, "encoder": model_id,
        "C": nd, "P@0%": p_ev["p0_pr_auc"], "P@100%": p_ev["p100_pr_auc"],
        "TE": p_ev["p0_pr_auc"] - nd,
        "IE": p_ev["p100_pr_auc"] - p_ev["p0_pr_auc"],
        "Total": p_ev["p100_pr_auc"] - nd,
        "IS_PFD": is_pfd["pr_auc"], "CF_PFD": cf_pfd["pr_auc"],
        "wss95": {
            "C": c_ev["wss95"], "P@0%": p_ev["p0_wss95"],
            "P@100%": p_ev["p100_wss95"],
            "TE": p_ev["p0_wss95"] - c_ev["wss95"],
            "IE": p_ev["p100_wss95"] - p_ev["p0_wss95"],
            "IS_PFD": is_pfd["wss95"], "CF_PFD": cf_pfd["wss95"],
        },
        "n_test": len(y_te), "n_pos": int(sum(y_te)),
    }
    result_path.write_text(json.dumps(r, indent=2))
    return r


def _p(values):
    values = [v for v in values if v != 0]
    if len(values) < 6:
        return float("nan")
    return float(wilcoxon(values)[1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default="biomedbert-abs",
                   help=f"Encoder key ({', '.join(ENCODERS)}) or HuggingFace id")
    p.add_argument("--datasets", default=",".join(DATASET_NAMES),
                   help="Comma-separated dataset list")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model_id = ENCODERS.get(args.encoder, args.encoder)
    out = OUT_ROOT / "crossfit" / args.encoder.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    ds_list = args.datasets.split(",")

    rows = []
    for ds in ds_list:
        for fold in range(5):
            rows.append(run_fold(ds, fold, model_id, out, args.device))

    log(f"\n{'=' * 60}\nAGGREGATE (n={len(rows)}, encoder={model_id})\n{'=' * 60}")
    for key in ["C", "P@0%", "P@100%", "TE", "IE", "Total", "IS_PFD", "CF_PFD"]:
        vals = [r[key] for r in rows]
        log(f"  {key:8s}: {np.mean(vals):+.4f} +/- {np.std(vals, ddof=1):.4f}")
    d_is = [r["IS_PFD"] - r["C"] for r in rows]
    d_cf = [r["CF_PFD"] - r["C"] for r in rows]
    log(f"  TE p={_p([r['TE'] for r in rows]):.4f}  "
        f"IE p={_p([r['IE'] for r in rows]):.4f}  "
        f"Total p={_p([r['Total'] for r in rows]):.4f}")
    log(f"  IS-PFD - C: {np.mean(d_is):+.4f} (p={_p(d_is):.4f})")
    log(f"  CF-PFD - C: {np.mean(d_cf):+.4f} (p={_p(d_cf):.4f})")

    print("\nPer-dataset:")
    for ds in ds_list:
        dr = [r for r in rows if r["dataset"] == ds]
        print(f"  {ds:10s}  TE:{np.mean([r['TE'] for r in dr]):+.4f}  "
              f"IE:{np.mean([r['IE'] for r in dr]):+.4f}  "
              f"CF-PFD:{np.mean([r['CF_PFD'] - r['C'] for r in dr]):+.4f}")

    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    log(f"Saved to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
