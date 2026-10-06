#!/usr/bin/env python3
"""Compute cosine silhouette scores across all datasets, folds, and conditions.

Conditions: C, P@0%, P@100%, C_PFD. Requires the C and P checkpoints and CF
logits written by run_crossfit.py (under $FTD_OUT_DIR/crossfit/biomedbert-abs/).
Also caches the CLS features of the first dataset-fold for the UMAP figure
(gen_figures.py).

Usage (from the repository root):
  python experiments/run_silhouette.py --gpu 0 --datasets pfas,bpa,cd008122,cd011431,cd012069,cd012661
"""
import gc, json, argparse
import numpy as np, torch
from sklearn.metrics import silhouette_score


from belt_plus.bert_classifier_truncated import BertClassifierTruncated
from belt_plus.bert_classifier_with_pooling import BertClassifierWithPooling
from _helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--datasets", type=str, required=True, help="Comma-separated")
args = parser.parse_args()

device = f"cuda:{args.gpu}"
MODEL = ENCODERS["biomedbert-abs"]
SEED = 42
DATASETS = args.datasets.split(",")

CROSSFIT = OUT_ROOT / "crossfit" / "biomedbert-abs"
C_CKPT_DIR = CROSSFIT / "checkpoints"
CF_LOGITS_DIR = CROSSFIT / "cached_logits"
P_CKPT_DIR = CROSSFIT / "checkpoints"
OUT = OUT_ROOT / "silhouette"
OUT.mkdir(parents=True, exist_ok=True)

def extract_cls(model, texts, device, batch_size=32):
    model.neural_network.eval()
    feats = []
    for i in range(0, len(texts), batch_size):
        enc = model.tokenizer(texts[i:i+batch_size], return_tensors="pt",
                              truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            feats.append(model.neural_network.encoder(**enc).last_hidden_state[:, 0, :].cpu())
    return torch.cat(feats).numpy()

def build_trunc(y_tr):
    c_bs = HP_TRUNC_BS // 2 if len(y_tr) > 2000 else HP_TRUNC_BS
    return BertClassifierTruncated(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
        batch_size=c_bs, epochs=20, max_length=512, use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)

def build_pooled(y_tr):
    return BertClassifierWithPooling(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=HP_LR, weight_decay=HP_WD, dropout_rate=HP_DR,
        batch_size=T_BS, accumulation_steps=T_ACCUM, epochs=20,
        chunk_size=510, stride=510, minimal_chunk_length=50, max_chunks=12,
        chunk_pooling_strategy="mean", use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)

rows = []

for ds in DATASETS:
    print(f"\n=== {ds} (GPU {args.gpu}) ===", flush=True)
    set_seeds(SEED)
    df, ft, med, labels = load_data(ds)
    splits = make_cv_splits(labels, SEED)

    for fold in range(5):
        tag = f"{ds}_f{fold}"
        result_path = OUT / f"{tag}.json"
        if result_path.exists():
            print(f"  SKIP {tag}", flush=True)
            rows.append(json.load(open(result_path)))
            continue

        print(f"  {tag}...", flush=True)
        train_idx, test_idx = splits[fold]
        train_sub, val_sub = make_val_split(train_idx, labels, SEED, fold)

        X_tr = [med[i] for i in train_sub]
        X_vl = [med[i] for i in val_sub]
        X_tr_ft = [ft[i] for i in train_sub]
        X_vl_ft = [ft[i] for i in val_sub]
        y_tr = [labels[i] for i in train_sub]
        y_vl = [labels[i] for i in val_sub]
        X_te = [med[i] for i in test_idx]
        X_te_ft = [ft[i] for i in test_idx]
        y_te = np.array([labels[i] for i in test_idx])

        c_ckpt = C_CKPT_DIR / ds / f"C_{ds}_s{SEED}_f{fold}.pt"
        p_ckpt = P_CKPT_DIR / ds / f"P_{ds}_s{SEED}_f{fold}.pt"
        cf_logits = CF_LOGITS_DIR / ds / f"CF5_{ds}_f{fold}.pt"

        if not c_ckpt.exists() or not p_ckpt.exists():
            print("    Missing checkpoint, skip (run run_crossfit.py first)", flush=True)
            continue

        row = {"dataset": ds, "fold": fold}

        # 1. C
        c = build_trunc(y_tr)
        c.neural_network.encoder.load_state_dict(
            torch.load(str(c_ckpt), map_location=device, weights_only=True), strict=False)
        c.fit(X_tr, y_tr, X_vl, y_vl, patience=PATIENCE_C, early_stopping=True)
        feats_c = extract_cls(c, X_te, device)
        del c; gc.collect(); torch.cuda.empty_cache()

        # 2. P (train on FT, extract CLS on abstracts = P@0%, on FT = P@100%)
        p = build_pooled(y_tr)
        p.neural_network.encoder.load_state_dict(
            torch.load(str(p_ckpt), map_location=device, weights_only=True), strict=False)
        p.fit(X_tr_ft, y_tr, X_vl_ft, y_vl, patience=2, early_stopping=True)
        feats_p0 = extract_cls(p, X_te, device)      # abstracts
        feats_p100 = extract_cls(p, X_te_ft, device)  # full-text
        del p; gc.collect(); torch.cuda.empty_cache()

        # 3. C_PFD
        if cf_logits.exists():
            cf_data = torch.load(str(cf_logits), map_location='cpu', weights_only=True)
            set_seeds(SEED)
            cpfd = BertClassifierTruncated(
                pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
                learning_rate=5e-6, weight_decay=0.01, dropout_rate=0.1,
                batch_size=32, epochs=3, max_length=512, use_amp=False)
            cpfd.neural_network.encoder.load_state_dict(
                torch.load(str(c_ckpt), map_location=device, weights_only=True), strict=False)
            cpfd.fit(X_tr, y_tr, X_vl, y_vl, patience=2, early_stopping=True,
                     kd_type="logit", kd_targets_train=cf_data['tr'], kd_targets_val=cf_data['vl'],
                     kd_alpha=0.1, kd_temperature=4.0)
            feats_pfd = extract_cls(cpfd, X_te, device)
            del cpfd; gc.collect(); torch.cuda.empty_cache()

            row["sil_pfd"] = float(silhouette_score(feats_pfd, y_te, metric='cosine'))
        else:
            row["sil_pfd"] = None

        umap_cache = OUT / "umap_cache.npz"
        if not umap_cache.exists() and cf_logits.exists():
            np.savez(umap_cache, feats_c=feats_c, feats_p0=feats_p0,
                     feats_p100=feats_p100, feats_pfd=feats_pfd, y_te=y_te,
                     dataset=ds, fold=fold)

        row["sil_c"] = float(silhouette_score(feats_c, y_te, metric='cosine'))
        row["sil_p0"] = float(silhouette_score(feats_p0, y_te, metric='cosine'))
        row["sil_p100"] = float(silhouette_score(feats_p100, y_te, metric='cosine'))
        row["n_test"] = len(y_te)
        row["n_pos"] = int(y_te.sum())

        print(f"    C={row['sil_c']:.3f} P@0%={row['sil_p0']:.3f} "
              f"P@100%={row['sil_p100']:.3f} PFD={row.get('sil_pfd', 'N/A')}", flush=True)

        rows.append(row)
        result_path.write_text(json.dumps(row, indent=2))

# Summary
print("\n=== SUMMARY ===")
import pandas as pd
df = pd.DataFrame(rows)
print(df.groupby("dataset")[["sil_c", "sil_p0", "sil_p100", "sil_pfd"]].mean().round(3))
print("\nOverall mean:")
print(df[["sil_c", "sil_p0", "sil_p100", "sil_pfd"]].mean().round(3))
df.to_csv(OUT / f"silhouette_gpu{args.gpu}.csv", index=False)
