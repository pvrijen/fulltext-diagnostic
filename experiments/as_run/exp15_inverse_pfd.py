"""EXP15: the inverse PFD route (abstract teacher -> full-text student), Paper B v2.10.

The paper distills the full-text model's predictions into an abstract-only
student. LUPI theory says the reverse cannot help: an abstract teacher sees a
strict subset of what the full-text student sees, so its soft labels carry no
privileged information. This experiment measures that on the same 30
dataset-folds, with the same protocol as exp8 (seed 42, same outer folds,
same validation split, same hyperparameters, same composite loss).

Per (dataset, fold):
  1. Abstract baseline C (truncated, 512 tokens), trained from scratch as in
     exp8. Its logits on the training and validation abstracts are the
     in-sample (IS) abstract soft labels.
  2. B = 5 cross-fitted (CF) abstract teachers, each trained on 80% of the
     training fold and predicting the held-out 20%, exactly as the full-text
     CF teachers were built; a sixth teacher on the whole training fold gives
     the validation-set logits (same convention as papers/warmstart_pfd_crossfit.py).
  3. Full-text teacher P (chunk-pooled, 12 x 510 tokens, mean pooling), as in
     exp8; evaluated on test abstracts (P@0%) and test full text (P@100%).
     This replicates TE and IE for the fold and gives the converged encoder
     state the students warm-start from.
  4. Three warm-started full-text students (encoder from P, LR 5e-6, 3 epochs,
     patience 2, composite loss with alpha = 0.1 and T = 4 as in the paper),
     trained and scored on FULL TEXT:
       INV_CF : soft labels = cross-fitted abstract logits   (the inverse route)
       INV_IS : soft labels = in-sample abstract logits
       INV_A0 : alpha = 0, hard labels only (continued fine-tuning control)
     Each student is also scored on the test abstracts, to see whether the
     abstract teacher pulls the full-text model back toward abstract competence.

The contrast of interest is INV_CF - P@100% (expected about zero or negative)
next to the paper's PFD - C (positive). INV_A0 separates distillation from
the three extra epochs.

GPU required; one process per GPU, split by dataset:
  CUDA_VISIBLE_DEVICES=2 python exp15_inverse_pfd.py --datasets pfas bpa
  CUDA_VISIBLE_DEVICES=0 python exp15_inverse_pfd.py --datasets cd012661 cd008122
  CUDA_VISIBLE_DEVICES=1 python exp15_inverse_pfd.py --datasets cd012069 cd011431
Cached stages (abstract logits, P checkpoint, per-condition results) make a
restart skip finished work.
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
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

os.chdir("/home/paul/fulltext_biocuration")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+",
                    default=["pfas", "bpa", "cd008122", "cd011431", "cd012069", "cd012661"])
parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--out", default="papers/paper_b_biocuration_guide/versions/2.10_jamia/results")
parser.add_argument("--work", default="papers/output_exp15_inverse_pfd")
parser.add_argument("--smoke", action="store_true", help="tiny subsets, 1 epoch: API check only")
args = parser.parse_args()

from belt_plus.bert_classifier_truncated import BertClassifierTruncated  # noqa: E402
from belt_plus.bert_classifier_with_pooling import BertClassifierWithPooling  # noqa: E402
from run_rq2_unified import (load_data, make_cv_splits, make_val_split, set_seeds,  # noqa: E402
                             ENCODERS, HP_LR, HP_WD, HP_DR, HP_LR_C, HP_WD_C, HP_DR_C,
                             HP_WARMUP, HP_GRADNORM, HP_TRUNC_BS, timed_eval)
from run_pooling_chapter import PATIENCE_C, PATIENCE_P, T_BS, T_ACCUM  # noqa: E402

MODEL = ENCODERS["biomedbert-abs"]
SEED = 42
B_CF = 5
NUDGE_LR, NUDGE_WD, NUDGE_DR, NUDGE_EPOCHS, NUDGE_ALPHA, NUDGE_TEMP = 5e-6, 0.01, 0.1, 3, 0.1, 4.0

OUT = Path(args.out)
WORK = Path(args.work)
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
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    probs = np.asarray(probs, dtype=float).reshape(-1)
    return {"prauc": float(ev["pr_auc"]), "wss95": wss95(probs, [int(y) for y in y_te]),
            "probs": probs.tolist()}


def get_logits(model, X):
    with torch.no_grad():
        scores = model.predict_scores(X)
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    scores = scores.detach().cpu().float().reshape(-1)
    return torch.logit(scores.clamp(1e-6, 1 - 1e-6)).unsqueeze(-1)


def encoder_state_of(model):
    enc = model.neural_network.encoder
    if hasattr(enc, "module"):
        enc = enc.module
    return {k: v.cpu().clone() for k, v in enc.state_dict().items()}


def build_c(n_train):
    if args.smoke:
        return BertClassifierTruncated(pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
            learning_rate=HP_LR_C, batch_size=8, epochs=1, max_length=512, use_amp=False)
    c_bs = HP_TRUNC_BS // 2 if n_train > 2000 else HP_TRUNC_BS
    c_acc = 2 if n_train > 2000 else 1
    return BertClassifierTruncated(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
        batch_size=c_bs, accumulation_steps=c_acc, epochs=20,
        max_length=512, use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


def build_p(lr=HP_LR, wd=HP_WD, dr=HP_DR, epochs=20):
    if args.smoke:
        epochs = 1
    return BertClassifierWithPooling(
        pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
        learning_rate=lr, weight_decay=wd, dropout_rate=dr,
        batch_size=T_BS, accumulation_steps=T_ACCUM, epochs=epochs,
        chunk_size=510, stride=510, minimal_chunk_length=50, max_chunks=12,
        chunk_pooling_strategy="mean", use_amp=False,
        warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


def fit_pooled_kd(model, X_tr, y_tr, X_vl, y_vl, kd_tr, kd_vl, alpha, temp, epochs, patience):
    """Composite-loss fine-tuning of a chunk-pooled model.

    Same loss as BertClassifierTruncated.fit(kd_type="logit"):
      (1 - alpha) * BCE(logits, y) + alpha * T^2 * BCE(sigmoid(logits/T), sigmoid(teacher/T))
    Labels passed to the dataset are row indices, so random shuffling keeps
    each document aligned with its own hard label and teacher logit.
    Validation on the same composite loss; early stopping restores the best state.
    """
    nn_ = model.neural_network
    dev = model.device
    ds_tr = model._create_dataset(model._tokenize(X_tr), list(range(len(y_tr))))
    ds_vl = model._create_dataset(model._tokenize(X_vl), list(range(len(y_vl))))
    loader_tr = DataLoader(ds_tr, sampler=RandomSampler(ds_tr), batch_size=model.batch_size,
                           collate_fn=model.collate_fn)
    loader_vl = DataLoader(ds_vl, sampler=SequentialSampler(ds_vl), batch_size=model.batch_size,
                           collate_fn=model.collate_fn)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32)
    kd_tr_t = kd_tr.detach().float().reshape(-1)
    kd_vl_t = kd_vl.detach().float().reshape(-1)
    assert len(kd_tr_t) == len(y_tr) and len(kd_vl_t) == len(y_vl)

    optimizer = model._create_optimizer(nn_.parameters())
    total_steps = max(1, len(loader_tr) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    accum = max(1, model.accumulation_steps)

    def composite(logits, idx, y_all, kd_all):
        s = logits.reshape(-1)
        y = y_all[idx].to(dev)
        sup = F.binary_cross_entropy_with_logits(s, y)
        if alpha <= 0:
            return sup
        t = kd_all[idx].to(dev)
        ps = torch.sigmoid(s / temp)
        pt = torch.sigmoid(t / temp)
        eps = 1e-6
        kd = -(pt * torch.log(ps + eps) + (1 - pt) * torch.log(1 - ps + eps)).mean() * temp * temp
        return (1.0 - alpha) * sup + alpha * kd

    best_val, wait, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.current_epoch = epoch
        nn_.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader_tr):
            idx = batch[-1].long()
            logits = model._evaluate_single_batch(batch)
            loss = composite(logits, idx, y_tr_t, kd_tr_t) / accum
            loss.backward()
            if ((step + 1) % accum == 0) or (step + 1 == len(loader_tr)):
                torch.nn.utils.clip_grad_norm_(nn_.parameters(), model.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if not (epoch == 0 and step == 0):
                    scheduler.step()
        nn_.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for batch in loader_vl:
                idx = batch[-1].long()
                logits = model._evaluate_single_batch(batch)
                vloss += float(composite(logits, idx, y_vl_t, kd_vl_t))
                nb += 1
        vloss /= max(1, nb)
        log(f"      epoch {epoch}: val composite loss {vloss:.5f}")
        if vloss < best_val:
            best_val, wait = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in nn_.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        nn_.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    return model


for ds in args.datasets:
    set_seeds(SEED)
    df, ft, med, labels = load_data(ds)
    splits = make_cv_splits(labels, SEED)
    (WORK / "cached_logits" / ds).mkdir(parents=True, exist_ok=True)
    (WORK / "checkpoints" / ds).mkdir(parents=True, exist_ok=True)
    (WORK / "probs" / ds).mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        tag = f"{ds}_f{fold}"
        outpath = OUT / f"exp15_{tag}.json"
        if outpath.exists():
            log(f"SKIP {tag} (done)")
            continue
        partial = WORK / f"partial_{tag}.json"
        row = json.load(open(partial)) if partial.exists() else {"dataset": ds, "fold": fold}

        def save_partial():
            with open(partial, "w") as f:
                json.dump(row, f, indent=2)

        log(f"\n=== {tag} ===")
        train_idx, test_idx = splits[fold]
        train_sub, val_sub = make_val_split(train_idx, labels, SEED, fold)
        X_tr_ft = [ft[i] for i in train_sub]
        X_vl_ft = [ft[i] for i in val_sub]
        X_te_ft = [ft[i] for i in test_idx]
        X_tr_med = [med[i] for i in train_sub]
        X_vl_med = [med[i] for i in val_sub]
        X_te_med = [med[i] for i in test_idx]
        y_tr = [labels[i] for i in train_sub]
        y_vl = [labels[i] for i in val_sub]
        y_te = [labels[i] for i in test_idx]
        if args.smoke:
            k = 40
            X_tr_ft, X_vl_ft, X_te_ft = X_tr_ft[:k], X_vl_ft[:k], X_te_ft[:k]
            X_tr_med, X_vl_med, X_te_med = X_tr_med[:k], X_vl_med[:k], X_te_med[:k]
            y_tr, y_vl, y_te = y_tr[:k], y_vl[:k], y_te[:k]
        probs_path = WORK / "probs" / ds / f"probs_{tag}.npz"
        probs = dict(np.load(probs_path)) if probs_path.exists() else {}
        probs["labels"] = np.array(y_te, dtype=int)

        def save_probs():
            np.savez(probs_path, **probs)

        # 1. abstract baseline C + in-sample abstract logits
        is_path = WORK / "cached_logits" / ds / f"AIS_{tag}.pt"
        if "C_prauc" in row and is_path.exists():
            log("  C cached")
        else:
            t0 = time.time()
            log("  Training C (abstract baseline) from scratch...")
            set_seeds(SEED)
            c = build_c(len(y_tr))
            c.fit(X_tr_med, y_tr, X_vl_med, y_vl, patience=PATIENCE_C, early_stopping=True)
            c_ev = eval_full(c, X_te_med, y_te)
            row["C_prauc"], row["C_wss95"] = c_ev["prauc"], c_ev["wss95"]
            probs["C"] = np.array(c_ev["probs"])
            torch.save({"tr": get_logits(c, X_tr_med), "vl": get_logits(c, X_vl_med)}, str(is_path))
            del c
            clear_gpu()
            save_partial(); save_probs()
            log(f"  C: prauc={row['C_prauc']:.4f} wss95={row['C_wss95']:.4f} ({time.time()-t0:.0f}s)")

        # 2. cross-fitted abstract logits
        cf_path = WORK / "cached_logits" / ds / f"ACF5_{tag}.pt"
        if cf_path.exists():
            log("  CF abstract logits cached")
        else:
            t0 = time.time()
            log(f"  Training {B_CF} cross-fitted abstract teachers...")
            n = len(y_tr)
            cf_tr = torch.zeros(n, 1)
            skf = StratifiedKFold(n_splits=B_CF, shuffle=True, random_state=SEED)
            for b, (inner_tr, inner_pred) in enumerate(skf.split(np.zeros(n), y_tr)):
                log(f"    CF fold {b+1}/{B_CF} (train {len(inner_tr)}, predict {len(inner_pred)})")
                X_i = [X_tr_med[i] for i in inner_tr]
                y_i = [y_tr[i] for i in inner_tr]
                n_v = max(1, len(inner_tr) // 10)
                set_seeds(SEED)
                t = build_c(len(y_i) - n_v)
                t.fit(X_i[:-n_v], y_i[:-n_v], X_i[-n_v:], y_i[-n_v:],
                      patience=PATIENCE_C, early_stopping=True)
                pred = get_logits(t, [X_tr_med[i] for i in inner_pred])
                for j, idx in enumerate(inner_pred):
                    cf_tr[idx] = pred[j]
                del t
                clear_gpu()
            log("    Training val teacher for CF logits...")
            set_seeds(SEED)
            vt = build_c(len(y_tr))
            vt.fit(X_tr_med, y_tr, X_vl_med, y_vl, patience=PATIENCE_C, early_stopping=True)
            cf_vl = get_logits(vt, X_vl_med)
            del vt
            clear_gpu()
            torch.save({"tr": cf_tr, "vl": cf_vl}, str(cf_path))
            log(f"  CF abstract logits done ({time.time()-t0:.0f}s)")

        # 3. full-text teacher P (the student's starting point)
        p_ckpt = WORK / "checkpoints" / ds / f"P_{tag}.pt"
        if "P100_prauc" in row and p_ckpt.exists():
            log("  P cached")
            encoder_state = torch.load(str(p_ckpt), map_location="cpu", weights_only=True)
        else:
            t0 = time.time()
            log("  Training P (full-text, pooled)...")
            set_seeds(SEED)
            p = build_p()
            p.fit(X_tr_ft, y_tr, X_vl_ft, y_vl, patience=PATIENCE_P, early_stopping=True)
            p0 = eval_full(p, X_te_med, y_te)
            p100 = eval_full(p, X_te_ft, y_te)
            row["P0_prauc"], row["P0_wss95"] = p0["prauc"], p0["wss95"]
            row["P100_prauc"], row["P100_wss95"] = p100["prauc"], p100["wss95"]
            row["P_best_epoch"] = p.best_epoch
            probs["P0"], probs["P100"] = np.array(p0["probs"]), np.array(p100["probs"])
            encoder_state = encoder_state_of(p)
            torch.save(encoder_state, str(p_ckpt))
            del p
            clear_gpu()
            save_partial(); save_probs()
            log(f"  P@0%: prauc={row['P0_prauc']:.4f}  P@100%: prauc={row['P100_prauc']:.4f} "
                f"wss95={row['P100_wss95']:.4f} ({time.time()-t0:.0f}s)")

        # 4. inverse students: full-text model nudged with abstract soft labels
        is_logits = torch.load(str(is_path), map_location="cpu", weights_only=True)
        cf_logits = torch.load(str(cf_path), map_location="cpu", weights_only=True)
        arms = [("INV_CF", cf_logits, NUDGE_ALPHA),
                ("INV_IS", is_logits, NUDGE_ALPHA),
                ("INV_A0", cf_logits, 0.0)]
        for name, lg, alpha in arms:
            if f"{name}_prauc" in row:
                log(f"  {name} cached")
                continue
            t0 = time.time()
            log(f"  {name}: nudging P on full text (alpha={alpha:g})...")
            set_seeds(SEED)
            s = build_p(lr=NUDGE_LR, wd=NUDGE_WD, dr=NUDGE_DR, epochs=NUDGE_EPOCHS)
            s.neural_network.encoder.load_state_dict(
                {k: v.to(device) for k, v in encoder_state.items()}, strict=False)
            fit_pooled_kd(s, X_tr_ft, y_tr, X_vl_ft, y_vl, lg["tr"], lg["vl"],
                          alpha=alpha, temp=NUDGE_TEMP, epochs=NUDGE_EPOCHS, patience=2)
            ev_ft = eval_full(s, X_te_ft, y_te)
            ev_abs = eval_full(s, X_te_med, y_te)
            row[f"{name}_prauc"], row[f"{name}_wss95"] = ev_ft["prauc"], ev_ft["wss95"]
            row[f"{name}_abs_prauc"], row[f"{name}_abs_wss95"] = ev_abs["prauc"], ev_abs["wss95"]
            probs[name], probs[f"{name}_abs"] = np.array(ev_ft["probs"]), np.array(ev_abs["probs"])
            del s
            clear_gpu()
            save_partial(); save_probs()
            log(f"  {name}: prauc={ev_ft['prauc']:.4f} (vs P@100% {ev_ft['prauc']-row['P100_prauc']:+.4f}) "
                f"wss95={ev_ft['wss95']:.4f}; on abstracts prauc={ev_abs['prauc']:.4f} ({time.time()-t0:.0f}s)")

        row["TE"] = row["P0_prauc"] - row["C_prauc"]
        row["IE"] = row["P100_prauc"] - row["P0_prauc"]
        row["Total"] = row["P100_prauc"] - row["C_prauc"]
        for name in ("INV_CF", "INV_IS", "INV_A0"):
            row[f"{name}_delta"] = row[f"{name}_prauc"] - row["P100_prauc"]
            row[f"{name}_wss_delta"] = row[f"{name}_wss95"] - row["P100_wss95"]
        with open(outpath, "w") as f:
            json.dump(row, f, indent=2)
        partial.unlink(missing_ok=True)
        del encoder_state
        clear_gpu()
        log(f"  DONE {tag}: TE={row['TE']:+.4f} IE={row['IE']:+.4f} "
            f"INV_CF={row['INV_CF_delta']:+.4f} INV_IS={row['INV_IS_delta']:+.4f} INV_A0={row['INV_A0_delta']:+.4f}")

log("all done")
