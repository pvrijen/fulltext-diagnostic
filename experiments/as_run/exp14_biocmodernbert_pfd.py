#!/usr/bin/env python3
"""exp14: second encoder family. BioClinical-ModernBERT (8,192 tokens, no chunking) 2x2 decomposition + PFD.

Protocol per (dataset, outer fold), same splits, hyperparameters and helpers as exp8/exp9 (the paper's pipeline):
  phase "is":
    C   abstract student (MEDLINE abstracts, 512 tokens), seed 42            -> C_abs  (= L_abs->abs)
        the same model re-read at 8,192 tokens on test full text              -> C_ft   (= A_abs->ft, same architecture)
    P   full-text teacher (8,192 tokens), seed 42                             -> P0 (L_ft->abs), P100 (L_ft->ft)
        in-sample logits on train_sub / val_sub full text (cached)
    IS  warm-start PFD nudge of C with the teacher's soft labels (alpha .1, T 4, 3 epochs, lr 5e-6)   -> IS_*
    HARD the same nudge with alpha = 0 (hard labels only)                                            -> HARD_*
  phase "cf":
    five inner teachers (StratifiedKFold, seed 42) -> out-of-fold logits; nudge C with them           -> CF_*

Both the student and the teacher are BertClassifierTruncated with the same checkpoint, so TE, IE and the
interaction are measured within one architecture. Teachers (in-sample and cross-fitted) all use one recipe:
the paper's hyperparameters with batch 8 x 8 accumulation (64 effective), bf16, gradient checkpointing and
flash attention, as in the thesis long-context run; students use the paper's abstract-model recipe unchanged. Jobs are claimed from a shared queue (one worker per GPU).
Usage:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python <this file> --worker 0 [--phases is cf]
Output: papers/output_exp14_biocmb_pfd/{ds}_f{k}.json, cached logits in papers/output_exp14_biocmb_pfd/logits/
Note: the paper's exp9 row labelled BiocModernBERT used ENCODERS.get("biocmodernbert", "answerdotai/ModernBERT-base"),
i.e. the generic ModernBERT-base; this script pins thomas-sounack/BioClinical-ModernBERT-base explicitly.
"""
import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

os.chdir("/home/paul/fulltext_biocuration")
# "scripts" only: belt_plus then resolves to the installed fulltext-diagnostic package, which supports
# gradient checkpointing and bf16 for 8,192-token ModernBERT training (the src/ copy does not).
sys.path.insert(0, "scripts")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

# the training loop reports epochs and steps through logging; make that visible in the worker logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
for _n in ("transformers", "urllib3", "filelock", "httpx", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)

from belt_plus.bert_classifier_truncated import BertClassifierTruncated
from run_rq2_unified import (load_data, make_cv_splits, make_val_split, set_seeds, HP_LR_C, HP_WD_C, HP_DR_C, HP_LR,
                             HP_WD, HP_DR, HP_WARMUP, HP_GRADNORM, HP_TRUNC_BS, timed_eval)
from run_pooling_chapter import PATIENCE_C, PATIENCE_P

MODEL = "thomas-sounack/BioClinical-ModernBERT-base"
MAX_LEN = 8192
SEED = 42
B_CF = 5
NUDGE = dict(learning_rate=5e-6, weight_decay=0.01, dropout_rate=0.1, epochs=3, batch_size=32, alpha=0.1, temperature=4.0)
DATASETS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
OUT = Path("papers/output_exp14_biocmb_pfd"); LOGITS = OUT / "logits"; QUEUE = OUT / "queue"
device = "cuda:0"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()


def metrics(ev, prefix):
    return {f"{prefix}_prauc": float(ev["pr_auc"]), f"{prefix}_wss95": float(ev["wss95"])}


def get_logits(model, X, bs):
    scores = model.predict_scores(X, batch_size=bs)
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    probs = scores.detach().cpu().float().clamp(1e-6, 1 - 1e-6)
    return torch.logit(probs).unsqueeze(-1)


def build_student(n_train):
    c_bs = HP_TRUNC_BS // 2 if n_train > 2000 else HP_TRUNC_BS
    c_acc = 2 if n_train > 2000 else 1
    return BertClassifierTruncated(pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
                                   learning_rate=HP_LR_C, weight_decay=HP_WD_C, dropout_rate=HP_DR_C,
                                   batch_size=c_bs, accumulation_steps=c_acc, epochs=20, max_length=512, use_amp=False,
                                   warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


def build_teacher():
    """Full-text teacher: the paper's hyperparameters (lr, wd, dropout, warmup, epochs, patience) with the thesis
    long-context training settings (batch 8 x 8 accumulation = 64 effective, bf16, gradient checkpointing,
    flash attention). One recipe for in-sample and cross-fitted teachers alike."""
    return BertClassifierTruncated(pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
                                   learning_rate=HP_LR, weight_decay=HP_WD, dropout_rate=HP_DR,
                                   batch_size=8, accumulation_steps=8, epochs=20, max_length=MAX_LEN,
                                   use_amp=True, amp_dtype="bf16", fused_adamw=True, reference_compile=True,
                                   flash_attention=True, gradient_checkpointing=True,
                                   warmup_ratio=HP_WARMUP, max_grad_norm=HP_GRADNORM)


def encoder_state(model):
    nn = model.neural_network
    if hasattr(nn, "module"):
        nn = nn.module
    return copy.deepcopy({k: v.detach().cpu() for k, v in nn.encoder.state_dict().items()})


def nudge(state, X_tr, y_tr, X_vl, y_vl, X_te, y_te, logits_tr, logits_vl, alpha):
    set_seeds(SEED)
    c = BertClassifierTruncated(pretrained_model_name_or_path=MODEL, num_labels=1, device=device,
                                learning_rate=NUDGE["learning_rate"], weight_decay=NUDGE["weight_decay"],
                                dropout_rate=NUDGE["dropout_rate"], batch_size=NUDGE["batch_size"], epochs=NUDGE["epochs"],
                                max_length=512, use_amp=False)
    c.neural_network.encoder.load_state_dict(state, strict=False)
    c.fit(X_tr, y_tr, X_vl, y_vl, patience=2, early_stopping=True, kd_type="logit", kd_targets_train=logits_tr,
          kd_targets_val=logits_vl, kd_alpha=alpha, kd_temperature=NUDGE["temperature"])
    ev = timed_eval(c, X_te, y_te)
    del c; clear_gpu()
    return ev


def load_fold(ds, fold):
    set_seeds(SEED)
    df, ft, med, labels = load_data(ds)
    splits = make_cv_splits(labels, SEED)
    train_idx, test_idx = splits[fold]
    train_sub, val_sub = make_val_split(train_idx, labels, SEED, fold)
    return dict(X_tr_ft=[ft[i] for i in train_sub], X_vl_ft=[ft[i] for i in val_sub], X_tr_med=[med[i] for i in train_sub],
                X_vl_med=[med[i] for i in val_sub], X_te_med=[med[i] for i in test_idx], X_te_ft=[ft[i] for i in test_idx],
                y_tr=[labels[i] for i in train_sub], y_vl=[labels[i] for i in val_sub], y_te=[labels[i] for i in test_idx])


def train_student(d, tag):
    log(f"  [{tag}] C student (BiocModernBERT, abstracts, 512)")
    set_seeds(SEED); t0 = time.time()
    c = build_student(len(d["y_tr"]))
    c.fit(d["X_tr_med"], d["y_tr"], d["X_vl_med"], d["y_vl"], patience=PATIENCE_C, early_stopping=True)
    r = metrics(timed_eval(c, d["X_te_med"], d["y_te"]), "C")
    # fourth cell within the same architecture: the abstract-trained model reading full text at 8,192 tokens
    c._max_length = MAX_LEN
    r.update(metrics(timed_eval_bs(c, d["X_te_ft"], d["y_te"], 4), "Aft"))
    c._max_length = 512
    r["train_time_C"] = time.time() - t0
    state = encoder_state(c)
    del c; clear_gpu()
    return r, state


def timed_eval_bs(model, texts, y_true, bs):
    """timed_eval with a smaller prediction batch (long inputs)."""
    old = model.batch_size; model.batch_size = bs
    try:
        return timed_eval(model, texts, y_true)
    finally:
        model.batch_size = old


def run_is(ds, fold):
    tag = f"{ds}_f{fold}"; out = OUT / f"{tag}.json"
    if out.exists() and "IS_prauc" in json.loads(out.read_text()):
        log(f"SKIP is {tag}"); return
    t0 = time.time(); log(f"=== is {tag} ===")
    d = load_fold(ds, fold)
    r = {"dataset": ds, "fold": fold, "encoder": MODEL, "seed": SEED, "n_train": len(d["y_tr"]), "n_test": len(d["y_te"])}
    rc, state = train_student(d, tag); r.update(rc)
    log(f"  [{tag}] C {r['C_prauc']:.4f}  A_abs->ft {r['Aft_prauc']:.4f}")
    LOGITS.mkdir(parents=True, exist_ok=True)
    lp = LOGITS / f"IS_{tag}.pt"
    if lp.exists():
        cached = torch.load(str(lp), map_location="cpu", weights_only=False)
        is_logits = {"tr": cached["tr"], "vl": cached["vl"]}; r.update(cached["teacher_metrics"]); r["train_time_P"] = cached.get("train_time_P")
        log(f"  [{tag}] teacher logits cached")
    else:
        log(f"  [{tag}] P teacher (BiocModernBERT, full text, {MAX_LEN})")
        set_seeds(SEED); t1 = time.time()
        p = build_teacher()
        p.fit(d["X_tr_ft"], d["y_tr"], d["X_vl_ft"], d["y_vl"], patience=PATIENCE_P, early_stopping=True)
        tm = {}
        tm.update(metrics(timed_eval_bs(p, d["X_te_med"], d["y_te"], 8), "P0"))      # L_ft->abs (MEDLINE abstract)
        tm.update(metrics(timed_eval_bs(p, d["X_te_ft"], d["y_te"], 4), "P100"))     # L_ft->ft
        # artefact check: the teacher on full text truncated to its first 512 tokens (training format, abstract length)
        p._max_length = 512
        tm.update(metrics(timed_eval_bs(p, d["X_te_ft"], d["y_te"], 8), "P0trunc"))
        p._max_length = MAX_LEN
        is_logits = {"tr": get_logits(p, d["X_tr_ft"], 4), "vl": get_logits(p, d["X_vl_ft"], 4)}
        r["train_time_P"] = time.time() - t1
        torch.save({**is_logits, "teacher_metrics": tm, "train_time_P": r["train_time_P"]}, str(lp))
        r.update(tm)
        del p; clear_gpu()
    log(f"  [{tag}] P0 {r['P0_prauc']:.4f}  P100 {r['P100_prauc']:.4f}  TE {r['P0_prauc'] - r['C_prauc']:+.4f}  IE {r['P100_prauc'] - r['P0_prauc']:+.4f}"
        + (f"  P0trunc {r['P0trunc_prauc']:.4f}" if 'P0trunc_prauc' in r else ""))
    log(f"  [{tag}] PFD nudge (in-sample soft labels)")
    r.update(metrics(nudge(state, d["X_tr_med"], d["y_tr"], d["X_vl_med"], d["y_vl"], d["X_te_med"], d["y_te"], is_logits["tr"], is_logits["vl"], NUDGE["alpha"]), "IS"))
    log(f"  [{tag}] hard-label control (alpha = 0)")
    r.update(metrics(nudge(state, d["X_tr_med"], d["y_tr"], d["X_vl_med"], d["y_vl"], d["X_te_med"], d["y_te"], is_logits["tr"], is_logits["vl"], 0.0), "HARD"))
    r["elapsed_min_is"] = round((time.time() - t0) / 60, 1)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(r); out.write_text(json.dumps(prev, indent=2))
    log(f"  [{tag}] DONE is in {r['elapsed_min_is']} min: C {r['C_prauc']:.4f} IS {r['IS_prauc']:.4f} ({r['IS_prauc'] - r['C_prauc']:+.4f}) HARD {r['HARD_prauc']:.4f} ({r['HARD_prauc'] - r['C_prauc']:+.4f})")


def run_cf(ds, fold):
    tag = f"{ds}_f{fold}"; out = OUT / f"{tag}.json"
    if out.exists() and "CF_prauc" in json.loads(out.read_text()):
        log(f"SKIP cf {tag}"); return
    lp = LOGITS / f"IS_{tag}.pt"
    if not lp.exists():
        log(f"  [{tag}] cf needs the in-sample teacher logits first; skipping"); return
    t0 = time.time(); log(f"=== cf {tag} ===")
    d = load_fold(ds, fold)
    rc, state = train_student(d, tag)
    vl = torch.load(str(lp), map_location="cpu", weights_only=False)["vl"]
    cp = LOGITS / f"CF{B_CF}_{tag}.pt"
    if cp.exists():
        cf_tr = torch.load(str(cp), map_location="cpu", weights_only=True)["tr"]
    else:
        n = len(d["y_tr"]); cf_tr = torch.zeros(n, 1)
        skf = StratifiedKFold(n_splits=B_CF, shuffle=True, random_state=SEED)
        for b, (inner_tr, inner_pred) in enumerate(skf.split(np.zeros(n), d["y_tr"])):
            log(f"  [{tag}] CF teacher {b + 1}/{B_CF} (train {len(inner_tr)}, predict {len(inner_pred)})")
            X_i = [d["X_tr_ft"][i] for i in inner_tr]; y_i = [d["y_tr"][i] for i in inner_tr]; n_v = max(1, len(inner_tr) // 10)
            set_seeds(SEED)
            p = build_teacher()
            p.fit(X_i[:-n_v], y_i[:-n_v], X_i[-n_v:], y_i[-n_v:], patience=PATIENCE_P, early_stopping=True)
            pred = get_logits(p, [d["X_tr_ft"][i] for i in inner_pred], 4)
            for j, idx in enumerate(inner_pred):
                cf_tr[idx] = pred[j]
            del p; clear_gpu()
        torch.save({"tr": cf_tr, "vl": vl}, str(cp))
    log(f"  [{tag}] PFD nudge (cross-fitted soft labels)")
    r = metrics(nudge(state, d["X_tr_med"], d["y_tr"], d["X_vl_med"], d["y_vl"], d["X_te_med"], d["y_te"], cf_tr, vl, NUDGE["alpha"]), "CF")
    r["elapsed_min_cf"] = round((time.time() - t0) / 60, 1)
    prev = json.loads(out.read_text()) if out.exists() else {"dataset": ds, "fold": fold}
    prev.update(r); out.write_text(json.dumps(prev, indent=2))
    log(f"  [{tag}] DONE cf in {r['elapsed_min_cf']} min: CF {r['CF_prauc']:.4f} ({r['CF_prauc'] - prev.get('C_prauc', float('nan')):+.4f})")


def claim(job):
    QUEUE.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(QUEUE / (job + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode()); os.close(fd); return True
    except FileExistsError:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--phases", nargs="+", default=["is", "cf"], choices=["is", "cf"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--reclaim", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(ph, ds, k) for ph in a.phases for ds in a.datasets for k in range(5)]  # all "is" jobs before any "cf"
    log(f"worker {a.worker}: {len(jobs)} jobs, model {MODEL}")
    for ph, ds, k in jobs:
        tag = f"{ph}_{ds}_f{k}"
        if not a.reclaim and not claim(tag):
            continue
        if a.reclaim:
            claim(tag)
        try:
            (run_is if ph == "is" else run_cf)(ds, k)
        except Exception as e:
            log(f"  [{tag}] FAILED: {e!r}"); (QUEUE / (tag + ".failed")).write_text(repr(e)); clear_gpu()
    log(f"worker {a.worker}: queue exhausted")
