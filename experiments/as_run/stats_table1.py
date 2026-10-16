#!/usr/bin/env python3
"""Table 1 (encoder comparison, abstract-only input): per-fold PR-AUC and WSS@95 from the frozen thesis runs,
so the table can carry SDs, a WSS@95 column, and paired differences against BioMedBERT with cluster CIs.

Sources (read-only; FROZEN_THESIS_DATA markers are respected, nothing is regenerated):
  output/rq1_baselines_pmc_only/{ds}/{enc}_{ds}_s42_f{k}.json   BERT, RoBERTa, BioBERT, Bio-Longformer   (evals.test)
  output/rq2_pooling_chapter/{ds}/C_{ds}_s42_f{k}.json            BioMedBERT, condition C                  (evals.test)
  output/rq2c_longctx/{ds}/bioclinical-modernbert_s42_f{k}.json  BiocModernBERT, abstract-only eval       (evals.T_abs)
  output/rq1_classical_baselines_pmc_only/{ds}/svm_s42_f{k}.json SVM (TF-IDF)                              (pr_auc, wss95)
Writes table1_stats.json next to this script.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("THESIS_OUTPUT", "/home/paul/fulltext_biocuration/output"))
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
rng = np.random.default_rng(42)


def load_enc(enc):
    vals = {}
    for d in DS:
        for k in range(5):
            f = OUT / "rq1_baselines_pmc_only" / d / f"{enc}_{d}_s42_f{k}.json"
            j = json.load(open(f))["evals"]["test"]
            vals[(d, k)] = (j["pr_auc"], j["wss95"])
    return vals


def load_biomedbert():
    vals = {}
    for d in DS:
        for k in range(5):
            j = json.load(open(OUT / "rq2_pooling_chapter" / d / f"C_{d}_s42_f{k}.json"))["evals"]["test"]
            vals[(d, k)] = (j["pr_auc"], j["wss95"])
    return vals


def load_longctx(enc, key):
    vals = {}
    for d in DS:
        for k in range(5):
            j = json.load(open(OUT / "rq2c_longctx" / d / f"{enc}_s42_f{k}.json"))["evals"][key]
            vals[(d, k)] = (j["pr_auc"], j["wss95"])
    return vals


def load_svm():
    vals = {}
    for d in DS:
        for k in range(5):
            j = json.load(open(OUT / "rq1_classical_baselines_pmc_only" / d / f"svm_s42_f{k}.json"))
            vals[(d, k)] = (j["pr_auc"], j["wss95"])
    return vals


def cluster_ci(d, g, B=20000):
    groups = sorted(set(g)); byg = {x: d[g == x] for x in groups}; out = np.empty(B)
    for b in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        out[b] = np.concatenate([rng.choice(byg[x], size=len(byg[x]), replace=True) for x in pick]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load_paper_svm():
    """SVM row of the paper (TF-IDF, k = 10,000; svm_decomposition_wss.json, S_abs->abs), same as Table 2."""
    rows = json.load(open(HERE / "svm_decomposition_wss.json"))
    return {(r["dataset"], r["fold"]): (r["C_prauc"], r["C_wss95"]) for r in rows}


# rq2c_longctx evaluation keys: T_* = model trained on abstracts (truncated), evaluated on abstracts (T_abs);
# P_ft = model trained on full text, evaluated on full text within the native window.
models = {
    "BioMedBERT": load_biomedbert(),
    "BiocModernBERT (abstract)": load_longctx("bioclinical-modernbert", "T_abs"),
    "BiocModernBERT (full text)": load_longctx("bioclinical-modernbert", "P_ft"),
    "Bio-Longformer (abstract)": load_longctx("bio-longformer", "T_abs"),
    "Bio-Longformer (full text)": load_longctx("bio-longformer", "P_ft"),
    "BioBERT": load_enc("biobert"),
    "RoBERTa": load_enc("roberta"),
    "BERT": load_enc("bert"),
    "SVM (TF-IDF)": load_paper_svm(),
}
alt_bl = load_enc("bio-longformer")  # rq1 run of Bio-Longformer on abstracts (seed 42), for the record
keys = [(d, k) for d in DS for k in range(5)]
g = np.array([d for d, _ in keys])
ref = models["BioMedBERT"]
res = {}
for name, v in models.items():
    pr = np.array([v[k][0] for k in keys]); ws = np.array([v[k][1] for k in keys])
    ent = {"prauc_mean": float(pr.mean()), "prauc_sd": float(pr.std(ddof=1)), "wss_mean": float(ws.mean()), "wss_sd": float(ws.std(ddof=1))}
    if name != "BioMedBERT":
        dpr = pr - np.array([ref[k][0] for k in keys]); dws = ws - np.array([ref[k][1] for k in keys])
        lo, hi = cluster_ci(dpr, g); wlo, whi = cluster_ci(dws, g)
        ent.update({"d_prauc": float(dpr.mean()), "d_prauc_ci": [lo, hi], "p_prauc": float(stats.wilcoxon(dpr).pvalue),
                    "d_wss": float(dws.mean()), "d_wss_ci": [wlo, whi], "p_wss": float(stats.wilcoxon(dws).pvalue)})
    res[name] = ent
    print(f"{name:15s} PR-AUC {ent['prauc_mean']:.3f} sd {ent['prauc_sd']:.3f} | WSS@95 {ent['wss_mean']:.3f} sd {ent['wss_sd']:.3f}"
          + (f" | vs BioMedBERT PR {ent['d_prauc']:+.3f} [{lo:+.3f},{hi:+.3f}] P={ent['p_prauc']:.3f}; WSS {ent['d_wss']:+.3f} [{wlo:+.3f},{whi:+.3f}] P={ent['p_wss']:.3f}" if name != "BioMedBERT" else ""))
print("Bio-Longformer from rq2c_longctx T_abs:", round(float(np.mean([alt_bl[k][0] for k in keys])), 3), "(Table 1 prints 0.695)")
json.dump(res, open(HERE / "table1_stats.json", "w"), indent=1)
print("wrote", HERE / "table1_stats.json")
