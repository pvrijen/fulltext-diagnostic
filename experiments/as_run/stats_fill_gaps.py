#!/usr/bin/env python3
"""Fill the statistics gaps found in the v2.8 table scan, from fold-level data only.

- Table 2: SD of the abstract-trained cells (exp12) and of the SVM conditions (svm_decomposition_wss.json), both metrics
- Table 3: SD of the per-seed PFD and hard-label gains (stats_v26.json)
- Table 4: dWSS@95 (mean, SD, cluster CI) for PFD and L_ft->ft; SD of recall at each workload per condition;
           direct contrast L_PFD - L_ft->ft on PR-AUC and WSS@95 (pfd_vs_ft.json)
- Table S3: cluster-bootstrap CI of the CF-PFD gain for every alpha (exp10 files; alpha = 0 from exp11 seed 42)
- Table S4: cluster-bootstrap CI and SD for TE, IE and total per encoder (exp9 files; BioMedBERT from exp8)
- Text: SVM-teacher PFD gain with CI, on PR-AUC and WSS@95 (results_svm_pfd)
Cluster bootstrap: datasets then folds, 20,000 replicates, seed 42. Writes gaps.json next to this script.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R21 = Path(os.environ.get("EXP8_DIR", HERE.parent / "2.1_jamia_experiments" / "results"))
R25 = Path(os.environ.get("V25_DIR", HERE.parent / "2.5_jamia"))
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
rng = np.random.default_rng(42)


def load(pattern):
    return {(r["dataset"], r["fold"]): r for r in (json.load(open(f)) for f in sorted(glob.glob(str(pattern))))}


def cluster(d, g, B=20000):
    groups = sorted(set(g)); byg = {x: d[g == x] for x in groups}; out = np.empty(B)
    for b in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        out[b] = np.concatenate([rng.choice(byg[x], size=len(byg[x]), replace=True) for x in pick]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_pooled": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


out = {}
e8 = load(R21 / "exp8_unified_*.json"); e12 = load(HERE / "results" / "exp12_*.json")
keys = sorted(k for k in e8 if k in e12); g = np.array([k[0] for k in keys])
# Table 2: A-cell SDs per dataset, both metrics
out["A_sd"] = {m: {d: {"abs": float(np.std([e12[k][f"Aabs_abs_{m}"] for k in keys if k[0] == d], ddof=1)),
                       "ft": float(np.std([e12[k][f"Aabs_ft_{m}"] for k in keys if k[0] == d], ddof=1))} for d in DS} for m in ("prauc", "wss95")}
# Table 2: SVM row SDs (across 30 folds) both metrics
svm = json.load(open(HERE / "svm_decomposition_wss.json"))
out["svm_sd"] = {m: {c: float(np.std([r[f"{c}_{m}"] for r in svm], ddof=1)) for c in ("C", "P0", "P100")} for m in ("prauc", "wss95")}
# Table 3: per-seed SDs
S = json.load(open(HERE / "stats_v26.json"))
out["seed_sd"] = {s: {"a01": S["exp11_per_seed"][s]["alpha01"]["sd"], "a0": S["exp11_per_seed"][s]["alpha0"]["sd"]} for s in ("42", "123", "456")}
# Table 4: dWSS@95 and recall SDs
W = json.load(open(HERE / "wss_block.json"))["contrasts"]
out["dwss"] = {"PFD": W["dPFD"], "ft": W["total"]}
rc = [json.load(open(f)) for f in sorted(glob.glob(str(R25 / "results_recall_curves" / "recall_*_f*.json")))]
def recall_at(sc, y, frac):
    sc, y = np.asarray(sc), np.asarray(y); k = int(np.ceil(frac * len(y))); o = np.argsort(-sc, kind="stable")
    return y[o][:k].sum() / y.sum()
out["recall_sd"] = {f"{int(b*100)}": {c: float(np.std([recall_at(r[f"scores_{c}"], r["y_test"], b) for r in rc], ddof=1) * 100)
                                       for c in ("c", "pfd", "p100")} for b in (0.10, 0.20, 0.30, 0.50)}
# Table S3: alpha CIs
alpha = {}
for a in ("0.05", "0.1", "0.2", "0.3", "0.5", "1.0"):
    e10 = load(R21 / f"exp10_alpha_a{a}_*.json"); ks = sorted(e10); gg = np.array([k[0] for k in ks])
    d = np.array([e10[k]["CF_prauc"] - e10[k]["C_prauc"] for k in ks])
    alpha[a] = cluster(d, gg)
alpha["0.0"] = {"mean": S["exp11_per_seed"]["42"]["alpha0"]["mean"], "sd": S["exp11_per_seed"]["42"]["alpha0"]["sd"],
                "ci_lo": S["exp11_per_seed"]["42"]["alpha0"]["ci_lo"], "ci_hi": S["exp11_per_seed"]["42"]["alpha0"]["ci_hi"],
                "p_pooled": S["exp11_per_seed"]["42"]["alpha0"]["p_pooled"]}
out["alpha"] = alpha
# Table S4: encoder CIs
enc = {}
for name, pat in (("biolongformer", "exp9_decomp_biolongformer_*.json"), ("biocmodernbert", "exp9_decomp_biocmodernbert_*.json")):
    e9 = load(R21 / pat); ks = sorted(e9); gg = np.array([k[0] for k in ks])
    C = np.array([e9[k]["C_prauc"] for k in ks]); P0 = np.array([e9[k]["P0_prauc"] for k in ks]); P1 = np.array([e9[k]["P100_prauc"] for k in ks])
    enc[name] = {"TE": cluster(P0 - C, gg), "IE": cluster(P1 - P0, gg), "Total": cluster(P1 - C, gg), "n": len(ks)}
out["encoders"] = enc
# text: SVM-teacher PFD gain with CI, both metrics
sp = [json.load(open(f)) for f in sorted(glob.glob(str(R25 / "results_svm_pfd" / "*.json")))]
gs = np.array([r["dataset"] for r in sp])
out["svm_pfd"] = {m: cluster(np.array([r[f"svm_ft_pfd_{m}"] - r[f"C_{m}"] for r in sp]), gs) for m in ("prauc", "wss95")}
out["svm_abs_ctrl"] = {m: cluster(np.array([r[f"svm_abs_ctrl_{m}"] - r[f"C_{m}"] for r in sp]), gs) for m in ("prauc", "wss95")}
json.dump(out, open(HERE / "gaps.json", "w"), indent=1)
for a in ("0.0", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0"):
    v = alpha[a]; print(f"alpha {a:>4}: {v['mean']:+.3f} sd {v['sd']:.3f} CI [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] P={v['p_pooled']:.3f}")
for n, v in enc.items():
    print(n, {k: (round(v[k]['mean'], 3), round(v[k]['ci_lo'], 3), round(v[k]['ci_hi'], 3)) for k in ('TE', 'IE', 'Total')})
print("svm_pfd", {m: (round(v['mean'], 4), round(v['ci_lo'], 3), round(v['ci_hi'], 3), round(v['p_pooled'], 3)) for m, v in out["svm_pfd"].items()})
print("svm_abs_ctrl", {m: (round(v['mean'], 4), round(v['ci_lo'], 3), round(v['ci_hi'], 3), round(v['p_pooled'], 3)) for m, v in out["svm_abs_ctrl"].items()})
print("recall sd", out["recall_sd"]["20"])
print("wrote", HERE / "gaps.json")
