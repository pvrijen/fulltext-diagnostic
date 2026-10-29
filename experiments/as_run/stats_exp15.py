"""Statistics for exp15 (inverse PFD route), Paper B v2.10.

Same conventions as stats_exp14.py: paired fold-level differences over the 30
dataset-folds, two-stage cluster bootstrap (datasets, then folds; 20,000
replicates; seed 42) for 95% CIs and a bootstrap P, pooled two-sided Wilcoxon
signed-rank, SD with ddof = 1, per-dataset means.

Effects on full text (student minus the full-text model it started from):
  INV_CF   : cross-fitted abstract soft labels, alpha 0.1   (the inverse route)
  INV_IS   : in-sample abstract soft labels, alpha 0.1
  INV_A0   : hard labels only (continued fine-tuning control)
  soft_minus_hard : INV_CF - INV_A0  (what the abstract soft labels add)
  IS_minus_hard   : INV_IS - INV_A0
Replicated decomposition (this run): TE, IE, Total.
Students on abstracts: INV_*_abs minus P@0% (does the abstract teacher pull
the full-text model back toward abstract competence?).
Writes exp15_stats.json next to this script and prints a summary.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
rng = np.random.default_rng(42)

rows = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "exp15_*_f*.json")))]
rows.sort(key=lambda r: (DS.index(r["dataset"]), r["fold"]))
assert len(rows) == 30, len(rows)
g = np.array([r["dataset"] for r in rows])


def boot(d, grp, B=20000):
    groups = sorted(set(grp)); byg = {x: d[grp == x] for x in groups}; out = np.empty(B)
    for b in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        out[b] = np.concatenate([rng.choice(byg[x], size=len(byg[x]), replace=True) for x in pick]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def summarize(d, grp):
    lo, hi, pb = boot(d, grp)
    try:
        pw = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        pw = float("nan")
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci": [lo, hi], "p_boot": pb, "p_wilcoxon": pw,
            "n_pos_folds": int((d > 0).sum()), "n_folds": int(len(d)),
            "n_datasets_pos": int(sum(d[grp == x].mean() > 0 for x in DS)), "n_datasets": len(DS),
            "per_dataset": {x: float(d[grp == x].mean()) for x in DS}}


def col(k, metric):
    return np.array([r[f"{k}_{metric}"] for r in rows], dtype=float)


out = {"n_folds": 30, "datasets": DS, "seed": 42}
for metric in ("prauc", "wss95"):
    C, P0, P1 = (col(k, metric) for k in ("C", "P0", "P100"))
    CF, IS, A0 = (col(k, metric) for k in ("INV_CF", "INV_IS", "INV_A0"))
    CFa, ISa, A0a = (col(k, metric) for k in ("INV_CF_abs", "INV_IS_abs", "INV_A0_abs"))
    m = {"cells": {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                       "per_dataset": {x: float(v[g == x].mean()) for x in DS}}
                   for k, v in (("C", C), ("P0", P0), ("P100", P1), ("INV_CF", CF), ("INV_IS", IS), ("INV_A0", A0),
                                ("INV_CF_abs", CFa), ("INV_IS_abs", ISa), ("INV_A0_abs", A0a))}}
    eff = {"TE": P0 - C, "IE": P1 - P0, "Total": P1 - C,
           "INV_CF": CF - P1, "INV_IS": IS - P1, "INV_A0": A0 - P1,
           "soft_minus_hard": CF - A0, "IS_minus_hard": IS - A0, "CF_minus_IS": CF - IS,
           "INV_CF_abs_vs_P0": CFa - P0, "INV_IS_abs_vs_P0": ISa - P0, "INV_A0_abs_vs_P0": A0a - P0,
           "INV_CF_abs_vs_C": CFa - C}
    for k, v in eff.items():
        m[k] = summarize(v, g)
    out[metric] = m

with open(HERE / "exp15_stats.json", "w") as f:
    json.dump(out, f, indent=2)

for metric, name in (("prauc", "PR-AUC"), ("wss95", "WSS@95")):
    m = out[metric]
    print(f"\n=== {name} (n = 30 dataset-folds) ===")
    print("cells: " + "  ".join(f"{k}={v['mean']:.3f}" for k, v in m["cells"].items()))
    for k in ("TE", "IE", "Total", "INV_CF", "INV_IS", "INV_A0", "soft_minus_hard", "IS_minus_hard", "CF_minus_IS",
              "INV_CF_abs_vs_P0", "INV_A0_abs_vs_P0", "INV_CF_abs_vs_C"):
        s = m[k]
        print(f"  {k:17s} {s['mean']:+.4f} (SD {s['sd']:.3f}; 95% CI {s['ci'][0]:+.3f} to {s['ci'][1]:+.3f}; "
              f"P_boot {s['p_boot']:.3f}, P_wilcoxon {s['p_wilcoxon']:.3f}; folds>0 {s['n_pos_folds']}/30; "
              f"datasets>0 {s['n_datasets_pos']}/6)")
