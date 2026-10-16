#!/usr/bin/env python3
"""Direct paired contrast of the cross-fitted PFD student against the full-text model (L_PFD - L_ft->ft),
on PR-AUC and WSS@95, from the exp8 fold files; plus per-dataset IE intervals for the CI-based decision rule.

Cluster bootstrap as in stats_reanalysis.py (datasets then folds, 20,000 replicates, seed 42); pooled
two-sided Wilcoxon (n = 30). Writes pfd_vs_ft.json next to this script.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP8 = Path(os.environ.get("EXP8_DIR", HERE.parent / "2.1_jamia_experiments" / "results"))
rows = [json.load(open(f)) for f in sorted(glob.glob(str(EXP8 / "exp8_unified_*.json")))]
assert len(rows) == 30
g = np.array([r["dataset"] for r in rows])
rng = np.random.default_rng(42)


def cluster_ci(d, B=20000):
    groups = sorted(set(g)); byg = {x: d[g == x] for x in groups}; out = np.empty(B)
    for b in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        out[b] = np.concatenate([rng.choice(byg[x], size=len(byg[x]), replace=True) for x in pick]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5]); p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


res = {}
for metric in ("prauc", "wss95"):
    d = np.array([r[f"CF_{metric}"] - r[f"P100_{metric}"] for r in rows])
    lo, hi, pb = cluster_ci(d)
    res[f"PFD_minus_ft_{metric}"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci_lo": lo, "ci_hi": hi, "p_boot": pb,
                                    "p_pooled": float(stats.wilcoxon(d).pvalue), "ds_positive": int(sum(d[g == x].mean() > 0 for x in sorted(set(g))))}
    print(f"PFD - L_ft ({metric}): {d.mean():+.4f} sd {d.std(ddof=1):.3f} CI [{lo:+.3f}, {hi:+.3f}] boot P={pb:.3f} pooled P={res[f'PFD_minus_ft_{metric}']['p_pooled']:.3f}")
# per-dataset IE (PR-AUC) with a fold-level 95% t-interval and a percentile bootstrap (5 folds), for the CI-based rule
ie = {}
for x in sorted(set(g)):
    v = np.array([r["P100_prauc"] - r["P0_prauc"] for r in rows if r["dataset"] == x])
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    t = stats.t.ppf(0.975, len(v) - 1)
    ie[x] = {"mean": float(m), "t_lo": float(m - t * se), "t_hi": float(m + t * se), "folds_positive": int((v > 0).sum())}
    print(f"IE {x:9s} {m:+.3f} t95 [{m - t * se:+.3f}, {m + t * se:+.3f}] folds>0 {int((v > 0).sum())}/5")
res["IE_per_dataset"] = ie
json.dump(res, open(HERE / "pfd_vs_ft.json", "w"), indent=1)
print("wrote", HERE / "pfd_vs_ft.json")
