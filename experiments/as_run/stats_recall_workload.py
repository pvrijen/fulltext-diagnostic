#!/usr/bin/env python3
"""Recall at fixed screening workload, paired by fold, from the stored recall-at-effort scores.

Reads recall_{ds}_f{fold}.json (30 folds; y_test, scores_c, scores_p100, scores_pfd; the same files
behind the recall-at-effort figure) from RECALL_DIR (default: ../2.5_jamia/results_recall_curves) and
reports, for each workload budget (fraction of candidates screened in score order), the mean recall of
L_abs->abs, L_ft->ft and L_PFD, and the paired fold-level differences PFD - baseline and ft - baseline with
a two-stage cluster-bootstrap 95% CI (datasets, then folds; 20,000 replicates, seed 0) and the pooled
two-sided Wilcoxon P (n = 30), as in stats_v26.py.

Writes recall_workload.json next to this script.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("RECALL_DIR", HERE.parent / "2.5_jamia" / "results_recall_curves"))
BUDGETS = [0.10, 0.20, 0.30, 0.50]
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
rng = np.random.default_rng(0)


def recall_at(scores, labels, frac):
    scores, labels = np.asarray(scores), np.asarray(labels)
    n_pos = labels.sum()
    if n_pos == 0:
        return np.nan
    k = int(np.ceil(frac * len(labels)))
    order = np.argsort(-scores, kind="stable")
    return labels[order][:k].sum() / n_pos


def cluster_ci(d, g, B=20000):
    groups = sorted(set(g))
    byg = {x: d[g == x] for x in groups}
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        vals = np.concatenate([rng.choice(byg[x], size=len(byg[x]), replace=True) for x in pick])
        out[b] = vals.mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


rows = [json.load(open(f)) for f in sorted(glob.glob(str(SRC / "recall_*_f*.json")))]
assert len(rows) == 30, (len(rows), SRC)
g = np.array([r["dataset"] for r in rows])
res = {"n_folds": len(rows), "source": str(SRC), "budgets": {}}
for frac in BUDGETS:
    c = np.array([recall_at(r["scores_c"], r["y_test"], frac) for r in rows])
    p = np.array([recall_at(r["scores_p100"], r["y_test"], frac) for r in rows])
    s = np.array([recall_at(r["scores_pfd"], r["y_test"], frac) for r in rows])
    key = f"{int(round(frac * 100))}"
    ent = {"recall_C": float(c.mean()), "recall_ft": float(p.mean()), "recall_PFD": float(s.mean())}
    for name, d in (("PFD_minus_C", s - c), ("ft_minus_C", p - c)):
        lo, hi = cluster_ci(d, g)
        ent[name] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci_lo": lo, "ci_hi": hi,
                     "p_pooled": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                     "ds_positive": int(sum(d[g == x].mean() > 0 for x in DS))}
    res["budgets"][key] = ent
    print(f"workload {key:>3}%: recall C {ent['recall_C']:.3f}  ft {ent['recall_ft']:.3f}  PFD {ent['recall_PFD']:.3f} | "
          f"PFD-C {ent['PFD_minus_C']['mean']:+.3f} CI [{ent['PFD_minus_C']['ci_lo']:+.3f}, {ent['PFD_minus_C']['ci_hi']:+.3f}] "
          f"P={ent['PFD_minus_C']['p_pooled']:.3f} ds {ent['PFD_minus_C']['ds_positive']}/6 | "
          f"ft-C {ent['ft_minus_C']['mean']:+.3f} CI [{ent['ft_minus_C']['ci_lo']:+.3f}, {ent['ft_minus_C']['ci_hi']:+.3f}]")
json.dump(res, open(HERE / "recall_workload.json", "w"), indent=1)
print("wrote", HERE / "recall_workload.json")
