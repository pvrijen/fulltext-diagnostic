#!/usr/bin/env python3
"""Table 3 WSS@95 block: the exp11 control contrasts on WSS@95 with the same statistics as the PR-AUC block.

Per-seed cells: mean +-SD of the paired fold-level gain over L_abs->abs (n = 30). Mean (95% CI): seed-averaged
fold-level gain with a two-stage cluster bootstrap (datasets then folds, 20,000 replicates, seed 42); P = pooled
two-sided Wilcoxon over the 30 seed-averaged dataset-folds. Soft minus hard: paired over all 90 seed-folds, with
the two-stage cluster-bootstrap P (seed-folds within a dataset resampled together). Mirrors stats_v26.py.
Writes exp11_wss95.json.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
SEEDS = [42, 123, 456]
rng = np.random.default_rng(42)
rows = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / "results" / "exp11_s*_f*.json")))]
assert len(rows) == 90


def boot(d, g, B=20000):
    groups = sorted(set(g)); byg = {x: d[g == x] for x in groups}; out = np.empty(B)
    for b in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        out[b] = np.concatenate([rng.choice(byg[x], size=len(byg[x]), replace=True) for x in pick]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def summarize(d, g):
    lo, hi, pb = boot(d, g)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci": [lo, hi], "p_boot": pb,
            "p_wilcoxon": float(stats.wilcoxon(d).pvalue), "n_datasets_pos": int(sum(d[g == x].mean() > 0 for x in DS))}


out = {}
for metric in ("wss95", "prauc"):
    m = {"per_seed": {}}
    per = {}
    for s in SEEDS:
        rs = sorted([r for r in rows if r["seed"] == s], key=lambda r: (DS.index(r["dataset"]), r["fold"]))
        g = np.array([r["dataset"] for r in rs])
        soft = np.array([r[f"a0.1_{metric}"] - r[f"C_{metric}"] for r in rs])
        hard = np.array([r[f"a0_{metric}"] - r[f"C_{metric}"] for r in rs])
        per[s] = (soft, hard, g)
        m["per_seed"][s] = {"soft_mean": float(soft.mean()), "soft_sd": float(soft.std(ddof=1)),
                            "hard_mean": float(hard.mean()), "hard_sd": float(hard.std(ddof=1))}
    g = per[42][2]
    S = np.mean([per[s][0] for s in SEEDS], axis=0); H = np.mean([per[s][1] for s in SEEDS], axis=0)
    m["soft_seedavg"] = summarize(S, g); m["hard_seedavg"] = summarize(H, g)
    rs = sorted(rows, key=lambda r: (DS.index(r["dataset"]), r["seed"], r["fold"]))
    g90 = np.array([r["dataset"] for r in rs])
    d90 = np.array([r[f"a0.1_{metric}"] - r[f"a0_{metric}"] for r in rs])
    m["soft_minus_hard_90"] = summarize(d90, g90)
    out[metric] = m
    print(metric, "soft", round(m["soft_seedavg"]["mean"], 3), m["soft_seedavg"]["ci"], "| hard", round(m["hard_seedavg"]["mean"], 3),
          m["hard_seedavg"]["ci"], "| soft-hard(90)", round(m["soft_minus_hard_90"]["mean"], 3), m["soft_minus_hard_90"]["ci"],
          "boot P", round(m["soft_minus_hard_90"]["p_boot"], 3), "datasets>0", m["soft_minus_hard_90"]["n_datasets_pos"])
json.dump(out, open(HERE / "exp11_wss95.json", "w"), indent=1)
print("wrote", HERE / "exp11_wss95.json")
