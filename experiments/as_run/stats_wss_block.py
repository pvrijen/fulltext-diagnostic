#!/usr/bin/env python3
"""WSS@95 block for Table 2: the six cells per dataset and the five contrasts, with cluster-bootstrap CIs.

Cells (same outer folds and test documents as the PR-AUC block):
  L_abs->abs, L_ft->abs, L_ft->ft, L_PFD (cross-fitted)  from exp8_unified_*.json (C_, P0_, P100_, CF_wss95)
  A_abs->abs, A_abs->ft                                  from exp12_*.json (Aabs_abs_wss95, Aabs_ft_wss95)
Contrasts: TE, IE, IE', TE', dPFD, total; two-stage cluster bootstrap (datasets, then folds; 20,000; seed 0)
and pooled two-sided Wilcoxon P (n = 30). Writes wss_block.json and prints the LaTeX rows.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
EXP8 = Path(os.environ.get("EXP8_DIR", HERE.parent / "2.1_jamia_experiments" / "results"))
DS = [("pfas", "PFAS"), ("bpa", "BPA"), ("cd012661", "CD012661"), ("cd012069", "CD012069"), ("cd008122", "CD008122"), ("cd011431", "CD011431")]
rng = np.random.default_rng(42)  # same bootstrap seed as stats_reanalysis.py so Table 2 and Table S10 agree

e8 = {(r["dataset"], r["fold"]): r for r in (json.load(open(f)) for f in sorted(glob.glob(str(EXP8 / "exp8_unified_*.json"))))}
e12 = {(r["dataset"], r["fold"]): r for r in (json.load(open(f)) for f in sorted(glob.glob(str(HERE / "results" / "exp12_*.json"))))}
keys = [k for k in e8 if k in e12]
assert len(keys) == 30, len(keys)
g = np.array([k[0] for k in keys])
col = lambda src, name: np.array([src[k][name] for k in keys])
L_abs, L_fta, L_ft, L_pfd = col(e8, "C_wss95"), col(e8, "P0_wss95"), col(e8, "P100_wss95"), col(e8, "CF_wss95")
A_abs, A_ft = col(e12, "Aabs_abs_wss95"), col(e12, "Aabs_ft_wss95")


def cluster_ci(d, B=20000):
    groups = sorted(set(g)); byg = {x: d[g == x] for x in groups}; out = np.empty(B)
    for b in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        out[b] = np.concatenate([rng.choice(byg[x], size=len(byg[x]), replace=True) for x in pick]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def summ(d):
    lo, hi = cluster_ci(d)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci_lo": lo, "ci_hi": hi, "p_pooled": float(wilcoxon(d).pvalue)}


contr = {"TE": L_fta - L_abs, "IE": L_ft - L_fta, "IEp": A_ft - L_abs, "TEp": L_ft - A_ft, "dPFD": L_pfd - L_abs, "total": L_ft - L_abs,
         "interaction": (L_ft - A_ft) - (L_fta - L_abs)}
res = {"n": 30, "contrasts": {k: summ(v) for k, v in contr.items()}, "rows": []}
sg = lambda v: (f"$-${abs(v):.3f}" if v < 0 else f"$+${v:.3f}") if round(v, 3) != 0 else "0.000"
sd = lambda v: f"\\sd{{{f'{v:.3f}'[1:]}}}"
for d, lab in DS:
    m = g == d
    vals = [L_abs[m], A_abs[m], A_ft[m], L_fta[m], L_ft[m], L_pfd[m]]
    means = [v.mean() for v in vals]
    best = int(np.argmax([means[0], means[3], means[4], means[5]]))  # bold among L_abs, L_fta, L_ft, L_pfd
    cells = [f"{means[0]:.3f}{sd(vals[0].std(ddof=1))}", f"{means[1]:.3f}", f"{means[2]:.3f}",
             f"{means[3]:.3f}{sd(vals[3].std(ddof=1))}", f"{means[4]:.3f}{sd(vals[4].std(ddof=1))}", f"{means[5]:.3f}{sd(vals[5].std(ddof=1))}"]
    for i, j in enumerate((0, 3, 4, 5)):
        if i == best:
            cells[j] = "\\textbf{" + cells[j] + "}"
    te, ie, iep, tep, dp = (means[3] - means[0], means[4] - means[3], means[2] - means[0], means[4] - means[2], means[5] - means[0])
    row = f"\t\t{lab} & " + " & ".join(cells) + f" & {sg(te)} & {sg(ie)} & {sg(iep)} & {sg(tep)} & {sg(dp)} \\\\"
    res["rows"].append(row)
mm = [L_abs.mean(), A_abs.mean(), A_ft.mean(), L_fta.mean(), L_ft.mean(), L_pfd.mean()]
c = res["contrasts"]
bestm = int(np.argmax([mm[0], mm[3], mm[4], mm[5]]))
mcells = [f"{mm[0]:.3f}", f"{mm[1]:.3f}", f"{mm[2]:.3f}", f"{mm[3]:.3f}", f"{mm[4]:.3f}", f"{mm[5]:.3f}"]
for i, j in enumerate((0, 3, 4, 5)):
    if i == bestm:
        mcells[j] = "\\textbf{" + mcells[j] + "}"
res["mean_row"] = "\t\tMean & " + " & ".join(mcells) + f" & {sg(c['TE']['mean'])} & {sg(c['IE']['mean'])} & {sg(c['IEp']['mean'])} & {sg(c['TEp']['mean'])} & {sg(c['dPFD']['mean'])}{sd(c['dPFD']['sd'])} \\\\"
fp = lambda p: ("$<$.001" if p < 0.001 else (f"{p:.3f}"[1:] if p < 0.01 or 0.04 <= p < 0.06 else f"{p:.2f}"[1:]))
res["p_row"] = f"\t\tWilcoxon $P$, $n = 30$ & & & & & & & {fp(c['TE']['p_pooled'])} & {fp(c['IE']['p_pooled'])} & {fp(c['IEp']['p_pooled'])} & {fp(c['TEp']['p_pooled'])} & {fp(c['dPFD']['p_pooled'])} \\\\"
# exact two-sided Wilcoxon on the six dataset means (same as the PR-AUC block's n = 6 row)
def ds_p(d):
    dm = np.array([d[g == x].mean() for x, _ in DS])
    return float(wilcoxon(dm, alternative="two-sided", method="exact").pvalue)
res["p6"] = {k: ds_p(v) for k, v in contr.items()}
res["p6_row"] = f"\t\tWilcoxon $P$, $n = 6$ & & & & & & & {fp(res['p6']['TE'])} & {fp(res['p6']['IE'])} & {fp(res['p6']['IEp'])} & {fp(res['p6']['TEp'])} & {fp(res['p6']['dPFD'])} \\\\"
json.dump(res, open(HERE / "wss_block.json", "w"), indent=1)
print("\n".join(res["rows"])); print(res["mean_row"]); print(res["p_row"])
for k, v in c.items():
    print(f"{k:12s} {v['mean']:+.3f} sd {v['sd']:.3f} CI [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] P={v['p_pooled']:.3f}")
