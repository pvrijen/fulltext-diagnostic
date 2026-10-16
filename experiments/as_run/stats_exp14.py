"""exp14: BioClinical-ModernBERT second encoder family, two-by-two decomposition + in-sample and cross-fitted PFD.

Reads papers/output_exp14_biocmb_pfd/<dataset>_f<k>.json (30 folds) and writes exp14_stats.json.
Same statistics as the BioMedBERT blocks (Tables 2 and 3): fold means, SD (ddof=1), two-stage cluster
bootstrap CI (datasets then folds, B=20000, seed 42), two-sided bootstrap P, Wilcoxon over folds, number
of datasets with a positive mean. Both metrics (PR-AUC, WSS@95). Run with --final to assert 30 folds.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = Path("/home/paul/fulltext_biocuration/papers/output_exp14_biocmb_pfd")
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
rng = np.random.default_rng(42)

rows = [json.load(open(f)) for f in sorted(glob.glob(str(OUT / "*_f*.json")))]
rows = [r for r in rows if "IS_prauc" in r]
rows.sort(key=lambda r: (DS.index(r["dataset"]), r["fold"]))
if "--final" in sys.argv:
    assert len(rows) == 30, len(rows)
g = np.array([r["dataset"] for r in rows])
present = [d for d in DS if d in set(g)]


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
    here = [x for x in DS if x in set(grp)]
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci": [lo, hi], "p_boot": pb, "p_wilcoxon": pw,
            "n_pos_folds": int((d > 0).sum()), "n_folds": int(len(d)),
            "n_datasets_pos": int(sum(d[grp == x].mean() > 0 for x in here)), "n_datasets": len(here),
            "per_dataset": {x: float(d[grp == x].mean()) for x in here}}


def col(k, metric):
    v = [r.get(f"{k}_{metric}") for r in rows]
    return np.array([np.nan if x is None else x for x in v], dtype=float)


out = {"n_folds": len(rows), "datasets": present, "encoder": rows[0]["encoder"] if rows else None}
for metric in ("prauc", "wss95"):
    C, A, P0, P1, IS, H, T, CF = (col(k, metric) for k in ("C", "Aft", "P0", "P100", "IS", "HARD", "P0trunc", "CF"))
    m = {"cells": {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)), "n": int((~np.isnan(v)).sum()),
                       "per_dataset": {x: float(np.nanmean(v[g == x])) for x in present if not np.all(np.isnan(v[g == x]))}}
                   for k, v in (("C", C), ("Aft", A), ("P0", P0), ("P100", P1), ("IS", IS), ("HARD", H), ("P0trunc", T), ("CF", CF))}}
    eff = {"TE": P0 - C, "IE": P1 - P0, "Total": P1 - C, "IEp": A - C, "TEp": P1 - A, "Int": (P1 - A) - (P0 - C),
           "IS": IS - C, "HARD": H - C, "soft_minus_hard": IS - H, "margin": P1 - C,
           "CF": CF - C, "CF_minus_hard": CF - H, "CF_minus_IS": CF - IS, "CF_vs_P100": CF - P1}
    for k, v in eff.items():
        ok = ~np.isnan(v)
        m[k] = summarize(v[ok], g[ok]) if ok.sum() >= 3 else None
    # length-matched check on the folds that have it: format component (trunc - abstract), length component (ft - trunc)
    ok = ~np.isnan(T)
    if ok.sum() >= 3:
        m["length_check"] = {"n_folds": int(ok.sum()),
                             "format_component_trunc_minus_abs": summarize((T - P0)[ok], g[ok]),
                             "length_component_ft_minus_trunc": summarize((P1 - T)[ok], g[ok]),
                             "TE_trunc_minus_C": summarize((T - C)[ok], g[ok])}
    # what predicts the PFD gain: teacher margin vs IE (dataset level n=6 and fold level)
    gain = IS - C
    cfgain = CF - C
    dm = np.array([gain[g == x].mean() for x in present]); mg = np.array([(P1 - C)[g == x].mean() for x in present])
    ie = np.array([(P1 - P0)[g == x].mean() for x in present])
    if len(present) >= 4:
        s1, s2 = stats.spearmanr(mg, dm), stats.spearmanr(ie, dm)
        f1, f2 = stats.spearmanr(P1 - C, gain), stats.spearmanr(P1 - P0, gain)
        m["predictors"] = {"spearman_margin_gain": [float(s1[0]), float(s1[1])], "spearman_IE_gain": [float(s2[0]), float(s2[1])],
                           "fold_spearman_margin_gain": [float(f1[0]), float(f1[1])], "fold_spearman_IE_gain": [float(f2[0]), float(f2[1])]}
        if (~np.isnan(cfgain)).all():
            dc = np.array([cfgain[g == x].mean() for x in present])
            c1, c2 = stats.spearmanr(mg, dc), stats.spearmanr(ie, dc)
            g1, g2 = stats.spearmanr(P1 - C, cfgain), stats.spearmanr(P1 - P0, cfgain)
            m["predictors_cf"] = {"spearman_margin_gain": [float(c1[0]), float(c1[1])], "spearman_IE_gain": [float(c2[0]), float(c2[1])],
                                  "fold_spearman_margin_gain": [float(g1[0]), float(g1[1])], "fold_spearman_IE_gain": [float(g2[0]), float(g2[1])]}
    out[metric] = m

json.dump(out, open(HERE / "exp14_stats.json", "w"), indent=1)

for metric in ("prauc", "wss95"):
    m = out[metric]
    print(f"\n== {metric}  ({out['n_folds']} folds, {len(present)} datasets) ==")
    print("cells:", "  ".join(f"{k} {v['mean']:.3f}" for k, v in m["cells"].items()))
    for k in ("TE", "IE", "Total", "IEp", "TEp", "Int", "IS", "HARD", "soft_minus_hard", "margin", "CF", "CF_minus_hard", "CF_minus_IS", "CF_vs_P100"):
        s = m[k]
        if s is None:
            continue
        print(f"{k:16s} {s['mean']:+.3f} ±{s['sd']:.3f}  CI [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  P_boot {s['p_boot']:.3f}"
              f"  P_wilc {s['p_wilcoxon']:.3f}  folds+ {s['n_pos_folds']}/{s['n_folds']}  ds+ {s['n_datasets_pos']}/{s['n_datasets']}")
    if "length_check" in m:
        lc = m["length_check"]
        print(f"length check (n={lc['n_folds']}): format {lc['format_component_trunc_minus_abs']['mean']:+.3f}"
              f" CI [{lc['format_component_trunc_minus_abs']['ci'][0]:+.3f}, {lc['format_component_trunc_minus_abs']['ci'][1]:+.3f}]"
              f" | length {lc['length_component_ft_minus_trunc']['mean']:+.3f}"
              f" CI [{lc['length_component_ft_minus_trunc']['ci'][0]:+.3f}, {lc['length_component_ft_minus_trunc']['ci'][1]:+.3f}]")
    if "predictors" in m:
        p = m["predictors"]
        print(f"predictors: dataset-level rho margin->gain {p['spearman_margin_gain'][0]:+.2f} (P {p['spearman_margin_gain'][1]:.2f}),"
              f" IE->gain {p['spearman_IE_gain'][0]:+.2f} (P {p['spearman_IE_gain'][1]:.2f});"
              f" fold-level margin {p['fold_spearman_margin_gain'][0]:+.2f}, IE {p['fold_spearman_IE_gain'][0]:+.2f}")
    print("per-dataset IS gain:", "  ".join(f"{x} {v:+.3f}" for x, v in m["IS"]["per_dataset"].items()))
    if m.get("CF"):
        print("per-dataset CF gain:", "  ".join(f"{x} {v:+.3f}" for x, v in m["CF"]["per_dataset"].items()))
    if "predictors_cf" in m:
        p = m["predictors_cf"]
        print(f"CF predictors: dataset-level rho margin->gain {p['spearman_margin_gain'][0]:+.2f} (P {p['spearman_margin_gain'][1]:.2f}),"
              f" IE->gain {p['spearman_IE_gain'][0]:+.2f} (P {p['spearman_IE_gain'][1]:.2f}); fold-level margin {p['fold_spearman_margin_gain'][0]:+.2f}, IE {p['fold_spearman_IE_gain'][0]:+.2f}")
