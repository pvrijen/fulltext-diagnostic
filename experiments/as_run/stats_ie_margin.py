#!/usr/bin/env python3
"""Does the cross-fitted PFD gain track IE or the full-text model's margin over the baseline?

Dataset-level (n = 6) and fold-level (n = 30) Spearman correlations on PR-AUC and WSS@95, from the exp8 fold files.
Writes ie_margin.json. Used by the decision framework (Step 2) and the Figure 4 caption.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP8 = Path(os.environ.get("EXP8_DIR", HERE.parent / "2.1_jamia_experiments" / "results"))
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
rows = [json.load(open(f)) for f in sorted(glob.glob(str(EXP8 / "exp8_unified_*.json")))]
assert len(rows) == 30
out = {}
for m in ("prauc", "wss95"):
    ie, tot, gain = [], [], []
    for d in DS:
        r = [x for x in rows if x["dataset"] == d]
        ie.append(np.mean([x[f"P100_{m}"] - x[f"P0_{m}"] for x in r]))
        tot.append(np.mean([x[f"P100_{m}"] - x[f"C_{m}"] for x in r]))
        gain.append(np.mean([x[f"CF_{m}"] - x[f"C_{m}"] for x in r]))
    fie = [x[f"P100_{m}"] - x[f"P0_{m}"] for x in rows]
    ftot = [x[f"P100_{m}"] - x[f"C_{m}"] for x in rows]
    fgain = [x[f"CF_{m}"] - x[f"C_{m}"] for x in rows]
    s_ie, s_tot = stats.spearmanr(ie, gain), stats.spearmanr(tot, gain)
    f_ie, f_tot = stats.spearmanr(fie, fgain), stats.spearmanr(ftot, fgain)
    out[m] = {"datasets": DS, "IE": ie, "total": tot, "pfd_gain": gain,
              "spearman_IE_gain": [float(s_ie[0]), float(s_ie[1])], "spearman_total_gain": [float(s_tot[0]), float(s_tot[1])],
              "fold_spearman_IE_gain": [float(f_ie[0]), float(f_ie[1])], "fold_spearman_total_gain": [float(f_tot[0]), float(f_tot[1])],
              "gain_over_total": [float(g / t) if abs(t) > 1e-9 else None for g, t in zip(gain, tot)]}
    print(f"{m}: dataset-level Spearman(IE, gain) = {s_ie[0]:.2f} (P = {s_ie[1]:.2f}); Spearman(total, gain) = {s_tot[0]:.2f} (P = {s_tot[1]:.3f}); "
          f"fold-level IE {f_ie[0]:.2f} (P = {f_ie[1]:.2f}), total {f_tot[0]:.2f} (P = {f_tot[1]:.3f})")
json.dump(out, open(HERE / "ie_margin.json", "w"), indent=1)
print("wrote", HERE / "ie_margin.json")
