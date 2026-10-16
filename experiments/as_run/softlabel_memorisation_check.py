"""Memorisation check of teacher soft labels, both encoder families, all 30 folds.

For each outer fold the in-sample teacher (trained on the full training split) and the five cross-fitted inner
teachers each produced a soft label for every training document. We compare both label sets with the ground truth:
mean |p - y|, the share of positives scored below 0.5, and the share of labels that are soft (0.05 < p < 0.95).
Primary pipeline: papers/output_warmstart_pfd_crossfit/cached_logits/<ds>/{IS,CF5}_<ds>_f<k>.pt
Second family:    papers/output_exp14_biocmb_pfd/logits/{IS,CF5}_<ds>_f<k>.pt
Fold labels come from the experiment script's load_fold (identical splits in both pipelines, seed 42).
Writes softlabel_memorisation.json next to this script and prints a summary. CPU only; run with CUDA_VISIBLE_DEVICES="".
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP14 = Path("/home/paul/fulltext_biocuration/.claude/worktrees/paper-b-v26/papers/paper_b_biocuration_guide/versions/2.8_jamia/scripts/exp14_biocmodernbert_pfd.py")
FAMILIES = {
    "primary": Path("/home/paul/fulltext_biocuration/papers/output_warmstart_pfd_crossfit/cached_logits"),
    "second": Path("/home/paul/fulltext_biocuration/papers/output_exp14_biocmb_pfd/logits"),
}
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]

spec = importlib.util.spec_from_file_location("exp14", EXP14)
m = importlib.util.module_from_spec(spec)
sys.argv = [str(EXP14)]
spec.loader.exec_module(m)


def path(fam, tag, ds, fold):
    d = FAMILIES[fam]
    return (d / ds / f"{tag}_{ds}_f{fold}.pt") if fam == "primary" else (d / f"{tag}_{ds}_f{fold}.pt")


def stats(z, y):
    p = torch.sigmoid(torch.as_tensor(z).float().flatten()).numpy()
    return {"mae": float(np.abs(p - y).mean()), "pos_below_half": float((p[y == 1] < .5).mean()),
            "soft": float(((p > .05) & (p < .95)).mean())}


out = {}
for ds in DS:
    for fold in range(5):
        y = np.asarray(m.load_fold(ds, fold)["y_tr"], dtype=float)
        for fam in FAMILIES:
            for tag in ("IS", "CF5"):
                z = torch.load(str(path(fam, tag, ds, fold)), map_location="cpu", weights_only=False)["tr"]
                assert len(z) == len(y), (fam, tag, ds, fold)
                out.setdefault(fam, {}).setdefault(tag, []).append({"dataset": ds, "fold": fold, **stats(z, y)})

summary = {}
for fam, tags in out.items():
    summary[fam] = {}
    for tag, rows in tags.items():
        a = {k: float(np.mean([r[k] for r in rows])) for k in ("mae", "pos_below_half", "soft")}
        per_ds = {d: {k: float(np.mean([r[k] for r in rows if r["dataset"] == d])) for k in ("mae", "pos_below_half", "soft")} for d in DS}
        summary[fam][tag] = {"mean": a, "per_dataset": per_ds, "n_folds": len(rows)}
        print(f"{fam:8s} {tag:4s} folds {len(rows):2d}  mean |p-y| {a['mae']:.3f}  positives below 0.5 {a['pos_below_half']:.2f}  soft labels {a['soft']:.3f}")
json.dump({"folds": out, "summary": summary}, open(HERE / "softlabel_memorisation.json", "w"), indent=1)
print("written", HERE / "softlabel_memorisation.json")
