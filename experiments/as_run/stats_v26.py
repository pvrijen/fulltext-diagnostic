#!/usr/bin/env python3
"""Paper B v2.6 statistics: alpha=0 control, student-side seed replicate, 2x2 factorial.

Inputs
  exp8  (versions/2.1_jamia_experiments/results/exp8_unified_*.json): seed-42 reference
        cells C=L_abs->abs (truncated), P0=L_ft->abs, P100=L_ft->ft, CF=CF-PFD.
  exp11 (results/exp11_s{seed}_{ds}_f{fold}.json): C retrained per seed; students
        alpha=0 (hard labels only) and alpha=0.1 (CF-PFD) from the same encoder state.
  exp12 (results/exp12_{ds}_f{fold}.json): pooled architecture trained on abstracts,
        evaluated on abstracts (A_abs->abs) and on full text (A_abs->ft).

Outputs (in --out dir): stats_v26.json and tab_v26_*.tex row files.
All CIs are two-stage cluster bootstrap (datasets, then folds; 20,000 replicates);
P values are two-sided pooled Wilcoxon over dataset-fold observations unless labelled.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def cboot(x, g, B=20000, seed=42):
    rng = np.random.default_rng(seed)
    labs = sorted(set(g))
    pools = {l: x[g == l] for l in labs}
    m = np.empty(B)
    for b in range(B):
        pk = rng.choice(labs, len(labs), replace=True)
        m[b] = np.concatenate([rng.choice(pools[l], len(pools[l]), replace=True) for l in pk]).mean()
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi), float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))


def wil(x):
    x = np.asarray(x, float)
    try:
        return float(stats.wilcoxon(x, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def ds_signs(x, g, sign=+1):
    return int(sum((sign * x[g == d]).mean() > 0 for d in sorted(set(g)))), len(set(g))


def summarize(name, x, g, sign=+1):
    lo, hi, pb = cboot(x, g)
    n_sign, n_ds = ds_signs(x, g, sign)
    r = dict(name=name, n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)),
             ci_lo=lo, ci_hi=hi, p_boot=pb, p_pooled=wil(x), ds_sign=f"{n_sign}/{n_ds}")
    print(f"{name:34s} n={len(x):3d} mean {x.mean():+.4f} sd {x.std(ddof=1):.3f}  CI [{lo:+.3f}, {hi:+.3f}]  pooled P={r['p_pooled']:.4f}  datasets {r['ds_sign']}")
    return r


def signed(v):
    return f"$-${abs(v):.3f}" if v < 0 else f"$+${v:.3f}"


def fmt_p(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "$<$.001"
    return f"{p:.3f}"[1:] if p < 0.01 else f"{p:.2f}"[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp8", default=str(HERE.parent / "2.1_jamia_experiments" / "results"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    out = {}

    e8 = load(f"{a.exp8}/exp8_unified_*.json")
    key8 = {(r["dataset"], r["fold"]): r for r in e8}

    # ---------------------------------------------------------------- exp11
    e11 = load(f"{a.results}/exp11_*.json")
    print(f"\n=== exp11: {len(e11)} folds, seeds {sorted(set(r['seed'] for r in e11))} ===")
    rows_tex = []
    per_seed = {}
    for seed in sorted(set(r["seed"] for r in e11)):
        rs = [r for r in e11 if r["seed"] == seed]
        g = np.array([r["dataset"] for r in rs])
        C = np.array([r["C_prauc"] for r in rs])
        a0 = np.array([r["a0_prauc"] for r in rs])
        a1 = np.array([r["a0.1_prauc"] for r in rs])
        per_seed[seed] = dict(
            C=float(C.mean()),
            alpha0=summarize(f"seed {seed}: alpha=0 gain", a0 - C, g),
            alpha01=summarize(f"seed {seed}: alpha=0.1 gain", a1 - C, g),
            soft_vs_hard=summarize(f"seed {seed}: alpha0.1 - alpha0", a1 - a0, g),
            complete=len(rs) == 30,
        )
        if seed == 42 and len(rs) == 30:
            ref = np.array([key8[(r["dataset"], r["fold"])]["CF_prauc"] - key8[(r["dataset"], r["fold"])]["C_prauc"] for r in rs])
            per_seed[seed]["rerun_vs_exp8"] = summarize("seed 42 rerun CF gain - exp8 CF gain", (a1 - C) - ref, g)
            per_seed[seed]["rerun_C_vs_exp8"] = summarize("seed 42 rerun C - exp8 C", C - np.array([key8[(r["dataset"], r["fold"])]["C_prauc"] for r in rs]), g)
    out["exp11_per_seed"] = per_seed

    complete_seeds = [s for s, v in per_seed.items() if v["complete"]]
    if len(complete_seeds) >= 2:
        # seed-averaged fold-level gains (each fold averaged over seeds)
        folds = sorted(set((r["dataset"], r["fold"]) for r in e11))
        def seedavg(key):
            vals, g = [], []
            for (d, f) in folds:
                rr = [r for r in e11 if r["dataset"] == d and r["fold"] == f and r["seed"] in complete_seeds]
                if len(rr) == len(complete_seeds):
                    vals.append(np.mean([r[key] - r["C_prauc"] for r in rr])); g.append(d)
            return np.array(vals), np.array(g)
        x, g = seedavg("a0.1_prauc")
        out["seedavg_cf_gain"] = summarize(f"seed-averaged CF-PFD gain ({len(complete_seeds)} seeds)", x, g)
        x0, g0 = seedavg("a0_prauc")
        out["seedavg_alpha0_gain"] = summarize(f"seed-averaged alpha=0 gain ({len(complete_seeds)} seeds)", x0, g0)
        # all seed-fold observations pooled, soft vs hard
        rs = [r for r in e11 if r["seed"] in complete_seeds]
        g = np.array([r["dataset"] for r in rs])
        out["all_seeds_soft_vs_hard"] = summarize(f"alpha0.1 - alpha0, all {len(rs)} seed-folds", np.array([r["a0.1_prauc"] - r["a0_prauc"] for r in rs]), g)
        # cluster-robust t on seed-averaged gains (df = 5)
        dm = np.array([x[g0 == d].mean() for d in sorted(set(g0))]) if False else None
        import statsmodels.api as sm
        codes = np.unique(seedavg("a0.1_prauc")[1], return_inverse=True)[1]
        m = sm.OLS(x, np.ones(len(x))).fit(cov_type="cluster", cov_kwds={"groups": codes, "use_correction": True, "df_correction": True})
        t = m.params[0] / m.bse[0]
        out["seedavg_cf_gain"]["p_cluster_t"] = float(2 * stats.t.sf(abs(t), df=5))
        print(f"seed-averaged CF gain cluster-robust t P = {out['seedavg_cf_gain']['p_cluster_t']:.3f}")

    # ---------------------------------------------------------------- exp12: 2x2 factorial
    e12 = load(f"{a.results}/exp12_*.json")
    print(f"\n=== exp12: {len(e12)} folds ===")
    if e12:
        g = np.array([r["dataset"] for r in e12])
        AA = np.array([r["Aabs_abs_prauc"] for r in e12])          # train abs, infer abs (pooled arch)
        AF = np.array([r["Aabs_ft_prauc"] for r in e12])           # train abs, infer ft
        FA = np.array([key8[(r["dataset"], r["fold"])]["P0_prauc"] for r in e12])    # train ft, infer abs
        FF = np.array([key8[(r["dataset"], r["fold"])]["P100_prauc"] for r in e12])  # train ft, infer ft
        CT = np.array([key8[(r["dataset"], r["fold"])]["C_prauc"] for r in e12])     # truncated baseline
        fac = {}
        fac["cells"] = {k: float(v.mean()) for k, v in dict(A_abs_abs=AA, A_abs_ft=AF, L_ft_abs=FA, L_ft_ft=FF, C_truncated=CT).items()}
        print("cell means:", {k: round(v, 4) for k, v in fac["cells"].items()})
        fac["TE_same_arch"] = summarize("TE, same architecture (ft->abs - abs->abs)", FA - AA, g, sign=-1)
        fac["TE_paper"] = summarize("TE, paper (ft->abs - C truncated)", FA - CT, g, sign=-1)
        fac["arch_cost_on_abs"] = summarize("pooled arch on abstracts - truncated C", AA - CT, g)
        fac["IE_ft_trained"] = summarize("IE | trained on ft (ft->ft - ft->abs)", FF - FA, g)
        fac["IE_abs_trained"] = summarize("IE | trained on abs (abs->ft - abs->abs)", AF - AA, g)
        fac["train_main"] = summarize("training main effect (ft - abs, avg over infer)", ((FA + FF) - (AA + AF)) / 2, g, sign=-1)
        fac["infer_main"] = summarize("inference main effect (ft - abs, avg over train)", ((AF + FF) - (AA + FA)) / 2, g)
        fac["interaction"] = summarize("interaction (IE|ft-trained - IE|abs-trained)", (FF - FA) - (AF - AA), g)
        fac["abs_model_on_ft_vs_abs"] = summarize("abstract model: ft input - abs input", AF - AA, g)
        # Knafou's second route through the square (slides "The Missing Corner", 5 Oct 2026):
        #   IE' = A->FT - C_truncated ; TE' = P@100 - A->FT ; interaction = IE - IE' = TE' - TE
        fac["IE_prime"] = summarize("IE' = A->FT - C (Knafou)", AF - CT, g)
        fac["TE_prime"] = summarize("TE' = P@100 - A->FT (Knafou)", FF - AF, g, sign=-1)
        fac["interaction_knafou"] = summarize("interaction = TE' - TE (Knafou route)", (FF - AF) - (FA - CT), g)
        # structure outcome: each model wins on its own input -> IE' < 0 AND TE' > 0
        # intersection-union test: each component one-sided at .05 (cluster bootstrap over datasets, then folds)
        def one_sided_p(x, direction):
            rng = np.random.default_rng(7)
            labs = sorted(set(g)); pools = {l: x[g == l] for l in labs}; m = np.empty(20000)
            for b in range(20000):
                pk = rng.choice(labs, len(labs), replace=True)
                m[b] = np.concatenate([rng.choice(pools[l], len(pools[l]), replace=True) for l in pk]).mean()
            return float((m >= 0).mean() if direction < 0 else (m <= 0).mean())
        p_ie_neg = one_sided_p(AF - CT, -1)   # share of resamples with IE' >= 0
        p_te_pos = one_sided_p(FF - AF, +1)   # share of resamples with TE' <= 0
        fac["structure_test"] = dict(p_IEprime_lt0=p_ie_neg, p_TEprime_gt0=p_te_pos,
                                     structure_shown=bool(p_ie_neg < 0.05 and p_te_pos < 0.05),
                                     ds_IEprime_neg=ds_signs(AF - CT, g, -1)[0], ds_TEprime_pos=ds_signs(FF - AF, g, +1)[0], n_datasets=len(set(g)))
        print(f"structure test: P(IE'>=0)={p_ie_neg:.3f}  P(TE'<=0)={p_te_pos:.3f}  -> structure shown: {fac['structure_test']['structure_shown']}  "
              f"(IE'<0 on {fac['structure_test']['ds_IEprime_neg']}/{len(set(g))} datasets, TE'>0 on {fac['structure_test']['ds_TEprime_pos']}/{len(set(g))})")
        out["factorial"] = fac
        rows_tex.append("\n".join(
            f"\t\t{lab} & {signed(fac[k]['mean'])}\\sd{{{f'{fac[k]['sd']:.3f}'[1:]}}} & {signed(fac[k]['ci_lo'])} to {signed(fac[k]['ci_hi'])} & {fmt_p(fac[k]['p_pooled'])} & {fac[k]['ds_sign']} \\\\"
            for lab, k in [("Training main effect", "train_main"), ("Inference main effect", "infer_main"), ("Interaction", "interaction"),
                           ("TE, single architecture", "TE_same_arch"), ("IE given full-text training", "IE_ft_trained"), ("IE given abstract training", "IE_abs_trained")]))

    Path(a.out, "stats_v26.json").write_text(json.dumps(out, indent=2, default=float))
    if rows_tex:
        Path(a.out, "tab_v26_factorial.tex").write_text("\n".join(rows_tex) + "\n")
    print("\nwrote", Path(a.out, "stats_v26.json"))


if __name__ == "__main__":
    main()
