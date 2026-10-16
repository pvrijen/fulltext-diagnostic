#!/usr/bin/env python3
"""Statistical re-analysis of the exp8 fold-level results (Paper B, Table S9).

For each effect (paired PR-AUC difference vs the abstract-only baseline) this
script computes, from the 30 dataset-fold observations:

  * mean and SD (ddof = 1) of the paired fold-level difference
  * pooled two-sided Wilcoxon signed-rank P (n = 30; the group's convention)
  * two-stage cluster bootstrap 95% percentile CI and P (resample datasets
    with replacement, then folds within each sampled dataset; 20,000 replicates)
  * cluster-robust one-sample t on the 30 differences with dataset clusters
    (t distribution with 5 df)
  * Nadeau-Bengio corrected resampled t within each dataset (test fraction 0.2),
    combined across datasets with Stouffer's method
  * exact two-sided Wilcoxon on the six dataset means (minimum attainable P .031)
  * number of datasets with the hypothesised sign
  * Bayesian two-level hierarchical normal model (fold differences within
    dataset means; half-Cauchy(0, 0.1) priors on both SDs; mu ~ N(0, 1);
    Gibbs sampler, 35,000 retained draws): posterior mean and 95% credible
    interval of the population effect, P(effect has the hypothesised sign),
    and the posterior predictive probability that a new dataset shows it.

Usage: python stats_reanalysis.py [--results DIR] [--out tab_stats_reanalysis.tex]
Writes the LaTeX rows for Table S9 and a JSON with all values.
"""
import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "2.1_jamia_experiments" / "results"

EFFECTS = [  # name, label, sign of hypothesis
    ("IE", r"IE ($\Mft - \Mfta$)", +1),
    ("TE", r"TE ($\Mfta - \Mabs$)", -1),
    ("Total", r"Total ($\Mft - \Mabs$)", +1),
    ("IEp", r"IE$'$ ($\Aft - \Mabs$)", -1),
    ("TEp", r"TE$'$ ($\Mft - \Aft$)", +1),
    ("Int", r"Interaction (TE$'$ $-$ TE)", +1),
    ("TEsame", r"TE, same architecture ($\Mfta - \Aabs$)", -1),
    ("CF", r"CF-PFD gain ($\Mpfd - \Mabs$)", +1),
    ("IS", r"IS-PFD gain", +1),
]


def load(results_dir, metric="prauc", exp12_dir=None):
    """metric: 'prauc' (Table 2 upper block) or 'wss95' (Table 2 lower block); same folds, same keys.
    exp12_dir holds the fourth-cell results (A_abs->abs, A_abs->ft) for IE', TE', the interaction and the
    single-architecture TE; defaults to ./results."""
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{results_dir}/exp8_unified_*.json"))]
    exp12_dir = exp12_dir or (HERE / "results")
    e12 = {(r["dataset"], r["fold"]): r for r in (json.load(open(f)) for f in sorted(glob.glob(f"{exp12_dir}/exp12_*.json")))}
    rows = [r for r in rows if (r["dataset"], r["fold"]) in e12]
    assert len(rows) == 30, len(rows)
    ds = np.array([r["dataset"] for r in rows])
    C = np.array([r[f"C_{metric}"] for r in rows])
    P0 = np.array([r[f"P0_{metric}"] for r in rows])
    P1 = np.array([r[f"P100_{metric}"] for r in rows])
    IS = np.array([r[f"IS_{metric}"] for r in rows])
    CF = np.array([r[f"CF_{metric}"] for r in rows])
    Aabs = np.array([e12[(r["dataset"], r["fold"])][f"Aabs_abs_{metric}"] for r in rows])
    Aft = np.array([e12[(r["dataset"], r["fold"])][f"Aabs_ft_{metric}"] for r in rows])
    return ds, {"IE": P1 - P0, "TE": P0 - C, "Total": P1 - C, "IEp": Aft - C, "TEp": P1 - Aft,
                "Int": (P1 - Aft) - (P0 - C), "TEsame": P0 - Aabs, "CF": CF - C, "IS": IS - C}


def cluster_bootstrap(x, groups, B=20000, seed=42):
    rng = np.random.default_rng(seed)
    labels = sorted(set(groups))
    pools = {g: x[groups == g] for g in labels}
    means = np.empty(B)
    for b in range(B):
        picked = rng.choice(labels, len(labels), replace=True)
        vals = np.concatenate([rng.choice(pools[g], len(pools[g]), replace=True) for g in picked])
        means[b] = vals.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return lo, hi, min(p, 1.0)


def cluster_robust_t(x, groups):
    import statsmodels.api as sm
    codes = np.unique(groups, return_inverse=True)[1]
    m = sm.OLS(x, np.ones(len(x))).fit(cov_type="cluster",
                                      cov_kwds={"groups": codes, "use_correction": True, "df_correction": True})
    t = m.params[0] / m.bse[0]
    return t, 2 * stats.t.sf(abs(t), df=len(set(groups)) - 1)


def nadeau_bengio_combined(x, groups, test_frac=0.2):
    """Per-dataset corrected resampled t (Nadeau & Bengio 2003), Stouffer-combined."""
    zs = []
    for g in sorted(set(groups)):
        d = x[groups == g]
        k = len(d)
        var = d.var(ddof=1) * (1 / k + test_frac / (1 - test_frac))
        t = d.mean() / np.sqrt(var)
        p_one = stats.t.sf(t, df=k - 1)  # one-sided in the positive direction
        zs.append(stats.norm.isf(p_one))
    z = np.sum(zs) / np.sqrt(len(zs))
    return 2 * stats.norm.sf(abs(z))


def dataset_level(x, groups, sign):
    dm = np.array([x[groups == g].mean() for g in sorted(set(groups))])
    p = stats.wilcoxon(dm, alternative="two-sided", method="exact").pvalue
    return p, int((sign * dm > 0).sum()), len(dm)


def gibbs(y, g, A=0.1, n_iter=40000, burn=5000, seed=0, mu_sd=1.0):
    rng = np.random.default_rng(seed)
    n = len(y)
    J = g.max() + 1
    nj = np.bincount(g, minlength=J)
    ybar = np.bincount(g, weights=y, minlength=J) / nj
    theta = ybar.copy()
    mu = y.mean()
    tau2 = max(ybar.var(ddof=1), 1e-4)
    sig2 = max(y.var(), 1e-4)
    a_t = a_s = 1.0
    out = np.empty((n_iter - burn, 3))
    for it in range(n_iter):
        prec = nj / sig2 + 1 / tau2
        theta = rng.normal((nj * ybar / sig2 + mu / tau2) / prec, np.sqrt(1 / prec))
        prec_mu = J / tau2 + 1 / mu_sd ** 2
        mu = rng.normal((theta.sum() / tau2) / prec_mu, np.sqrt(1 / prec_mu))
        tau2 = stats.invgamma.rvs((J + 1) / 2, scale=((theta - mu) ** 2).sum() / 2 + 1 / a_t, random_state=rng)
        a_t = stats.invgamma.rvs(1.0, scale=1 / tau2 + 1 / A ** 2, random_state=rng)
        resid = y - theta[g]
        sig2 = stats.invgamma.rvs((n + 1) / 2, scale=(resid ** 2).sum() / 2 + 1 / a_s, random_state=rng)
        a_s = stats.invgamma.rvs(1.0, scale=1 / sig2 + 1 / A ** 2, random_state=rng)
        if it >= burn:
            out[it - burn] = (mu, np.sqrt(tau2), np.sqrt(sig2))
    return out


def bayes(x, groups, sign):
    codes = np.unique(groups, return_inverse=True)[1]
    s = gibbs(x, codes)
    mu, tau = s[:, 0], s[:, 1]
    rng = np.random.default_rng(1)
    theta_new = rng.normal(mu, tau)
    return (mu.mean(), *np.percentile(mu, [2.5, 97.5]), (sign * mu > 0).mean(),
            (sign * theta_new > 0).mean(), np.median(tau))


def fmt_p(p):
    if p < 0.001:
        return "$<$.001"
    return f"{p:.3f}"[1:] if p < 0.01 else f"{p:.2f}"[1:]


def signed(v):
    return f"$-${abs(v):.3f}" if v < 0 else f"$+${v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out", default=None, help="default: tab_stats_reanalysis.tex (prauc) or tab_stats_reanalysis_wss95.tex")
    ap.add_argument("--metric", default="prauc", choices=["prauc", "wss95"])
    args = ap.parse_args()
    if args.out is None:
        args.out = str(HERE / ("tab_stats_reanalysis.tex" if args.metric == "prauc" else "tab_stats_reanalysis_wss95.tex"))
    groups, eff = load(args.results, args.metric)

    lines, record = [], {}
    for key, label, sign in EFFECTS:
        x = eff[key]
        lo, hi, p_boot = cluster_bootstrap(x, groups)
        t_cl, p_cl = cluster_robust_t(x, groups)
        p_nb = nadeau_bengio_combined(x, groups)
        p_pool = stats.wilcoxon(x, alternative="two-sided").pvalue
        p_ds, n_sign, n_ds = dataset_level(x, groups, sign)
        mu, mu_lo, mu_hi, p_dir, p_new, tau = bayes(x, groups, sign)
        record[key] = dict(mean=x.mean(), sd=x.std(ddof=1), boot_lo=lo, boot_hi=hi, p_boot=p_boot,
                           p_cluster_t=p_cl, p_nb=p_nb, p_pooled=p_pool, p_dataset=p_ds,
                           n_sign=n_sign, mu=mu, mu_lo=mu_lo, mu_hi=mu_hi, p_dir=p_dir, p_newds=p_new, tau=tau)
        lines.append(
            f"\t\t{label} & {signed(x.mean())}\\sd{{{f'{x.std(ddof=1):.3f}'[1:]}}} & "
            f"{signed(lo)} to {signed(hi)} & {fmt_p(p_boot)} & {fmt_p(p_pool)} & {fmt_p(p_cl)} & {fmt_p(p_nb)} & "
            f"{fmt_p(p_ds)} & {n_sign}/{n_ds} & "
            f"{signed(mu_lo)} to {signed(mu_hi)} & {p_dir:.2f} & {p_new:.2f} \\\\"
        )
        print(f"{key:6s} mean {x.mean():+.4f} boot [{lo:+.3f},{hi:+.3f}] P={p_boot:.3f} | pooled {p_pool:.4f} | "
              f"cluster t {p_cl:.3f} | NB {p_nb:.3f} | n=6 {p_ds:.3f} ({n_sign}/{n_ds}) | "
              f"Bayes mu {mu:+.3f} [{mu_lo:+.3f},{mu_hi:+.3f}] P(dir) {p_dir:.2f} P(new ds) {p_new:.2f}")

    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(record, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
