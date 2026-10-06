# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Compute the TE/IE decomposition and generate actionable recommendations.

The decomposition splits the total effect of full-text processing into:
  - Training Effect (TE) = P@0% - C
  - Inference Effect (IE) = P@100% - P@0%
  - Total = TE + IE

Statistical significance is assessed via the Wilcoxon signed-rank test on
fold-level paired differences. With fewer than six folds the two-sided
exact test cannot reach P < .05, so no P value is reported; the paper's P
values pool folds across six datasets (n = 30).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from rich.console import Console
from rich.table import Table

console = Console()


MIN_N_FOR_TEST = 6


def _safe_wilcoxon(x: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test.

    Returns NaN when the test is not meaningful: fewer than
    ``MIN_N_FOR_TEST`` observations, or all differences zero.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < MIN_N_FOR_TEST or np.all(x == 0):
        return float("nan")
    try:
        _, p = wilcoxon(x, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def _fmt_p(p: float, n: int) -> str:
    if np.isnan(p):
        return f"n/a (n={n}<{MIN_N_FOR_TEST})" if n < MIN_N_FOR_TEST else "n/a"
    return f"{p:.4f}"


def _json_p(p: float):
    return None if np.isnan(p) else float(p)


def run_diagnosis(results_dir: Path, output_path: Optional[str] = None) -> None:
    """Compute decomposition from saved diagnostic results and print a report.

    Reads ``diagnostic_results.csv`` from *results_dir*, computes aggregate
    statistics and Wilcoxon p-values, prints a formatted table, and saves a
    structured JSON report to ``diagnostic_report.json``.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``diagnostic_results.csv`` (produced by the
        ``train`` command).
    output_path : str, optional
        If given, also write the JSON report to this path.
    """
    results_file = results_dir / "diagnostic_results.csv"
    if not results_file.exists():
        console.print(
            f"[red]No diagnostic_results.csv in {results_dir}. "
            f"Run 'fulltext-diagnostic train' first.[/red]"
        )
        return

    df = pd.read_csv(results_file)

    # Aggregate statistics
    n = len(df)
    c_mean, c_std = df["C"].mean(), df["C"].std(ddof=1)
    p0_mean, p0_std = df["P@0%"].mean(), df["P@0%"].std(ddof=1)
    p100_mean, p100_std = df["P@100%"].mean(), df["P@100%"].std(ddof=1)
    te_mean = df["TE"].mean()
    ie_mean = df["IE"].mean()
    total_mean = df["Total"].mean()

    # Fold-level values for statistical tests
    te_vals = df["TE"].values
    ie_vals = df["IE"].values
    total_vals = df["Total"].values

    te_p = _safe_wilcoxon(te_vals)
    ie_p = _safe_wilcoxon(ie_vals)
    total_p = _safe_wilcoxon(total_vals)

    # var(TE) as a predictor of PFD suitability
    te_var = float(np.var(te_vals, ddof=1))

    # Build report table
    table = Table(title="Full-Text Diagnostic Decomposition")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_column("p-value")
    table.add_column("Interpretation")

    table.add_row("C (abstract)", f"{c_mean:.3f} +/- {c_std:.3f}",
                  "", "Abstract-only baseline")
    table.add_row("P@0% (ft->abs)", f"{p0_mean:.3f} +/- {p0_std:.3f}",
                  "", "Full-text model on abstracts")
    table.add_row("P@100% (ft->ft)", f"{p100_mean:.3f} +/- {p100_std:.3f}",
                  "", "Full-text model on full-text")
    table.add_row("", "", "", "")
    table.add_row(
        "Training Effect (TE)", f"{te_mean:+.3f}", _fmt_p(te_p, n),
        "Negative = FT training hurts" if te_mean < 0
        else "Positive = FT training helps",
    )
    table.add_row(
        "Inference Effect (IE)", f"{ie_mean:+.3f}", _fmt_p(ie_p, n),
        "Positive = body text adds signal" if ie_mean > 0
        else "Negative = body text hurts",
    )
    table.add_row(
        "Total Effect", f"{total_mean:+.3f}", _fmt_p(total_p, n),
        "Net benefit of full-text processing",
    )
    table.add_row("", "", "", "")
    table.add_row("var(TE)", f"{te_var:.4f}", "",
                  "Higher = more training disruption")

    console.print(table)
    if n < MIN_N_FOR_TEST:
        console.print(
            f"[dim]P values need at least {MIN_N_FOR_TEST} paired folds; "
            f"pool folds across datasets for inference.[/dim]"
        )

    # Decision recommendation
    console.print("\n[bold]Recommendation[/bold]")

    if ie_mean > 0 and te_mean < 0:
        console.print(
            "[yellow]Pattern: IE positive, TE negative (cancellation).[/yellow]"
        )
        console.print(
            "Body text carries signal, but full-text training degrades "
            "abstract predictions."
        )
        console.print(
            "[green]-> PFD is recommended.[/green] Run: "
            "fulltext-diagnostic pfd --results-dir " + str(results_dir)
        )
    elif ie_mean > 0 and te_mean >= 0:
        console.print(
            "[green]Pattern: Both IE and TE are favorable.[/green]"
        )
        console.print("Full-text processing helps at both training and inference.")
        console.print(
            "-> Deploy the full-text model directly (P@100%). PFD is not needed."
        )
    elif ie_mean <= 0:
        console.print("[red]Pattern: IE is zero or negative.[/red]")
        console.print(
            "Body text does not add discriminative signal for this task."
        )
        console.print(
            "-> Deploy the abstract baseline (C). "
            "Full-text processing is not worthwhile."
        )

    # Save structured results
    report = {
        "n_folds": int(n),
        "C": {"mean": float(c_mean), "std": float(c_std)},
        "P@0%": {"mean": float(p0_mean), "std": float(p0_std)},
        "P@100%": {"mean": float(p100_mean), "std": float(p100_std)},
        "TE": {"mean": float(te_mean), "p": _json_p(te_p),
               "values": te_vals.tolist()},
        "IE": {"mean": float(ie_mean), "p": _json_p(ie_p),
               "values": ie_vals.tolist()},
        "Total": {"mean": float(total_mean), "p": _json_p(total_p),
                  "values": total_vals.tolist()},
        "var_TE": te_var,
        "recommendation": (
            "pfd" if (ie_mean > 0 and te_mean < 0)
            else "deploy_fulltext" if (ie_mean > 0 and te_mean >= 0)
            else "deploy_abstract"
        ),
    }

    report_path = results_dir / "diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\nReport saved to {report_path}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
