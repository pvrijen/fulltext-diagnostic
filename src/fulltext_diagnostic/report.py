# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Generate a formatted diagnostic report with per-fold tables and
deployment recommendations.

Uses pipeline state to determine which results (decomposition, PFD) are
available, and produces a consolidated report with actionable guidance.
"""

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .pipeline import build_dag, get_summary, load_state, mark_done

console = Console()


def generate_report(results_dir: Path) -> None:
    """Generate a formatted report and mark the 'report' job as done.

    Reads ``diagnostic_results.csv`` (and optionally ``pfd_results.csv``)
    from *results_dir* and prints a formatted summary to the console
    including per-fold decomposition values and a deployment recommendation.

    Parameters
    ----------
    results_dir : Path
        Directory containing pipeline outputs.
    """
    config = json.load(open(results_dir / "config.json"))
    n_folds = config["folds"]
    dag = build_dag(n_folds)
    dag = load_state(results_dir, dag)
    s = get_summary(dag)

    console.print(Panel(
        f"[bold]Diagnostic Report[/bold]  "
        f"({s['done']}/{s['total']} jobs complete)",
        style="blue",
    ))

    # Decomposition table
    diag_file = results_dir / "diagnostic_results.csv"
    if not diag_file.exists():
        console.print("[red]No diagnostic results. Run 'train' first.[/red]")
        return

    diag = pd.read_csv(diag_file)
    table = Table(title="Decomposition (PR-AUC)")
    for col in ["Fold", "C", "P@0%", "P@100%", "TE", "IE"]:
        table.add_column(col, justify="right" if col != "Fold" else "left")
    for _, r in diag.iterrows():
        table.add_row(
            str(int(r["fold"])),
            f"{r['C']:.3f}", f"{r['P@0%']:.3f}", f"{r['P@100%']:.3f}",
            f"{r['TE']:+.3f}", f"{r['IE']:+.3f}",
        )
    table.add_row(
        "Mean",
        f"{diag['C'].mean():.3f}", f"{diag['P@0%'].mean():.3f}",
        f"{diag['P@100%'].mean():.3f}",
        f"{diag['TE'].mean():+.3f}", f"{diag['IE'].mean():+.3f}",
    )
    console.print(table)

    # PFD table (if available)
    pfd_file = results_dir / "pfd_results.csv"
    if pfd_file.exists():
        pfd_df = pd.read_csv(pfd_file)
        console.print()
        pt = Table(title="PFD Transfer")
        for col in ["Fold", "C", "C_PFD", "delta"]:
            pt.add_column(col, justify="right" if col != "Fold" else "left")
        for _, r in pfd_df.iterrows():
            pt.add_row(
                str(int(r["fold"])),
                f"{r['C']:.3f}", f"{r['C_PFD']:.3f}", f"{r['delta']:+.3f}",
            )
        pt.add_row(
            "Mean",
            f"{pfd_df['C'].mean():.3f}", f"{pfd_df['C_PFD'].mean():.3f}",
            f"{pfd_df['delta'].mean():+.3f}",
        )
        console.print(pt)

    # Recommendation
    te = diag["TE"].mean()
    ie = diag["IE"].mean()
    console.print()

    if ie > 0 and te < 0:
        if pfd_file.exists() and pd.read_csv(pfd_file)["delta"].mean() > 0:
            d = pd.read_csv(pfd_file)["delta"].mean()
            console.print(Panel(
                f"[green bold]DEPLOY C_PFD[/green bold]\n\n"
                f"IE={ie:+.3f} (signal present), TE={te:+.3f} (training hurts).\n"
                f"PFD transferred +{d:.3f} PR-AUC. Deploy student on abstracts.",
                title="Recommendation", style="green",
            ))
        else:
            console.print(Panel(
                f"[yellow bold]RUN PFD[/yellow bold]\n\n"
                f"IE={ie:+.3f} (signal present), TE={te:+.3f} (training hurts).\n"
                f"Run: fulltext-diagnostic pfd --results-dir {results_dir} "
                f"--data DATA",
                title="Recommendation", style="yellow",
            ))
    elif ie > 0 and te >= 0:
        console.print(Panel(
            f"[green bold]DEPLOY P@100%[/green bold]\n\n"
            f"IE={ie:+.3f}, TE={te:+.3f}. Full-text helps everywhere.\n"
            f"Deploy full-text model directly.",
            title="Recommendation", style="green",
        ))
    else:
        console.print(Panel(
            f"[blue bold]DEPLOY C (ABSTRACT)[/blue bold]\n\n"
            f"IE={ie:+.3f}. Body text adds no signal.\n"
            f"Abstract baseline is sufficient.",
            title="Recommendation", style="blue",
        ))

    mark_done(results_dir, "report")
