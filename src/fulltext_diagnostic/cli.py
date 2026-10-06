# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Command-line interface for the fulltext-diagnostic pipeline.

Provides subcommands for the complete diagnostic workflow:
  - train: Run the full diagnostic pipeline (sequential, resumable)
  - diagnose: Compute TE/IE decomposition from trained models
  - pfd: Apply Privileged Features Distillation
  - report: Generate a formatted summary with recommendations
  - status / dag: Inspect pipeline progress
  - slurm: Generate SLURM batch scripts for HPC deployment
  - run-job: Execute a single pipeline step (for SLURM arrays)
  - reset: Clear a job's state marker so it reruns
"""

import sys
from pathlib import Path

import click

DEFAULT_ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
_ENCODER_HELP = (
    "HuggingFace model name or path (default: the abstract-only BioMedBERT "
    "checkpoint used in the paper)"
)


@click.group()
@click.version_option()
def main() -> None:
    """Diagnose whether full-text helps biomedical screening, and transfer knowledge via PFD.

    \b
    Quick start (single GPU, sequential):
      fulltext-diagnostic train --data dataset.csv --output-dir results/
      fulltext-diagnostic diagnose --results-dir results/
      fulltext-diagnostic pfd --results-dir results/ --data dataset.csv

    \b
    HPC (SLURM, parallel across folds):
      fulltext-diagnostic slurm --data dataset.csv --output-dir results/
      bash results/slurm/submit_all.sh
    """
    pass


# --- High-level commands ---------------------------------------------------

@main.command()
@click.option("--data", required=True, type=click.Path(exists=True),
              help="CSV with columns: id, text_abstract, text_fulltext, label")
@click.option("--encoder", default=DEFAULT_ENCODER, help=_ENCODER_HELP)
@click.option("--output-dir", default="ft_diagnostic_output",
              help="Directory for all pipeline outputs")
@click.option("--folds", default=5, type=int, help="Number of cross-validation folds")
@click.option("--seed", default=42, type=int,
              help="Random seed for fold splits and model initialization")
@click.option("--max-length", default=512, type=int,
              help="Max token length for the abstract model C")
@click.option("--chunks", default=12, type=int,
              help="Max chunks for the full-text model P")
@click.option("--chunk-size", default=510, type=int,
              help="Content tokens per chunk for the full-text model P")
@click.option("--epochs", default=20, type=int,
              help="Max training epochs (early stopping, patience 3)")
@click.option("--batch-size", default=32, type=int,
              help="Batch size for the abstract model C")
@click.option("--lr", default=3e-5, type=float,
              help="Learning rate for the abstract model C")
@click.option("--batch-size-fulltext", default=8, type=int,
              help="Micro-batch size for P (effective batch stays 64)")
@click.option("--lr-fulltext", default=1e-5, type=float,
              help="Learning rate for the full-text model P")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def train(data: str, encoder: str, output_dir: str, folds: int, seed: int,
          max_length: int, chunks: int, chunk_size: int, epochs: int,
          batch_size: int, lr: float, batch_size_fulltext: int,
          lr_fulltext: float, device: str) -> None:
    """Run the full diagnostic pipeline (sequential). Resumable.

    Defaults follow the paper's hyperparameter table. Exits with status 1
    if any job fails.
    """
    from .train import run_training
    ok = run_training(Path(data), encoder, Path(output_dir), folds, seed,
                      max_length, chunks, chunk_size, epochs, batch_size,
                      lr, device, lr_fulltext=lr_fulltext,
                      batch_size_fulltext=batch_size_fulltext)
    if not ok:
        sys.exit(1)


@main.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True),
              help="Directory containing diagnostic_results.csv")
@click.option("--output", default=None, type=click.Path(),
              help="Optional path to save JSON report")
def diagnose(results_dir: str, output: str) -> None:
    """Compute TE/IE decomposition from trained model predictions."""
    from .diagnose import run_diagnosis
    run_diagnosis(Path(results_dir), output)


@main.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True),
              help="Directory containing trained model cache")
@click.option("--data", required=True, type=click.Path(exists=True),
              help="CSV dataset (same as used for training)")
@click.option("--encoder", default=DEFAULT_ENCODER,
              help="Fallback encoder if no C checkpoint exists to warm-start from")
@click.option("--alpha", default=0.1, type=float,
              help="Soft-target weight in PFD loss (0=hard labels only, "
                   "1=soft targets only)")
@click.option("--temperature", default=4.0, type=float,
              help="Temperature for soft-label distillation")
@click.option("--epochs", default=3, type=int,
              help="Max PFD epochs (early stopping, patience 2)")
@click.option("--lr", default=5e-6, type=float, help="PFD learning rate")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def pfd(results_dir: str, data: str, encoder: str, alpha: float,
        temperature: float, epochs: int, lr: float, device: str) -> None:
    """Run Privileged Features Distillation using cached teacher logits.

    The student is warm-started from the trained abstract model C of each
    fold. Exits with status 1 if any fold fails.
    """
    from .pfd import run_pfd
    ok = run_pfd(Path(results_dir), Path(data), encoder, alpha, temperature,
                 epochs, lr, device)
    if not ok:
        sys.exit(1)


@main.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True),
              help="Directory containing pipeline outputs")
def report(results_dir: str) -> None:
    """Generate a full diagnostic report with recommendations."""
    from .report import generate_report
    generate_report(Path(results_dir))


# --- Pipeline management ---------------------------------------------------

@main.command()
@click.option("--output-dir", required=True, type=click.Path(),
              help="Pipeline output directory")
@click.option("--folds", default=5, type=int, help="Number of folds")
def status(output_dir: str, folds: int) -> None:
    """Show pipeline status: which jobs are done, pending, or failed."""
    from .pipeline import build_dag, load_state, show_status
    dag = build_dag(folds)
    dag = load_state(Path(output_dir), dag)
    show_status(dag, folds)


@main.command()
@click.option("--output-dir", required=True, type=click.Path(),
              help="Pipeline output directory")
@click.option("--folds", default=5, type=int, help="Number of folds")
def dag(output_dir: str, folds: int) -> None:
    """Show the job dependency tree."""
    from .pipeline import build_dag, load_state, show_dag
    dag_obj = build_dag(folds)
    dag_obj = load_state(Path(output_dir), dag_obj)
    show_dag(dag_obj, folds)


@main.command()
@click.option("--data", required=True, type=click.Path(exists=True),
              help="CSV dataset")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Pipeline output directory")
@click.option("--encoder", default=DEFAULT_ENCODER, help=_ENCODER_HELP)
@click.option("--folds", default=5, type=int, help="Number of folds")
@click.option("--partition", default="gpu", help="SLURM partition for GPU jobs")
@click.option("--time", "time_limit", default="04:00:00",
              help="SLURM time limit per job")
@click.option("--mem", default="32G", help="SLURM memory per job")
@click.option("--gpus", default=1, type=int, help="GPUs per job")
def slurm(data: str, output_dir: str, encoder: str, folds: int,
          partition: str, time_limit: str, mem: str, gpus: int) -> None:
    """Generate SLURM batch scripts with dependency chains.

    \b
    Creates slurm/ directory with:
      01_split.sbatch      (CPU, seconds)
      02_train_C.sbatch    (GPU array, ~10min/fold on H100)
      03_train_P.sbatch    (GPU array, ~40min/fold on H100)
      04_eval.sbatch       (GPU array, minutes)
      05_diagnose.sbatch   (CPU, seconds)
      06_pfd.sbatch        (GPU array, ~10min/fold on H100)
      07_report.sbatch     (CPU, seconds)
      submit_all.sh        (submits chain with --dependency)
    """
    from .pipeline import generate_slurm_scripts
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    generate_slurm_scripts(od, data, encoder, folds, partition,
                           time_limit, mem, gpus)


@main.command()
@click.option("--output-dir", required=True, type=click.Path(exists=True),
              help="Pipeline output directory")
@click.argument("job_name")
def reset(output_dir: str, job_name: str) -> None:
    """Reset a job (remove its done/failed marker so it reruns).

    \b
    Examples:
      fulltext-diagnostic reset --output-dir results/ train_C_f0
      fulltext-diagnostic reset --output-dir results/ diagnose
    """
    od = Path(output_dir)
    for marker in [od / f".done_{job_name}", od / f".failed_{job_name}"]:
        if marker.exists():
            marker.unlink()
            click.echo(f"Reset: {marker.name}")
    click.echo(f"Job '{job_name}' will rerun on next execution.")


# --- Individual job runner --------------------------------------------------

JOB_TYPES = ["split", "train_C", "train_P", "eval_C", "eval_P",
             "diagnose", "pfd", "report"]


@main.command(name="run-job")
@click.argument("job_type", type=click.Choice(JOB_TYPES))
@click.option("--fold", default=0, type=int, help="Fold index")
@click.option("--data", required=True, type=click.Path(exists=True),
              help="CSV dataset")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Pipeline output directory")
@click.option("--encoder", default=DEFAULT_ENCODER, help=_ENCODER_HELP)
@click.option("--max-length", default=512, type=int,
              help="Max token length for the abstract model C")
@click.option("--chunks", default=12, type=int,
              help="Max chunks for the full-text model P")
@click.option("--chunk-size", default=510, type=int,
              help="Content tokens per chunk for the full-text model P")
@click.option("--epochs", default=20, type=int, help="Max training epochs")
@click.option("--batch-size", default=32, type=int,
              help="Batch size for C (and C_PFD evaluation)")
@click.option("--lr", default=3e-5, type=float, help="Learning rate for C")
@click.option("--batch-size-fulltext", default=8, type=int,
              help="Micro-batch size for P")
@click.option("--lr-fulltext", default=1e-5, type=float,
              help="Learning rate for P")
@click.option("--alpha", default=0.1, type=float,
              help="PFD soft-target weight")
@click.option("--temperature", default=4.0, type=float,
              help="PFD temperature")
@click.option("--pfd-epochs", default=3, type=int, help="Max PFD epochs")
@click.option("--pfd-lr", default=5e-6, type=float, help="PFD learning rate")
@click.option("--folds", default=5, type=int, help="Number of folds")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def run_job(job_type: str, fold: int, data: str, output_dir: str,
            encoder: str, max_length: int, chunks: int, chunk_size: int,
            epochs: int, batch_size: int, lr: float,
            batch_size_fulltext: int, lr_fulltext: float, alpha: float,
            temperature: float, pfd_epochs: int, pfd_lr: float,
            folds: int, seed: int, device: str) -> None:
    """Run a single job step (for SLURM array or manual use).

    Exits with status 1 if the job fails, so SLURM dependency chains stop.

    \b
    SLURM array example:
      #SBATCH --array=0-4
      fulltext-diagnostic run-job train_C --fold $SLURM_ARRAY_TASK_ID \\
        --data dataset.csv --output-dir results/
    """
    from .train import (job_split_folds, job_train_C, job_train_P,
                        job_eval_C, job_eval_P, job_diagnose)
    from .pfd import job_pfd, summarize_pfd
    from .report import generate_report
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    dp = Path(data)

    def _report() -> bool:
        summarize_pfd(od, folds)
        generate_report(od)
        return True

    dispatch = {
        "split": lambda: job_split_folds(dp, od, folds, seed),
        "train_C": lambda: job_train_C(fold, dp, od, encoder, max_length,
                                       epochs, batch_size, lr, device),
        "train_P": lambda: job_train_P(fold, dp, od, encoder, epochs,
                                       batch_size_fulltext, lr_fulltext,
                                       device, n_chunks=chunks,
                                       chunk_size=chunk_size),
        "eval_C": lambda: job_eval_C(fold, dp, od, max_length, batch_size,
                                     device),
        "eval_P": lambda: job_eval_P(fold, dp, od, device),
        "diagnose": lambda: job_diagnose(od, folds),
        "pfd": lambda: job_pfd(fold, od, dp, encoder, alpha, temperature,
                               pfd_epochs, pfd_lr, device),
        "report": _report,
    }
    if not dispatch[job_type]():
        sys.exit(1)


if __name__ == "__main__":
    main()
