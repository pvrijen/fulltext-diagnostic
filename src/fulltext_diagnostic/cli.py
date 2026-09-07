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

import click
from pathlib import Path


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
@click.option("--encoder",
              default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
              help="HuggingFace model name or path")
@click.option("--output-dir", default="ft_diagnostic_output",
              help="Directory for all pipeline outputs")
@click.option("--folds", default=5, type=int, help="Number of cross-validation folds")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("--max-length", default=512, type=int,
              help="Max token length for abstract models")
@click.option("--chunks", default=12, type=int,
              help="Max chunks for full-text model")
@click.option("--chunk-size", default=510, type=int,
              help="Tokens per chunk for full-text model")
@click.option("--epochs", default=10, type=int, help="Max training epochs")
@click.option("--batch-size", default=16, type=int, help="Training batch size")
@click.option("--lr", default=2e-5, type=float, help="Learning rate")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def train(data: str, encoder: str, output_dir: str, folds: int, seed: int,
          max_length: int, chunks: int, chunk_size: int, epochs: int,
          batch_size: int, lr: float, device: str) -> None:
    """Run the full diagnostic pipeline (sequential). Resumable."""
    from .train import run_training
    run_training(Path(data), encoder, Path(output_dir), folds, seed,
                 max_length, chunks, chunk_size, epochs, batch_size, lr, device)


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
@click.option("--encoder",
              default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
              help="HuggingFace model name or path")
@click.option("--alpha", default=0.1, type=float,
              help="Hard-label weight in PFD loss (0=all soft, 1=all hard)")
@click.option("--temperature", default=4.0, type=float,
              help="Temperature for soft-label distillation")
@click.option("--epochs", default=3, type=int, help="PFD training epochs")
@click.option("--lr", default=5e-6, type=float, help="PFD learning rate")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def pfd(results_dir: str, data: str, encoder: str, alpha: float,
        temperature: float, epochs: int, lr: float, device: str) -> None:
    """Run Privileged Features Distillation using cached teacher logits."""
    from .pfd import run_pfd
    run_pfd(Path(results_dir), Path(data), encoder, alpha, temperature,
            epochs, lr, device)


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
@click.option("--encoder",
              default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
              help="HuggingFace model name or path")
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
      02_train_C.sbatch    (GPU array, ~1h/fold)
      03_train_P.sbatch    (GPU array, ~2h/fold)
      04_eval.sbatch       (GPU array, minutes)
      05_diagnose.sbatch   (CPU, seconds)
      06_pfd.sbatch        (GPU array, ~30min/fold)
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

@main.command(name="run-job")
@click.argument("job_type", type=click.Choice([
    "split", "train_C", "train_P", "eval_C", "eval_P", "diagnose"]))
@click.option("--fold", default=0, type=int, help="Fold index")
@click.option("--data", required=True, type=click.Path(exists=True),
              help="CSV dataset")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Pipeline output directory")
@click.option("--encoder",
              default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
              help="HuggingFace model name or path")
@click.option("--max-length", default=512, type=int)
@click.option("--epochs", default=10, type=int)
@click.option("--batch-size", default=16, type=int)
@click.option("--lr", default=2e-5, type=float)
@click.option("--folds", default=5, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--device", default="cuda")
def run_job(job_type: str, fold: int, data: str, output_dir: str,
            encoder: str, max_length: int, epochs: int, batch_size: int,
            lr: float, folds: int, seed: int, device: str) -> None:
    """Run a single job step (for SLURM array or manual use).

    \b
    SLURM array example:
      #SBATCH --array=0-4
      fulltext-diagnostic run-job train_C --fold $SLURM_ARRAY_TASK_ID \\
        --data dataset.csv --output-dir results/
    """
    from .train import (job_split_folds, job_train_C, job_train_P,
                        job_eval_C, job_eval_P, job_diagnose)
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    dp = Path(data)

    dispatch = {
        "split": lambda: job_split_folds(dp, od, folds, seed),
        "train_C": lambda: job_train_C(fold, dp, od, encoder, max_length,
                                       epochs, batch_size, lr, device),
        "train_P": lambda: job_train_P(fold, dp, od, encoder, max_length,
                                       epochs, batch_size, lr, device),
        "eval_C": lambda: job_eval_C(fold, dp, od, max_length, batch_size,
                                     device),
        "eval_P": lambda: job_eval_P(fold, dp, od, max_length, batch_size,
                                     device),
        "diagnose": lambda: job_diagnose(od, folds),
    }
    dispatch[job_type]()


if __name__ == "__main__":
    main()
