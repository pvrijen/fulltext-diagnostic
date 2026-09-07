# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Pipeline manager: single source of truth for job state and dependencies.

All modules (train, pfd, diagnose, report) use this module for:
  - Checking whether a job is done, failed, or running
  - Marking a job as done or failed
  - Querying which jobs are runnable (all dependencies satisfied)
  - Displaying status tables and dependency trees
  - Generating SLURM batch scripts with dependency chains
"""

import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


# --- State Management -------------------------------------------------------

class State(str, Enum):
    """Pipeline job state."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


def job_done(output_dir: Path, job_name: str) -> bool:
    """Check whether a job has completed successfully."""
    return (output_dir / f".done_{job_name}").exists()


def job_failed(output_dir: Path, job_name: str) -> bool:
    """Check whether a job has failed."""
    return (output_dir / f".failed_{job_name}").exists()


def mark_done(output_dir: Path, job_name: str) -> None:
    """Mark a job as successfully completed."""
    (output_dir / f".done_{job_name}").touch()
    failed = output_dir / f".failed_{job_name}"
    if failed.exists():
        failed.unlink()


def mark_failed(output_dir: Path, job_name: str, error: str = "") -> None:
    """Mark a job as failed, storing the first 500 chars of the error."""
    (output_dir / f".failed_{job_name}").write_text(error[:500])


def mark_running(output_dir: Path, job_name: str) -> None:
    """Mark a job as currently running."""
    (output_dir / f".running_{job_name}").touch()


def clear_running(output_dir: Path, job_name: str) -> None:
    """Remove the running marker for a job."""
    r = output_dir / f".running_{job_name}"
    if r.exists():
        r.unlink()


def reset_job(output_dir: Path, job_name: str) -> None:
    """Remove all state markers for a job so it can be rerun."""
    for prefix in [".done_", ".failed_", ".running_"]:
        m = output_dir / f"{prefix}{job_name}"
        if m.exists():
            m.unlink()


def run_guarded(output_dir: Path, job_name: str, fn: Callable) -> bool:
    """Run a function with automatic state tracking.

    Skips the job if already done. On success, marks the job as done. On
    failure, marks the job as failed and prints the error.

    Parameters
    ----------
    output_dir : Path
        Pipeline output directory.
    job_name : str
        Unique job identifier.
    fn : Callable
        Zero-argument callable to execute.

    Returns
    -------
    bool
        True if the job completed (or was already done), False on failure.
    """
    if job_done(output_dir, job_name):
        console.print(f"  [dim]{job_name}: skip (done)[/dim]")
        return True

    mark_running(output_dir, job_name)
    try:
        fn()
        clear_running(output_dir, job_name)
        mark_done(output_dir, job_name)
        return True
    except Exception as e:
        clear_running(output_dir, job_name)
        mark_failed(output_dir, job_name, traceback.format_exc())
        console.print(f"  [red]{job_name}: FAILED -- {e}[/red]")
        return False


# --- Job Definitions ---------------------------------------------------------

@dataclass
class JobDef:
    """Definition of a pipeline job with its dependencies and metadata."""

    name: str
    job_type: str
    fold: Optional[int]
    depends_on: list[str]
    resource: str       # "cpu" or "gpu"
    est_minutes: int
    state: State = State.PENDING
    error: str = ""

    @property
    def cli_args(self) -> str:
        """Command-line arguments for ``run-job``."""
        if self.fold is not None:
            return f"{self.job_type} --fold {self.fold}"
        return self.job_type


def build_dag(n_folds: int) -> dict[str, JobDef]:
    """Build the full dependency graph for the diagnostic pipeline.

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds.

    Returns
    -------
    dict[str, JobDef]
        Mapping from job name to its definition.
    """
    jobs: dict[str, JobDef] = {}

    jobs["split"] = JobDef("split", "split", None, [], "cpu", 1)

    for f in range(n_folds):
        jobs[f"train_C_f{f}"] = JobDef(
            f"train_C_f{f}", "train_C", f, ["split"], "gpu", 60)
        jobs[f"train_P_f{f}"] = JobDef(
            f"train_P_f{f}", "train_P", f, ["split"], "gpu", 120)
        jobs[f"eval_C_f{f}"] = JobDef(
            f"eval_C_f{f}", "eval_C", f, [f"train_C_f{f}"], "gpu", 5)
        jobs[f"eval_P_f{f}"] = JobDef(
            f"eval_P_f{f}", "eval_P", f, [f"train_P_f{f}"], "gpu", 10)

    all_evals = (
        [f"eval_C_f{f}" for f in range(n_folds)]
        + [f"eval_P_f{f}" for f in range(n_folds)]
    )
    jobs["diagnose"] = JobDef("diagnose", "diagnose", None, all_evals, "cpu", 1)

    for f in range(n_folds):
        jobs[f"pfd_f{f}"] = JobDef(
            f"pfd_f{f}", "pfd", f,
            ["diagnose", f"eval_C_f{f}", f"eval_P_f{f}"], "gpu", 30,
        )

    all_pfd = [f"pfd_f{f}" for f in range(n_folds)]
    jobs["report"] = JobDef("report", "report", None, all_pfd, "cpu", 1)

    return jobs


def load_state(output_dir: Path, dag: dict[str, JobDef]) -> dict[str, JobDef]:
    """Synchronise job states from disk markers.

    Parameters
    ----------
    output_dir : Path
        Pipeline output directory containing ``.done_*`` / ``.failed_*``
        marker files.
    dag : dict[str, JobDef]
        The dependency graph to update in place.

    Returns
    -------
    dict[str, JobDef]
        The same *dag* dict, updated with states read from disk.
    """
    for name, job in dag.items():
        if job_done(output_dir, name):
            job.state = State.DONE
        elif job_failed(output_dir, name):
            job.state = State.FAILED
            try:
                job.error = (output_dir / f".failed_{name}").read_text()[:100]
            except Exception:
                pass
        elif (output_dir / f".running_{name}").exists():
            job.state = State.RUNNING
    return dag


def is_ready(job: JobDef, dag: dict[str, JobDef]) -> bool:
    """Check whether all dependencies of a job are satisfied."""
    return all(dag[d].state == State.DONE for d in job.depends_on)


def get_runnable(dag: dict[str, JobDef]) -> list[JobDef]:
    """Return all pending jobs whose dependencies are satisfied."""
    return [j for j in dag.values()
            if j.state == State.PENDING and is_ready(j, dag)]


def get_summary(dag: dict[str, JobDef]) -> dict[str, int]:
    """Return a summary of job states."""
    done = sum(1 for j in dag.values() if j.state == State.DONE)
    failed = sum(1 for j in dag.values() if j.state == State.FAILED)
    running = sum(1 for j in dag.values() if j.state == State.RUNNING)
    pending = sum(1 for j in dag.values() if j.state == State.PENDING)
    return {
        "done": done, "failed": failed, "running": running,
        "pending": pending, "total": len(dag),
    }


# --- Display -----------------------------------------------------------------

STEP_LABELS = {
    "split": "0", "train_C": "1a", "train_P": "1b",
    "eval_C": "2a", "eval_P": "2b", "diagnose": "3",
    "pfd": "4", "report": "5",
}

STATE_FMT = {
    State.PENDING: "[dim]PENDING[/dim]",
    State.RUNNING: "[yellow]RUNNING[/yellow]",
    State.DONE: "[green]DONE[/green]",
    State.FAILED: "[red]FAILED[/red]",
}


def show_status(dag: dict[str, JobDef], n_folds: int) -> None:
    """Print a formatted status table to the console."""
    s = get_summary(dag)
    pct = s["done"] / s["total"] * 100 if s["total"] else 0

    header = f"[bold]{s['done']}/{s['total']}[/bold] done ({pct:.0f}%)"
    if s["failed"]:
        header += f"  [red]{s['failed']} failed[/red]"
    if s["running"]:
        header += f"  [yellow]{s['running']} running[/yellow]"
    console.print(Panel(header, title="Pipeline", style="blue"))

    table = Table(show_lines=False, pad_edge=False)
    table.add_column("Step", width=5)
    table.add_column("Job", width=16)
    table.add_column("Res", width=4)
    table.add_column("~Min", width=5)
    table.add_column("State", width=10)

    for step_key in ["split", "train_C", "train_P", "eval_C", "eval_P",
                     "diagnose", "pfd", "report"]:
        for name, job in dag.items():
            if job.job_type == step_key:
                table.add_row(
                    STEP_LABELS.get(job.job_type, "?"),
                    name, job.resource,
                    str(job.est_minutes),
                    STATE_FMT[job.state],
                )
    console.print(table)

    runnable = get_runnable(dag)
    if runnable:
        console.print(f"\n[bold]Next ({len(runnable)} ready):[/bold]")
        for j in runnable[:8]:
            console.print(
                f"  fulltext-diagnostic run-job {j.cli_args} ..."
            )
    elif s["failed"]:
        console.print(
            "\n[red]Fix failures, then: "
            "fulltext-diagnostic reset --output-dir DIR JOB_NAME[/red]"
        )


def show_dag(dag: dict[str, JobDef], n_folds: int) -> None:
    """Print the job dependency tree to the console."""
    tree = Tree("[bold]Pipeline[/bold]")
    split_node = tree.add(_node(dag["split"]))
    for f in range(n_folds):
        fb = split_node.add(f"[bold]fold {f}[/bold]")
        fb.add(_node(dag[f"train_C_f{f}"])).add(_node(dag[f"eval_C_f{f}"]))
        fb.add(_node(dag[f"train_P_f{f}"])).add(_node(dag[f"eval_P_f{f}"]))
    diag = tree.add(_node(dag["diagnose"]))
    for f in range(n_folds):
        diag.add(_node(dag[f"pfd_f{f}"]))
    tree.add(_node(dag["report"]))
    console.print(tree)


def _node(j: JobDef) -> str:
    """Format a job as a Rich-markup tree node."""
    icons = {
        State.DONE: "OK", State.PENDING: "o",
        State.RUNNING: ">", State.FAILED: "X",
    }
    colors = {
        State.DONE: "green", State.PENDING: "dim",
        State.RUNNING: "yellow", State.FAILED: "red",
    }
    return f"[{colors[j.state]}]{icons[j.state]} {j.name}[/{colors[j.state]}]"


# --- SLURM Generation -------------------------------------------------------

def generate_slurm_scripts(
    output_dir: Path,
    data_path: str,
    encoder: str,
    n_folds: int,
    partition: str = "gpu",
    time_limit: str = "04:00:00",
    mem: str = "32G",
    gpus: int = 1,
) -> None:
    """Generate SLURM batch scripts with dependency chains.

    Creates a ``slurm/`` subdirectory inside *output_dir* containing one
    ``.sbatch`` file per pipeline step and a ``submit_all.sh`` script that
    submits them with ``--dependency`` flags.

    Parameters
    ----------
    output_dir : Path
        Pipeline output directory.
    data_path : str
        Path to the CSV dataset.
    encoder : str
        HuggingFace model identifier.
    n_folds : int
        Number of cross-validation folds.
    partition : str
        SLURM partition for GPU jobs.
    time_limit : str
        Wall-clock time limit per job.
    mem : str
        Memory limit per job.
    gpus : int
        Number of GPUs per job.
    """
    slurm_dir = output_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    common = (
        f"--data {data_path} --output-dir {output_dir} "
        f"--encoder {encoder} --folds {n_folds}"
    )

    scripts = [
        ("01_split.sbatch", {
            "job-name": "ftd-split", "partition": "short",
            "time": "00:05:00", "mem": "4G",
            "command": f"fulltext-diagnostic run-job split {common}",
        }),
        ("02_train_C.sbatch", {
            "job-name": "ftd-trainC", "partition": partition,
            "time": time_limit, "mem": mem,
            "gres": f"gpu:{gpus}", "array": f"0-{n_folds-1}",
            "command": (
                f"fulltext-diagnostic run-job train_C "
                f"--fold $SLURM_ARRAY_TASK_ID {common}"
            ),
        }),
        ("03_train_P.sbatch", {
            "job-name": "ftd-trainP", "partition": partition,
            "time": time_limit, "mem": mem,
            "gres": f"gpu:{gpus}", "array": f"0-{n_folds-1}",
            "command": (
                f"fulltext-diagnostic run-job train_P "
                f"--fold $SLURM_ARRAY_TASK_ID {common}"
            ),
        }),
        ("04_eval.sbatch", {
            "job-name": "ftd-eval", "partition": partition,
            "time": "00:30:00", "mem": mem,
            "gres": f"gpu:{gpus}", "array": f"0-{n_folds-1}",
            "command": (
                f"fulltext-diagnostic run-job eval_C "
                f"--fold $SLURM_ARRAY_TASK_ID {common}\n"
                f"fulltext-diagnostic run-job eval_P "
                f"--fold $SLURM_ARRAY_TASK_ID {common}"
            ),
        }),
        ("05_diagnose.sbatch", {
            "job-name": "ftd-diagnose", "partition": "short",
            "time": "00:05:00", "mem": "4G",
            "command": f"fulltext-diagnostic run-job diagnose {common}",
        }),
        ("06_pfd.sbatch", {
            "job-name": "ftd-pfd", "partition": partition,
            "time": "01:00:00", "mem": mem,
            "gres": f"gpu:{gpus}", "array": f"0-{n_folds-1}",
            "command": (
                f"fulltext-diagnostic run-job pfd "
                f"--fold $SLURM_ARRAY_TASK_ID {common}"
            ),
        }),
    ]

    for fname, opts in scripts:
        _write_sbatch(slurm_dir / fname, opts, slurm_dir)

    submit_script = (
        "#!/bin/bash\nset -e\n"
        f"S=$(sbatch --parsable {slurm_dir}/01_split.sbatch)\n"
        f"C=$(sbatch --parsable --dependency=afterok:$S "
        f"{slurm_dir}/02_train_C.sbatch)\n"
        f"P=$(sbatch --parsable --dependency=afterok:$S "
        f"{slurm_dir}/03_train_P.sbatch)\n"
        f"E=$(sbatch --parsable --dependency=afterok:$C,afterok:$P "
        f"{slurm_dir}/04_eval.sbatch)\n"
        f"D=$(sbatch --parsable --dependency=afterok:$E "
        f"{slurm_dir}/05_diagnose.sbatch)\n"
        f"F=$(sbatch --parsable --dependency=afterok:$D "
        f"{slurm_dir}/06_pfd.sbatch)\n"
        'echo "Submitted: split=$S trainC=$C trainP=$P eval=$E '
        'diag=$D pfd=$F"\n'
    )
    (slurm_dir / "submit_all.sh").write_text(submit_script)
    (slurm_dir / "submit_all.sh").chmod(0o755)

    console.print(f"[bold]SLURM scripts -> {slurm_dir}/[/bold]")
    console.print(
        f"  One command: [green]bash {slurm_dir}/submit_all.sh[/green]"
    )


def _write_sbatch(path: Path, opts: dict, log_dir: Path) -> None:
    """Write a single SLURM batch script."""
    lines = ["#!/bin/bash"]
    for k in ["job-name", "partition", "time", "mem", "gres", "array"]:
        if k in opts:
            lines.append(f"#SBATCH --{k}={opts[k]}")
    lines.append(f"#SBATCH --output={log_dir}/%x_%A_%a.out")
    lines.append(f"#SBATCH --error={log_dir}/%x_%A_%a.err")
    lines.extend(["", "# source /path/to/venv/bin/activate", "",
                   opts["command"], ""])
    path.write_text("\n".join(lines))
