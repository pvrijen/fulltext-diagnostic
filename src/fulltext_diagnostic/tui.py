# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Interactive TUI (Terminal User Interface) for the fulltext-diagnostic pipeline.

Launch with ``ftd-monitor`` after installing the package. Provides four tabs:

  1. **Dashboard** -- Progress gauges, resource breakdown, next actions
  2. **Jobs** -- squeue-style table with state filters and drill-down
  3. **DAG** -- Dependency tree with live state
  4. **SLURM** -- Script generation, submission, squeue output

Auto-refreshes every 5 seconds by re-reading disk state markers.
"""

import json
import os
import subprocess
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import (
    Button, DataTable, Footer, Header, ProgressBar, RichLog,
    Rule, Static, TabbedContent, TabPane, Tree,
)
from rich.text import Text

from .pipeline import (
    STEP_LABELS, State, build_dag, generate_slurm_scripts,
    get_runnable, get_summary, load_state, reset_job,
)


CSS = """
Screen { background: $surface; }

#top-stats { height: 5; dock: top; background: $boost; padding: 0 1; }
.gauge { width: 1fr; height: 5; content-align: center middle; padding: 0 1; }
.gauge-done { background: $success 20%; }
.gauge-run  { background: $warning 20%; }
.gauge-fail { background: $error 20%; }
.gauge-pend { background: $surface; }

#progress { height: 1; margin: 0 1; }

.section-title { color: $text; text-style: bold; margin: 1 0 0 1; }

#next-panel { height: auto; max-height: 10; margin: 0 1; padding: 1;
              border: solid $primary; }
#config-panel { height: auto; max-height: 8; margin: 0 1; padding: 1;
                border: solid $accent; }

.filter-bar { height: 3; dock: top; padding: 0 1; }
.filter-bar Button { margin: 0 1; min-width: 10; }

#detail-panel { height: 14; margin: 0 1; padding: 1; border: solid $accent; }
#detail-actions { height: 3; margin: 0 1; }
#detail-actions Button { margin: 0 1; }

DataTable { height: 1fr; margin: 0 1; }
Tree { height: 1fr; margin: 0 1; }
RichLog { height: 1fr; margin: 0 1; border: solid $accent; }

#slurm-actions { height: 3; padding: 0 1; }
#slurm-actions Button { margin: 0 1; }
#slurm-info { height: auto; max-height: 6; margin: 0 1; padding: 1;
              border: solid $primary; }
"""


class PipelineMonitor(App):
    """Interactive pipeline monitor for the fulltext-diagnostic toolkit."""

    TITLE = "ftd-monitor"
    SUB_TITLE = "fulltext-diagnostic pipeline manager"
    CSS = CSS
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
        ("1", "tab('dashboard')", "Dashboard"),
        ("2", "tab('jobs')", "Jobs"),
        ("3", "tab('dag')", "DAG"),
        ("4", "tab('slurm')", "SLURM"),
    ]

    selected_job: reactive[str] = reactive("")

    def __init__(
        self,
        output_dir: str,
        n_folds: int = 5,
        data_path: str = "",
        encoder: str = "",
    ) -> None:
        super().__init__()
        self._output_dir = Path(output_dir)
        self._n_folds = n_folds
        self._data_path = data_path
        self._encoder = encoder
        self._dag: dict = {}
        self._current_filter: State | None = None

    def _load(self) -> None:
        """Re-read job states from disk."""
        self._dag = build_dag(self._n_folds)
        if self._output_dir.exists():
            self._dag = load_state(self._output_dir, self._dag)

    # --- Layout ---------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                yield from self._compose_dashboard()
            with TabPane("Jobs", id="jobs"):
                yield from self._compose_jobs()
            with TabPane("DAG", id="dag"):
                yield from self._compose_dag()
            with TabPane("SLURM", id="slurm"):
                yield from self._compose_slurm()
        yield Footer()

    # === TAB 1: DASHBOARD =====================================================

    def _compose_dashboard(self) -> ComposeResult:
        with Horizontal(id="top-stats"):
            yield Static("", id="g-done", classes="gauge gauge-done")
            yield Static("", id="g-run", classes="gauge gauge-run")
            yield Static("", id="g-fail", classes="gauge gauge-fail")
            yield Static("", id="g-pend", classes="gauge gauge-pend")
        yield ProgressBar(total=100, id="progress", show_eta=False)
        yield Static("[bold]Pipeline Overview[/bold]", classes="section-title")
        yield Static("", id="overview-text")
        yield Rule()
        yield Static("[bold]> Next Actions[/bold]", classes="section-title")
        yield Static("", id="next-panel")
        yield Rule()
        yield Static("[bold]Configuration[/bold]", classes="section-title")
        yield Static("", id="config-panel")

    def _refresh_dashboard(self) -> None:
        s = get_summary(self._dag)
        pct = s["done"] / s["total"] * 100 if s["total"] else 0

        self.query_one("#g-done", Static).update(
            f"[bold green]OK {s['done']}[/bold green]\nDone")
        self.query_one("#g-run", Static).update(
            f"[bold yellow]> {s['running']}[/bold yellow]\nRunning")
        self.query_one("#g-fail", Static).update(
            f"[bold red]X {s['failed']}[/bold red]\nFailed")
        self.query_one("#g-pend", Static).update(
            f"[dim]o {s['pending']}[/dim]\nPending")
        self.query_one("#progress", ProgressBar).update(progress=pct)

        gpu_remain = sum(
            j.est_minutes for j in self._dag.values()
            if j.state not in (State.DONE,) and j.resource == "gpu"
        )
        gpu_done = sum(
            j.est_minutes for j in self._dag.values()
            if j.state == State.DONE and j.resource == "gpu"
        )
        self.query_one("#overview-text", Static).update(
            f"  Progress: {s['done']}/{s['total']} ({pct:.0f}%)    "
            f"GPU done: ~{gpu_done}min ({gpu_done/60:.1f}h)    "
            f"GPU remaining: ~{gpu_remain}min "
            f"(seq: {gpu_remain/60:.1f}h, "
            f"{self._n_folds}-par: {gpu_remain/self._n_folds/60:.1f}h)"
        )

        runnable = get_runnable(self._dag)
        failed = [j for j in self._dag.values() if j.state == State.FAILED]

        lines: list[str] = []
        if s["done"] == s["total"]:
            lines.append("[green bold]Pipeline complete.[/green bold]")
            lines.append(
                f"  fulltext-diagnostic report "
                f"--results-dir {self._output_dir}"
            )
        elif failed:
            lines.append(f"[red bold]{len(failed)} job(s) failed:[/red bold]")
            for j in failed[:3]:
                lines.append(f"  [red]X {j.name}[/red]: {j.error[:60]}")
                lines.append(
                    f"    -> fulltext-diagnostic reset "
                    f"--output-dir {self._output_dir} {j.name}"
                )
        if runnable:
            lines.append(f"[green]{len(runnable)} job(s) ready:[/green]")
            for j in runnable[:4]:
                lines.append(
                    f"  fulltext-diagnostic run-job {j.cli_args} "
                    f"--data {self._data_path or 'DATA'} "
                    f"--output-dir {self._output_dir}"
                )
            if len(runnable) > 4:
                lines.append(f"  [dim]... +{len(runnable)-4} more[/dim]")
        elif not failed and s["done"] < s["total"]:
            lines.append("[yellow]Waiting for running jobs...[/yellow]")
        self.query_one("#next-panel", Static).update(
            "\n".join(lines) if lines else "(idle)"
        )

        cfg_path = self._output_dir / "config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            cfg_lines = [f"  {k}: [bold]{v}[/bold]" for k, v in cfg.items()]
            self.query_one("#config-panel", Static).update(
                "\n".join(cfg_lines)
            )
        else:
            self.query_one("#config-panel", Static).update(
                f"  [dim]No config.json in {self._output_dir}[/dim]\n"
                f"  Run: fulltext-diagnostic train --data DATA "
                f"--output-dir {self._output_dir}"
            )

    # === TAB 2: JOBS (squeue) =================================================

    def _compose_jobs(self) -> ComposeResult:
        with Horizontal(classes="filter-bar"):
            yield Button("All", id="f-all", variant="default")
            yield Button("Pending", id="f-pend", variant="default")
            yield Button("Running", id="f-run", variant="warning")
            yield Button("Done", id="f-done", variant="success")
            yield Button("Failed", id="f-fail", variant="error")
            yield Button("Refresh", id="f-refresh", variant="primary")
        yield DataTable(id="jobs-table", cursor_type="row")
        yield Static(
            "[dim]Select a job for details[/dim]", id="detail-panel"
        )
        with Horizontal(id="detail-actions"):
            yield Button("Reset Job", id="act-reset", variant="error",
                         disabled=True)
            yield Button("Run Job", id="act-run", variant="success",
                         disabled=True)
            yield Button("View Output", id="act-output", variant="default",
                         disabled=True)

    def _refresh_jobs(self, state_filter: State | None = None) -> None:
        self._current_filter = state_filter
        table = self.query_one("#jobs-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Step", "Job", "Fold", "Res", "~Min", "State",
                          "Ready")

        state_fmt = {
            State.DONE: Text("DONE", style="bold green"),
            State.RUNNING: Text("RUNNING", style="bold yellow"),
            State.FAILED: Text("FAILED", style="bold red"),
            State.PENDING: Text("PENDING", style="dim"),
        }

        for name, job in self._dag.items():
            if state_filter and job.state != state_filter:
                continue
            ready = all(
                self._dag[d].state == State.DONE for d in job.depends_on
            )
            table.add_row(
                STEP_LABELS.get(job.job_type, "?"),
                name,
                str(job.fold) if job.fold is not None else "--",
                Text(job.resource,
                     style="cyan" if job.resource == "gpu" else "dim"),
                str(job.est_minutes),
                state_fmt[job.state],
                Text("OK yes", style="green") if ready
                else Text("X no", style="red"),
                key=name,
            )

    def _show_detail(self, job_name: str) -> None:
        if job_name not in self._dag:
            return
        self.selected_job = job_name
        job = self._dag[job_name]

        dep_lines = []
        for d in job.depends_on:
            ds = self._dag[d].state
            icon = ("OK" if ds == State.DONE
                    else "X" if ds == State.FAILED else "o")
            dep_lines.append(f"{icon} {d}")
        deps_str = ", ".join(dep_lines) if dep_lines else "(none)"

        lines = [
            f"[bold]{job.name}[/bold]  |  type: {job.job_type}  |  "
            f"fold: {job.fold if job.fold is not None else '--'}  |  "
            f"resource: {job.resource}  |  est: {job.est_minutes}min",
            f"State: [{'green' if job.state==State.DONE else 'red' if job.state==State.FAILED else 'yellow' if job.state==State.RUNNING else 'dim'}]{job.state.value}[/]",
            f"Dependencies: {deps_str}",
        ]

        if job.state == State.FAILED:
            lines.append(f"[red]Error: {job.error}[/red]")
            lines.append("[yellow]-> Reset this job, then rerun[/yellow]")
        elif job.state == State.DONE:
            cache = self._output_dir / "cache"
            outputs = []
            for pat in [f"C_f{job.fold}.npz", f"P_f{job.fold}.npz",
                        f"PFD_f{job.fold}.npz"]:
                p = cache / pat
                if p.exists():
                    outputs.append(
                        f"{p.name} ({p.stat().st_size//1024}KB)"
                    )
            model_dir = self._output_dir / (
                f"models/{job.job_type.replace('train_', '')}_f{job.fold}"
            )
            if model_dir.exists():
                outputs.append(f"model -> {model_dir.name}/")
            if outputs:
                lines.append(f"Outputs: {', '.join(outputs)}")
        elif job.state == State.PENDING:
            ready = all(
                self._dag[d].state == State.DONE for d in job.depends_on
            )
            if ready:
                lines.append("[green]Ready to run[/green]")
            else:
                blocked = [d for d in job.depends_on
                           if self._dag[d].state != State.DONE]
                lines.append(f"[dim]Blocked by: {', '.join(blocked)}[/dim]")

        self.query_one("#detail-panel", Static).update("\n".join(lines))

        self.query_one("#act-reset", Button).disabled = (
            job.state not in (State.FAILED, State.DONE)
        )
        ready = (
            job.state == State.PENDING
            and all(self._dag[d].state == State.DONE for d in job.depends_on)
        )
        self.query_one("#act-run", Button).disabled = not ready
        self.query_one("#act-output", Button).disabled = (
            job.state != State.DONE
        )

    # === TAB 3: DAG ===========================================================

    def _compose_dag(self) -> ComposeResult:
        yield Tree("Pipeline DAG", id="dag-tree")

    def _refresh_dag_tree(self) -> None:
        tree = self.query_one("#dag-tree", Tree)
        tree.clear()
        tree.root.expand()

        icons = {
            State.DONE: "OK", State.PENDING: "o",
            State.RUNNING: ">", State.FAILED: "X",
        }
        colors = {
            State.DONE: "green", State.PENDING: "dim",
            State.RUNNING: "yellow", State.FAILED: "red",
        }

        def _n(j):
            return (
                f"[{colors[j.state]}]{icons[j.state]} {j.name} "
                f"({j.resource}, ~{j.est_minutes}m)[/{colors[j.state]}]"
            )

        s = self._dag["split"]
        sn = tree.root.add(_n(s), expand=True, data="split")

        for f in range(self._n_folds):
            fn = sn.add(f"[bold]Fold {f}[/bold]", expand=True)
            tc = self._dag[f"train_C_f{f}"]
            ec = self._dag[f"eval_C_f{f}"]
            tcn = fn.add(_n(tc), data=f"train_C_f{f}")
            tcn.add_leaf(_n(ec), data=f"eval_C_f{f}")
            tp = self._dag[f"train_P_f{f}"]
            ep = self._dag[f"eval_P_f{f}"]
            tpn = fn.add(_n(tp), data=f"train_P_f{f}")
            tpn.add_leaf(_n(ep), data=f"eval_P_f{f}")

        d = self._dag["diagnose"]
        dn = tree.root.add(_n(d), expand=True, data="diagnose")
        for f in range(self._n_folds):
            pfd_job = self._dag[f"pfd_f{f}"]
            dn.add_leaf(_n(pfd_job), data=f"pfd_f{f}")

        rpt = self._dag["report"]
        tree.root.add_leaf(_n(rpt), data="report")

    # === TAB 4: SLURM =========================================================

    def _compose_slurm(self) -> ComposeResult:
        with Horizontal(id="slurm-actions"):
            yield Button("Generate Scripts", id="sl-gen", variant="primary")
            yield Button("Check squeue", id="sl-squeue", variant="default")
            yield Button("Show submit_all.sh", id="sl-show", variant="default")
        yield Static("", id="slurm-info")
        yield RichLog(id="slurm-log", highlight=True, markup=True)

    def _refresh_slurm_info(self) -> None:
        slurm_dir = self._output_dir / "slurm"
        if slurm_dir.exists():
            scripts = sorted(slurm_dir.glob("*.sbatch"))
            self.query_one("#slurm-info", Static).update(
                f"[bold]SLURM directory:[/bold] {slurm_dir}\n"
                f"[bold]Scripts:[/bold] {len(scripts)} "
                f"({', '.join(s.name for s in scripts)})\n"
                f"[bold]Submit:[/bold] bash {slurm_dir}/submit_all.sh"
            )
        else:
            self.query_one("#slurm-info", Static).update(
                "[dim]No SLURM scripts generated yet. "
                "Click 'Generate Scripts'.[/dim]"
            )

    # --- Event Handlers -------------------------------------------------------

    def on_mount(self) -> None:
        self._load()
        self._refresh_all()
        self.set_interval(5, self._auto_refresh)

    def _auto_refresh(self) -> None:
        self._load()
        self._refresh_dashboard()

    def _refresh_all(self) -> None:
        self._load()
        self._refresh_dashboard()
        self._refresh_jobs(self._current_filter)
        self._refresh_dag_tree()
        self._refresh_slurm_info()

    def action_refresh(self) -> None:
        self._refresh_all()
        self.notify("Refreshed", timeout=2)

    def action_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    # Filter buttons
    @on(Button.Pressed, "#f-all")
    def _f_all(self) -> None:
        self._refresh_jobs()

    @on(Button.Pressed, "#f-pend")
    def _f_pend(self) -> None:
        self._refresh_jobs(State.PENDING)

    @on(Button.Pressed, "#f-run")
    def _f_run(self) -> None:
        self._refresh_jobs(State.RUNNING)

    @on(Button.Pressed, "#f-done")
    def _f_done(self) -> None:
        self._refresh_jobs(State.DONE)

    @on(Button.Pressed, "#f-fail")
    def _f_fail(self) -> None:
        self._refresh_jobs(State.FAILED)

    @on(Button.Pressed, "#f-refresh")
    def _f_refresh(self) -> None:
        self._refresh_all()

    # Job table drill-down
    @on(DataTable.RowSelected, "#jobs-table")
    def _job_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self._show_detail(str(event.row_key.value))

    # DAG tree drill-down
    @on(Tree.NodeSelected, "#dag-tree")
    def _dag_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data and event.node.data in self._dag:
            self.action_tab("jobs")
            self._show_detail(str(event.node.data))

    # Action buttons
    @on(Button.Pressed, "#act-reset")
    def _act_reset(self) -> None:
        if self.selected_job:
            reset_job(self._output_dir, self.selected_job)
            self.notify(f"Reset: {self.selected_job}", timeout=3)
            self._refresh_all()
            self._show_detail(self.selected_job)

    @on(Button.Pressed, "#act-run")
    def _act_run(self) -> None:
        if self.selected_job and self.selected_job in self._dag:
            job = self._dag[self.selected_job]
            cmd = (
                f"fulltext-diagnostic run-job {job.cli_args} "
                f"--data {self._data_path or 'DATA'} "
                f"--output-dir {self._output_dir}"
            )
            self.notify(f"Command copied: {cmd}", timeout=5)

    @on(Button.Pressed, "#act-output")
    def _act_output(self) -> None:
        if self.selected_job and self.selected_job in self._dag:
            job = self._dag[self.selected_job]
            cache = self._output_dir / "cache"
            files = []
            for pat in [f"C_f{job.fold}.npz", f"P_f{job.fold}.npz",
                        f"PFD_f{job.fold}.npz",
                        f"../diagnostic_results.csv",
                        f"../pfd_results.csv"]:
                p = cache / pat
                if p.exists():
                    files.append(str(p))
            if files:
                self.query_one("#detail-panel", Static).update(
                    f"[bold]Output files for {self.selected_job}:[/bold]\n"
                    + "\n".join(f"  {f}" for f in files)
                )

    # SLURM actions
    @on(Button.Pressed, "#sl-gen")
    def _sl_gen(self) -> None:
        log = self.query_one("#slurm-log", RichLog)
        try:
            generate_slurm_scripts(
                self._output_dir,
                self._data_path or "DATA.csv",
                self._encoder or "BioMedBERT",
                self._n_folds,
            )
            log.write("[green]Scripts generated successfully.[/green]")
            self._refresh_slurm_info()
        except Exception as e:
            log.write(f"[red]Error: {e}[/red]")

    @on(Button.Pressed, "#sl-squeue")
    def _sl_squeue(self) -> None:
        log = self.query_one("#slurm-log", RichLog)
        try:
            r = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""),
                 "--format=%.10i %.12P %.25j %.8T %.10M %.4D"],
                capture_output=True, text=True, timeout=5,
            )
            log.write("[bold]squeue output:[/bold]")
            log.write(
                r.stdout if r.stdout.strip()
                else "[dim](no active jobs)[/dim]"
            )
        except FileNotFoundError:
            log.write(
                "[yellow]squeue not available (not on SLURM cluster)[/yellow]"
            )
        except Exception as e:
            log.write(f"[red]{e}[/red]")

    @on(Button.Pressed, "#sl-show")
    def _sl_show(self) -> None:
        log = self.query_one("#slurm-log", RichLog)
        submit = self._output_dir / "slurm" / "submit_all.sh"
        if submit.exists():
            log.write(f"[bold]-- {submit} --[/bold]")
            log.write(submit.read_text())
        else:
            log.write(
                "[dim]No submit_all.sh found. Generate scripts first.[/dim]"
            )


def run() -> None:
    """Entry point for the ``ftd-monitor`` command."""
    import click

    @click.command()
    @click.option("--output-dir", default="ft_diagnostic_output",
                  help="Pipeline output directory")
    @click.option("--folds", default=5, type=int, help="Number of folds")
    @click.option("--data", default="",
                  help="Dataset path (for SLURM generation)")
    @click.option("--encoder", default="",
                  help="Encoder name (for SLURM generation)")
    def _run(output_dir: str, folds: int, data: str, encoder: str) -> None:
        """Interactive pipeline monitor."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        app = PipelineMonitor(output_dir, folds, data, encoder)
        app.run()

    _run()
