"""Smoke and unit tests for the diagnostic pipeline.

Runs the full train -> diagnose -> pfd -> report chain on a tiny synthetic
dataset with a 2-layer BERT on CPU, and unit-tests the pieces that do not
need a model (PFD loss, decomposition statistics, CLI wiring).
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

TINY = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a minimal CSV with 40 articles (8 positive, 32 negative)."""
    np.random.seed(42)
    n = 40
    labels = [1] * 8 + [0] * 32
    abstracts = [f"Abstract text for article {i}." for i in range(n)]
    fulltexts = [
        f"Abstract text for article {i}. Introduction body text with more "
        f"detail about article {i}. Methods section. Results section."
        for i in range(n)
    ]
    df = pd.DataFrame({
        "id": range(n),
        "text_abstract": abstracts,
        "text_fulltext": fulltexts,
        "label": labels,
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d


# --- Unit tests (no model) ---------------------------------------------------

def test_pfd_loss_alpha_semantics():
    """alpha weights the soft term: alpha=0 is pure BCE, alpha=1 pure KD."""
    from fulltext_diagnostic.pfd import pfd_loss

    student = torch.tensor([2.0, -1.0, 0.5])
    teacher = torch.tensor([1.5, -2.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 1.0])

    hard_only = pfd_loss(student, teacher, labels, alpha=0.0, temperature=4.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(student, labels)
    assert torch.isclose(hard_only, bce)

    soft_only = pfd_loss(student, teacher, labels, alpha=1.0, temperature=4.0)
    soft = torch.nn.functional.binary_cross_entropy(
        torch.sigmoid(student / 4.0), torch.sigmoid(teacher / 4.0))
    assert torch.isclose(soft_only, 16.0 * soft)

    # Student matching teacher exactly gives the minimal soft loss
    same = pfd_loss(teacher, teacher, labels, alpha=1.0, temperature=4.0)
    assert same <= soft_only


def test_safe_wilcoxon_small_n_is_nan():
    from fulltext_diagnostic.diagnose import _safe_wilcoxon

    assert np.isnan(_safe_wilcoxon(np.array([0.1, 0.2, -0.1, 0.05, 0.3])))
    assert np.isnan(_safe_wilcoxon(np.zeros(10)))
    p = _safe_wilcoxon(np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.12, 0.2, 0.1]))
    assert 0.0 < p < 0.05


def test_run_diagnosis_from_csv(tmp_path):
    """run_diagnosis consumes the columns train.py writes."""
    from fulltext_diagnostic.diagnose import run_diagnosis
    from fulltext_diagnostic.train import RESULT_COLUMNS

    rows = []
    for f in range(6):
        c, p0, p100 = 0.70 + 0.01 * f, 0.65 + 0.01 * f, 0.72 + 0.01 * f
        rows.append({"fold": f, "C": c, "P@0%": p0, "P@100%": p100,
                     "TE": p0 - c, "IE": p100 - p0, "Total": p100 - c})
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
        tmp_path / "diagnostic_results.csv", index=False)

    run_diagnosis(tmp_path)
    rep = json.load(open(tmp_path / "diagnostic_report.json"))
    assert rep["recommendation"] == "pfd"
    assert rep["n_folds"] == 6
    assert rep["TE"]["p"] is not None and rep["TE"]["p"] < 0.05
    np.testing.assert_allclose(rep["TE"]["mean"], -0.05, atol=1e-9)
    np.testing.assert_allclose(rep["IE"]["mean"], 0.07, atol=1e-9)


def test_cli_help_and_run_job_choices():
    from fulltext_diagnostic.cli import JOB_TYPES, main

    runner = CliRunner()
    assert runner.invoke(main, ["--help"]).exit_code == 0
    for cmd in ["train", "diagnose", "pfd", "report", "status", "dag",
                "slurm", "reset", "run-job"]:
        res = runner.invoke(main, [cmd, "--help"])
        assert res.exit_code == 0, cmd
    assert "pfd" in JOB_TYPES and "report" in JOB_TYPES
    res = runner.invoke(main, ["pfd", "--help"])
    assert "Soft-target weight" in res.output


def test_split_rejects_missing_column(tmp_path, output_dir):
    from fulltext_diagnostic.train import job_split_folds

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1, 2], "text_abstract": ["a", "b"],
                  "label": [0, 1]}).to_csv(bad, index=False)
    assert job_split_folds(bad, output_dir, n_folds=2, seed=42) is False
    assert (output_dir / ".failed_split").exists()
    assert "text_fulltext" in (output_dir / ".failed_split").read_text()


# --- Smoke tests (tiny model, CPU) ------------------------------------------

def test_split_folds(tiny_dataset, output_dir):
    from fulltext_diagnostic.train import job_split_folds, load_folds

    assert job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    folds = load_folds(output_dir)
    assert len(folds) == 2
    for fi in folds.values():
        assert set(fi["train"]) | set(fi["val"]) == set(fi["train_all"])
        assert not (set(fi["train"]) & set(fi["val"]))
        assert not (set(fi["train_all"]) & set(fi["test"]))
    all_idx = set(folds[0]["test"]) | set(folds[1]["test"])
    assert len(all_idx) == 40
    # Stratified validation split keeps at least one positive
    labels = pd.read_csv(tiny_dataset)["label"].values
    assert labels[folds[0]["val"]].sum() >= 1


def test_train_C_runs(tiny_dataset, output_dir):
    from fulltext_diagnostic.train import job_split_folds, job_train_C

    job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    assert job_train_C(
        fold=0, data_path=tiny_dataset, output_dir=output_dir,
        encoder=TINY, max_length=64, epochs=2, batch_size=8, lr=5e-4,
        device="cpu",
    )
    assert (output_dir / "models/C_f0").exists()


def test_train_P_respects_chunk_options(tiny_dataset, output_dir):
    from fulltext_diagnostic.train import job_split_folds, job_train_P

    job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    assert job_train_P(
        fold=0, data_path=tiny_dataset, output_dir=output_dir,
        encoder=TINY, epochs=2, batch_size=4, lr=5e-4, device="cpu",
        n_chunks=3, chunk_size=62,
    )
    model_dir = output_dir / "models/P_f0"
    assert (model_dir / "encoder.pt").exists()
    cfg = json.load(open(model_dir / "config.json"))
    assert cfg["chunk_size"] == 62
    assert cfg["max_chunks"] == 3
    assert cfg["pooling"] == "mean"


def _run_pipeline(tiny_dataset, output_dir, n_folds=2):
    from fulltext_diagnostic.train import (
        job_split_folds, job_train_C, job_train_P,
        job_eval_C, job_eval_P, job_diagnose,
    )
    assert job_split_folds(tiny_dataset, output_dir, n_folds=n_folds, seed=42)
    for fold in range(n_folds):
        assert job_train_C(fold, tiny_dataset, output_dir, TINY,
                           64, 2, 8, 5e-4, "cpu")
        assert job_train_P(fold, tiny_dataset, output_dir, TINY,
                           2, 4, 5e-4, "cpu", n_chunks=3, chunk_size=62)
        assert job_eval_C(fold, tiny_dataset, output_dir, 64, 8, "cpu")
        assert job_eval_P(fold, tiny_dataset, output_dir, "cpu")
    assert job_diagnose(output_dir, n_folds=n_folds)


def test_end_to_end_train_diagnose_pfd_report(tiny_dataset, output_dir):
    """Full chain on CPU. Guards the CSV schema shared by all stages."""
    from fulltext_diagnostic.pfd import job_pfd, summarize_pfd
    from fulltext_diagnostic.report import generate_report
    from fulltext_diagnostic.train import RESULT_COLUMNS

    _run_pipeline(tiny_dataset, output_dir)

    df = pd.read_csv(output_dir / "diagnostic_results.csv")
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 2
    np.testing.assert_allclose(df["TE"] + df["IE"], df["Total"], atol=1e-6)
    np.testing.assert_allclose(df["P@0%"] - df["C"], df["TE"], atol=1e-6)
    assert (output_dir / "diagnostic_report.json").exists()
    assert not (output_dir / ".failed_diagnose").exists()

    # PFD warm-starts from the saved C and writes a per-fold cache
    for fold in range(2):
        assert job_pfd(fold, output_dir, tiny_dataset, TINY,
                       alpha=0.1, temperature=4.0, epochs=1, lr=5e-5,
                       device="cpu")
        assert (output_dir / f"cache/PFD_f{fold}.npz").exists()
    rdf = summarize_pfd(output_dir, 2)
    assert list(rdf.columns) == ["fold", "C", "C_PFD", "delta"]
    np.testing.assert_allclose(rdf["C_PFD"] - rdf["C"], rdf["delta"],
                               atol=1e-6)

    # Report renders from both CSVs and marks itself done
    generate_report(output_dir)
    assert (output_dir / ".done_report").exists()
