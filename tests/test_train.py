"""Smoke tests for the training pipeline.

Verifies that job_train_C, job_train_P, and job_eval_P run without errors
on a tiny synthetic dataset. Uses CPU to avoid GPU requirements in CI.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


def test_split_folds(tiny_dataset, output_dir):
    from fulltext_diagnostic.train import job_split_folds, load_folds

    job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    folds = load_folds(output_dir)
    assert len(folds) == 2
    assert "train" in folds[0]
    assert "val" in folds[0]
    assert "test" in folds[0]
    # All indices accounted for
    all_idx = set(folds[0]["test"]) | set(folds[1]["test"])
    assert len(all_idx) == 40


def test_train_C_runs(tiny_dataset, output_dir):
    """Smoke test: train_C completes on CPU with a tiny model."""
    from fulltext_diagnostic.train import job_split_folds, job_train_C

    job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    # Use a tiny model for speed
    job_train_C(
        fold=0, data_path=tiny_dataset, output_dir=output_dir,
        encoder="google/bert_uncased_L-2_H-128_A-2", max_length=64,
        epochs=2, batch_size=8, lr=5e-4, device="cpu",
    )
    model_dir = output_dir / "models/C_f0"
    assert model_dir.exists()


def test_train_P_runs(tiny_dataset, output_dir):
    """Smoke test: train_P with chunk-mean pooling completes on CPU."""
    from fulltext_diagnostic.train import job_split_folds, job_train_P

    job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    job_train_P(
        fold=0, data_path=tiny_dataset, output_dir=output_dir,
        encoder="google/bert_uncased_L-2_H-128_A-2", max_length=64,
        epochs=2, batch_size=4, lr=5e-4, device="cpu",
    )
    model_dir = output_dir / "models/P_f0"
    assert model_dir.exists()
    assert (model_dir / "encoder.pt").exists()
    assert (model_dir / "config.json").exists()
    cfg = json.load(open(model_dir / "config.json"))
    assert cfg["chunk_size"] == 510
    assert cfg["pooling"] == "mean"


def test_eval_P_runs(tiny_dataset, output_dir):
    """Smoke test: eval_P produces L_ft→abs and L_ft→ft scores."""
    from fulltext_diagnostic.train import (
        job_split_folds, job_train_P, job_eval_P,
    )

    job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    job_train_P(
        fold=0, data_path=tiny_dataset, output_dir=output_dir,
        encoder="google/bert_uncased_L-2_H-128_A-2", max_length=64,
        epochs=2, batch_size=4, lr=5e-4, device="cpu",
    )
    job_eval_P(
        fold=0, data_path=tiny_dataset, output_dir=output_dir,
        max_length=64, batch_size=8, device="cpu",
    )
    cache = output_dir / "cache" / "P_f0.npz"
    assert cache.exists()
    data = np.load(cache)
    assert "p0_prauc" in data
    assert "p100_prauc" in data
    assert "teacher_logits" in data
    # Both should be valid PR-AUC values
    assert 0.0 <= float(data["p0_prauc"]) <= 1.0
    assert 0.0 <= float(data["p100_prauc"]) <= 1.0


def test_diagnose_runs(tiny_dataset, output_dir):
    """Smoke test: full pipeline through diagnosis produces CSV."""
    from fulltext_diagnostic.train import (
        job_split_folds, job_train_C, job_train_P,
        job_eval_C, job_eval_P, job_diagnose,
    )

    job_split_folds(tiny_dataset, output_dir, n_folds=2, seed=42)
    for fold in range(2):
        job_train_C(fold, tiny_dataset, output_dir,
                    "google/bert_uncased_L-2_H-128_A-2", 64, 2, 8, 5e-4, "cpu")
        job_train_P(fold, tiny_dataset, output_dir,
                    "google/bert_uncased_L-2_H-128_A-2", 64, 2, 4, 5e-4, "cpu")
        job_eval_C(fold, tiny_dataset, output_dir, 64, 8, "cpu")
        job_eval_P(fold, tiny_dataset, output_dir, 64, 8, "cpu")

    job_diagnose(output_dir, n_folds=2)
    results = output_dir / "diagnostic_results.csv"
    assert results.exists()
    df = pd.read_csv(results)
    assert len(df) == 2
    assert "TE" in df.columns
    assert "IE" in df.columns
    # TE + IE = Total (algebraic identity)
    np.testing.assert_allclose(
        df["TE"] + df["IE"], df["Total"], atol=1e-6,
    )
