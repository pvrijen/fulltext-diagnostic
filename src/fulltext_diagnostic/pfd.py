# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Privileged Features Distillation (PFD): transfer full-text knowledge to an
abstract-only student model.

PFD trains a student on abstract text using a combination of hard labels and
soft targets from a full-text teacher. The teacher's logits are cached during
the ``train`` step, so PFD requires no additional GPU time for teacher
inference. Each fold is a separate resumable job.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rich.console import Console

from .pipeline import build_dag, load_state, reset_job, run_guarded, show_status

console = Console()


def pfd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    alpha: float = 0.1,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Compute the PFD distillation loss.

    Combines binary cross-entropy on hard labels with a soft-target loss
    scaled by temperature squared.

    Parameters
    ----------
    student_logits : Tensor
        Raw logits from the student model.
    teacher_logits : Tensor
        Cached logits from the full-text teacher.
    hard_labels : Tensor
        Ground-truth binary labels.
    alpha : float
        Weight for the hard-label loss. The soft loss receives weight
        ``(1 - alpha) * temperature**2``.
    temperature : float
        Temperature for softening probability distributions.

    Returns
    -------
    Tensor
        Scalar loss value.
    """
    hard = torch.nn.functional.binary_cross_entropy_with_logits(
        student_logits, hard_labels
    )
    soft_s = torch.sigmoid(student_logits / temperature)
    soft_t = torch.sigmoid(teacher_logits / temperature)
    soft = torch.nn.functional.binary_cross_entropy(soft_s, soft_t)
    return (1 - alpha) * hard + alpha * temperature ** 2 * soft


def _run_pfd_fold(
    fold: int,
    results_dir: Path,
    data_path: Path,
    encoder_name: str,
    alpha: float,
    temperature: float,
    epochs: int,
    lr: float,
    device: str,
) -> None:
    """Run PFD for a single cross-validation fold."""
    from .train import load_folds

    fi = load_folds(results_dir)[fold]
    df = pd.read_csv(data_path)
    abstracts = df["text_abstract"].fillna("").tolist()
    labels = df["label"].values

    cache = results_dir / "cache"
    c_data = np.load(cache / f"C_f{fold}.npz")
    p_data = np.load(cache / f"P_f{fold}.npz")
    teacher_logits = p_data["teacher_logits"]

    tok = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_name, num_labels=1
    ).to(device)

    train_idx = fi["train_all"]
    train_enc = tok(
        [abstracts[i] for i in train_idx],
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    train_ds = TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        torch.tensor(labels[train_idx], dtype=torch.float32),
        torch.tensor(teacher_logits, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for ids, mask, hard, teach in train_loader:
            out = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            )
            loss = pfd_loss(
                out.logits.squeeze(), teach.to(device), hard.to(device),
                alpha, temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Evaluate on test abstracts
    test_idx = fi["test"]
    test_enc = tok(
        [abstracts[i] for i in test_idx],
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    test_ds = TensorDataset(
        test_enc["input_ids"],
        test_enc["attention_mask"],
        torch.tensor(labels[test_idx]),
    )
    test_loader = DataLoader(test_ds, batch_size=16)

    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for ids, mask, lab in test_loader:
            out = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            )
            all_logits.extend(out.logits.squeeze().cpu().numpy())
            all_labels.extend(lab.numpy())

    probs = 1 / (1 + np.exp(-np.array(all_logits)))
    pfd_prauc = average_precision_score(all_labels, probs)
    c_prauc = average_precision_score(c_data["labels"], c_data["probs"])
    delta = pfd_prauc - c_prauc

    np.savez(
        cache / f"PFD_f{fold}.npz",
        logits=np.array(all_logits), probs=probs,
        labels=np.array(all_labels), test_idx=np.array(test_idx),
        prauc=pfd_prauc, c_prauc=c_prauc, delta=delta,
    )

    console.print(
        f"  pfd_f{fold}: C_PFD={pfd_prauc:.3f} "
        f"(C={c_prauc:.3f}, delta={delta:+.3f})"
    )
    del model
    torch.cuda.empty_cache()


def run_pfd(
    results_dir: Path,
    data_path: Path,
    encoder_name: str,
    alpha: float,
    temperature: float,
    epochs: int,
    lr: float,
    device: str,
) -> None:
    """Run PFD for all folds. Each fold is a resumable job.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``config.json`` and the ``cache/`` subdirectory
        with teacher logits from the ``train`` step.
    data_path : Path
        Path to the CSV dataset.
    encoder_name : str
        HuggingFace model identifier for the student encoder.
    alpha : float
        Hard-label weight in PFD loss.
    temperature : float
        Distillation temperature.
    epochs : int
        Number of PFD training epochs.
    lr : float
        Learning rate for the PFD student.
    device : str
        PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
    """
    console.print("[bold]fulltext-diagnostic pfd[/bold]")
    console.print(f"  alpha={alpha}, T={temperature}, epochs={epochs}")

    config = json.load(open(results_dir / "config.json"))
    n_folds = config["folds"]

    for fold in range(n_folds):
        run_guarded(
            results_dir, f"pfd_f{fold}",
            lambda f=fold: _run_pfd_fold(
                f, results_dir, data_path, encoder_name,
                alpha, temperature, epochs, lr, device,
            ),
        )

    # Summary
    cache = results_dir / "cache"
    rows = []
    for f in range(n_folds):
        pf = cache / f"PFD_f{f}.npz"
        if pf.exists():
            d = np.load(pf)
            rows.append({
                "fold": f,
                "C": float(d["c_prauc"]),
                "C_PFD": float(d["prauc"]),
                "delta": float(d["delta"]),
            })

    if rows:
        rdf = pd.DataFrame(rows)
        rdf.to_csv(results_dir / "pfd_results.csv", index=False)
        console.print("\n[bold]PFD Summary[/bold]")
        console.print(f"  C:     {rdf['C'].mean():.3f} +/- {rdf['C'].std():.3f}")
        console.print(
            f"  C_PFD: {rdf['C_PFD'].mean():.3f} +/- {rdf['C_PFD'].std():.3f}"
        )
        console.print(f"  delta: {rdf['delta'].mean():+.3f}")

    # Mark report as pending (needs rerun after PFD)
    reset_job(results_dir, "report")

    # Show pipeline status
    console.print()
    dag = build_dag(n_folds)
    dag = load_state(results_dir, dag)
    show_status(dag, n_folds)
