# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Privileged Features Distillation (PFD): transfer full-text knowledge to an
abstract-only student model.

PFD warm-starts the student from the converged abstract baseline C and
fine-tunes it for a few epochs on abstract text with a combination of hard
labels and soft targets from the full-text teacher P. The teacher's logits
are cached during the ``train`` step, so PFD requires no additional GPU time
for teacher inference. Each fold is a separate resumable job.

The student is trained on the same articles as C (the ``train`` split) and
early-stopped on the same validation split, so the C vs C_PFD comparison is
data-matched and isolates the effect of distillation.
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

PFD_BATCH_SIZE = 32
PFD_WEIGHT_DECAY = 0.01
PFD_PATIENCE = 2


def pfd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    alpha: float = 0.1,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Compute the PFD distillation loss.

    ``(1 - alpha) * BCE(student, labels)
    + alpha * T^2 * BCE(sigmoid(student / T), sigmoid(teacher / T))``

    Parameters
    ----------
    student_logits : Tensor
        Raw logits from the student model.
    teacher_logits : Tensor
        Cached logits from the full-text teacher.
    hard_labels : Tensor
        Ground-truth binary labels.
    alpha : float
        Weight of the soft-target (distillation) term. 0 = hard labels only,
        1 = soft targets only. The paper uses 0.1.
    temperature : float
        Temperature for softening both logit distributions. The paper uses 4.

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


def _make_pfd_loader(tok, texts, labels, teacher_logits, shuffle):
    enc = tok(texts, padding=True, truncation=True, max_length=512,
              return_tensors="pt")
    ds = TensorDataset(
        enc["input_ids"], enc["attention_mask"],
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(teacher_logits, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=PFD_BATCH_SIZE, shuffle=shuffle)


def _epoch_loss(model, loader, device, alpha, temperature) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ids, mask, hard, teach in loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            loss = pfd_loss(out.logits.squeeze(-1), teach.to(device),
                            hard.to(device), alpha, temperature)
            total += float(loss) * len(hard)
            n += len(hard)
    return total / max(1, n)


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
    from .train import _read_seed, load_folds, set_seed

    set_seed(_read_seed(results_dir) + fold)
    fi = load_folds(results_dir)[fold]
    df = pd.read_csv(data_path)
    abstracts = df["text_abstract"].fillna("").tolist()
    labels = df["label"].values

    cache = results_dir / "cache"
    c_data = np.load(cache / f"C_f{fold}.npz")
    p_data = np.load(cache / f"P_f{fold}.npz")
    # Teacher logits were cached over train_all; map article index -> logit.
    teacher_by_idx = dict(zip(p_data["train_all_idx"].tolist(),
                              p_data["teacher_logits"].tolist()))

    # Warm start from the converged abstract baseline (paper protocol).
    student_dir = results_dir / f"models/C_f{fold}"
    if student_dir.exists():
        source = student_dir
    else:
        console.print(
            f"  [yellow]pfd_f{fold}: no C checkpoint at {student_dir}; "
            f"cold-starting from {encoder_name}. The paper warm-starts; "
            f"cold-start results are not comparable.[/yellow]"
        )
        source = encoder_name
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=1
    ).to(device)

    train_idx = fi["train"]
    val_idx = fi["val"]
    train_loader = _make_pfd_loader(
        tok, [abstracts[i] for i in train_idx], labels[train_idx],
        [teacher_by_idx[i] for i in train_idx], shuffle=True,
    )
    val_loader = _make_pfd_loader(
        tok, [abstracts[i] for i in val_idx], labels[val_idx],
        [teacher_by_idx[i] for i in val_idx], shuffle=False,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=PFD_WEIGHT_DECAY)
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for ids, mask, hard, teach in train_loader:
            out = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            )
            loss = pfd_loss(
                out.logits.squeeze(-1), teach.to(device), hard.to(device),
                alpha, temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        vloss = _epoch_loss(model, val_loader, device, alpha, temperature)
        if vloss < best_loss:
            best_loss, wait = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PFD_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test abstracts
    test_idx = fi["test"]
    test_enc = tok(
        [abstracts[i] for i in test_idx],
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    test_loader = DataLoader(
        TensorDataset(test_enc["input_ids"], test_enc["attention_mask"],
                      torch.tensor(labels[test_idx])),
        batch_size=PFD_BATCH_SIZE,
    )

    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for ids, mask, lab in test_loader:
            out = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            )
            all_logits.extend(out.logits.squeeze(-1).cpu().numpy())
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


def job_pfd(
    fold: int, results_dir: Path, data_path: Path, encoder_name: str,
    alpha: float = 0.1, temperature: float = 4.0, epochs: int = 3,
    lr: float = 5e-6, device: str = "cuda",
) -> bool:
    """Resumable single-fold PFD job (used by ``run-job pfd``)."""
    return run_guarded(
        results_dir, f"pfd_f{fold}",
        lambda: _run_pfd_fold(fold, results_dir, data_path, encoder_name,
                              alpha, temperature, epochs, lr, device),
    )


def summarize_pfd(results_dir: Path, n_folds: int) -> pd.DataFrame | None:
    """Collect per-fold PFD results into ``pfd_results.csv``."""
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
    if not rows:
        return None
    rdf = pd.DataFrame(rows)
    rdf.to_csv(results_dir / "pfd_results.csv", index=False)
    return rdf


def run_pfd(
    results_dir: Path,
    data_path: Path,
    encoder_name: str,
    alpha: float,
    temperature: float,
    epochs: int,
    lr: float,
    device: str,
) -> bool:
    """Run PFD for all folds. Each fold is a resumable job.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``config.json``, ``models/C_f*`` and the
        ``cache/`` subdirectory with teacher logits from the ``train`` step.
    data_path : Path
        Path to the CSV dataset.
    encoder_name : str
        HuggingFace model identifier. Only used as a fallback when no C
        checkpoint is available for warm-starting.
    alpha : float
        Weight of the soft-target term in the PFD loss (paper: 0.1).
    temperature : float
        Distillation temperature (paper: 4.0).
    epochs : int
        Maximum PFD training epochs (paper: 3).
    lr : float
        Learning rate for the PFD student (paper: 5e-6).
    device : str
        PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns
    -------
    bool
        True if every fold completed.
    """
    console.print("[bold]fulltext-diagnostic pfd[/bold]")
    console.print(f"  alpha={alpha}, T={temperature}, epochs={epochs}, lr={lr}")

    config = json.load(open(results_dir / "config.json"))
    n_folds = config["folds"]

    ok = True
    for fold in range(n_folds):
        ok &= job_pfd(fold, results_dir, data_path, encoder_name,
                      alpha, temperature, epochs, lr, device)

    rdf = summarize_pfd(results_dir, n_folds)
    if rdf is not None:
        console.print("\n[bold]PFD Summary[/bold]")
        console.print(f"  C:     {rdf['C'].mean():.3f} +/- {rdf['C'].std(ddof=1):.3f}")
        console.print(
            f"  C_PFD: {rdf['C_PFD'].mean():.3f} +/- {rdf['C_PFD'].std(ddof=1):.3f}"
        )
        console.print(f"  delta: {rdf['delta'].mean():+.3f}")

    # The report must be regenerated after PFD.
    reset_job(results_dir, "report")

    console.print()
    dag = build_dag(n_folds)
    dag = load_state(results_dir, dag)
    show_status(dag, n_folds)
    return bool(ok)
