# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Training jobs for the diagnostic pipeline.

Each public function corresponds to one SLURM-style job that can be executed
independently or orchestrated by the sequential ``run_training`` driver.
Jobs are resumable: if a job's done marker exists, it is skipped.

The two models trained are:
  - **C** (abstract baseline): trained and evaluated on abstracts only.
  - **P** (full-text model): trained on full-text, evaluated on abstracts
    (P@0%) and on full-text (P@100%). Teacher logits for PFD are also
    extracted during evaluation.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rich.console import Console

from .pipeline import build_dag, load_state, mark_done, run_guarded, show_status

console = Console()


# --- Shared Utilities --------------------------------------------------------

def _make_loader(
    texts: list[str],
    labels: list[int] | np.ndarray,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Tokenise texts and wrap them in a DataLoader."""
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return DataLoader(
        TensorDataset(enc["input_ids"], enc["attention_mask"],
                      torch.tensor(labels)),
        batch_size=batch_size, shuffle=shuffle,
    )


def _train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    patience: int = 3,
) -> tuple[torch.nn.Module, float]:
    """Fine-tune a HuggingFace classification model with early stopping.

    Returns the model loaded with the best checkpoint and the best
    validation PR-AUC score.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_score = -1.0
    best_state: dict | None = None
    wait = 0
    for _ in range(epochs):
        model.train()
        for ids, mask, lab in train_loader:
            out = model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                labels=lab.to(device).float(),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        score = _eval_prauc(model, val_loader, device)
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_score


def _eval_prauc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """Compute PR-AUC on a DataLoader."""
    model.eval()
    logits_list: list[float] = []
    labels_list: list[int] = []
    with torch.no_grad():
        for ids, mask, lab in loader:
            out = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            )
            logit_vals = (
                out.logits[:, 1] if out.logits.shape[-1] > 1
                else out.logits.squeeze()
            )
            logits_list.extend(logit_vals.cpu().numpy())
            labels_list.extend(lab.numpy())
    probs = 1 / (1 + np.exp(-np.array(logits_list)))
    return float(average_precision_score(labels_list, probs))


def _get_preds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract logits, probabilities, and labels from a DataLoader."""
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for ids, mask, lab in loader:
            out = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            )
            logit_vals = (
                out.logits[:, 1] if out.logits.shape[-1] > 1
                else out.logits.squeeze()
            )
            all_logits.extend(logit_vals.cpu().numpy())
            all_labels.extend(lab.numpy())
    logits = np.array(all_logits)
    probs = 1 / (1 + np.exp(-logits))
    return logits, probs, np.array(all_labels)


def load_folds(output_dir: Path) -> dict[int, dict]:
    """Load the fold-index mapping saved by ``job_split_folds``."""
    with open(output_dir / "folds.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


# --- Job Functions -----------------------------------------------------------

def job_split_folds(
    data_path: Path, output_dir: Path, n_folds: int, seed: int,
) -> None:
    """Create stratified k-fold splits and save to ``folds.json``."""
    def _run() -> None:
        df = pd.read_csv(data_path)
        labels = df["label"].values
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds: dict[int, dict] = {}
        for fi, (train_idx, test_idx) in enumerate(skf.split(df, labels)):
            rng = np.random.RandomState(seed + fi)
            vs = int(0.3 * len(train_idx))
            perm = rng.permutation(len(train_idx))
            folds[fi] = {
                "train": train_idx[perm[vs:]].tolist(),
                "val": train_idx[perm[:vs]].tolist(),
                "test": test_idx.tolist(),
                "train_all": train_idx.tolist(),
            }
        with open(output_dir / "folds.json", "w") as f:
            json.dump(folds, f)
        console.print(
            f"  split: {n_folds} folds, {len(df)} articles, "
            f"{labels.sum()} pos ({labels.mean()*100:.1f}%)"
        )
    run_guarded(output_dir, "split", _run)


def job_train_C(
    fold: int, data_path: Path, output_dir: Path, encoder: str,
    max_length: int, epochs: int, batch_size: int, lr: float, device: str,
) -> None:
    """Train the abstract-only baseline model C for one fold."""
    def _run() -> None:
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        texts = df["text_abstract"].fillna("").tolist()
        labels = df["label"].values
        tok = AutoTokenizer.from_pretrained(encoder)
        model = AutoModelForSequenceClassification.from_pretrained(
            encoder, num_labels=2
        ).to(device)
        tl = _make_loader(
            [texts[i] for i in fi["train"]], labels[fi["train"]],
            tok, max_length, batch_size, True,
        )
        vl = _make_loader(
            [texts[i] for i in fi["val"]], labels[fi["val"]],
            tok, max_length, batch_size,
        )
        model, vs = _train_loop(model, tl, vl, epochs, lr, device)
        md = output_dir / f"models/C_f{fold}"
        md.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(md)
        tok.save_pretrained(md)
        console.print(f"  train_C_f{fold}: val={vs:.3f}")
        del model
        torch.cuda.empty_cache()
    run_guarded(output_dir, f"train_C_f{fold}", _run)


def job_train_P(
    fold: int, data_path: Path, output_dir: Path, encoder: str,
    max_length: int, epochs: int, batch_size: int, lr: float, device: str,
) -> None:
    """Train the full-text model P for one fold."""
    def _run() -> None:
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        texts = df["text_fulltext"].fillna("").tolist()
        labels = df["label"].values
        ft_max = max_length * 2
        tok = AutoTokenizer.from_pretrained(encoder)
        model = AutoModelForSequenceClassification.from_pretrained(
            encoder, num_labels=2
        ).to(device)
        tl = _make_loader(
            [texts[i] for i in fi["train"]], labels[fi["train"]],
            tok, ft_max, batch_size, True,
        )
        vl = _make_loader(
            [texts[i] for i in fi["val"]], labels[fi["val"]],
            tok, ft_max, batch_size,
        )
        model, vs = _train_loop(model, tl, vl, epochs, lr, device)
        md = output_dir / f"models/P_f{fold}"
        md.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(md)
        tok.save_pretrained(md)
        console.print(f"  train_P_f{fold}: val={vs:.3f}")
        del model
        torch.cuda.empty_cache()
    run_guarded(output_dir, f"train_P_f{fold}", _run)


def job_eval_C(
    fold: int, data_path: Path, output_dir: Path,
    max_length: int, batch_size: int, device: str,
) -> None:
    """Evaluate model C on the test fold and cache predictions."""
    def _run() -> None:
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        texts = df["text_abstract"].fillna("").tolist()
        labels = df["label"].values
        md = output_dir / f"models/C_f{fold}"
        tok = AutoTokenizer.from_pretrained(md)
        model = AutoModelForSequenceClassification.from_pretrained(md).to(device)
        tl = _make_loader(
            [texts[i] for i in fi["test"]], labels[fi["test"]],
            tok, max_length, batch_size,
        )
        logits, probs, labs = _get_preds(model, tl, device)
        prauc = average_precision_score(labs, probs)
        cache = output_dir / "cache"
        cache.mkdir(exist_ok=True)
        np.savez(
            cache / f"C_f{fold}.npz",
            logits=logits, probs=probs, labels=labs,
            test_idx=np.array(fi["test"]), prauc=prauc,
        )
        console.print(f"  eval_C_f{fold}: PR-AUC={prauc:.3f}")
        del model
        torch.cuda.empty_cache()
    run_guarded(output_dir, f"eval_C_f{fold}", _run)


def job_eval_P(
    fold: int, data_path: Path, output_dir: Path,
    max_length: int, batch_size: int, device: str,
) -> None:
    """Evaluate model P at 0% and 100% coverage and cache teacher logits."""
    def _run() -> None:
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        abstracts = df["text_abstract"].fillna("").tolist()
        fulltexts = df["text_fulltext"].fillna("").tolist()
        labels = df["label"].values
        ft_max = max_length * 2
        md = output_dir / f"models/P_f{fold}"
        tok = AutoTokenizer.from_pretrained(md)
        model = AutoModelForSequenceClassification.from_pretrained(md).to(device)
        # P@0%
        tl_abs = _make_loader(
            [abstracts[i] for i in fi["test"]], labels[fi["test"]],
            tok, max_length, batch_size,
        )
        p0_l, p0_p, p0_y = _get_preds(model, tl_abs, device)
        p0 = average_precision_score(p0_y, p0_p)
        # P@100%
        tl_ft = _make_loader(
            [fulltexts[i] for i in fi["test"]], labels[fi["test"]],
            tok, ft_max, batch_size,
        )
        p100_l, p100_p, p100_y = _get_preds(model, tl_ft, device)
        p100 = average_precision_score(p100_y, p100_p)
        # Teacher logits on training data (for PFD)
        tl_teach = _make_loader(
            [fulltexts[i] for i in fi["train_all"]],
            labels[fi["train_all"]],
            tok, ft_max, batch_size,
        )
        teach_l, teach_p, _ = _get_preds(model, tl_teach, device)
        cache = output_dir / "cache"
        cache.mkdir(exist_ok=True)
        np.savez(
            cache / f"P_f{fold}.npz",
            p0_logits=p0_l, p0_probs=p0_p, p0_prauc=p0,
            p100_logits=p100_l, p100_probs=p100_p, p100_prauc=p100,
            teacher_logits=teach_l, teacher_probs=teach_p,
            test_idx=np.array(fi["test"]),
            train_all_idx=np.array(fi["train_all"]),
            labels=p100_y,
        )
        console.print(f"  eval_P_f{fold}: P@0%={p0:.3f} P@100%={p100:.3f}")
        del model
        torch.cuda.empty_cache()
    run_guarded(output_dir, f"eval_P_f{fold}", _run)


def job_diagnose(output_dir: Path, n_folds: int) -> None:
    """Assemble per-fold results into ``diagnostic_results.csv`` and print
    the decomposition."""
    def _run() -> None:
        cache = output_dir / "cache"
        rows = []
        for f in range(n_folds):
            c = np.load(cache / f"C_f{f}.npz")
            p = np.load(cache / f"P_f{f}.npz")
            cv = float(c["prauc"])
            p0v = float(p["p0_prauc"])
            p100v = float(p["p100_prauc"])
            rows.append({
                "fold": f, "C": cv, "P@0%": p0v, "P@100%": p100v,
                "TE": p0v - cv, "IE": p100v - p0v, "Total": p100v - cv,
            })
        pd.DataFrame(rows).to_csv(
            output_dir / "diagnostic_results.csv", index=False
        )
        from .diagnose import run_diagnosis
        run_diagnosis(output_dir)
    run_guarded(output_dir, "diagnose", _run)


# --- Orchestrator ------------------------------------------------------------

def run_training(
    data_path: Path,
    encoder_name: str,
    output_dir: Path,
    n_folds: int,
    seed: int,
    max_length: int,
    n_chunks: int,
    chunk_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    """Run the full diagnostic pipeline sequentially. Each step is resumable.

    Parameters
    ----------
    data_path : Path
        Path to the CSV dataset.
    encoder_name : str
        HuggingFace model identifier.
    output_dir : Path
        Directory for all pipeline outputs.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed.
    max_length : int
        Max token length for abstract models.
    n_chunks : int
        Max chunks for the full-text model (reserved for future use).
    chunk_size : int
        Tokens per chunk (reserved for future use).
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    device : str
        PyTorch device string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "data": str(data_path), "encoder": encoder_name,
            "folds": n_folds, "seed": seed, "max_length": max_length,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
        }, f, indent=2)

    console.print("[bold]fulltext-diagnostic train[/bold]")

    # Show current state
    dag = build_dag(n_folds)
    dag = load_state(output_dir, dag)
    show_status(dag, n_folds)
    console.print()

    # Execute in order
    job_split_folds(data_path, output_dir, n_folds, seed)

    for fold in range(n_folds):
        console.print(f"\n[bold]Fold {fold}[/bold]")
        job_train_C(fold, data_path, output_dir, encoder_name, max_length,
                    epochs, batch_size, lr, device)
        job_train_P(fold, data_path, output_dir, encoder_name, max_length,
                    epochs, batch_size, lr, device)
        job_eval_C(fold, data_path, output_dir, max_length, batch_size, device)
        job_eval_P(fold, data_path, output_dir, max_length, batch_size, device)

    console.print("\n[bold]Diagnose[/bold]")
    job_diagnose(output_dir, n_folds)

    # Final status
    console.print()
    dag = build_dag(n_folds)
    dag = load_state(output_dir, dag)
    show_status(dag, n_folds)
