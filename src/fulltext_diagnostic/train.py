# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Training jobs for the diagnostic pipeline.

Each public function corresponds to one SLURM-style job that can be executed
independently or orchestrated by the sequential ``run_training`` driver.
Jobs are resumable: if a job's done marker exists, it is skipped.

The two models trained are:
  - **C** (abstract baseline): BertClassifierTruncated, trained and evaluated
    on abstracts only (max 512 tokens).
  - **P** (full-text model): BertClassifierWithPooling, trained on full-text
    with chunk-mean pooling (up to 12 chunks of 510 tokens), evaluated on
    abstracts (L_ft→abs) and on full-text (L_ft→ft). Teacher logits for PFD
    are also extracted during evaluation.
"""

import json
import sys
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

# Add belt_plus to path (sibling directory in repo)
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from belt_plus.bert_classifier_truncated import BertClassifierTruncated
from belt_plus.bert_classifier_with_pooling import BertClassifierWithPooling

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


def _eval_prauc_belt(model, texts, labels):
    """Compute PR-AUC using a belt_plus model's predict_scores."""
    with torch.no_grad():
        scores = model.predict_scores(texts)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().float().numpy().flatten()
    return float(average_precision_score(labels, scores))


def _get_preds_belt(model, texts):
    """Extract logits and probabilities from a belt_plus model."""
    with torch.no_grad():
        scores = model.predict_scores(texts)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().float().numpy().flatten()
    probs = np.array(scores)
    logits = np.log(probs / (1 - np.clip(probs, 1e-7, 1 - 1e-7)))
    return logits, probs


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
            encoder, num_labels=1
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
    """Train the full-text model P for one fold.

    Uses BertClassifierWithPooling with chunk-mean pooling (up to 12
    non-overlapping chunks of 510 content tokens), matching the paper's
    decomposition protocol (Section 3.3).
    """
    def _run() -> None:
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        texts = df["text_fulltext"].fillna("").tolist()
        labels = df["label"].values
        X_tr = [texts[i] for i in fi["train"]]
        y_tr = labels[fi["train"]].tolist()
        X_vl = [texts[i] for i in fi["val"]]
        y_vl = labels[fi["val"]].tolist()

        model = BertClassifierWithPooling(
            pretrained_model_name_or_path=encoder, num_labels=1, device=device,
            learning_rate=lr, weight_decay=0.01, dropout_rate=0.1,
            batch_size=8, accumulation_steps=8, epochs=epochs,
            chunk_size=510, stride=510, minimal_chunk_length=50, max_chunks=12,
            chunk_pooling_strategy="mean", use_amp=False,
            warmup_ratio=0.05, max_grad_norm=1.0)
        model.fit(X_tr, y_tr, X_vl, y_vl, patience=3, early_stopping=True)

        md = output_dir / f"models/P_f{fold}"
        md.mkdir(parents=True, exist_ok=True)
        nn = model.neural_network
        if hasattr(nn, "module"):
            nn = nn.module
        torch.save(nn.encoder.state_dict(), md / "encoder.pt")
        # Save full model config for reloading
        json.dump({"encoder": encoder, "chunk_size": 510, "stride": 510,
                    "max_chunks": 12, "pooling": "mean"},
                   open(md / "config.json", "w"), indent=2)

        vs = _eval_prauc_belt(model, X_vl, y_vl)
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
    """Evaluate model P on abstracts (L_ft→abs) and full text (L_ft→ft).

    Also caches teacher logits on training data for PFD.
    Uses BertClassifierWithPooling with chunk-mean pooling.
    """
    def _run() -> None:
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        abstracts = df["text_abstract"].fillna("").tolist()
        fulltexts = df["text_fulltext"].fillna("").tolist()
        labels = df["label"].values

        md = output_dir / f"models/P_f{fold}"
        cfg = json.load(open(md / "config.json"))

        # Rebuild pooled model and load saved encoder weights
        model = BertClassifierWithPooling(
            pretrained_model_name_or_path=cfg["encoder"], num_labels=1,
            device=device, learning_rate=1e-5, weight_decay=0.01,
            dropout_rate=0.1, batch_size=8, accumulation_steps=8, epochs=1,
            chunk_size=cfg["chunk_size"], stride=cfg["stride"],
            minimal_chunk_length=50, max_chunks=cfg["max_chunks"],
            chunk_pooling_strategy=cfg["pooling"], use_amp=False)
        model.neural_network.encoder.load_state_dict(
            torch.load(str(md / "encoder.pt"), map_location=device,
                        weights_only=True), strict=False)

        # Evaluate on abstracts (L_ft→abs)
        X_te_abs = [abstracts[i] for i in fi["test"]]
        y_te = labels[fi["test"]].tolist()
        p0_l, p0_p = _get_preds_belt(model, X_te_abs)
        p0 = average_precision_score(y_te, p0_p)

        # Evaluate on full text (L_ft→ft)
        X_te_ft = [fulltexts[i] for i in fi["test"]]
        p100_l, p100_p = _get_preds_belt(model, X_te_ft)
        p100 = average_precision_score(y_te, p100_p)

        # Teacher logits on training full text (for PFD)
        X_tr_ft = [fulltexts[i] for i in fi["train_all"]]
        teach_l, teach_p = _get_preds_belt(model, X_tr_ft)

        cache = output_dir / "cache"
        cache.mkdir(exist_ok=True)
        np.savez(
            cache / f"P_f{fold}.npz",
            p0_logits=p0_l, p0_probs=p0_p, p0_prauc=p0,
            p100_logits=p100_l, p100_probs=p100_p, p100_prauc=p100,
            teacher_logits=teach_l, teacher_probs=teach_p,
            test_idx=np.array(fi["test"]),
            train_all_idx=np.array(fi["train_all"]),
            labels=np.array(y_te),
        )
        console.print(
            f"  eval_P_f{fold}: L_ft→abs={p0:.3f} L_ft→ft={p100:.3f}"
        )
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
                "fold": f, "L_abs": cv, "L_fta": p0v, "L_ff": p100v,
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
        Max token length for the abstract model (C). Default 512.
    n_chunks : int
        Max chunks for the full-text model (P). Default 12.
    chunk_size : int
        Content tokens per chunk for the full-text model. Default 510.
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
