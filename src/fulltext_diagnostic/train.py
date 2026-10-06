# Copyright (c) 2026 Paul van Rijen, Patrick Ruch
# HES-SO Geneva / SIB Swiss Institute of Bioinformatics
# Licensed under the MIT License

"""Training jobs for the diagnostic pipeline.

Each public function corresponds to one SLURM-style job that can be executed
independently or orchestrated by the sequential ``run_training`` driver.
Jobs are resumable: if a job's done marker exists, it is skipped.

The two models trained are:
  - **C** (abstract baseline): a standard sequence classifier trained and
    evaluated on abstracts only (max 512 tokens).
  - **P** (full-text model): BertClassifierWithPooling, trained on full text
    with chunk-mean pooling (up to 12 non-overlapping chunks of 510 tokens),
    evaluated on abstracts (P@0%) and on full text (P@100%). Teacher logits
    for PFD are also extracted during evaluation.

Default hyperparameters follow the paper's hyperparameter table: both models
use AdamW, 5% linear warmup, gradient clipping at 1.0, up to 20 epochs and
early stopping on validation loss with patience 3.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from rich.console import Console

# Add belt_plus to path (sibling directory in repo)
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from belt_plus.bert import set_seed
from belt_plus.bert_classifier_with_pooling import BertClassifierWithPooling

from .pipeline import build_dag, load_state, run_guarded, show_status

console = Console()

# Column names of ``diagnostic_results.csv``. Shared with diagnose.py,
# report.py and the tests.
COL_C = "C"
COL_P0 = "P@0%"
COL_P100 = "P@100%"
RESULT_COLUMNS = ["fold", COL_C, COL_P0, COL_P100, "TE", "IE", "Total"]

# Paper hyperparameters (abstract baseline C)
C_WEIGHT_DECAY = 0.10
C_DROPOUT = 0.5
# Paper hyperparameters (full-text model P)
P_WEIGHT_DECAY = 0.01
P_DROPOUT = 0.1
P_EFFECTIVE_BATCH = 64
# Shared
WARMUP_RATIO = 0.05
PATIENCE = 3
VAL_FRACTION = 0.2


# --- Shared Utilities --------------------------------------------------------

def _read_seed(output_dir: Path, default: int = 42) -> int:
    """Read the pipeline seed from ``config.json`` if present."""
    cfg = output_dir / "config.json"
    if cfg.exists():
        try:
            return int(json.load(open(cfg)).get("seed", default))
        except (ValueError, json.JSONDecodeError):
            return default
    return default


def _make_loader(
    texts: list[str],
    labels: list[int] | np.ndarray,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Tokenize texts and wrap them in a DataLoader."""
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return DataLoader(
        TensorDataset(enc["input_ids"], enc["attention_mask"],
                      torch.tensor(labels)),
        batch_size=batch_size, shuffle=shuffle,
    )


def _logits_1d(logits: torch.Tensor) -> torch.Tensor:
    """Return a 1-D tensor of positive-class logits for a batch."""
    if logits.shape[-1] > 1:
        return logits[:, 1]
    return logits.squeeze(-1)


def _val_loss(
    model: torch.nn.Module, loader: DataLoader, device: str,
) -> float:
    """Mean binary cross-entropy on a DataLoader."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ids, mask, lab in loader:
            out = model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                labels=lab.to(device).float(),
            )
            total += float(out.loss) * len(lab)
            n += len(lab)
    return total / max(1, n)


def _train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = C_WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
    patience: int = PATIENCE,
) -> tuple[torch.nn.Module, float]:
    """Fine-tune a HuggingFace classification model.

    Uses AdamW with linear warmup and decay, gradient clipping at 1.0, and
    early stopping on validation loss (the same criterion BELT uses for the
    full-text model, so both arms of the decomposition are selected the same
    way). Returns the model loaded with the best checkpoint and its
    validation PR-AUC.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps,
    )
    best_loss = float("inf")
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
            scheduler.step()
            optimizer.zero_grad()
        loss = _val_loss(model, val_loader, device)
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, _eval_prauc(model, val_loader, device)


def _eval_prauc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """Compute PR-AUC on a DataLoader."""
    _, probs, labels = _get_preds(model, loader, device)
    return float(average_precision_score(labels, probs))


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
            all_logits.extend(_logits_1d(out.logits).cpu().numpy())
            all_labels.extend(lab.numpy())
    logits = np.array(all_logits)
    probs = 1 / (1 + np.exp(-logits))
    return logits, probs, np.array(all_labels)


def _eval_prauc_belt(model, texts, labels):
    """Compute PR-AUC using a belt_plus model's predict_scores."""
    _, probs = _get_preds_belt(model, texts)
    return float(average_precision_score(labels, probs))


def _get_preds_belt(model, texts):
    """Extract logits and probabilities from a belt_plus model.

    ``predict_scores`` returns sigmoid probabilities; logits are reconstructed
    from clipped probabilities so that 0 and 1 stay finite.
    """
    with torch.no_grad():
        scores = model.predict_scores(texts)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().float().numpy().flatten()
    probs = np.asarray(scores, dtype=float)
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    return logits, probs


def load_folds(output_dir: Path) -> dict[int, dict]:
    """Load the fold-index mapping saved by ``job_split_folds``."""
    with open(output_dir / "folds.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def _build_P(
    encoder: str, device: str, lr: float, epochs: int, batch_size: int,
    n_chunks: int, chunk_size: int,
) -> BertClassifierWithPooling:
    """Construct the full-text pooled model with the paper's settings."""
    accumulation = max(1, P_EFFECTIVE_BATCH // batch_size)
    return BertClassifierWithPooling(
        pretrained_model_name_or_path=encoder, num_labels=1, device=device,
        learning_rate=lr, weight_decay=P_WEIGHT_DECAY, dropout_rate=P_DROPOUT,
        batch_size=batch_size, accumulation_steps=accumulation, epochs=epochs,
        chunk_size=chunk_size, stride=chunk_size, minimal_chunk_length=50,
        max_chunks=n_chunks, chunk_pooling_strategy="mean", use_amp=False,
        warmup_ratio=WARMUP_RATIO, max_grad_norm=1.0)


# --- Job Functions -----------------------------------------------------------

def job_split_folds(
    data_path: Path, output_dir: Path, n_folds: int, seed: int,
) -> bool:
    """Create stratified k-fold splits and save to ``folds.json``.

    Within each training fold, a stratified 20% validation split is held out
    for early stopping. ``train_all`` is the union of ``train`` and ``val``.
    """
    def _run() -> None:
        df = pd.read_csv(data_path)
        for col in ("text_abstract", "text_fulltext", "label"):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' missing from {data_path}. Expected "
                    f"columns: id, text_abstract, text_fulltext, label."
                )
        labels = df["label"].values
        if set(np.unique(labels)) - {0, 1}:
            raise ValueError("Column 'label' must contain only 0 and 1.")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds: dict[int, dict] = {}
        for fi, (train_idx, test_idx) in enumerate(skf.split(df, labels)):
            tr, vl = train_test_split(
                train_idx, test_size=VAL_FRACTION, random_state=seed + fi,
                stratify=labels[train_idx],
            )
            folds[fi] = {
                "train": sorted(int(i) for i in tr),
                "val": sorted(int(i) for i in vl),
                "test": test_idx.tolist(),
                "train_all": train_idx.tolist(),
            }
        with open(output_dir / "folds.json", "w") as f:
            json.dump(folds, f)
        cfg_path = output_dir / "config.json"
        if not cfg_path.exists():
            json.dump({"folds": n_folds, "seed": seed},
                      open(cfg_path, "w"), indent=2)
        console.print(
            f"  split: {n_folds} folds, {len(df)} articles, "
            f"{labels.sum()} pos ({labels.mean()*100:.1f}%)"
        )
    return run_guarded(output_dir, "split", _run)


def job_train_C(
    fold: int, data_path: Path, output_dir: Path, encoder: str,
    max_length: int, epochs: int, batch_size: int, lr: float, device: str,
) -> bool:
    """Train the abstract-only baseline model C for one fold."""
    def _run() -> None:
        set_seed(_read_seed(output_dir) + fold)
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        texts = df["text_abstract"].fillna("").tolist()
        labels = df["label"].values
        tok = AutoTokenizer.from_pretrained(encoder)
        config = AutoConfig.from_pretrained(encoder, num_labels=1)
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = C_DROPOUT
        model = AutoModelForSequenceClassification.from_pretrained(
            encoder, config=config
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
        console.print(f"  train_C_f{fold}: val PR-AUC={vs:.3f}")
        del model
        torch.cuda.empty_cache()
    return run_guarded(output_dir, f"train_C_f{fold}", _run)


def job_train_P(
    fold: int, data_path: Path, output_dir: Path, encoder: str,
    epochs: int, batch_size: int, lr: float, device: str,
    n_chunks: int = 12, chunk_size: int = 510,
) -> bool:
    """Train the full-text model P for one fold.

    Uses BertClassifierWithPooling with chunk-mean pooling over up to
    ``n_chunks`` non-overlapping chunks of ``chunk_size`` content tokens,
    matching the paper's decomposition protocol. ``batch_size`` is the
    per-step micro-batch; gradient accumulation keeps the effective batch
    at 64.
    """
    def _run() -> None:
        set_seed(_read_seed(output_dir) + fold)
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        texts = df["text_fulltext"].fillna("").tolist()
        labels = df["label"].values
        X_tr = [texts[i] for i in fi["train"]]
        y_tr = labels[fi["train"]].tolist()
        X_vl = [texts[i] for i in fi["val"]]
        y_vl = labels[fi["val"]].tolist()

        model = _build_P(encoder, device, lr, epochs, batch_size,
                         n_chunks, chunk_size)
        model.fit(X_tr, y_tr, X_vl, y_vl, patience=PATIENCE,
                  early_stopping=True)

        md = output_dir / f"models/P_f{fold}"
        md.mkdir(parents=True, exist_ok=True)
        nn = model.neural_network
        if hasattr(nn, "module"):
            nn = nn.module
        torch.save(nn.encoder.state_dict(), md / "encoder.pt")
        json.dump({"encoder": encoder, "chunk_size": chunk_size,
                   "stride": chunk_size, "max_chunks": n_chunks,
                   "pooling": "mean", "batch_size": batch_size},
                  open(md / "config.json", "w"), indent=2)

        vs = _eval_prauc_belt(model, X_vl, y_vl)
        console.print(f"  train_P_f{fold}: val PR-AUC={vs:.3f}")
        del model
        torch.cuda.empty_cache()
    return run_guarded(output_dir, f"train_P_f{fold}", _run)


def job_eval_C(
    fold: int, data_path: Path, output_dir: Path,
    max_length: int, batch_size: int, device: str,
) -> bool:
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
    return run_guarded(output_dir, f"eval_C_f{fold}", _run)


def job_eval_P(
    fold: int, data_path: Path, output_dir: Path, device: str,
) -> bool:
    """Evaluate model P on abstracts (P@0%) and full text (P@100%).

    Also caches teacher logits on the training articles (``train_all``)
    for PFD.
    """
    def _run() -> None:
        fi = load_folds(output_dir)[fold]
        df = pd.read_csv(data_path)
        abstracts = df["text_abstract"].fillna("").tolist()
        fulltexts = df["text_fulltext"].fillna("").tolist()
        labels = df["label"].values

        md = output_dir / f"models/P_f{fold}"
        cfg = json.load(open(md / "config.json"))

        model = _build_P(cfg["encoder"], device, lr=1e-5, epochs=1,
                         batch_size=cfg.get("batch_size", 8),
                         n_chunks=cfg["max_chunks"],
                         chunk_size=cfg["chunk_size"])
        model.neural_network.encoder.load_state_dict(
            torch.load(str(md / "encoder.pt"), map_location=device,
                       weights_only=True), strict=False)

        X_te_abs = [abstracts[i] for i in fi["test"]]
        y_te = labels[fi["test"]].tolist()
        p0_l, p0_p = _get_preds_belt(model, X_te_abs)
        p0 = average_precision_score(y_te, p0_p)

        X_te_ft = [fulltexts[i] for i in fi["test"]]
        p100_l, p100_p = _get_preds_belt(model, X_te_ft)
        p100 = average_precision_score(y_te, p100_p)

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
            f"  eval_P_f{fold}: P@0%={p0:.3f} P@100%={p100:.3f}"
        )
        del model
        torch.cuda.empty_cache()
    return run_guarded(output_dir, f"eval_P_f{fold}", _run)


def job_diagnose(output_dir: Path, n_folds: int) -> bool:
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
                "fold": f, COL_C: cv, COL_P0: p0v, COL_P100: p100v,
                "TE": p0v - cv, "IE": p100v - p0v, "Total": p100v - cv,
            })
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
            output_dir / "diagnostic_results.csv", index=False
        )
        from .diagnose import run_diagnosis
        run_diagnosis(output_dir)
    return run_guarded(output_dir, "diagnose", _run)


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
    lr_fulltext: float = 1e-5,
    batch_size_fulltext: int = 8,
) -> bool:
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
        Random seed for fold splits and model initialization.
    max_length : int
        Max token length for the abstract model (C). Default 512.
    n_chunks : int
        Max chunks for the full-text model (P). Default 12.
    chunk_size : int
        Content tokens per chunk for the full-text model. Default 510.
    epochs : int
        Maximum training epochs (both models).
    batch_size : int
        Training batch size for C.
    lr : float
        Learning rate for C.
    device : str
        PyTorch device string.
    lr_fulltext : float
        Learning rate for P.
    batch_size_fulltext : int
        Micro-batch size for P (effective batch stays 64 via accumulation).

    Returns
    -------
    bool
        True if every job completed, False if any job failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "data": str(data_path), "encoder": encoder_name,
            "folds": n_folds, "seed": seed, "max_length": max_length,
            "n_chunks": n_chunks, "chunk_size": chunk_size,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "lr_fulltext": lr_fulltext,
            "batch_size_fulltext": batch_size_fulltext,
        }, f, indent=2)

    console.print("[bold]fulltext-diagnostic train[/bold]")

    dag = build_dag(n_folds)
    dag = load_state(output_dir, dag)
    show_status(dag, n_folds)
    console.print()

    ok = job_split_folds(data_path, output_dir, n_folds, seed)
    if not ok:
        console.print("[red]Fold split failed; stopping.[/red]")
        return False

    for fold in range(n_folds):
        console.print(f"\n[bold]Fold {fold}[/bold]")
        ok &= job_train_C(fold, data_path, output_dir, encoder_name,
                          max_length, epochs, batch_size, lr, device)
        ok &= job_train_P(fold, data_path, output_dir, encoder_name,
                          epochs, batch_size_fulltext, lr_fulltext, device,
                          n_chunks=n_chunks, chunk_size=chunk_size)
        ok &= job_eval_C(fold, data_path, output_dir, max_length,
                         batch_size, device)
        ok &= job_eval_P(fold, data_path, output_dir, device)

    console.print("\n[bold]Diagnose[/bold]")
    ok &= job_diagnose(output_dir, n_folds)

    console.print()
    dag = build_dag(n_folds)
    dag = load_state(output_dir, dag)
    show_status(dag, n_folds)
    return bool(ok)
