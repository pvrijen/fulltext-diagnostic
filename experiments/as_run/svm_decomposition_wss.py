"""SVM TE/IE decomposition on PR-AUC and WSS@95 (CPU only; same folds as svm_decomposition.py).

Conditions (TF-IDF, k = 10,000, calibrated linear SVM):
  S_abs->abs: trained on abstracts, evaluated on abstracts
  S_ft->abs:  trained on full text, evaluated on abstracts (through the full-text vocabulary)
  S_ft->ft:   trained on full text, evaluated on full text
Writes svm_decomposition_wss.json next to this script with per-fold PR-AUC and WSS@95 for the three
conditions. WSS@95 = (N - n95)/N - 0.05, n95 = rank at which 95% of positives have been seen.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
os.chdir("/home/paul/fulltext_biocuration")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from run_rq2_unified import load_data, make_cv_splits, set_seeds  # noqa: E402

SEED = 42
DATASETS = ["pfas", "bpa", "cd008122", "cd011431", "cd012069", "cd012661"]
MAX_FEATURES = 10000


def wss95(y, scores):
    y = np.asarray(y); scores = np.asarray(scores)
    n, n_pos = len(y), int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    cum = np.cumsum(y[order])
    n95 = int(np.searchsorted(cum, np.ceil(0.95 * n_pos))) + 1
    return (n - n95) / n - 0.05


def make_svm():
    return CalibratedClassifierCV(SVC(kernel="linear", C=1.0), cv=3, n_jobs=-1)


rows = []
out = HERE / "svm_decomposition_wss.json"
for ds in DATASETS:
    set_seeds(SEED)
    df, texts_full, texts_medline, labels = load_data(ds)
    splits = make_cv_splits(labels, SEED)
    for fold in range(5):
        train_idx, test_idx = splits[fold]
        y_train = [labels[i] for i in train_idx]
        y_test = np.array([labels[i] for i in test_idx])
        tfidf_abs = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
        X_train_abs = tfidf_abs.fit_transform([texts_medline[i] for i in train_idx])
        X_test_abs = tfidf_abs.transform([texts_medline[i] for i in test_idx])
        tfidf_ft = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
        X_train_ft = tfidf_ft.fit_transform([texts_full[i] for i in train_idx])
        X_test_ft = tfidf_ft.transform([texts_full[i] for i in test_idx])
        X_test_abs_ftvocab = tfidf_ft.transform([texts_medline[i] for i in test_idx])
        svm_c = make_svm().fit(X_train_abs, y_train)
        svm_p = make_svm().fit(X_train_ft, y_train)
        s_c = svm_c.predict_proba(X_test_abs)[:, 1]
        s_p0 = svm_p.predict_proba(X_test_abs_ftvocab)[:, 1]
        s_p1 = svm_p.predict_proba(X_test_ft)[:, 1]
        row = {"dataset": ds, "fold": fold,
               "C_prauc": float(average_precision_score(y_test, s_c)), "P0_prauc": float(average_precision_score(y_test, s_p0)),
               "P100_prauc": float(average_precision_score(y_test, s_p1)),
               "C_wss95": wss95(y_test, s_c), "P0_wss95": wss95(y_test, s_p0), "P100_wss95": wss95(y_test, s_p1)}
        rows.append(row)
        print(f"{ds:10s} f{fold}  PR-AUC C={row['C_prauc']:.3f} P0={row['P0_prauc']:.3f} P100={row['P100_prauc']:.3f} | "
              f"WSS C={row['C_wss95']:.3f} P0={row['P0_wss95']:.3f} P100={row['P100_wss95']:.3f}", flush=True)
        json.dump(rows, open(out, "w"), indent=1)
print("saved", out)
