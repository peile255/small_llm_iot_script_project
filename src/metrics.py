from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "n": int(len(y_true)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else None,
    }

def bootstrap_metric(y_true, y_pred, metric_name="f1", n_resamples=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_pred[idx]).get(metric_name)
        if m is not None and np.isfinite(m):
            vals.append(m)
    if not vals:
        return None
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"low": float(lo), "high": float(hi)}
