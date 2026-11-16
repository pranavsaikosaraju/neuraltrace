"""Evaluation metrics: AUROC with bootstrap CIs and risk-coverage curves."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y_true, scores) -> float:
    """AUROC; returns NaN when only one class is present."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def bootstrap_ci(y_true, scores, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for AUROC. Returns (point, lo, hi)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auroc(y_true[idx], scores[idx])
        if not np.isnan(a):
            stats.append(a)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return auroc(y_true, scores), float(lo), float(hi)


def risk_coverage(confidence, correct):
    """Risk-coverage curve.

    Sort answers by descending confidence; at each coverage level report the
    accuracy of the answers kept. Returns (coverage, accuracy) arrays.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-confidence, kind="stable")
    c = correct[order]
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    accuracy = np.cumsum(c) / np.arange(1, n + 1)
    return coverage, accuracy


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve using error rate (lower is better)."""
    _, acc = risk_coverage(confidence, correct)
    return float(np.mean(1.0 - acc))
