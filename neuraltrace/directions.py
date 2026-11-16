"""Finding 'known vs unknown' directions in activations.

All functions take X of shape (n_examples, d_model) and binary y
(1 = model knows the entity, 0 = it does not).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import auroc


def diff_of_means(X, y) -> np.ndarray:
    """Unit vector pointing from the unknown mean to the known mean."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    d = X[y].mean(0) - X[~y].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def random_directions(d_model: int, n: int, seed: int = 0) -> np.ndarray:
    """n random unit vectors, used as the null for steering/ablation."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d_model))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fit_probe(X, y, C: float = 1.0, seed: int = 0):
    """L2 logistic regression on standardised activations.

    Returns (scale_fn, direction, bias) such that
    score = scale_fn(x) @ direction + bias, all in the original activation space
    so the direction can be used for steering.
    """
    X = np.asarray(X, dtype=np.float64)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced", random_state=seed).fit(sc.transform(X), y)
    w = clf.coef_[0] / sc.scale_  # direction in raw activation space
    b = float(clf.intercept_[0] - (sc.mean_ * w).sum())
    return w, b


def score(X, direction, bias: float = 0.0) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ direction + bias


def layerwise_auroc(acts_train, y_train, acts_test, y_test, method: str = "dom"):
    """Per-layer test AUROC. acts_*: (n_layers, n_examples, d_model)."""
    out = []
    for L in range(acts_train.shape[0]):
        if method == "dom":
            d = diff_of_means(acts_train[L], y_train)
            out.append(auroc(y_test, score(acts_test[L], d)))
        elif method == "probe":
            w, b = fit_probe(acts_train[L], y_train)
            out.append(auroc(y_test, score(acts_test[L], w, b)))
        else:
            raise ValueError(method)
    return np.array(out)
