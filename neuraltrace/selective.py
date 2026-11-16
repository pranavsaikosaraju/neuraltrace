"""Selective answering: use the detection score to decide when to abstain.

The model abstains on an item when its score (higher = 'I know this entity')
falls below a threshold. Two views of the same trade-off:

* catch rate on unknown entities at a given false-abstention rate on known ones
* accuracy among answered items at a given coverage (fraction answered)

The prompt-based abstention baseline is a single point in both views, so the
detector is compared with it at a matched false-abstention rate (threshold
chosen on the validation split) and at matched coverage.
"""
from __future__ import annotations

import numpy as np

FPR_GRID = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
COV_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def catch_curve(s, known, grid=FPR_GRID):
    """Fraction of unknown items abstained on when a fraction f of known items is."""
    s, known = np.asarray(s, float), np.asarray(known, bool)
    return [float((s[~known] < np.quantile(s[known], f)).mean()) for f in grid]


def accuracy_at_coverage(s, correct, cov: float) -> float:
    """Accuracy among the top `cov` fraction of items by score."""
    s, correct = np.asarray(s, float), np.asarray(correct, bool)
    k = max(1, int(round(cov * len(s))))
    return float(correct[np.argsort(-s, kind="stable")[:k]].mean())


def _point(abstain, known, correct):
    ans = ~abstain
    return {"fa": float(abstain[known].mean()), "catch": float(abstain[~known].mean()),
            "cov": float(ans.mean()), "acc": float(correct[ans].mean()) if ans.any() else float("nan")}


def evaluate(s_val, known_val, s_test, known_test, correct_test, p_abstain, p_correct, n_boot: int = 500, seed: int = 0):
    """Compare score-thresholded abstention with prompt-based abstention on the test split.

    correct_test: correctness of the plain-prompt answer (used when the detector lets the model answer).
    p_abstain / p_correct: abstention and correctness under the abstention prompt.
    """
    s_val, s_test = np.asarray(s_val, float), np.asarray(s_test, float)
    kv, kt = np.asarray(known_val, bool), np.asarray(known_test, bool)
    co, pa, pc = (np.asarray(a, bool) for a in (correct_test, p_abstain, p_correct))
    prompt = _point(pa, kt, pc)
    thr = float(np.quantile(s_val[kv], prompt["fa"])) if kv.any() else float("-inf")
    det_ab = s_test < thr
    det = _point(det_ab, kt, co)
    acc_cov = accuracy_at_coverage(s_test, co, prompt["cov"])
    rng = np.random.default_rng(seed)
    n = len(s_test)
    d_catch, d_acc, c_det = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if kt[i].all() or not kt[i].any() or pa[i].all():
            continue
        c_det.append(det_ab[i][~kt[i]].mean())
        d_catch.append(c_det[-1] - pa[i][~kt[i]].mean())
        ans = ~pa[i]
        d_acc.append(accuracy_at_coverage(s_test[i], co[i], ans.mean()) - pc[i][ans].mean())
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
    return {
        "n": int(n), "n_known": int(kt.sum()),
        "roc": catch_curve(s_test, kt), "rc": [accuracy_at_coverage(s_test, co, c) for c in COV_GRID],
        "prompt": prompt, "det": det, "det_catch_ci": ci(c_det),
        "d_catch": [det["catch"] - prompt["catch"]] + ci(d_catch),
        "acc_at_prompt_cov": acc_cov, "d_acc": [acc_cov - prompt["acc"]] + ci(d_acc),
    }
