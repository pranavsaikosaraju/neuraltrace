"""Steering experiment design (torch-free parts).

Sign convention: the direction points from 'unknown' towards 'known'.
  alpha > 0  pushes the model towards 'I know this'  (should induce hallucination)
  alpha < 0  pushes towards 'I don't know this'      (should induce abstention)
alpha is measured in units of the gap between the two class means along the
direction, so alpha = 1 moves an unknown entity to the average known one.

Nulls: isotropic random unit directions, and difference-of-means directions
computed from shuffled labels (these live in the data's own subspace).
"""
from __future__ import annotations

import numpy as np

CLASSES = ("known", "unk_real", "fict")


def build_conditions(alphas=(-2.0, -1.0, 1.0, 2.0), probe_alphas=(-1.0, 1.0), n_random: int = 8, n_shuffle: int = 4):
    conds = [{"name": "baseline", "kind": "none"}, {"name": "prompt_abstain", "kind": "prompt"}]
    conds += [{"name": f"dom_a{a:+g}", "kind": "dom", "alpha": float(a)} for a in alphas]
    conds += [{"name": f"probe_a{a:+g}", "kind": "probe", "alpha": float(a)} for a in probe_alphas]
    conds.append({"name": "dom_ablate", "kind": "ablate"})
    for k in range(n_random):
        conds += [{"name": f"rand{k}_a{a:+g}", "kind": "random", "k": k, "alpha": a} for a in (-1.0, 1.0)]
    for k in range(n_shuffle):
        conds += [{"name": f"shuf{k}_a{a:+g}", "kind": "shuffle", "k": k, "alpha": a} for a in (-1.0, 1.0)]
    return conds


def summarize(cls_items, abstain, correct):
    """Counts per class: n, abstentions, correct answers."""
    cls_items = np.asarray(cls_items)
    abstain = np.asarray(abstain, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    out = {}
    for c in CLASSES:
        m = cls_items == c
        out[c] = {"n": int(m.sum()), "abstain": int(abstain[m].sum()), "correct": int(correct[m].sum())}
    return out


def sample_test_items(cls, split, per_class: int, seed: int = 0):
    """Stratified sample of test-split indices, up to per_class from each class."""
    cls = np.asarray(cls)
    split = np.asarray(split)
    rng = np.random.default_rng(seed)
    picked = []
    for c in CLASSES:
        pool = np.flatnonzero((split == "test") & (cls == c))
        k = min(per_class, len(pool))
        if k:
            picked += list(rng.choice(pool, k, replace=False))
    return np.sort(np.array(picked, dtype=int))


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, float(centre - half), float(centre + half))
