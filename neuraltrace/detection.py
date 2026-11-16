"""Detection analysis: how well do activations separate known from unknown?

Classes (per entity, from the model's own behaviour):
  known     real entity the model answers correctly
  unk_real  real entity the model gets wrong or abstains on
  fict      invented name (no correct answer exists)
Tasks:
  A  known vs fict      (cleanest; most exposed to name-surface artifacts)
  B  known vs unk_real  (the realistic 'will it hallucinate' case)
  C  known vs everything else
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import match_strata
from .directions import diff_of_means, fit_probe, score
from .metrics import auroc, bootstrap_ci

TASKS = ("A", "B", "C")


def task_masks(cls):
    cls = np.asarray(cls)
    known = cls == "known"
    return {
        "A": (known, cls == "fict"),
        "B": (known, cls == "unk_real"),
        "C": (known, cls != "known"),
    }


def _idx(split, name, mask):
    return np.flatnonzero((np.asarray(split) == name) & mask)


def _task_split(cls, split, task):
    pos, neg = task_masks(cls)[task]
    m = pos | neg
    out = {}
    for name in ("train", "val", "test"):
        i = _idx(split, name, m)
        out[name] = (i, pos[i].astype(int))
    return out


def _ok(y):
    return len(np.unique(y)) == 2


def layerwise(ent, cls, split, do_probe: bool = True):
    """Per-layer val/test AUROC for difference-of-means and probe directions.

    ent: (n_layers, n, d) float16/32. Returns {task: {metric: [per-layer]}}.
    """
    n_layers = ent.shape[0]
    res = {}
    for task in TASKS:
        ts = _task_split(cls, split, task)
        (itr, ytr), (iva, yva), (ite, yte) = ts["train"], ts["val"], ts["test"]
        if not (_ok(ytr) and _ok(yva) and _ok(yte)):
            continue
        r = {k: [] for k in ("dom_val", "dom_test", "probe_val", "probe_test")}
        for l in range(n_layers):
            X = ent[l]
            Xtr = X[itr].astype(np.float32)
            d = diff_of_means(Xtr, ytr)
            r["dom_val"].append(auroc(yva, score(X[iva].astype(np.float32), d)))
            r["dom_test"].append(auroc(yte, score(X[ite].astype(np.float32), d)))
            if do_probe:
                w, b = fit_probe(Xtr, ytr)
                r["probe_val"].append(auroc(yva, score(X[iva].astype(np.float32), w, b)))
                r["probe_test"].append(auroc(yte, score(X[ite].astype(np.float32), w, b)))
        res[task] = {k: [float(v) for v in vals] for k, vals in r.items()}
    return res


def best_layer(per_layer, task: str = "C", key: str = "dom_val", skip_last: int = 1) -> int:
    v = np.array(per_layer[task][key], dtype=float)
    if skip_last:
        v = v[:-skip_last]
    return int(np.nanargmax(v))


def _surface_auroc(S, itr, ytr, ite, yte):
    sc = StandardScaler().fit(S[itr])
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(S[itr]), ytr)
    return auroc(yte, clf.decision_function(sc.transform(S[ite])))


def at_layer(ent, layer, cls, split, typ, sitelinks, surface, n_boot: int = 500):
    """Detailed statistics at one layer.

    surface: (n, 3) array [n_chars, n_words, n_entity_tokens].
    Returns (stats_dict, directions_dict); directions are unit/raw vectors.
    """
    cls = np.asarray(cls)
    typ = np.asarray(typ)
    surface = np.asarray(surface, dtype=float)
    X = ent[layer].astype(np.float32)
    stats: dict = {"layer": int(layer), "tasks": {}}
    dirs: dict = {}
    for task in TASKS:
        ts = _task_split(cls, split, task)
        (itr, ytr), (_, _), (ite, yte) = ts["train"], ts["val"], ts["test"]
        if not (_ok(ytr) and _ok(yte)):
            continue
        d = diff_of_means(X[itr], ytr)
        w, b = fit_probe(X[itr], ytr)
        s_dom, s_probe = score(X[ite], d), score(X[ite], w, b)
        dirs[task] = {"dom": d, "probe_w": w, "probe_b": b}
        p, lo, hi = bootstrap_ci(yte, s_dom, n_boot=n_boot)
        pp, plo, phi = bootstrap_ci(yte, s_probe, n_boot=n_boot)
        t = {
            "n_test": int(len(ite)),
            "dom": [p, lo, hi],
            "probe": [pp, plo, phi],
            "surface_baseline": _surface_auroc(surface, itr, ytr, ite, yte),
        }
        if task == "A":  # token-count-matched control
            pos, neg = task_masks(cls)["A"]
            a = _idx(split, "test", pos)
            f = _idx(split, "test", neg)
            if len(a) and len(f):
                ia, ib = match_strata(surface[a, 2], surface[f, 2])
                if len(ia) >= 20:
                    m = np.concatenate([a[ia], f[ib]])
                    ym = np.concatenate([np.ones(len(ia), int), np.zeros(len(ib), int)])
                    t["matched"] = {
                        "n": int(len(m)),
                        "frac_kept": float(len(m) / (len(a) + len(f))),
                        "dom": auroc(ym, score(X[m], d)),
                        "probe": auroc(ym, score(X[m], w, b)),
                        "surface_baseline": _surface_auroc(surface, itr, ytr, m, ym),
                    }
        stats["tasks"][task] = t

    # cross-type transfer (task C, difference of means)
    types = sorted(set(typ.tolist()))
    pos, neg = task_masks(cls)["C"]
    transfer = {}
    for tx in types:
        itr = _idx(split, "train", (pos | neg) & (typ == tx))
        ytr = pos[itr].astype(int)
        if not _ok(ytr):
            continue
        d = diff_of_means(X[itr], ytr)
        for ty in types:
            ite = _idx(split, "test", (pos | neg) & (typ == ty))
            yte = pos[ite].astype(int)
            if _ok(yte):
                transfer[f"{tx}->{ty}"] = auroc(yte, score(X[ite], d))
    stats["cross_type"] = transfer

    # does the score track popularity among real entities?
    if "C" in dirs:
        real = np.flatnonzero((np.asarray(split) == "test") & (cls != "fict"))
        if len(real) > 10:
            rho = spearmanr(score(X[real], dirs["C"]["dom"]), np.log1p(np.asarray(sitelinks)[real]))[0]
            stats["popularity_spearman"] = float(rho)
    return stats, dirs
