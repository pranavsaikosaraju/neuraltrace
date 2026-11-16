"""Benchmark construction helpers.

The key control: fictional names are built by recombining the *tokens* of real
names so that known/unknown classes match on token count, length and token
rarity. This targets the artifact where an 'unknown entity' feature actually
tracks rare subword sequences (see the Qwen3.6-27B replication).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def make_fictional_names(
    real_names: Sequence[str],
    tokenize: Callable[[str], list[str]],
    detokenize: Callable[[list[str]], str],
    n: int,
    seed: int = 0,
    max_tries: int = 50,
) -> list[str]:
    """Create n names absent from real_names with matched token counts.

    For each target, pick a real name as a length template, then fill each
    position with a token drawn from the same position in other real names.
    """
    rng = np.random.default_rng(seed)
    real_set = {r.lower() for r in real_names}
    toks = [tokenize(r) for r in real_names]
    toks = [t for t in toks if len(t) >= 2]
    if not toks:
        raise ValueError("need real names with >=2 tokens")
    by_pos: dict[int, list[str]] = {}
    for t in toks:
        for i, tok in enumerate(t):
            by_pos.setdefault(i, []).append(tok)
    out: list[str] = []
    seen: set[str] = set()
    tries = 0
    while len(out) < n and tries < n * max_tries:
        tries += 1
        template = toks[rng.integers(len(toks))]
        new = [by_pos[i][rng.integers(len(by_pos[i]))] for i in range(len(template))]
        name = detokenize(new)
        key = name.lower()
        if key in real_set or key in seen or not name.strip():
            continue
        seen.add(key)
        out.append(name)
    return out


def split_by_entity(entity_ids: Sequence[str], test_frac: float = 0.3, seed: int = 0):
    """Boolean test mask where every entity falls wholly in train or test."""
    ids = list(dict.fromkeys(entity_ids))
    rng = np.random.default_rng(seed)
    test_ids = set(rng.choice(ids, size=int(round(len(ids) * test_frac)), replace=False))
    return np.array([e in test_ids for e in entity_ids])


def match_report(a_names: Sequence[str], b_names: Sequence[str], tokenize) -> dict:
    """Compare two name sets on token count and character length."""
    def stats(names):
        n_tok = np.array([len(tokenize(x)) for x in names], dtype=float)
        n_chr = np.array([len(x) for x in names], dtype=float)
        return n_tok, n_chr

    ta, ca = stats(a_names)
    tb, cb = stats(b_names)
    return {
        "mean_tokens": (float(ta.mean()), float(tb.mean())),
        "mean_chars": (float(ca.mean()), float(cb.mean())),
    }


# ---------------------------------------------------------------------------
# Single-word names, deterministic splits, and surface-feature matching
# ---------------------------------------------------------------------------
import hashlib
import unicodedata


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def splice_single_words(
    real_names: Sequence[str], n: int, seed: int = 0, max_tries: int = 100
) -> list[str]:
    """Invent single-word names by splicing the prefix of one real name onto
    the suffix of another, keeping length within one character of a template.
    Cuts never fall before a combining mark, so Devanagari stays well-formed.
    """
    rng = np.random.default_rng(seed)
    pool = [r for r in real_names if " " not in r and len(r) >= 4]
    real_set = {r.lower() for r in real_names}
    if len(pool) < 3:
        raise ValueError("need at least 3 single-word real names")
    out: list[str] = []
    seen: set[str] = set()
    tries = 0
    while len(out) < n and tries < n * max_tries:
        tries += 1
        a, b, t = (pool[i] for i in rng.integers(0, len(pool), 3))
        ca = int(rng.integers(2, max(3, len(a))))
        cb = int(rng.integers(1, max(2, len(b) - 1)))
        if _is_mark(b[cb]) or (ca < len(a) and _is_mark(a[ca])):
            continue
        name = a[:ca] + b[cb:]
        if abs(len(name) - len(t)) > 1 or len(name) < 3:
            continue
        key = name.lower()
        if key in real_set or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def make_fictional(real_names: Sequence[str], n: int, seed: int = 0) -> list[str]:
    """Fictional names matching the word-count mix of the real names."""
    multi = [r for r in real_names if " " in r]
    single = [r for r in real_names if " " not in r]
    frac_multi = len(multi) / max(1, len(real_names))
    n_multi = int(round(n * frac_multi)) if len(multi) >= 8 else 0
    out: list[str] = []
    if n_multi:
        out += make_fictional_names(
            multi, lambda s: s.split(" "), lambda t: " ".join(t), n_multi, seed=seed
        )
    if len(out) < n and len(single) >= 3:
        out += splice_single_words(single, n - len(out), seed=seed + 1)
    return out[:n]


def bucket(entity_id: str, mod: int = 100) -> int:
    """Stable hash bucket so an entity lands in the same split in every
    language and every run."""
    return int(hashlib.sha256(entity_id.encode("utf-8")).hexdigest(), 16) % mod


def split_of(entity_id: str, train: int = 60, val: int = 15) -> str:
    b = bucket(entity_id)
    return "train" if b < train else ("val" if b < train + val else "test")


def match_strata(real_feat: Sequence[float], fict_feat: Sequence[float], seed: int = 0):
    """Indices that equalise two groups on a discrete surface feature.

    For every feature value present in both groups, keep min(count_a, count_b)
    random members of each. Returns (idx_a, idx_b).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(real_feat)
    b = np.asarray(fict_feat)
    keep_a: list[int] = []
    keep_b: list[int] = []
    for v in np.intersect1d(a, b):
        ia = np.flatnonzero(a == v)
        ib = np.flatnonzero(b == v)
        k = min(len(ia), len(ib))
        keep_a += list(rng.choice(ia, k, replace=False))
        keep_b += list(rng.choice(ib, k, replace=False))
    return np.sort(np.array(keep_a, dtype=int)), np.sort(np.array(keep_b, dtype=int))
