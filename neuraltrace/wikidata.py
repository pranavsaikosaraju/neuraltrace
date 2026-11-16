"""Fetch real entities from Wikidata and assemble the benchmark.

Each real entity has a gold answer (country / director / performer) in the
question language plus the English label as an alias. Entities are sampled in
popularity bands (number of Wikipedia sitelinks) so the 'known' and 'unknown'
classes both have real mass. Fictional entities are added by name splicing
(see data.py), then every record gets a deterministic train/val/test split.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np

from .data import make_fictional, split_of

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "NeuralTrace/0.1 (https://github.com/pranavsaikosaraju/neuraltrace)"

TYPE_SPEC = {
    "city": {"cls": "Q515", "prop": "P17", "surname": False},
    "film": {"cls": "Q11424", "prop": "P57", "surname": True},
    "album": {"cls": "Q482994", "prop": "P175", "surname": True},
}

BANDS = [(1, 3), (3, 6), (6, 12), (12, 25), (25, 100000)]


def build_query(type_: str, lang: str, lo: int, hi: int, limit: int) -> str:
    spec = TYPE_SPEC[type_]
    en_opt = "" if lang == "en" else '?a rdfs:label ?ae . FILTER(LANG(?ae) = "en")'
    en_sel = "" if lang == "en" else "?ae"
    return f"""
SELECT ?e ?el ?al {en_sel} ?s WHERE {{
  ?e wdt:P31 wd:{spec['cls']} ; wdt:{spec['prop']} ?a ; wikibase:sitelinks ?s .
  FILTER(?s >= {lo} && ?s < {hi})
  ?e rdfs:label ?el . FILTER(LANG(?el) = "{lang}")
  ?a rdfs:label ?al . FILTER(LANG(?al) = "{lang}")
  OPTIONAL {{ {en_opt} }}
}} LIMIT {limit}
"""


def run_sparql(query: str, retries: int = 5, timeout: int = 75):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:  # network, 429, 504 ...
            last = e
            time.sleep(6 * (k + 1))
    raise RuntimeError(f"SPARQL failed after {retries} tries: {last}")


def _val(row, key):
    return row[key]["value"] if key in row else None


def group_rows(rows, lang: str):
    """Collapse rows to one record per entity with all gold answers."""
    by_q: dict[str, dict] = {}
    for r in rows:
        qid = _val(r, "e").rsplit("/", 1)[-1]
        rec = by_q.setdefault(
            qid,
            {"qid": qid, "entity": _val(r, "el"), "golds": [], "sitelinks": int(float(_val(r, "s")))},
        )
        for key in ("al", "ae"):
            v = _val(r, key)
            if v and v not in rec["golds"]:
                rec["golds"].append(v)
    return list(by_q.values())


def clean_entities(recs):
    """Drop entities whose label is shared with another entity, or that look
    like placeholders (very short, or containing digits)."""
    count: dict[str, int] = defaultdict(int)
    for r in recs:
        count[r["entity"].lower()] += 1
    out = []
    for r in recs:
        e = r["entity"]
        if count[e.lower()] > 1 or len(e) < 3 or any(c.isdigit() for c in e):
            continue
        out.append(r)
    return out


def sample_band(recs, lo: int, hi: int, n: int, rng):
    pool = [r for r in recs if lo <= r["sitelinks"] < hi]
    if len(pool) <= n:
        return pool
    idx = rng.choice(len(pool), n, replace=False)
    return [pool[i] for i in idx]


def fetch_type_lang(type_: str, lang: str, n_per_band: int, seed: int, log=print):
    rng = np.random.default_rng(seed)
    pooled = []
    for lo, hi in BANDS:
        rows = run_sparql(build_query(type_, lang, lo, hi, limit=max(1500, n_per_band * 12)))
        recs = clean_entities(group_rows(rows, lang))
        take = sample_band(recs, lo, hi, n_per_band, rng)
        log(f"  {type_}/{lang} sitelinks[{lo},{hi}): fetched {len(recs)} -> kept {len(take)}")
        pooled += take
        time.sleep(2)
    return pooled


def assemble(
    langs=("en", "es", "hi"),
    types=("city", "film", "album"),
    n_per_band: int = 120,
    seed: int = 0,
    log=print,
):
    """Return a list of records (real + fictional) for all langs and types."""
    records = []
    for lang in langs:
        for t in types:
            real = fetch_type_lang(t, lang, n_per_band, seed, log)
            if len(real) < 20:
                log(f"  WARNING: only {len(real)} real {t}/{lang} entities; skipping type")
                continue
            for r in real:
                records.append(
                    {
                        "id": r["qid"],
                        "type": t,
                        "lang": lang,
                        "entity": r["entity"],
                        "real": True,
                        "golds": r["golds"],
                        "sitelinks": r["sitelinks"],
                        "surname": TYPE_SPEC[t]["surname"],
                    }
                )
            names = [r["entity"] for r in real]
            fakes = make_fictional(names, len(real), seed=seed)
            for k, f in enumerate(fakes):
                records.append(
                    {
                        "id": f"F-{t}-{lang}-{k}",
                        "type": t,
                        "lang": lang,
                        "entity": f,
                        "real": False,
                        "golds": [],
                        "sitelinks": 0,
                        "surname": False,
                    }
                )
            log(f"{lang}/{t}: {len(real)} real + {len(fakes)} fictional")
    for r in records:
        r["split"] = split_of(r["id"])
    return records


def write_jsonl(records, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
