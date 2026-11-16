"""Command line: build the dataset, run one model, print compact results."""
from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neuraltrace")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data", help="fetch entities from Wikidata and add fictional names")
    b.add_argument("--out", default="data/entities.jsonl")
    b.add_argument("--langs", nargs="+", default=["en", "es", "hi"])
    b.add_argument("--n-per-band", type=int, default=120)
    b.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="run all experiments for one model")
    r.add_argument("--model", required=True)
    r.add_argument("--data", default="data/entities.jsonl")
    r.add_argument("--out", default="results")
    r.add_argument("--langs", nargs="+", default=["en", "es", "hi"])
    r.add_argument("--steer-langs", nargs="*", default=["en"])
    r.add_argument("--limit", type=int, default=None, help="records per language (smoke test)")
    r.add_argument("--steer-per-class", type=int, default=120)
    r.add_argument("--n-random", type=int, default=8)
    r.add_argument("--n-shuffle", type=int, default=4)
    r.add_argument("--gen-batch", type=int, default=32)
    r.add_argument("--no-probe", action="store_true")
    r.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    if a.cmd == "build-data":
        from .wikidata import assemble, write_jsonl

        recs = assemble(tuple(a.langs), n_per_band=a.n_per_band, seed=a.seed)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        write_jsonl(recs, a.out)
        print(f"wrote {len(recs)} records to {a.out}")
    else:
        from .pipeline import Config, run_model
        from .wikidata import read_jsonl

        cfg = Config(
            langs=tuple(a.langs),
            steer_langs=tuple(a.steer_langs),
            limit_per_lang=a.limit,
            steer_per_class=a.steer_per_class,
            n_random=a.n_random,
            n_shuffle=a.n_shuffle,
            gen_batch=a.gen_batch,
            do_probe=not a.no_probe,
            seed=a.seed,
        )
        res = run_model(a.model, read_jsonl(a.data), a.out, cfg)
        print("NT_RESULT_BEGIN")
        print(json.dumps(res, separators=(",", ":")))
        print("NT_RESULT_END")


if __name__ == "__main__":
    main()
