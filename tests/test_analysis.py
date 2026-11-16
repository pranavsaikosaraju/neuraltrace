import json

import numpy as np

from neuraltrace import analysis


def _counts(ab_u, ab_k):
    return {"known": {"n": 40, "abstain": ab_k, "correct": 20},
            "unk_real": {"n": 40, "abstain": ab_u, "correct": 1},
            "fict": {"n": 40, "abstain": ab_u, "correct": 0}}


def _fake(path, name, params):
    n = 12
    lw = {"dom_val": list(np.linspace(0.5, 0.9, n)), "dom_test": list(np.linspace(0.5, 0.88, n)),
          "probe_val": list(np.linspace(0.5, 0.92, n)), "probe_test": list(np.linspace(0.5, 0.9, n))}
    task = {"n_test": 100, "dom": [0.88, 0.8, 0.94], "probe": [0.9, 0.82, 0.95], "surface_baseline": 0.6,
            "matched": {"n": 50, "frac_kept": 0.5, "dom": 0.8, "probe": 0.82, "surface_baseline": 0.5}}
    conds = {"baseline": {"kind": "plain", "counts": _counts(0, 0)},
             "prompt_abstain": {"kind": "prompt", "counts": _counts(20, 5)},
             "dom_a-1": {"kind": "dom", "alpha": -1.0, "counts": _counts(30, 10)},
             "dom_a+1": {"kind": "dom", "alpha": 1.0, "counts": _counts(0, 0)},
             "rand0_a-1": {"kind": "random", "alpha": -1.0, "counts": _counts(1, 0)},
             "shuf0_a-1": {"kind": "shuffle", "alpha": -1.0, "counts": _counts(2, 0)}}
    lang = {"n": 300, "behaviour": {"overall": {"acc_plain": 0.2, "abstain_real_plain": 0.0,
                                                 "abstain_fict_plain": 0.0, "abstain_fict_prompted": 0.4}},
            "layerwise": {"C": lw, "A": lw}, "best_layer": 8,
            "at_best_layer": {"layer": 8, "tasks": {"A": task, "C": task}, "cross_type": {}, "popularity_spearman": 0.4},
            "steering": {"layer": 8, "delta": 3.0, "n_items": 120, "conditions": conds}}
    d = {"model": f"Qwen/{name}", "n_layers": n, "d_model": 64, "n_params_b": params,
         "langs": {"en": lang, "es": lang, "hi": {"behaviour": {}, "n": 10}},
         "crosslingual": {"layer": 8, "auroc": {"en->es": 0.8, "en->en": 0.9, "es->es": 0.9, "es->en": 0.85}, "cosine": {}}}
    json.dump(d, open(path, "w"))


def test_analysis_end_to_end(tmp_path):
    _fake(tmp_path / "a.json", "Qwen2.5-1.5B-Instruct", 1.5)
    _fake(tmp_path / "b.json", "Qwen2.5-0.5B-Instruct", 0.5)
    (tmp_path / "b.partial.json").write_text("{}")
    res = analysis.load_results(str(tmp_path))
    assert [r["n_params_b"] for r in res] == [0.5, 1.5]
    md = analysis.run_all(str(tmp_path), str(tmp_path / "figs"))
    assert "insufficient data" in md and "dom_a-1" in md and "random dirs" in md
    for f in ("fig1_layerwise", "fig2_scale", "fig3_steering", "fig4_crosslingual"):
        assert (tmp_path / "figs" / f"{f}.png").stat().st_size > 5000


def test_rates_and_null_band():
    r = analysis.rates(_counts(30, 10))
    assert abs(r["abs_unknown"][0] - 0.75) < 1e-9 and abs(r["abs_known"][0] - 0.25) < 1e-9
