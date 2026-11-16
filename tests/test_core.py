import numpy as np

from neuraltrace.data import make_fictional_names, match_report, split_by_entity
from neuraltrace.directions import diff_of_means, fit_probe, layerwise_auroc, random_directions, score
from neuraltrace.metrics import aurc, auroc, bootstrap_ci, risk_coverage


def _toy(n=400, d=32, shift=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d))
    X[:, 0] += shift * y
    return X, y


def test_auroc_perfect_and_chance():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert np.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_bootstrap_ci_contains_point():
    X, y = _toy()
    s = X[:, 0]
    p, lo, hi = bootstrap_ci(y, s, n_boot=200)
    assert lo <= p <= hi and p > 0.9


def test_risk_coverage_monotone_for_good_confidence():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])
    cov, acc = risk_coverage(conf, correct)
    assert acc[1] == 1.0 and acc[-1] == 0.5
    assert aurc(conf, correct) < aurc(conf[::-1], correct)


def test_diff_of_means_recovers_signal_axis():
    X, y = _toy()
    d = diff_of_means(X, y)
    assert abs(np.linalg.norm(d) - 1) < 1e-9
    assert d[0] > 0.9


def test_probe_direction_in_raw_space_scores_correctly():
    X, y = _toy(seed=1)
    w, b = fit_probe(X, y)
    assert auroc(y, score(X, w, b)) > 0.9


def test_random_directions_do_not_detect():
    X, y = _toy(n=2000, seed=2)
    R = random_directions(X.shape[1], 20, seed=3)
    aucs = [auroc(y, X @ r) for r in R]
    assert abs(np.mean(aucs) - 0.5) < 0.1


def test_layerwise_auroc_shape():
    Xtr, ytr = _toy(seed=4)
    Xte, yte = _toy(seed=5)
    acts_tr = np.stack([Xtr, np.random.default_rng(0).standard_normal(Xtr.shape)])
    acts_te = np.stack([Xte, np.random.default_rng(1).standard_normal(Xte.shape)])
    out = layerwise_auroc(acts_tr, ytr, acts_te, yte)
    assert out.shape == (2,) and out[0] > 0.9 and out[1] < 0.65


def _tok(s):
    return s.split(" ")


def _detok(t):
    return " ".join(t)


def test_fictional_names_are_novel_and_length_matched():
    real = ["John Carter Smith", "Maria Lopez Diaz", "Wei Chen Lee", "Anna Berg Holm",
            "Omar Haddad Said", "Lena Fischer Roth", "Raj Patel Shah", "Ivan Petrov Volkov"]
    fake = make_fictional_names(real, _tok, _detok, n=20, seed=0)
    assert len(fake) == 20 and len(set(f.lower() for f in fake)) == 20
    assert not ({f.lower() for f in fake} & {r.lower() for r in real})
    rep = match_report(real, fake, _tok)
    assert rep["mean_tokens"][0] == rep["mean_tokens"][1] == 3.0


def test_split_by_entity_has_no_leakage():
    ids = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
    mask = split_by_entity(ids, test_frac=0.4, seed=0)
    for e in set(ids):
        vals = {bool(m) for i, m in zip(ids, mask) if i == e}
        assert len(vals) == 1


# --- prompts / scoring / data extensions ---------------------------------
from neuraltrace.data import bucket, make_fictional, match_strata, splice_single_words, split_of
from neuraltrace.prompts import build_messages
from neuraltrace.scoring import is_abstention, matches_gold, normalize, outcome, outcome_fictional


def test_normalize_folds_latin_accents_but_keeps_devanagari():
    assert normalize("São Paulo!", "en") == "sao paulo"
    assert normalize("फ़िल्म", "hi") != ""
    assert len(normalize("हिन्दी", "hi")) >= 4  # combining marks kept


def test_matches_gold_and_surname_rule():
    assert matches_gold("It is in Germany.", ["Germany", "Deutschland"])
    assert not matches_gold("It is in Austria.", ["Germany"])
    assert matches_gold("Directed by Kubrick", ["Stanley Kubrick"], surname=True)
    assert not matches_gold("Directed by Kubrick", ["Stanley Kubrick"], surname=False)


def test_abstention_detection_three_languages():
    assert is_abstention("I don't know.", "en")
    assert is_abstention("Lo siento, no tengo información sobre eso.", "es")
    assert is_abstention("मुझे नहीं पता।", "hi")
    assert not is_abstention("France", "en")


def test_outcome_labels():
    assert outcome("France", ["France"]) == "correct"
    assert outcome("I don't know", ["France"]) == "abstain"
    assert outcome("Italy", ["France"]) == "wrong"
    assert outcome_fictional("I don't know") == "abstain"
    assert outcome_fictional("Brazil") == "hallucinate"


def test_build_messages_contains_entity_in_user_turn():
    msgs, user = build_messages("film", "es", "Foo Bar", abstain=True)
    assert "Foo Bar" in user and msgs[1]["content"] == user and msgs[0]["role"] == "system"


def test_splice_single_words_novel_and_wellformed():
    real = ["Hyderabad", "Bangalore", "Chennai", "Mumbai", "Kolkata", "Lucknow", "Nagpur", "Indore"]
    fake = splice_single_words(real, 15, seed=1)
    assert len(fake) == 15 and not ({f.lower() for f in fake} & {r.lower() for r in real})


def test_make_fictional_mix_and_split_stability():
    real = [f"Alpha{i} Beta{j}" for i in range(6) for j in range(3)] + ["Solo", "Moon", "Starlight"]
    out = make_fictional(real, 10, seed=0)
    assert len(out) == 10
    assert split_of("Q42") == split_of("Q42") and 0 <= bucket("x") < 100
    assert {split_of(f"Q{i}") for i in range(300)} == {"train", "val", "test"}


def test_match_strata_equalises_counts():
    a = [3, 3, 3, 4, 4, 5]
    b = [3, 4, 4, 4, 6]
    ia, ib = match_strata(a, b, seed=0)
    assert len(ia) == len(ib) == 3  # one 3 and two 4s
    assert sorted(np.array(a)[ia]) == sorted(np.array(b)[ib])


# --- wikidata parsing (offline) ------------------------------------------
from neuraltrace.wikidata import build_query, clean_entities, group_rows, sample_band


def _row(q, el, al, s, ae=None):
    r = {"e": {"value": f"http://www.wikidata.org/entity/{q}"}, "el": {"value": el},
         "al": {"value": al}, "s": {"value": str(s)}}
    if ae:
        r["ae"] = {"value": ae}
    return r


def test_group_rows_merges_golds_and_clean_drops_duplicates():
    rows = [_row("Q1", "Ciudad Uno", "Francia", 10, "France"), _row("Q1", "Ciudad Uno", "Francia", 10, "France"),
            _row("Q2", "Villa Dos", "Italia", 4), _row("Q3", "Villa Dos", "Italia", 5), _row("Q4", "X1", "Y", 3)]
    recs = group_rows(rows, "es")
    q1 = next(r for r in recs if r["qid"] == "Q1")
    assert q1["golds"] == ["Francia", "France"] and q1["sitelinks"] == 10
    kept = {r["qid"] for r in clean_entities(recs)}
    assert kept == {"Q1"}  # Q2/Q3 share a label, Q4 has a digit


def test_build_query_has_language_and_bounds():
    q = build_query("film", "hi", 3, 6, 100)
    assert 'LANG(?el) = "hi"' in q and "?s >= 3 && ?s < 6" in q and "wd:Q11424" in q


def test_sample_band_respects_bounds():
    recs = [{"qid": f"Q{i}", "entity": f"E{i}", "golds": [], "sitelinks": i} for i in range(1, 40)]
    out = sample_band(recs, 5, 10, 3, np.random.default_rng(0))
    assert len(out) == 3 and all(5 <= r["sitelinks"] < 10 for r in out)


# --- detection on synthetic activations -----------------------------------
from neuraltrace.detection import at_layer, best_layer, layerwise, task_masks


def _synthetic(n=900, L=4, D=24, seed=0, signal_layer=2):
    rng = np.random.default_rng(seed)
    cls = rng.choice(["known", "unk_real", "fict"], n, p=[0.4, 0.3, 0.3])
    split = rng.choice(["train", "val", "test"], n, p=[0.6, 0.15, 0.25])
    typ = rng.choice(["city", "film"], n)
    ent = rng.standard_normal((L, n, D)).astype(np.float16)
    ent[signal_layer, :, 0] += 2.5 * (cls == "known")
    sitelinks = rng.integers(1, 50, n)
    surface = np.stack([rng.integers(4, 12, n), rng.integers(1, 3, n), rng.integers(1, 5, n)], 1)
    return ent, cls, split, typ, sitelinks, surface


def test_task_masks_partition_logic():
    cls = np.array(["known", "unk_real", "fict"])
    m = task_masks(cls)
    assert m["C"][1].tolist() == [False, True, True] and m["A"][1].tolist() == [False, False, True]


def test_layerwise_finds_signal_layer_and_best_layer():
    ent, cls, split, *_ = _synthetic()
    pl = layerwise(ent, cls, split)
    assert set(pl) == {"A", "B", "C"}
    assert best_layer(pl, "C", "dom_val", skip_last=1) == 2
    assert pl["C"]["probe_test"][2] > 0.9 and pl["C"]["dom_test"][0] < 0.65


def test_at_layer_reports_controls_and_transfer():
    ent, cls, split, typ, sl, surf = _synthetic()
    stats, dirs = at_layer(ent, 2, cls, split, typ, sl, surf, n_boot=50)
    a = stats["tasks"]["A"]
    assert a["dom"][0] > 0.9 and a["surface_baseline"] < 0.65
    assert a["matched"]["dom"] > 0.9 and a["matched"]["surface_baseline"] < 0.7
    assert stats["cross_type"]["city->film"] > 0.9
    assert set(dirs["C"]) == {"dom", "probe_w", "probe_b"}


# --- steering design -------------------------------------------------------
from neuraltrace.steering import build_conditions, sample_test_items, summarize, wilson


def test_build_conditions_counts_and_names_unique():
    c = build_conditions(alphas=(-1, 1), probe_alphas=(-1, 1), n_random=3, n_shuffle=2)
    names = [x["name"] for x in c]
    assert len(names) == len(set(names)) == 2 + 2 + 2 + 1 + 3 * 2 + 2 * 2
    assert names[:2] == ["baseline", "prompt_abstain"]


def test_summarize_and_sampling_and_wilson():
    cls = np.array(["known", "known", "fict", "unk_real"])
    ab = np.array([0, 1, 1, 0], bool)
    co = np.array([1, 0, 0, 0], bool)
    s = summarize(cls, ab, co)
    assert s["known"] == {"n": 2, "abstain": 1, "correct": 1} and s["fict"]["abstain"] == 1
    split = np.array(["test"] * 40)
    cl = np.array(["known"] * 20 + ["fict"] * 20)
    idx = sample_test_items(cl, split, 5, seed=0)
    assert len(idx) == 10
    p, lo, hi = wilson(50, 100)
    assert abs(p - 0.5) < 1e-9 and lo < 0.5 < hi


def test_judge_and_behaviour_summary_run_without_torch():
    import importlib.util
    spec = importlib.util.find_spec("torch")
    from neuraltrace.pipeline import judge, behaviour_summary  # imports fine without torch
    recs = [{"real": True, "golds": ["France"], "type": "city", "sitelinks": 5, "surname": False},
            {"real": True, "golds": ["Italy"], "type": "city", "sitelinks": 30, "surname": False},
            {"real": False, "golds": [], "type": "city", "sitelinks": 0, "surname": False}]
    cls, ab, co = judge(recs, ["France", "I don't know", "Brazil"], "en")
    assert cls.tolist() == ["known", "unk_real", "fict"] and ab.tolist() == [False, True, False]
    b = behaviour_summary(recs, ab, co, ab, co)
    assert b["overall"]["n_real"] == 2 and b["by_band"]["3-6"]["acc_plain"] == 1.0
