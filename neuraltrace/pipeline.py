"""End-to-end experiment for one model: behaviour, detection, steering,
cross-lingual transfer. Needs torch/transformers (see modelio.py)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import modelio
from .detection import at_layer, best_layer, layerwise
from .directions import diff_of_means, fit_probe, random_directions, score
from .metrics import auroc
from .prompts import build_messages
from .scoring import outcome, outcome_fictional
from . import selective
from .steering import build_conditions, sample_test_items, summarize, sweep_layers, triple

BANDS = [(1, 3), (3, 6), (6, 12), (12, 25), (25, 100000)]


@dataclass
class Config:
    langs: tuple = ("en", "es", "hi")
    steer_langs: tuple = ("en",)
    max_new_tokens: int = 24
    gen_batch: int = 32
    act_batch: int = 16
    steer_per_class: int = 120
    alphas: tuple = (-2.0, -1.0, 1.0, 2.0)
    n_random: int = 8
    n_shuffle: int = 4
    sweep_langs: tuple = ()          # languages for the steering layer sweep
    sweep_alphas: tuple = (-1.0, -2.0)
    sweep_null: int = 3              # random null directions per layer and alpha
    limit_per_lang: int | None = None  # smoke tests
    n_boot: int = 500
    do_probe: bool = True
    seed: int = 0
    extra: dict = field(default_factory=dict)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def judge(recs, outs, lang):
    """Per-item class, abstain flag, correct flag from raw generations."""
    cls, ab, co = [], [], []
    for r, o in zip(recs, outs):
        if r["real"]:
            oc = outcome(o, r["golds"], lang, r.get("surname", False))
            cls.append("known" if oc == "correct" else "unk_real")
            ab.append(oc == "abstain")
            co.append(oc == "correct")
        else:
            oc = outcome_fictional(o, lang)
            cls.append("fict")
            ab.append(oc == "abstain")
            co.append(False)
    return np.array(cls), np.array(ab), np.array(co)


def _rate(x):
    return float(np.mean(x)) if len(x) else None


def behaviour_summary(recs, ab0, co0, ab1, co1):
    real = np.array([r["real"] for r in recs])
    typ = np.array([r["type"] for r in recs])
    sl = np.array([r["sitelinks"] for r in recs])

    def block(mask):
        rl, fc = mask & real, mask & ~real
        return {
            "n_real": int(rl.sum()),
            "n_fict": int(fc.sum()),
            "acc_plain": _rate(co0[rl]),
            "abstain_real_plain": _rate(ab0[rl]),
            "abstain_fict_plain": _rate(ab0[fc]),
            "acc_prompted": _rate(co1[rl]),
            "abstain_real_prompted": _rate(ab1[rl]),
            "abstain_fict_prompted": _rate(ab1[fc]),
        }

    out = {"overall": block(np.ones(len(recs), bool)), "by_type": {}, "by_band": {}}
    for t in sorted(set(typ.tolist())):
        out["by_type"][t] = block(typ == t)
    for lo, hi in BANDS:
        m = real & (sl >= lo) & (sl < hi)
        if m.sum():
            out["by_band"][f"{lo}-{hi}"] = {
                "n": int(m.sum()),
                "acc_plain": _rate(co0[m]),
                "abstain_plain": _rate(ab0[m]),
            }
    return out


def prepare(tok, recs, lang):
    plain, prompted, pos, nent = [], [], [], []
    for r in recs:
        m0, _ = build_messages(r["type"], lang, r["entity"], abstain=False)
        t0 = modelio.render(tok, m0)
        plain.append(t0)
        pos.append(modelio.entity_pos_from_end(tok, t0, r["entity"]))
        m1, _ = build_messages(r["type"], lang, r["entity"], abstain=True)
        prompted.append(modelio.render(tok, m1))
        nent.append(modelio.n_tokens(tok, r["entity"]))
    surface = np.array([[len(r["entity"]), len(r["entity"].split()), n] for r, n in zip(recs, nent)])
    return plain, prompted, pos, surface


def run_steering(model, tok, recs, lang, plain, prompted, state, layer, cfg, ab_fn):
    """Generate under each steering condition on a stratified test sample."""
    cls, split, ent = state["cls"], state["split"], state["ent"]
    items = sample_test_items(cls, split, cfg.steer_per_class, cfg.seed)
    sub_recs = [recs[i] for i in items]
    X = ent[layer].astype(np.float32)
    pos_t, neg_t = cls == "known", cls != "known"
    itr = np.flatnonzero((split == "train"))
    ytr = pos_t[itr].astype(int)
    d_dom = diff_of_means(X[itr], ytr)
    delta = float(score(X[itr][ytr == 1], d_dom).mean() - score(X[itr][ytr == 0], d_dom).mean())
    w, _ = fit_probe(X[itr], ytr)
    d_probe = w / (np.linalg.norm(w) + 1e-12)
    rng = np.random.default_rng(cfg.seed + 7)
    rand_dirs = random_directions(X.shape[1], cfg.n_random, seed=cfg.seed + 11)
    shuf_dirs = []
    for k in range(cfg.n_shuffle):
        yp = rng.permutation(ytr)
        shuf_dirs.append(diff_of_means(X[itr], yp))
    log(f"  steering lang={lang} layer={layer} items={len(items)} delta={delta:.2f}")

    results = {}
    for cond in build_conditions(cfg.alphas, n_random=cfg.n_random, n_shuffle=cfg.n_shuffle):
        steer, texts = None, [plain[i] for i in items]
        k, a = cond["kind"], cond.get("alpha", 0.0)
        if k == "prompt":
            texts = [prompted[i] for i in items]
        elif k == "dom":
            steer = modelio.Steer(model, layer, d_dom, "add", a * delta)
        elif k == "probe":
            steer = modelio.Steer(model, layer, d_probe, "add", a * delta)
        elif k == "ablate":
            steer = modelio.Steer(model, layer, d_dom, "ablate")
        elif k == "random":
            steer = modelio.Steer(model, layer, rand_dirs[cond["k"]], "add", a * delta)
        elif k == "shuffle":
            steer = modelio.Steer(model, layer, shuf_dirs[cond["k"]], "add", a * delta)
        t0 = time.time()
        outs = modelio.generate(model, tok, texts, cfg.max_new_tokens, cfg.gen_batch, steer=steer)
        _, ab, co = judge(sub_recs, outs, lang)
        # class labels come from the *unsteered* plain run, not the steered output
        results[cond["name"]] = {**{kk: v for kk, v in cond.items() if kk != "name"},
                                 "counts": summarize(cls[items], ab, co),
                                 "sample_outputs": outs[:3]}
        log(f"    {cond['name']:<16} {time.time() - t0:5.1f}s")
    return {"layer": int(layer), "delta": delta, "n_items": int(len(items)), "conditions": results}


def run_sweep(model, tok, recs, lang, plain, state, cfg):
    """Steer with each layer's own difference-of-means direction, with random
    nulls of the same norm at the same layer. Tests whether the failure to
    induce abstention at the detection layer is specific to that layer."""
    cls, split, ent = state["cls"], state["split"], state["ent"]
    items = sample_test_items(cls, split, cfg.steer_per_class, cfg.seed)
    sub, texts = [recs[i] for i in items], [plain[i] for i in items]
    itr = np.flatnonzero(split == "train")
    ytr = (cls[itr] == "known").astype(int)

    def gen(steer=None):
        outs = modelio.generate(model, tok, texts, cfg.max_new_tokens, cfg.gen_batch, steer=steer)
        _, ab, co = judge(sub, outs, lang)
        return triple(summarize(cls[items], ab, co))

    out = {"n_items": int(len(items)), "base": gen(), "layers": {}}
    for L in sweep_layers(ent.shape[0]):
        X = ent[L].astype(np.float32)
        d = diff_of_means(X[itr], ytr)
        delta = float(score(X[itr][ytr == 1], d).mean() - score(X[itr][ytr == 0], d).mean())
        rd = random_directions(X.shape[1], cfg.sweep_null, seed=cfg.seed + 100 + L)
        e = {"delta": round(delta, 3), "dom": {}, "rand": {}}
        for a in cfg.sweep_alphas:
            e["dom"][f"{a:g}"] = gen(modelio.Steer(model, L, d, "add", a * delta))
            e["rand"][f"{a:g}"] = [gen(modelio.Steer(model, L, r, "add", a * delta)) for r in rd]
        out["layers"][str(L)] = e
        log(f"    sweep layer {L}: delta={delta:.2f} dom={e['dom']}")
    return out


def run_selective(X, cls, split, co0, ab1, co1, cfg):
    """Detector-as-abstention-rule vs the abstention prompt (see selective.py)."""
    known = cls == "known"
    tr, va, te = (np.flatnonzero(split == k) for k in ("train", "val", "test"))
    if known[va].sum() < 8 or known[te].sum() < 8:
        return None
    d = diff_of_means(X[tr], known[tr].astype(int))
    w, b = fit_probe(X[tr], known[tr].astype(int))
    out = {}
    for name, s in (("dom", score(X, d)), ("probe", score(X, w, b))):
        out[name] = selective.evaluate(s[va], known[va], s[te], known[te], co0[te], ab1[te], co1[te], cfg.n_boot, cfg.seed)
    return out


def run_language(model, tok, records, lang, cfg):
    recs = [r for r in records if r["lang"] == lang]
    if cfg.limit_per_lang and len(recs) > cfg.limit_per_lang:
        rng = np.random.default_rng(cfg.seed)
        recs = [recs[i] for i in sorted(rng.choice(len(recs), cfg.limit_per_lang, replace=False))]
    log(f"[{lang}] {len(recs)} records")
    plain, prompted, pos, surface = prepare(tok, recs, lang)

    out0 = modelio.generate(model, tok, plain, cfg.max_new_tokens, cfg.gen_batch)
    out1 = modelio.generate(model, tok, prompted, cfg.max_new_tokens, cfg.gen_batch)
    cls, ab0, co0 = judge(recs, out0, lang)
    _, ab1, co1 = judge(recs, out1, lang)
    beh = behaviour_summary(recs, ab0, co0, ab1, co1)
    log(f"[{lang}] behaviour: {beh['overall']}")
    examples = [
        {"entity": r["entity"], "type": r["type"], "class": c, "output": o}
        for r, c, o in list(zip(recs, cls, out0))[:: max(1, len(recs) // 24)]
    ]

    ent, last = modelio.extract_activations(model, tok, plain, pos, cfg.act_batch)
    split = np.array([r["split"] for r in recs])
    typ = np.array([r["type"] for r in recs])
    sitelinks = np.array([r["sitelinks"] for r in recs])
    per_layer = layerwise(ent, cls, split, do_probe=cfg.do_probe)
    last_layer = layerwise(last, cls, split, do_probe=False)
    del last
    if "C" not in per_layer:
        log(f"[{lang}] not enough data per class for detection; skipping")
        return {"behaviour": beh, "examples": examples, "n": len(recs)}, None
    bl = best_layer(per_layer, "C", "dom_val")
    stats, dirs = at_layer(ent, bl, cls, split, typ, sitelinks, surface, cfg.n_boot)
    log(f"[{lang}] best layer {bl}/{ent.shape[0]}: C dom AUROC {stats['tasks']['C']['dom'][0]:.3f}")

    result = {
        "n": len(recs),
        "class_counts": {c: int((cls == c).sum()) for c in ("known", "unk_real", "fict")},
        "behaviour": beh,
        "examples": examples,
        "layerwise": per_layer,
        "layerwise_last_token_dom": last_layer,
        "best_layer": bl,
        "at_best_layer": stats,
    }
    state = {"ent": ent, "cls": cls, "split": split, "typ": typ, "dirs": dirs}
    sel = run_selective(ent[bl].astype(np.float32), cls, split, co0, ab1, co1, cfg)
    if sel:
        result["selective"] = sel
        log(f"[{lang}] selective: prompt {sel['dom']['prompt']} detector {sel['dom']['det']}")
    if lang in cfg.steer_langs:
        result["steering"] = run_steering(model, tok, recs, lang, plain, prompted, state, bl, cfg, None)
    if lang in cfg.sweep_langs:
        result["sweep"] = run_sweep(model, tok, recs, lang, plain, state, cfg)
    return result, state


def crosslingual(states, per_lang_results):
    langs = list(states)
    if len(langs) < 2:
        return None
    n_layers = len(next(iter(per_lang_results.values()))["layerwise"]["C"]["dom_val"])
    mean_val = np.nanmean([per_lang_results[lg]["layerwise"]["C"]["dom_val"] for lg in langs], axis=0)
    layer = int(np.nanargmax(mean_val[: n_layers - 1]))
    dirs, tests = {}, {}
    for lg in langs:
        s = states[lg]
        X = s["ent"][layer].astype(np.float32)
        pos = s["cls"] == "known"
        itr = np.flatnonzero(s["split"] == "train")
        dirs[lg] = diff_of_means(X[itr], pos[itr].astype(int))
        ite = np.flatnonzero(s["split"] == "test")
        tests[lg] = (X[ite], pos[ite].astype(int))
    auc, cos = {}, {}
    for a in langs:
        for b in langs:
            Xb, yb = tests[b]
            auc[f"{a}->{b}"] = auroc(yb, score(Xb, dirs[a]))
            cos[f"{a}|{b}"] = float(dirs[a] @ dirs[b])
    return {"layer": layer, "auroc": auc, "cosine": cos}


def run_model(model_name, records, out_dir, cfg: Config):
    import torch

    torch.manual_seed(cfg.seed)
    short = model_name.split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tok, model = modelio.load(model_name)
    n_layers = model.config.num_hidden_layers
    log(f"loaded {model_name}: {n_layers} layers, d={model.config.hidden_size}")
    langs_out, states = {}, {}
    for lang in cfg.langs:
        res, state = run_language(model, tok, records, lang, cfg)
        langs_out[lang] = res
        if state is not None:
            states[lang] = state
        with open(os.path.join(out_dir, f"{short}.partial.json"), "w") as f:
            json.dump({"model": model_name, "langs": langs_out}, f)
    cl = crosslingual(states, {k: v for k, v in langs_out.items() if k in states})
    result = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": model.config.hidden_size,
        "n_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
        "config": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
        "langs": langs_out,
        "crosslingual": cl,
        "minutes": (time.time() - t0) / 60,
    }
    with open(os.path.join(out_dir, f"{short}.json"), "w") as f:
        json.dump(result, f)
    return result
