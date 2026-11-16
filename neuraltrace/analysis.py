"""Turn results/*.json into figures and a numbers-only summary.

Everything printed in the README/paper is read from the tables built here,
so numbers cannot drift from the raw results.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
LANG_COLOR = {"en": BLUE, "es": ORANGE, "hi": AQUA}
LANG_NAME = {"en": "English", "es": "Spanish", "hi": "Hindi"}
MARK = {"en": "o", "es": "s", "hi": "^"}
UNKNOWN = ("unk_real", "fict")


# --------------------------------------------------------------------- loading
def load_results(path: str = "results"):
    """Final per-model result files, smallest model first."""
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        if f.endswith(".partial.json") or os.path.basename(f) == "summary.json":
            continue
        with open(f) as fh:
            d = json.load(fh)
        if "langs" in d and "n_params_b" in d:
            out.append(d)
    return sorted(out, key=lambda d: d["n_params_b"])


def short(d) -> str:
    return d["model"].split("/")[-1].replace("-Instruct", "").replace("Qwen2.5-", "Qwen2.5 ")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


# ---------------------------------------------------------------- steering math
def rates(counts: dict) -> dict:
    """Abstention on unknowns (unk_real+fict), false abstention and accuracy on known."""
    nu = sum(counts[c]["n"] for c in UNKNOWN)
    au = sum(counts[c]["abstain"] for c in UNKNOWN)
    kn = counts["known"]
    return {
        "abs_unknown": wilson(au, nu),
        "abs_known": wilson(kn["abstain"], kn["n"]),
        "acc_known": wilson(kn["correct"], kn["n"]),
        "n_unknown": nu,
        "n_known": kn["n"],
    }


def null_band(conditions: dict, prefix: tuple, alpha: float, key: str = "abs_unknown"):
    """Mean and range of a metric over null directions at one alpha."""
    vals = []
    for name, c in conditions.items():
        if name.startswith(prefix) and c.get("alpha") == alpha:
            vals.append(rates(c["counts"])[key][0])
    if not vals:
        return None
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)), len(vals)


def steering_table(d: dict, lang: str = "en"):
    """Rows of (condition, alpha, abs_unknown, abs_known, acc_known) for one model/language."""
    st = d["langs"][lang].get("steering")
    if not st:
        return None
    rows = {}
    for name, c in st["conditions"].items():
        r = rates(c["counts"])
        rows[name] = {"kind": c["kind"], "alpha": c.get("alpha"), **{k: r[k] for k in ("abs_unknown", "abs_known", "acc_known")}}
    return rows


# ---------------------------------------------------------------------- figures
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def fig_layerwise(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cols = [BLUE, ORANGE, AQUA, MAGENTA]
    for i, d in enumerate(results):
        en = d["langs"].get("en", {}).get("layerwise", {}).get("C")
        if not en:
            continue
        v = np.array(en["dom_val"], dtype=float)
        x = np.linspace(0, 1, len(v))
        ax.plot(x[:-1], v[:-1], color=cols[i % 4], lw=2, ls=["-", "--", "-.", ":"][i % 4], label=f"{short(d)} ({d['n_params_b']:.1f}B)")
    ax.axhline(0.5, color=INK2, lw=1)
    ax.set_xlabel("relative depth (layer / n_layers)")
    ax.set_ylabel("AUROC, known vs not-known (validation)")
    ax.set_title("Where the 'do I know this entity?' signal lives (English)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_scale(results, out):
    """AUROC at the selected layer vs model size, per language, with surface baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task, title in zip(axes, ("A", "C"), ("known vs fictional (A)", "known vs everything else (C)")):
        for lg in ("en", "es", "hi"):
            xs, ys, lo, hi = [], [], [], []
            for d in results:
                t = d["langs"].get(lg, {}).get("at_best_layer", {}).get("tasks", {}).get(task)
                if t:
                    xs.append(d["n_params_b"]); ys.append(t["dom"][0]); lo.append(t["dom"][1]); hi.append(t["dom"][2])
            if xs:
                ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], color=LANG_COLOR[lg], marker=MARK[lg],
                            ms=6, lw=1.8, capsize=3, label=LANG_NAME[lg])
        sb = [d["langs"]["en"]["at_best_layer"]["tasks"][task]["surface_baseline"] for d in results
              if d["langs"].get("en", {}).get("at_best_layer", {}).get("tasks", {}).get(task)]
        if sb:
            ax.plot([d["n_params_b"] for d in results][: len(sb)], sb, color=INK2, ls=":", lw=1.5, label="surface-only baseline (en)")
        ax.axhline(0.5, color=INK2, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (billions)")
        ax.set_title(title)
    axes[0].set_ylabel("test AUROC (difference-of-means, 95% bootstrap CI)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_steering(results, out, lang: str = "en"):
    """Abstention on unknown entities vs steering strength, with null bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    ms = [d for d in results if d["langs"].get(lang, {}).get("steering")]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.8), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ms):
        conds = d["langs"][lang]["steering"]["conditions"]
        pts = sorted((c["alpha"], rates(c["counts"])["abs_unknown"], n) for n, c in conds.items() if c["kind"] == "dom")
        if pts:
            a = [p[0] for p in pts]; y = [p[1][0] for p in pts]
            ax.errorbar(a, y, yerr=[np.array(y) - [p[1][1] for p in pts], [p[1][2] for p in pts] - np.array(y)],
                        color=BLUE, marker="o", lw=2, capsize=3, label="difference-of-means")
        pp = sorted((c["alpha"], rates(c["counts"])["abs_unknown"][0]) for n, c in conds.items() if c["kind"] == "probe")
        if pp:
            ax.plot([p[0] for p in pp], [p[1] for p in pp], color=MAGENTA, marker="s", ls="--", lw=1.6, label="probe")
        for pref, col, nm in ((("rand",), INK2, "random dirs"), (("shuf",), ORANGE, "shuffled-label dirs")):
            xs, mids, los, his = [], [], [], []
            for a in (-1.0, 1.0):
                nb = null_band(conds, pref, a)
                if nb:
                    xs.append(a); mids.append(nb[0]); los.append(nb[1]); his.append(nb[2])
            if xs:
                ax.errorbar(xs, mids, yerr=[np.array(mids) - los, np.array(his) - mids], color=col, marker="D", ms=5,
                            ls="none", capsize=3, label=nm + " (range)")
        base = rates(conds["baseline"]["counts"])["abs_unknown"][0]
        pa = rates(conds["prompt_abstain"]["counts"])["abs_unknown"][0]
        ax.axhline(base, color=INK2, lw=1, ls=":")
        ax.axhline(pa, color=AQUA, lw=1.4, ls="-.", label="prompt-based abstention")
        ax.set_title(short(d))
        ax.set_xlabel("steering strength alpha (class-gap units)")
    axes[0][0].set_ylabel("abstention rate on unknown entities")
    axes[0][0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Steering the knowledge direction (alpha<0 pushes toward 'I don't know')", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_crosslingual(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    ms = [d for d in results if d.get("crosslingual")]
    if not ms:
        return
    langs = ["en", "es", "hi"]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms), 3.4), squeeze=False)
    for ax, d in zip(axes[0], ms):
        auc = d["crosslingual"]["auroc"]
        M = np.full((3, 3), np.nan)
        for i, a in enumerate(langs):
            for j, b in enumerate(langs):
                if f"{a}->{b}" in auc:
                    M[i, j] = auc[f"{a}->{b}"]
        ax.imshow(M, vmin=0.5, vmax=1.0, cmap="Blues")
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color=INK, fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(langs); ax.set_yticks(range(3)); ax.set_yticklabels(langs)
        ax.set_xlabel("test language"); ax.set_title(short(d))
    axes[0][0].set_ylabel("direction fitted on")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------- summary
def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def summary_markdown(results) -> str:
    L = []
    L.append("## Detection (test AUROC of the difference-of-means direction at the validation-selected layer)\n")
    L.append("| model | lang | layer / n | A: known vs fictional | A token-matched | C: known vs rest | probe C | surface-only C | pop. Spearman |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in results:
        for lg in ("en", "es", "hi"):
            b = d["langs"].get(lg, {}).get("at_best_layer")
            if not b:
                L.append(f"| {short(d)} | {lg} | – | insufficient data | | | | | |")
                continue
            A, C = b["tasks"].get("A"), b["tasks"].get("C")
            ci = lambda t, k="dom": f"{t[k][0]:.3f} [{t[k][1]:.3f}, {t[k][2]:.3f}]" if t else "n/a"
            am = f3(A["matched"]["dom"]) if A and "matched" in A else "n/a"
            L.append(f"| {short(d)} | {lg} | {b['layer']}/{d['n_layers']} | {ci(A)} | {am} | {ci(C)} | {ci(C, 'probe')} | "
                     f"{f3(C['surface_baseline']) if C else 'n/a'} | {f3(b.get('popularity_spearman'))} |")
    L.append("\n## Behaviour of the unmodified model (plain prompt)\n")
    L.append("| model | lang | accuracy on real | abstain on real | abstain on fictional | with abstention prompt: abstain on fictional |")
    L.append("|---|---|---|---|---|---|")
    for d in results:
        for lg in ("en", "es", "hi"):
            o = d["langs"].get(lg, {}).get("behaviour", {})
            o = o.get("overall", o)
            if o:
                L.append(f"| {short(d)} | {lg} | {f3(o.get('acc_plain'))} | {f3(o.get('abstain_real_plain'))} | "
                         f"{f3(o.get('abstain_fict_plain'))} | {f3(o.get('abstain_fict_prompted'))} |")
    L.append("\n## Steering (English): abstention rate on unknown entities / on known entities, 95% Wilson CI\n")
    L.append("| model | condition | abstain (unknown) | abstain (known) | correct (known) |")
    L.append("|---|---|---|---|---|")
    for d in results:
        rows = steering_table(d, "en")
        if not rows:
            continue
        for name in ("baseline", "prompt_abstain", "dom_a-2", "dom_a-1", "dom_a+1", "dom_a+2", "probe_a-1", "probe_a+1", "dom_ablate"):
            r = rows.get(name)
            if r:
                f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
                L.append(f"| {short(d)} | {name} | {f(r['abs_unknown'])} | {f(r['abs_known'])} | {f(r['acc_known'])} |")
        conds = d["langs"]["en"]["steering"]["conditions"]
        for pref, nm in ((("rand",), "random dirs"), (("shuf",), "shuffled-label dirs")):
            for a in (-1.0, 1.0):
                nb = null_band(conds, pref, a)
                if nb:
                    L.append(f"| {short(d)} | {nm} alpha={a:+g} (n={nb[3]}) | mean {nb[0]:.3f}, range [{nb[1]:.3f}, {nb[2]:.3f}] | | |")
    return "\n".join(L) + "\n"


def run_all(results_dir="results", fig_dir="figures"):
    res = load_results(results_dir)
    if not res:
        raise SystemExit(f"no result files in {results_dir}")
    os.makedirs(fig_dir, exist_ok=True)
    fig_layerwise(res, os.path.join(fig_dir, "fig1_layerwise.png"))
    fig_scale(res, os.path.join(fig_dir, "fig2_scale.png"))
    fig_steering(res, os.path.join(fig_dir, "fig3_steering.png"))
    fig_crosslingual(res, os.path.join(fig_dir, "fig4_crosslingual.png"))
    md = summary_markdown(res)
    with open(os.path.join(results_dir, "summary.md"), "w") as f:
        f.write(md)
    return md


if __name__ == "__main__":
    import sys
    print(run_all(*(sys.argv[1:3])))
