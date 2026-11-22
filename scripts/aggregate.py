"""Turn results/runs/*.json into tables, figures and headline numbers.

Usage:  python scripts/aggregate.py [--runs results/runs] [--out results]

Writes
  results/RESULTS.md          all tables (markdown, mean ± std over seeds)
  results/summary.json        headline numbers used in the README / resume
  results/figures/*.png       transfer heat-map, rank and data-size curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

LANG_ORDER = ["pa", "hi", "kn", "ur", "en"]
LANG_NAME = {"pa": "Punjabi", "hi": "Hindi", "kn": "Kannada", "ur": "Urdu", "en": "English"}
SHORT = {"muril-base-cased": "MuRIL", "IndicBERTv2-MLM-only": "IndicBERT-v2",
         "xlm-roberta-base": "XLM-R", "gemma-2-2b": "Gemma-2-2B"}


# ----------------------------------------------------------------------------- loading
def load(runs_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows, raw = [], {}
    for f in sorted(runs_dir.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        raw[r["run_id"]] = r
        c = r["config"]
        for lang, m in r["test"].items():
            rows.append({
                "run_id": r["run_id"], "experiment": c["experiment"],
                "model": SHORT.get(c["model"].split("/")[-1], c["model"].split("/")[-1]),
                "method": c["method"], "train_lang": c["train_lang"], "test_lang": lang,
                "seed": c["seed"], "train_size": r["n_train"], "lora_r": c["lora_r"],
                "targets": c["lora_targets"], "macro_f1": 100 * m["macro_f1"],
                "accuracy": 100 * m["accuracy"], "trainable_pct": r["trainable_pct"],
                "trainable_params": r["trainable_params"],
                "peak_mem_gb": r["peak_gpu_mem_gb"], "train_s": r["train_seconds"]})
    return pd.DataFrame(rows), raw


def pm(s: pd.Series) -> str:
    return f"{s.mean():.1f} ± {s.std(ddof=1) if len(s) > 1 else 0:.1f}"


# ----------------------------------------------------------------------------- statistics
def paired_bootstrap(raw: dict, ids_a: list[str], ids_b: list[str], lang: str,
                     n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Paired bootstrap over test examples, pooling seeds.

    Returns (delta, ci_low, ci_high, p) for macro-F1(a) - macro-F1(b), in points.
    Seeds are matched by sorting run ids, so seed k of A is paired with seed k of B.
    """
    rng = np.random.default_rng(seed)
    ya, pa_, pb_ = [], [], []
    for ia, ib in zip(sorted(ids_a), sorted(ids_b)):
        ta, tb = raw[ia]["test"][lang], raw[ib]["test"][lang]
        ya.append(np.array(ta["y_true"]))
        pa_.append(np.array(ta["y_pred"]))
        pb_.append(np.array(tb["y_pred"]))
    n = len(ya[0])
    def score(idx):
        fa = np.mean([f1_score(y[idx], p[idx], average="macro") for y, p in zip(ya, pa_)])
        fb = np.mean([f1_score(y[idx], p[idx], average="macro") for y, p in zip(ya, pb_)])
        return 100 * (fa - fb)
    full = score(np.arange(n))
    boots = np.array([score(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return full, lo, hi, max(p, 1 / n_boot)


# ----------------------------------------------------------------------------- sections
def section_main(df, raw, out_md, figs, summary):
    m = df[df.experiment == "main"]
    if m.empty:
        return
    inl = m[m.train_lang == m.test_lang]
    langs = [l for l in LANG_ORDER if l in set(inl.test_lang)]

    out_md.append("## Table 1 — In-language macro-F1 (mean ± std over seeds)\n")
    hdr = "| Model | Method | " + " | ".join(LANG_NAME[l] for l in langs) + \
          " | Avg | Trainable % | Peak GPU mem (GB) | Train time (s) |"
    out_md += [hdr, "|" + "---|" * (len(langs) + 6)]
    for (model, method), g in inl.groupby(["model", "method"], sort=False):
        cells = [pm(g[g.test_lang == l].macro_f1) for l in langs]
        avg = g.groupby("seed").macro_f1.mean()
        mem = g.peak_mem_gb.dropna()
        mem_s = f"{mem.mean():.2f}" if not mem.empty else "–"
        out_md.append(f"| {model} | {method} | " + " | ".join(cells) +
                      f" | {pm(avg)} | {g.trainable_pct.iloc[0]:.2f} | {mem_s} | {g.train_s.mean():.0f} |")

    # LoRA - full with paired bootstrap significance
    out_md.append("\n## Table 2 — LoRA minus full fine-tuning (macro-F1 points, paired bootstrap)\n")
    out_md += ["| Model | Language | Δ (LoRA − full) | 95% CI | p |", "|---|---|---|---|---|"]
    gaps = []
    for model in inl.model.unique():
        for l in langs:
            a = inl[(inl.model == model) & (inl.method == "lora") & (inl.test_lang == l)].run_id.tolist()
            b = inl[(inl.model == model) & (inl.method == "full") & (inl.test_lang == l)].run_id.tolist()
            if not a or len(a) != len(b):
                continue
            d, lo, hi, p = paired_bootstrap(raw, a, b, l)
            gaps.append({"model": model, "lang": l, "delta": d, "p": p})
            out_md.append(f"| {model} | {LANG_NAME[l]} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {p:.3f} |")

    # Cross-lingual transfer matrix per model/method
    for (model, method), g in m.groupby(["model", "method"], sort=False):
        tr = [l for l in LANG_ORDER if l in set(g.train_lang)]
        te = [l for l in LANG_ORDER if l in set(g.test_lang)]
        mat = np.array([[g[(g.train_lang == a) & (g.test_lang == b)].macro_f1.mean()
                         for b in te] for a in tr])
        if method == "full" and model == inl.model.unique()[0]:
            out_md.append(f"\n## Table 3 — Cross-lingual transfer, {model} ({method}); rows = train, cols = test\n")
            out_md += ["| train ↓ / test → | " + " | ".join(LANG_NAME[b] for b in te) + " |",
                       "|---" * (len(te) + 1) + "|"]
            for a, row in zip(tr, mat):
                out_md.append(f"| {LANG_NAME[a]} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(te)), [LANG_NAME[b] for b in te], rotation=30)
        ax.set_yticks(range(len(tr)), [LANG_NAME[a] for a in tr])
        for i in range(len(tr)):
            for j in range(len(te)):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        color="white" if mat[i, j] < 60 else "black", fontsize=9)
        ax.set_xlabel("test language"); ax.set_ylabel("train language")
        ax.set_title(f"Transfer macro-F1: {model} ({method})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figs / f"transfer_{model}_{method}.png", dpi=160)
        plt.close(fig)

    # Headline numbers (best Indic encoder = highest avg in-language full FT)
    full = inl[inl.method == "full"].groupby("model").macro_f1.mean()
    best = full.idxmax()
    b_inl = inl[(inl.model == best) & (inl.test_lang != "en")]  # Indic languages only
    lora_b = b_inl[b_inl.method == "lora"]
    full_b = b_inl[b_inl.method == "full"]
    mem_full, mem_lora = full_b.peak_mem_gb.mean(), lora_b.peak_mem_gb.mean()
    bm = m[(m.model == best) & (m.method == "full")]
    def tf(a, b):
        return bm[(bm.train_lang == a) & (bm.test_lang == b)].macro_f1.mean()
    summary.update({
        "best_encoder": best,
        "lora_minus_full_avg_points_indic": float(lora_b.macro_f1.mean() - full_b.macro_f1.mean()),
        "lora_minus_full_by_lang": {g["lang"]: {"delta": round(g["delta"], 2), "p": round(g["p"], 3)}
                                    for g in gaps if g["model"] == best},
        "lora_trainable_pct": float(lora_b.trainable_pct.iloc[0]) if len(lora_b) else None,
        "peak_mem_full_gb": None if np.isnan(mem_full) else float(mem_full),
        "peak_mem_lora_gb": None if np.isnan(mem_lora) else float(mem_lora),
        "mem_reduction_x": None if np.isnan(mem_full) or np.isnan(mem_lora) else float(mem_full / mem_lora),
        "zero_shot_to_pa": {"from_hi": float(tf("hi", "pa")), "from_en": float(tf("en", "pa")),
                             "from_ur": float(tf("ur", "pa"))},
        "hi_to_pa_minus_en_to_pa_points": float(tf("hi", "pa") - tf("en", "pa")),
        "script_barrier_hi_to_ur_vs_hi_to_hi": float(tf("hi", "ur") - tf("hi", "hi")),
    })


def section_rank(df, out_md, figs, summary):
    r = df[(df.experiment == "rank") & (df.train_lang == df.test_lang)]
    if r.empty:
        return
    out_md.append("\n## Table 4 — LoRA rank / target-module ablation (MuRIL, Punjabi)\n")
    out_md += ["| Targets | Rank | Trainable params | Trainable % | Macro-F1 |", "|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for tgt, g in r.groupby("targets"):
        agg = g.groupby("lora_r").agg(f1=("macro_f1", "mean"), sd=("macro_f1", "std"),
                                      n=("trainable_params", "first"), pct=("trainable_pct", "first"))
        for rank, row in agg.iterrows():
            out_md.append(f"| {tgt} | {rank} | {row.n:,.0f} | {row.pct:.2f} | {row.f1:.1f} ± {row.sd:.1f} |")
        ax.errorbar(agg.n, agg.f1, yerr=agg.sd, marker="o", capsize=3, label=tgt)
    full = df[(df.experiment == "main") & (df.model == "MuRIL") & (df.method == "full")
              & (df.train_lang == "pa") & (df.test_lang == "pa")]
    if not full.empty:
        ax.axhline(full.macro_f1.mean(), color="k", ls="--", lw=1, label="full fine-tuning")
    ax.set_xscale("log"); ax.set_xlabel("trainable parameters (log)"); ax.set_ylabel("macro-F1")
    ax.set_title("Punjabi: accuracy vs. trainable parameters"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "rank_ablation.png", dpi=160); plt.close(fig)
    best = r.groupby(["targets", "lora_r"]).macro_f1.mean().idxmax()
    summary["best_lora_setting"] = {"targets": best[0], "rank": int(best[1])}


def section_size(df, out_md, figs, summary):
    s = df[(df.experiment == "size") & (df.train_lang == df.test_lang)]
    if s.empty:
        return
    out_md.append("\n## Table 5 — Training-set size (MuRIL)\n")
    sizes = sorted(s.train_size.unique())
    out_md += ["| Language | Method | " + " | ".join(map(str, sizes)) + " |", "|---" * (len(sizes) + 2) + "|"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for (lang, method), g in s.groupby(["train_lang", "method"]):
        agg = g.groupby("train_size").macro_f1.agg(["mean", "std"])
        out_md.append(f"| {LANG_NAME[lang]} | {method} | " +
                      " | ".join(f"{agg.loc[n, 'mean']:.1f} ± {agg.loc[n, 'std']:.1f}" if n in agg.index else "–"
                                 for n in sizes) + " |")
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", capsize=3,
                    ls="-" if method == "lora" else "--", label=f"{LANG_NAME[lang]} / {method}")
    ax.set_xscale("log"); ax.set_xlabel("labelled training examples"); ax.set_ylabel("macro-F1")
    ax.set_title("Data efficiency: LoRA vs. full fine-tuning"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "data_size.png", dpi=160); plt.close(fig)
    small = s[s.train_size == sizes[0]].groupby("method").macro_f1.mean()
    if {"lora", "full"} <= set(small.index):
        summary["lora_minus_full_at_smallest_n"] = {"n": int(sizes[0]),
                                                    "delta": float(small["lora"] - small["full"])}


def section_llm(df, out_md, summary):
    g = df[(df.experiment == "llm") & (df.train_lang == df.test_lang)]
    if g.empty:
        return
    out_md.append("\n## Table 6 — Gemma-2-2B + QLoRA (in-language macro-F1)\n")
    out_md += ["| Language | Macro-F1 | Trainable % | Peak GPU mem (GB) |", "|---|---|---|---|"]
    for lang, h in g.groupby("test_lang"):
        out_md.append(f"| {LANG_NAME[lang]} | {pm(h.macro_f1)} | {h.trainable_pct.iloc[0]:.2f} | "
                      f"{h.peak_mem_gb.mean():.2f} |")
    summary["gemma_qlora_avg"] = float(g.macro_f1.mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out, figs = Path(a.out), Path(a.out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    df, raw = load(Path(a.runs))
    if df.empty:
        raise SystemExit(f"No runs found in {a.runs}")
    md = ["# Results\n", f"_Generated from {df.run_id.nunique()} runs._\n"]
    summary: dict = {}
    section_main(df, raw, md, figs, summary)
    section_rank(df, md, figs, summary)
    section_size(df, md, figs, summary)
    section_llm(df, md, summary)
    smoke = df[df.experiment == "smoke"]
    if not smoke.empty:
        md.append("\n## Smoke test\n")
        md += ["| Method | Test lang | Macro-F1 |", "|---|---|---|"]
        for (meth, l), g in smoke.groupby(["method", "test_lang"]):
            md.append(f"| {meth} | {l} | {pm(g.macro_f1)} |")
    (out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
