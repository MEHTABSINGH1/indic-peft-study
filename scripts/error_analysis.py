"""Per-class F1, confusion matrices and misclassified examples for one setting.

Usage:
  python scripts/error_analysis.py --experiment main --model muril-base-cased \
      --method full --train-lang hi --test-langs hi pa ur

Pools predictions over seeds. Writes results/error_analysis/<setting>.md and a
confusion-matrix PNG per test language. Because SIB-200 is sentence-parallel,
the same sentence can be compared across languages: the report lists
sentences that are right in one language and wrong in another, which shows
what script/vocabulary differences break transfer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from indicpeft.data import LABELS, load_lang  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results/error_analysis")
    p.add_argument("--experiment", default="main")
    p.add_argument("--model", default="muril-base-cased")
    p.add_argument("--method", default="full")
    p.add_argument("--train-lang", default="hi")
    p.add_argument("--test-langs", nargs="+", default=["hi", "pa", "ur"])
    p.add_argument("--train-size", default="all", help="'all' or the n used in the run")
    p.add_argument("--examples", type=int, default=8)
    a = p.parse_args()

    pattern = f"{a.experiment}__{a.model}__{a.method}*__{a.train_lang}__n{a.train_size}__s*.json"
    files = sorted(Path(a.runs).glob(pattern))
    if not files:
        raise SystemExit(f"No runs match {pattern}")
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    tag = f"{a.model}_{a.method}_train-{a.train_lang}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    md = [f"# Error analysis — {a.model}, {a.method}, trained on `{a.train_lang}` ({len(runs)} seeds)\n",
          "## Per-class F1 (pooled over seeds)\n",
          "| Test lang | " + " | ".join(LABELS) + " | macro |", "|---" * (len(LABELS) + 2) + "|"]
    # majority vote over seeds per example, for the cross-language comparison
    vote = {}
    for lang in a.test_langs:
        y = np.array(runs[0]["test"][lang]["y_true"])
        preds = np.stack([r["test"][lang]["y_pred"] for r in runs])
        y_all, p_all = np.tile(y, len(runs)), preds.ravel()
        per = f1_score(y_all, p_all, average=None, labels=range(len(LABELS)))
        md.append(f"| {lang} | " + " | ".join(f"{100*v:.0f}" for v in per) +
                  f" | {100*f1_score(y_all, p_all, average='macro'):.1f} |")
        vote[lang] = (y, np.array([np.bincount(c, minlength=len(LABELS)).argmax() for c in preds.T]))

        cm = confusion_matrix(y_all, p_all, labels=range(len(LABELS)), normalize="true")
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS)), [l[:8] for l in LABELS], rotation=45, ha="right")
        ax.set_yticks(range(len(LABELS)), [l[:8] for l in LABELS])
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"train {a.train_lang} → test {lang}")
        fig.tight_layout(); fig.savefig(out / f"cm_{tag}_test-{lang}.png", dpi=160); plt.close(fig)

    # sentences correct in the training language but wrong in another
    texts = {l: load_lang(l)["test"]["text"] for l in a.test_langs}
    src = a.train_lang if a.train_lang in vote else a.test_langs[0]
    ys, ps = vote[src]
    for lang in a.test_langs:
        if lang == src:
            continue
        yt, pt = vote[lang]
        idx = np.where((ps == ys) & (pt != yt))[0]
        md.append(f"\n## Right in `{src}`, wrong in `{lang}`: {len(idx)} of {len(ys)} sentences\n")
        for i in idx[: a.examples]:
            md.append(f"- **true:** {LABELS[yt[i]]} · **pred ({lang}):** {LABELS[pt[i]]}  \n"
                      f"  `{src}`: {texts[src][i]}  \n  `{lang}`: {texts[lang][i]}")
    (out / f"{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
