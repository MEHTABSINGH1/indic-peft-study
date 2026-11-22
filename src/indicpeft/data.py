"""SIB-200 loading, label handling and stratified sub-sampling.

SIB-200 (Adelani et al., 2024) is a topic-classification benchmark built on
FLORES-200. Every language has the *same* sentences (701 train / 99 dev /
204 test), which makes cross-lingual comparisons clean: differences in score
come from the language/script, not from different data.
"""
from __future__ import annotations

from collections import defaultdict
import random

from datasets import Dataset, DatasetDict, load_dataset

DATASET_ID = "Davlan/sib200"

# Short code -> SIB-200 config. Hindi (Devanagari) and Urdu (Perso-Arabic)
# are close to one spoken language written in two scripts, which lets us
# isolate the effect of script on transfer.
LANGS: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "pa": "pan_Guru",
    "kn": "kan_Knda",
    "ur": "urd_Arab",
}

LABELS: list[str] = [
    "entertainment",
    "geography",
    "health",
    "politics",
    "science/technology",
    "sports",
    "travel",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_lang(lang: str) -> DatasetDict:
    """Load one language and add an integer ``label`` column."""
    if lang not in LANGS:
        raise ValueError(f"Unknown language '{lang}'. Choose from {sorted(LANGS)}")
    ds = load_dataset(DATASET_ID, LANGS[lang])
    return ds.map(lambda ex: {"label": LABEL2ID[ex["category"]]})


def stratified_subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    """Return ``n`` examples keeping the label distribution (None = all)."""
    if n is None or n >= len(ds):
        return ds
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(ds["label"]):
        by_label[y].append(i)
    rng = random.Random(seed)
    picked: list[int] = []
    for idxs in by_label.values():
        k = max(1, round(n * len(idxs) / len(ds)))
        picked.extend(rng.sample(idxs, min(k, len(idxs))))
    rng.shuffle(picked)
    return ds.select(sorted(picked[:n]))
