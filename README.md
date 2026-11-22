# How Much Does LoRA Lose? Parameter-Efficient Adaptation of Multilingual Encoders for Low-Resource Indic Languages

**Mehtab Singh** · M.Tech Data Science & AI, IIIT Dharwad

> **Status:** code complete and tested; the numbers below are filled in by `scripts/aggregate.py` after running the sweeps. Until then they are marked `TBD`.

## TL;DR

On SIB-200 topic classification for **Punjabi, Hindi, Kannada and Urdu**, LoRA trains **TBD %** of the parameters of full fine-tuning and uses **TBD× less GPU memory**. Its macro-F1 comes within **TBD points** of full fine-tuning. Zero-shot transfer **Hindi → Punjabi** beats **English → Punjabi** by **TBD points**. **Hindi → Urdu** (the same spoken language in a different script) loses **TBD points**, which isolates the cost of a script change.

## 1. Motivation

Most Indian languages have only a few hundred labelled examples for any given task. Full fine-tuning of every weight is expensive and can overfit at this scale. Parameter-efficient fine-tuning (PEFT) is cheap, but most results for it come from English or high-resource benchmarks. This project measures, under controlled conditions, **how much accuracy PEFT gives up, and when, for low-resource Indic languages**.

## 2. Research questions

| | Question | Experiment |
|---|---|---|
| **RQ1** | How close does LoRA get to full fine-tuning on Indic languages, and does the gap differ between Indic-specific (MuRIL, IndicBERT-v2) and general multilingual (XLM-R) encoders? | `configs/main.yaml` |
| **RQ2** | How do LoRA rank, target modules and the amount of labelled data affect the gap? | `configs/ablation_rank.yaml`, `configs/ablation_size.yaml` |
| **RQ3** | Which source language transfers best zero-shot to Punjabi, and how much does a script change alone cost (Hindi → Urdu)? | transfer matrix from `main.yaml` |
| **RQ4** *(optional)* | Does a 2B decoder LLM (Gemma-2-2B) with 4-bit QLoRA beat 237M-parameter Indic encoders? | `configs/llm_qlora.yaml` |

## 3. Setup

**Data: [SIB-200](https://huggingface.co/datasets/Davlan/sib200)** (Adelani et al., 2024). It has 7 topic classes and 701 / 99 / 204 train / dev / test sentences per language. The benchmark is **sentence-parallel**: every language has the same sentences, so score differences across languages come from the language and script, not from different data.

| Code | Language | Script |
|---|---|---|
| `pa` | Punjabi | Gurmukhi |
| `hi` | Hindi | Devanagari |
| `kn` | Kannada | Kannada |
| `ur` | Urdu | Perso-Arabic |
| `en` | English | Latin (transfer source only) |

**Models:** MuRIL-base, IndicBERT-v2 (MLM-only), XLM-RoBERTa-base and, optionally, Gemma-2-2B.

**Methods:** full fine-tuning, LoRA (r = 8 on query/value by default) and QLoRA (4-bit NF4 with double quantization).

**Protocol:**
- 10 epochs, with the best checkpoint chosen on dev macro-F1.
- Learning rate 3e-5 for full fine-tuning and 5e-4 for LoRA, with 10% warm-up.
- Max length 128 and batch size 16.
- **3 seeds** per configuration; results are reported as mean ± std.
- Significance of LoRA − full is tested with a **paired bootstrap** over test sentences (2,000 resamples, seeds pooled).
- Peak GPU memory is measured with `torch.cuda.max_memory_allocated`.

## 4. Results

All tables are generated in [`results/RESULTS.md`](results/RESULTS.md).

- **Table 1:** in-language macro-F1, trainable %, memory and time
- **Table 2:** LoRA − full, with 95% CI and p-value
- **Table 3 / figures:** cross-lingual transfer matrices (`results/figures/transfer_*.png`)
- **Table 4 / figure:** rank × target-module ablation (`results/figures/rank_ablation.png`)
- **Table 5 / figure:** data-size curve (`results/figures/data_size.png`)
- **Table 6:** Gemma-2-2B + QLoRA
- **Error analysis:** per-class F1, confusion matrices and parallel sentences that are right in Hindi but wrong in Urdu or Punjabi (`results/error_analysis/`)

### Key findings

*(Write 3–5 bullet points here after looking at the tables. Say what the numbers show, and be honest about surprises and negative results.)*

## 5. Limitations

- SIB-200 is topic classification on short, formal FLORES sentences. The results may not hold for code-mixed or social-media text.
- There are 204 test sentences per language, so differences under about 2 points are often not significant. Table 2 shows which ones are.
- Hyper-parameters are fixed across languages rather than tuned per language.

## 6. Reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # ~1 min on CPU
python scripts/sweep.py configs/smoke.yaml  # tiny CPU end-to-end run
python scripts/sweep.py configs/main.yaml           # GPU, ~2 h on a T4
python scripts/sweep.py configs/ablation_rank.yaml  # ~1 h
python scripts/sweep.py configs/ablation_size.yaml  # ~1 h
python scripts/aggregate.py
python scripts/error_analysis.py --model muril-base-cased --method full --train-lang hi --test-langs hi pa ur
```

There is also a one-click Colab notebook: [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb). Runs are resumable, and finished configurations are skipped after a disconnect.

```
src/indicpeft/   data.py (SIB-200, stratified sub-sampling) · modeling.py (full / LoRA / QLoRA) · train.py (one run → JSON)
scripts/         sweep.py (YAML grid runner) · aggregate.py (tables, bootstrap, figures) · error_analysis.py
configs/         one YAML per experiment
results/runs/    one JSON per run, with raw predictions for re-analysis
```

## References

- Adelani et al. *SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects.* EACL 2024.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- Khanuja et al. *MuRIL: Multilingual Representations for Indian Languages.* 2021.
- Doddapaneni et al. *Towards Leaving No Indic Language Behind (IndicBERT-v2 / IndicXTREME).* ACL 2023.
- Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).* ACL 2020.
