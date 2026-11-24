# Results

_Generated from 204 runs._

## Table 1 — In-language macro-F1 (mean ± std over seeds)

| Model | Method | Punjabi | Hindi | Kannada | Urdu | English | Avg | Trainable % | Peak GPU mem (GB) | Train time (s, median) |
|---|---|---|---|---|---|---|---|---|---|---|
| IndicBERT-v2 | full | 86.2 ± 1.3 | 86.8 ± 0.1 | 88.6 ± 3.9 | 86.1 ± 0.5 | 90.6 ± 1.7 | 87.7 ± 0.7 | 100.00 | 4.50 | 141 |
| IndicBERT-v2 | lora | 85.1 ± 2.8 | 87.7 ± 0.3 | 87.7 ± 1.8 | 85.3 ± 1.6 | 90.6 ± 1.2 | 87.3 ± 0.6 | 0.11 | 2.25 | 98 |
| MuRIL | full | 84.8 ± 1.9 | 85.8 ± 2.4 | 87.8 ± 2.9 | 85.7 ± 3.3 | 88.1 ± 0.3 | 86.4 ± 0.4 | 100.00 | 3.91 | 146 |
| MuRIL | lora | 84.9 ± 1.2 | 87.7 ± 1.7 | 78.4 ± 14.3 | 87.5 ± 0.5 | 90.1 ± 1.0 | 85.7 ± 2.7 | 0.13 | 1.92 | 89 |
| XLM-R | full | 76.2 ± 1.4 | 83.6 ± 2.1 | 82.6 ± 1.0 | 81.6 ± 1.1 | 90.3 ± 1.0 | 82.9 ± 1.0 | 100.00 | 4.67 | 203 |
| XLM-R | lora | 80.0 ± 2.1 | 82.7 ± 2.5 | 79.7 ± 3.4 | 76.2 ± 14.3 | 89.6 ± 1.3 | 81.6 ± 2.6 | 0.11 | 2.19 | 129 |

## Table 2 — LoRA minus full fine-tuning (macro-F1 points, paired bootstrap)

| Model | Language | Δ (LoRA − full) | 95% CI | p |
|---|---|---|---|---|
| IndicBERT-v2 | Punjabi | -1.1 | [-3.6, +1.3] | 0.367 |
| IndicBERT-v2 | Hindi | +0.8 | [-1.8, +3.6] | 0.529 |
| IndicBERT-v2 | Kannada | -0.9 | [-3.6, +1.5] | 0.463 |
| IndicBERT-v2 | Urdu | -0.8 | [-3.8, +1.9] | 0.601 |
| IndicBERT-v2 | English | -0.0 | [-2.5, +2.5] | 0.981 |
| MuRIL | Punjabi | +0.1 | [-2.7, +3.1] | 0.958 |
| MuRIL | Hindi | +1.9 | [-1.3, +5.2] | 0.250 |
| MuRIL | Kannada | -9.5 | [-12.3, -6.5] | 0.001 |
| MuRIL | Urdu | +1.8 | [-1.4, +5.2] | 0.275 |
| MuRIL | English | +2.0 | [-1.2, +5.3] | 0.221 |
| XLM-R | Punjabi | +3.8 | [+0.0, +7.6] | 0.050 |
| XLM-R | Hindi | -0.8 | [-3.5, +1.8] | 0.549 |
| XLM-R | Kannada | -2.9 | [-6.2, +0.7] | 0.095 |
| XLM-R | Urdu | -5.4 | [-8.4, -2.4] | 0.001 |
| XLM-R | English | -0.8 | [-3.2, +1.8] | 0.560 |

## Table 3 — Cross-lingual transfer, IndicBERT-v2 (full); rows = train, cols = test

| train ↓ / test → | Punjabi | Hindi | Kannada | Urdu | English |
|---|---|---|---|---|---|
| Punjabi | 86.2 | 88.2 | 88.2 | 84.7 | 90.1 |
| Hindi | 85.8 | 86.8 | 88.5 | 85.3 | 90.5 |
| Kannada | 86.3 | 87.8 | 88.6 | 83.7 | 89.6 |
| Urdu | 86.7 | 87.7 | 88.1 | 86.1 | 89.3 |
| English | 85.8 | 86.5 | 87.8 | 84.2 | 90.6 |

## Table 4 — LoRA rank / target-module ablation (MuRIL, Punjabi)

| Targets | Rank | Trainable params | Trainable % | Macro-F1 |
|---|---|---|---|---|
| all-linear | 1 | 172,807 | 0.07 | 86.1 ± 0.6 |
| all-linear | 2 | 340,231 | 0.14 | 85.5 ± 3.2 |
| all-linear | 4 | 675,079 | 0.28 | 83.7 ± 1.2 |
| all-linear | 8 | 1,344,775 | 0.56 | 65.9 ± 31.6 |
| all-linear | 16 | 2,684,167 | 1.12 | 20.3 ± 18.3 |
| all-linear | 64 | 10,720,519 | 4.32 | 15.1 ± 9.7 |
| attn_all | 1 | 79,111 | 0.03 | 84.2 ± 0.7 |
| attn_all | 2 | 152,839 | 0.06 | 84.1 ± 0.4 |
| attn_all | 4 | 300,295 | 0.13 | 83.9 ± 2.2 |
| attn_all | 8 | 595,207 | 0.25 | 83.3 ± 1.4 |
| attn_all | 16 | 1,185,031 | 0.50 | 48.8 ± 38.9 |
| attn_all | 64 | 4,723,975 | 1.95 | 15.7 ± 9.1 |
| attn_qv | 1 | 42,247 | 0.02 | 85.7 ± 1.5 |
| attn_qv | 2 | 79,111 | 0.03 | 84.8 ± 1.5 |
| attn_qv | 4 | 152,839 | 0.06 | 85.5 ± 0.7 |
| attn_qv | 8 | 300,295 | 0.13 | 84.9 ± 1.2 |
| attn_qv | 16 | 595,207 | 0.25 | 82.5 ± 1.9 |
| attn_qv | 64 | 2,364,679 | 0.99 | 5.9 ± 0.4 |

## Table 5 — Training-set size (MuRIL)

| Language | Method | 50 | 100 | 200 | 400 | 701 |
|---|---|---|---|---|---|---|
| Kannada | full | 49.0 ± 16.6 | 76.9 ± 3.2 | 79.3 ± 3.1 | 84.9 ± 0.6 | 86.8 ± 1.3 |
| Kannada | lora | 37.6 ± 10.3 | 62.5 ± 17.0 | 79.7 ± 2.3 | 85.0 ± 2.7 | 87.4 ± 2.1 |
| Punjabi | full | 43.8 ± 13.5 | 74.7 ± 1.5 | 78.8 ± 0.2 | 83.4 ± 1.6 | 85.1 ± 0.6 |
| Punjabi | lora | 34.6 ± 3.9 | 61.5 ± 11.4 | 80.6 ± 2.2 | 83.1 ± 1.2 | 84.9 ± 2.2 |
