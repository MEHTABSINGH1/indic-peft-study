"""Train one configuration and evaluate it on every requested test language.

Each run writes a single JSON file (metrics + raw predictions) so that
results can be aggregated, bootstrapped and error-analysed later without
re-training. Runs are idempotent: an existing JSON means "already done".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

from . import data as D
from .modeling import build, count_params


@dataclass
class RunConfig:
    model: str = "google/muril-base-cased"
    method: str = "lora"                 # full | lora | qlora
    train_lang: str = "pa"
    eval_langs: list[str] = field(default_factory=lambda: list(D.LANGS))
    train_size: int | None = None        # None = all 701
    seed: int = 0
    lora_r: int = 8
    lora_targets: str = "attn_qv"
    epochs: int = 10
    lr: float | None = None              # default depends on method
    batch_size: int = 16
    grad_accum: int = 1
    max_len: int = 128
    experiment: str = "main"

    def resolved_lr(self) -> float:
        if self.lr is not None:
            return self.lr
        # chosen on the Punjabi dev set (see README, section 3.1)
        return 5e-5 if self.method == "full" else 2e-3

    def run_id(self) -> str:
        m = self.model.split("/")[-1]
        size = self.train_size or "all"
        r = f"_r{self.lora_r}-{self.lora_targets}" if self.method != "full" else ""
        return f"{self.experiment}__{m}__{self.method}{r}__{self.train_lang}__n{size}__s{self.seed}"


class BestEpoch(TrainerCallback):
    """Remember the trainable weights of the epoch with the best dev macro-F1.

    Only parameters that are being trained can change, so for LoRA this copies
    a few MB; for full fine-tuning it is one CPU copy of the model.
    """

    def __init__(self, model: torch.nn.Module):
        self.model, self.score, self.epoch, self.state = model, -1.0, None, None

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        score = (metrics or {}).get("eval_macro_f1", -1.0)
        if score > self.score:
            self.score, self.epoch = score, round(state.epoch or 0)
            self.state = {n: p.detach().to("cpu", copy=True)
                          for n, p in self.model.named_parameters() if p.requires_grad}

    def restore(self) -> None:
        if self.state is None:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, saved in self.state.items():
                params[n].copy_(saved.to(params[n].device))


def _metrics(pred) -> dict:
    y_pred = np.argmax(pred.predictions, axis=-1)
    return {"macro_f1": f1_score(pred.label_ids, y_pred, average="macro"),
            "accuracy": accuracy_score(pred.label_ids, y_pred)}


def run(cfg: RunConfig, out_dir: str | Path = "results/runs") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg.run_id()}.json"
    if out_file.exists():
        print(f"[skip] {out_file.name} exists")
        return json.loads(out_file.read_text(encoding="utf-8"))

    t_start = time.time()
    set_seed(cfg.seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, tok = build(cfg.model, cfg.method, lora_r=cfg.lora_r, lora_targets=cfg.lora_targets)
    params = count_params(model)

    def encode(ds):
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_len),
                      batched=True, remove_columns=["index_id", "category", "text"])

    train_ds = D.load_lang(cfg.train_lang)
    train = encode(D.stratified_subsample(train_ds["train"], cfg.train_size, cfg.seed))
    dev = encode(train_ds["validation"])

    # INDICPEFT_FP32=1 disables mixed precision (GTX 16xx cards give NaN losses in fp16)
    force_fp32 = os.environ.get("INDICPEFT_FP32", "") == "1"
    bf16 = cuda and not force_fp32 and torch.cuda.is_bf16_supported()
    ckpt_dir = tempfile.mkdtemp(prefix="indicpeft_")
    args = TrainingArguments(
        output_dir=ckpt_dir, seed=cfg.seed, num_train_epochs=cfg.epochs,
        # 10% linear warm-up, given as steps (works on transformers 4.x and 5.x)
        learning_rate=cfg.resolved_lr(), weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * cfg.epochs * math.ceil(len(train) / (cfg.batch_size * cfg.grad_accum)))),
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=64,
        gradient_accumulation_steps=cfg.grad_accum,
        # Best epoch is kept in memory by BestEpoch (no checkpoint files): much
        # faster than writing ~1 GB per epoch, and works for the custom head.
        eval_strategy="epoch", save_strategy="no",
        bf16=bf16, fp16=cuda and not force_fp32 and not bf16 and cfg.method != "qlora",
        # QLoRA: gradient checkpointing is already enabled on the encoder by
        # prepare_model_for_kbit_training (the wrapper has no such method).
        report_to=[], logging_steps=20, disable_tqdm=True)
    best = BestEpoch(model)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=_metrics,
                      callbacks=[best])

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    best.restore()
    print(f"  best epoch {best.epoch} (dev macro-F1={best.score:.4f})")
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else None

    results = {}
    for lang in cfg.eval_langs:
        test = encode(D.load_lang(lang)["test"])
        out = trainer.predict(test)
        y_pred = np.argmax(out.predictions, axis=-1)
        results[lang] = {
            "macro_f1": float(f1_score(out.label_ids, y_pred, average="macro")),
            "accuracy": float(accuracy_score(out.label_ids, y_pred)),
            "y_true": out.label_ids.tolist(), "y_pred": y_pred.tolist(),
        }
        print(f"  test[{lang}] macro-F1={results[lang]['macro_f1']:.4f}")

    record = {
        "run_id": cfg.run_id(), "config": asdict(cfg), "lr": cfg.resolved_lr(),
        "trainable_params": params.trainable, "total_params": params.total,
        "trainable_pct": params.pct, "peak_gpu_mem_gb": peak_mem_gb,
        "train_seconds": train_secs, "n_train": len(train),
        "best_epoch": best.epoch, "best_dev_macro_f1": best.score,
        "total_seconds": time.time() - t_start,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "test": results,
    }
    out_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    del trainer, model
    if cuda:
        torch.cuda.empty_cache()
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for f, default in asdict(RunConfig()).items():
        if f == "eval_langs":
            p.add_argument("--eval-langs", nargs="+", default=default)
        elif f in {"train_size", "lr"}:
            p.add_argument(f"--{f.replace('_', '-')}", type=int if f == "train_size" else float, default=None)
        else:
            p.add_argument(f"--{f.replace('_', '-')}", type=type(default), default=default)
    p.add_argument("--out-dir", default="results/runs")
    a = vars(p.parse_args())
    out = a.pop("out_dir")
    run(RunConfig(**a), out)


if __name__ == "__main__":
    main()
