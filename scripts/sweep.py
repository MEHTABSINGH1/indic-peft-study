"""Expand a YAML experiment grid and run every configuration.

Usage:  python scripts/sweep.py configs/main.yaml [--dry-run] [--shard 0/2]

Finished runs are skipped, so the sweep can be restarted after a Colab
disconnect and will pick up where it stopped. ``--shard i/k`` runs every
k-th config, which lets two notebooks share one sweep.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from indicpeft.train import RunConfig, run  # noqa: E402


def expand(spec: dict) -> list[RunConfig]:
    fixed = spec.get("fixed", {})
    grid = spec.get("grid", {})
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        kw = {**fixed, **dict(zip(keys, values)), "experiment": spec["experiment"]}
        if kw.get("train_size") == "all":
            kw["train_size"] = None
        cfg = RunConfig(**kw)
        # LoRA-only knobs are meaningless for full fine-tuning: keep one copy.
        if cfg.method == "full":
            cfg.lora_r, cfg.lora_targets = RunConfig.lora_r, RunConfig.lora_targets
        configs.append(cfg)
    # de-duplicate while keeping order
    seen, unique = set(), []
    for c in configs:
        if c.run_id() not in seen:
            seen.add(c.run_id())
            unique.append(c)
    return unique


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()

    spec = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    configs = expand(spec)
    i, k = map(int, a.shard.split("/"))
    configs = configs[i::k]
    print(f"{len(configs)} runs in {a.config} (shard {a.shard})")
    failed = []
    for n, cfg in enumerate(configs, 1):
        print(f"[{n}/{len(configs)}] {cfg.run_id()}")
        if a.dry_run:
            continue
        try:
            run(cfg, a.out_dir)
        except Exception:  # keep the sweep going; report at the end
            traceback.print_exc()
            failed.append(cfg.run_id())
    if failed:
        print("\nFAILED runs:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
