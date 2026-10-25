"""Paired comparison against config 0 over the same 99 pairs: is a config's seed diversity (DreamSim, per pair) or
description length (Gemini, per pair, mean of its seeds and of both runs) really different from the baseline?

For each metric: mean paired difference (config - config 0), a 95% bootstrap interval over pairs (10,000 resamples,
fixed seed), and the share of pairs where the config is lower. Writes analysis/paired.json.

Usage: uv run analysis/paired.py --store gs://inspiration-seeds-speed [--configs 1,2,3,4,5,6]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.store import Store, open_store  # noqa: E402


def diversity(store: Store, cfg: str) -> dict[str, float]:
    return json.loads(store.get(f"eval/{cfg}/dreamsim.json"))["per_pair_diversity"]


def words(store: Store, cfg: str) -> dict[str, float]:
    per_pair: dict[str, list[int]] = {}
    for k in (1, 2):
        items = json.loads(store.get(f"eval/{cfg}/gemini_run{k}.json"))["items"]
        for name, item in items.items():
            per_pair.setdefault(re.sub(r"_seed\d+$", "", name), []).append(item["words"])
    return {p: statistics.fmean(v) for p, v in per_pair.items()}


def compare(base: dict[str, float], other: dict[str, float], rng: np.random.Generator) -> dict:
    pairs = sorted(set(base) & set(other))
    d = np.array([other[p] - base[p] for p in pairs])
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"pairs": len(pairs), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "share_lower": float((d < 0).mean()), "base_mean": float(np.mean([base[p] for p in pairs]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--configs", default="1,2,3,4,5,6,7")
    args = ap.parse_args()
    store = open_store(args.store)
    rng = np.random.default_rng(0)
    out = {}
    base_div, base_words = diversity(store, "0"), words(store, "0")
    for c in args.configs.split(","):
        out[c] = {"diversity": compare(base_div, diversity(store, c), rng),
                  "words": compare(base_words, words(store, c), rng)}
        for m in ("diversity", "words"):
            r = out[c][m]
            print(f"config {c} {m:9s}: diff {r['mean_diff']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
                  f"lower in {r['share_lower']:.0%} of pairs")
    (ROOT / "analysis/paired.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
