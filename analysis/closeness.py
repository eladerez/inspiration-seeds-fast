"""Per-pair closeness of one config to the baseline: DreamSim distance between each output and the baseline's output
for the same pair and seed (eval/<config>/dreamsim.json, per_image_fidelity), averaged over the 4 seeds, sorted from
closest to furthest. For scale, each row also shows the baseline's own seed-to-seed distance for that pair.
Writes analysis/closeness_<config>.md.

Usage: uv run analysis/closeness.py [--config 3] [--store data/bucket]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="3")
    ap.add_argument("--store", type=Path, default=ROOT / "data/bucket")
    args = ap.parse_args()
    fid = json.loads((args.store / f"eval/{args.config}/dreamsim.json").read_text())["per_image_fidelity"]
    base_div = json.loads((args.store / "eval/0/dreamsim.json").read_text())["per_pair_diversity"]
    cats = {p["pair_id"]: f'{p["category_1"]} × {p["category_2"]}'
            for p in json.loads((ROOT / "data/benchmark/pairs.json").read_text())}
    per_pair: dict[str, list[float]] = {}
    for name, d in fid.items():
        per_pair.setdefault(re.sub(r"_seed\d+$", "", name), []).append(d)
    rows = sorted(((statistics.fmean(v), max(v), pid) for pid, v in per_pair.items()))
    lines = [f"# Config {args.config} vs baseline, per pair (closest first)\n",
             "Distance: DreamSim between config output and baseline output, same pair and seed; mean of 4 seeds.",
             "Baseline seeds: the baseline's own mean distance between its 4 seeds for that pair (scale).\n",
             "| Rank | Pair | Categories | Mean distance | Worst seed | Baseline seeds apart | Distance ÷ seeds apart |",
             "|---:|---|---|---:|---:|---:|---:|"]
    for i, (mean, worst, pid) in enumerate(rows, 1):
        bd = base_div[pid]
        lines.append(f"| {i} | {pid[4:]} | {cats[pid]} | {mean:.3f} | {worst:.3f} | {bd:.3f} | {mean / bd:.2f} |")
    means = [r[0] for r in rows]
    lines.append(f"\nAll {len(rows)} pairs: median {statistics.median(means):.3f}, mean {statistics.fmean(means):.3f}, "
                 f"closest {means[0]:.3f}, furthest {means[-1]:.3f}.")
    out = ROOT / f"analysis/closeness_{args.config}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
