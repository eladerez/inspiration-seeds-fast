"""Faster drop-in for Inspiration Seeds' generate.py (github.com/kfirgoldberg/InspirationSeeds): same canvas, prompt,
LoRA, guidance and seeds, with one of the speedups measured in this repo. Each preset is one of the benchmarked
configs in runner/configs/, built by the same code (runner/pipeline.py); the only difference is that the prompt is
encoded once at start-up instead of loaded from a cache file (identical images; see tests/test_runner.py).

Presets (speed on one RTX PRO 6000 vs the original's 20.91 s per image; see analysis/results.md):
  nvfp4          config 3   8.69 s   2.4x  no loss in creativity or seed diversity (diversity +3%)    (Blackwell)
  nvfp4-cache    config 4   4.24 s   4.9x  -2.8% description length, -3% seed diversity            (Blackwell)
  nvfp4-turbo    config 7   2.77 s   7.5x  same description length, -7% seed diversity             (Blackwell)

Usage:
  uv run --group gen --group nunchaku generate_fast.py --image1 a.jpg --image2 b.jpg --preset nvfp4 \\
      [--output-dir outputs] [--seed 1] [--num-seeds 4]
FLUX.1 Kontext [dev] weights are non-commercial: accept their license on Hugging Face and set HF_TOKEN.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from runner import pipeline as pl
from runner.canvas import PROMPT, create_input_grid, load_rgb

ROOT = Path(__file__).resolve().parent
PRESETS = {"nvfp4": "3", "nvfp4-cache": "4", "nvfp4-turbo": "7"}


def preset_config(name: str) -> dict:
    cfg = yaml.safe_load((ROOT / f"runner/configs/{PRESETS[name]}.yaml").read_text())
    cfg["text_embedding"] = "encode"  # benchmark runs loaded a cached embedding of this same prompt
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image1", type=Path, required=True, help="first input (top-left)")
    ap.add_argument("--image2", type=Path, required=True, help="second input (bottom-right)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="nvfp4")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-seeds", type=int, default=4)
    args = ap.parse_args()

    cfg = preset_config(args.preset)
    grid = create_input_grid(load_rgb(args.image1), load_rgb(args.image2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"preset {args.preset} (config {cfg['id']}): loading pipeline...")
    pipe = pl.build(cfg, device="cuda")
    kwargs = {"guidance_scale": cfg["guidance_scale"]}
    if cfg.get("num_inference_steps"):
        kwargs["num_inference_steps"] = cfg["num_inference_steps"]
    stem = f"{args.image1.stem}_x_{args.image2.stem}"
    for seed in range(args.seed, args.seed + args.num_seeds):
        t0 = time.perf_counter()
        img = pipe(image=grid, prompt=PROMPT, generator=torch.Generator(device="cuda").manual_seed(seed),
                   **kwargs).images[0]
        out = args.output_dir / f"{stem}__seed_{seed:03d}.png"
        img.save(out)
        combined = Image.new("RGB", (grid.width + img.width, max(grid.height, img.height)))
        combined.paste(grid, (0, 0))
        combined.paste(img, (grid.width, 0))
        combined.save(args.output_dir / f"{stem}__seed_{seed:03d}_combined.jpeg", quality=92)
        print(f"seed {seed}: {time.perf_counter() - t0:.1f} s -> {out}")


if __name__ == "__main__":
    main()
