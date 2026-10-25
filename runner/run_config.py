"""Resumable batch runner: one config, every benchmark pair x seed, each image and its timing written to the store
the moment it exists (spot VMs can vanish).

Layout: outputs/<config>/<pair_id>_seed<k>.png, logs/<config>/timing.jsonl, logs/<config>/env.json.
Per-image timing objects go to logs/<config>/timing/<pair_id>_seed<k>.json first (timing before image, so a resume
that finds the image always has its timing) and are merged into timing.jsonl at the end.

Usage:
  uv run runner/run_config.py --config runner/configs/0.yaml --store gs://inspiration-seeds-speed [--smoke]
  --smoke                 2 pairs x seed 1 (a quick check before any full run)
  --write-text-cache      after the run, save the fixed prompt's embedding to cache/prompt_embeds.pt
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench import StageTimer  # noqa: E402
from runner import pipeline as pl  # noqa: E402
from runner.canvas import PROMPT, create_input_grid, load_rgb  # noqa: E402
from runner.store import Store, open_store  # noqa: E402

TEXT_CACHE = "cache/prompt_embeds.pt"
WARMUP = 2


def load_config(path: Path) -> tuple[dict, str]:
    cfg = yaml.safe_load(path.read_text())
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return cfg, digest


def jobs(pairs: list[dict], seeds: list[int]) -> Iterator[tuple[dict, int]]:
    for pair in pairs:
        for seed in seeds:
            yield pair, seed


def version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def env_info(cfg_id: str, cfg_hash: str) -> dict:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    driver = None
    if gpu:
        import subprocess

        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip() or None
    return {"config": cfg_id, "config_hash": cfg_hash, "gpu": gpu, "driver": driver, "cuda": torch.version.cuda,
            "torch": torch.__version__, "diffusers": version("diffusers"), "transformers": version("transformers"),
            "peft": version("peft"), "nunchaku": version("nunchaku"), "python": platform.python_version(),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def check_hash(store: Store, cfg_id: str, cfg_hash: str) -> None:
    """Never mix two different settings under one config id (a config's settings never change silently)."""
    key = f"logs/{cfg_id}/env.json"
    if store.exists(key):
        old = json.loads(store.get(key))["config_hash"]
        if old != cfg_hash:
            raise SystemExit(f"config {cfg_id} changed (hash {old} in the store, {cfg_hash} now); use a new config id")


def generate(pipe: Any, timer: StageTimer, cfg: dict, grid: Any, seed: int, cache: dict | None, device: str) -> tuple:
    import torch

    kwargs: dict[str, Any] = {"image": grid, "guidance_scale": cfg["guidance_scale"],
                              "generator": torch.Generator(device=device).manual_seed(seed)}
    if cfg.get("num_inference_steps"):
        kwargs["num_inference_steps"] = cfg["num_inference_steps"]
    if cache:
        kwargs.update(cache)
    else:
        kwargs["prompt"] = PROMPT
    out, timing = timer.measure(pipe, **kwargs)
    return out.images[0], timing


def png(img: Any) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_text_cache(pipe: Any, store: Store, cfg: dict) -> None:
    import torch

    with torch.no_grad():
        embeds, pooled, _ = pipe.encode_prompt(prompt=PROMPT, prompt_2=None)
    buf = io.BytesIO()
    torch.save({"prompt_embeds": embeds.cpu(), "pooled_prompt_embeds": pooled.cpu(), "prompt": PROMPT,
                "model": cfg.get("model", pl.KONTEXT), "revision": cfg.get("revision"),
                "loras": cfg.get("loras")}, buf)
    store.put(TEXT_CACHE, buf.getvalue())
    print(f"wrote {TEXT_CACHE}", flush=True)


def merge_timing(store: Store, cfg_id: str) -> int:
    keys = [k for k in store.list(f"logs/{cfg_id}/timing/") if k.endswith(".json")]
    lines = [json.dumps(json.loads(store.get(k))) for k in keys]
    store.put(f"logs/{cfg_id}/timing.jsonl", ("\n".join(lines) + "\n").encode())
    return len(lines)


def run(cfg_path: Path, store: Store, bench_dir: Path, smoke: bool = False, text_cache: bool = False,
        device: str = "cuda", pipeline_cls: Any = None) -> dict:
    cfg, cfg_hash = load_config(cfg_path)
    cfg_id = str(cfg["id"])
    pairs = json.loads((bench_dir / "pairs.json").read_text())
    seeds = cfg["seeds"]
    if smoke:
        pairs, seeds = pairs[:2], seeds[:1]
    check_hash(store, cfg_id, cfg_hash)
    todo = [(p, s) for p, s in jobs(pairs, seeds) if not store.exists(f"outputs/{cfg_id}/{p['pair_id']}_seed{s}.png")]
    print(f"config {cfg_id}: {len(todo)} of {len(pairs) * len(seeds)} images to generate", flush=True)
    if not todo and not text_cache:
        return {"generated": 0, "timing_lines": merge_timing(store, cfg_id)}

    pipe = pl.build(cfg, device=device, pipeline_cls=pipeline_cls)
    if not store.exists(f"logs/{cfg_id}/env.json"):
        store.put(f"logs/{cfg_id}/env.json", json.dumps(env_info(cfg_id, cfg_hash), indent=1).encode())
    cache = pl.load_text_cache(store.get(TEXT_CACHE), device) if cfg["text_embedding"] == "cached" else None
    timer = StageTimer(pipe, count_forwards=not cfg.get("compile"))
    grids: dict[str, Any] = {}

    def grid_for(pair: dict) -> Any:
        if pair["pair_id"] not in grids:
            grids[pair["pair_id"]] = create_input_grid(load_rgb(bench_dir / "images" / pair["image_1"]),
                                                       load_rgb(bench_dir / "images" / pair["image_2"]))
        return grids[pair["pair_id"]]

    if todo:
        for w in range(WARMUP):  # untimed: compile, autotune, allocator, caches
            generate(pipe, timer, cfg, grid_for(todo[0][0]), seeds[0] + w, cache, device)
    for n, (pair, seed) in enumerate(todo, 1):
        img, timing = generate(pipe, timer, cfg, grid_for(pair), seed, cache, device)
        name = f"{pair['pair_id']}_seed{seed}"
        record = {"config": cfg_id, "config_hash": cfg_hash, "pair_id": pair["pair_id"], "seed": seed,
                  "smoke": smoke, "timing": timing}
        store.put(f"logs/{cfg_id}/timing/{name}.json", json.dumps(record).encode())
        store.put(f"outputs/{cfg_id}/{name}.png", png(img))
        print(f"[{n}/{len(todo)}] {name} total {timing['total']:.2f}s transformer {timing['transformer']:.2f}s "
              f"peak {timing['peak_vram_gb']:.1f} GB", flush=True)
    if text_cache:
        write_text_cache(pipe, store, cfg)
    return {"generated": len(todo), "timing_lines": merge_timing(store, cfg_id)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--store", required=True, help="gs://bucket or a local folder")
    ap.add_argument("--bench-dir", type=Path, default=ROOT / "data/benchmark")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-text-cache", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(args.config, open_store(args.store), args.bench_dir, args.smoke, args.write_text_cache)))


if __name__ == "__main__":
    main()
