"""DreamSim fidelity and diversity for one config. Runs at the end of each GPU job (or anywhere with the images).

  fidelity   DreamSim(output, config-0 output) for the same pair and seed; lower = closer to the baseline
  diversity  mean pairwise DreamSim across the seeds of one pair; lower = less diverse

DreamSim's distance is 1 - cosine(embed(a), embed(b)) for the default ensemble, so every image is embedded once.
Writes eval/<config>/dreamsim.json to the store.

Usage: uv run --group eval eval/eval_dreamsim.py --config 1 --store gs://inspiration-seeds-speed [--reference 0]
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.store import Store, open_store  # noqa: E402

NAME = re.compile(r"(?P<pair>.+)_seed(?P<seed>\d+)\.png$")


def outputs(store: Store, cfg: str) -> dict[tuple[str, int], str]:
    found = {}
    for key in store.list(f"outputs/{cfg}/"):
        m = NAME.match(key.rsplit("/", 1)[-1])
        if m:
            found[(m["pair"], int(m["seed"]))] = key
    return found


@torch.no_grad()
def embed_all(store: Store, keys: dict, model, preprocess, device: str) -> dict:
    emb = {}
    for k, key in keys.items():
        img = Image.open(io.BytesIO(store.get(key))).convert("RGB")
        emb[k] = model.embed(preprocess(img).to(device)).float().squeeze(0)
    return emb


def dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1 - F.cosine_similarity(a, b, dim=-1))


def stats(xs: list[float]) -> dict | None:
    return {"mean": statistics.fmean(xs), "std": statistics.pstdev(xs), "median": statistics.median(xs),
            "n": len(xs)} if xs else None


def evaluate(store: Store, cfg: str, reference: str, model, preprocess, device: str) -> dict:
    mine = outputs(store, cfg)
    emb = embed_all(store, mine, model, preprocess, device)
    fidelity = {}
    if cfg != reference:
        ref = outputs(store, reference)
        shared = {k: ref[k] for k in mine if k in ref}
        ref_emb = embed_all(store, shared, model, preprocess, device)
        fidelity = {f"{p}_seed{s}": dist(emb[(p, s)], ref_emb[(p, s)]) for (p, s) in shared}
    diversity = {}
    for pair in sorted({p for p, _ in mine}):
        seeds = sorted(s for p, s in mine if p == pair)
        if len(seeds) > 1:
            diversity[pair] = statistics.fmean(dist(emb[(pair, a)], emb[(pair, b)])
                                               for a, b in itertools.combinations(seeds, 2))
    return {"config": cfg, "reference": reference, "images": len(mine),
            "fidelity": stats(list(fidelity.values())), "diversity": stats(list(diversity.values())),
            "per_image_fidelity": fidelity, "per_pair_diversity": diversity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--reference", default="0")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache/dreamsim"))
    args = ap.parse_args()
    from dreamsim import dreamsim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True, device=device, cache_dir=args.cache_dir)
    store = open_store(args.store)
    res = evaluate(store, args.config, args.reference, model, preprocess, device)
    store.put(f"eval/{args.config}/dreamsim.json", json.dumps(res, indent=1).encode())
    print(json.dumps({k: res[k] for k in ("config", "images", "fidelity", "diversity")}))


if __name__ == "__main__":
    main()
