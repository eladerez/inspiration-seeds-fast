"""Results table (and later charts) from the store: logs/<c>/timing.jsonl, eval/<c>/dreamsim.json, gemini_run*.json.
Every number in the table comes from those files. Writes analysis/results.json and analysis/results.md.

Usage: uv run analysis/plots.py --store gs://inspiration-seeds-speed [--configs 0,1,2]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.bench import summarize  # noqa: E402
from runner.store import Store, open_store  # noqa: E402


def load_json(store: Store, key: str) -> dict | None:
    return json.loads(store.get(key)) if store.exists(key) else None


def _nunchaku(store: Store, cfg: str) -> bool:
    env = load_json(store, f"logs/{cfg}/env.json") or {}
    return bool(env.get("nunchaku")) or (ROOT / f"runner/configs/{cfg}.yaml").read_text().count("backend: nunchaku") > 0


def row(store: Store, cfg: str) -> dict | None:
    if not store.exists(f"logs/{cfg}/timing.jsonl"):
        return None
    lines = [json.loads(l) for l in store.get(f"logs/{cfg}/timing.jsonl").decode().splitlines() if l.strip()]
    t = summarize(lines)  # smoke images are part of the full set (the full run reuses them), timed after warm-up too
    ds = load_json(store, f"eval/{cfg}/dreamsim.json") or {}
    runs = [load_json(store, f"eval/{cfg}/gemini_run{k}.json") for k in (1, 2)]
    words = [r["summary"]["words_mean"] for r in runs if r]
    gpu = (load_json(store, f"logs/{cfg}/env.json") or {}).get("gpu")
    return {"config": cfg, "gpu": gpu, "images": t["images"], "s_per_image_median": t["total"]["median"],
            "s_per_image_p90": t["total"]["p90"], "transformer_median": t["transformer"]["median"],
            "text_encode_median": t["text_encode"]["median"], "vae_decode_median": t["vae_decode"]["median"],
            # whole-GPU peak where recorded (configs 4+); PyTorch's counter misses Nunchaku's own allocations, so
            # a Nunchaku config without the whole-GPU figure (config 3) has no valid peak
            "peak_vram_gb": t.get("peak_device_used_gb") or (None if _nunchaku(store, cfg) else t["peak_vram_gb"]),
            "words_mean": statistics.fmean(words) if words else None,
            "words_run_spread": (max(words) - min(words)) if len(words) == 2 else None,
            "rate_copy": runs[0]["summary"]["rate_copy"] if runs[0] else None,
            "rate_insertion": runs[0]["summary"]["rate_insertion"] if runs[0] else None,
            "rate_split": runs[0]["summary"]["rate_split"] if runs[0] else None,
            "fidelity": (ds.get("fidelity") or {}).get("mean"), "diversity": (ds.get("diversity") or {}).get("mean")}


def fmt(v: float | None, spec: str) -> str:
    return "not measured" if v is None else format(v, spec)


def markdown(rows: list[dict]) -> str:
    base = next((r for r in rows if r["config"] == "0"), None)
    head = ("| Config | Images | s/image (median) | p90 | Speedup vs 0 | Transformer s | Peak VRAM GB | Words | "
            "Run spread | Fidelity (DreamSim to 0) | Diversity |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    out = [head]
    for r in rows:
        # a speedup only means something on the same GPU as the baseline
        speed = base["s_per_image_median"] / r["s_per_image_median"] if base and r.get("gpu") == base.get("gpu") else None
        out.append(f"| {r['config']} | {r['images']} | {fmt(r['s_per_image_median'], '.2f')} | "
                   f"{fmt(r['s_per_image_p90'], '.2f')} | {fmt(speed, '.2f') + '×' if speed else '—'} | {fmt(r['transformer_median'], '.2f')} | "
                   f"{fmt(r['peak_vram_gb'], '.1f')} | {fmt(r['words_mean'], '.1f')} | {fmt(r['words_run_spread'], '.1f')} | "
                   f"{fmt(r['fidelity'], '.3f')} | {fmt(r['diversity'], '.3f')} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--configs", default="0,1,2,3,4,5,6,7")  # RTX PRO 6000 only
    args = ap.parse_args()
    store = open_store(args.store)
    rows = [r for c in args.configs.split(",") if (r := row(store, c))]
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "analysis/results.json").write_text(json.dumps(rows, indent=1))
    (ROOT / "analysis/results.md").write_text(markdown(rows))
    print(markdown(rows))


if __name__ == "__main__":
    main()
