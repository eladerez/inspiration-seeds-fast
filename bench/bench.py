"""Per-stage timing and peak VRAM for one FluxKontextPipeline call, plus a summary CLI.

Stages are bounded with CUDA events, so nothing inside the transformer is touched (works with torch.compile,
Nunchaku and step caching):
  text_encode  pipe.encode_prompt                   (~0 when the embedding is cached)
  vae_encode   pipe.vae.encode of the input canvas
  transformer  end of vae_encode -> start of vae_decode: every denoising step, scheduler included
  vae_decode   pipe.vae.decode
  total        wall clock around the whole call, torch.cuda.synchronize() on both sides (pre/post-processing included)
Per-step times come from callback_on_step_end events.

Usage (summary): uv run bench/bench.py logs/0/timing.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

STAGES = ("total", "text_encode", "vae_encode", "transformer", "vae_decode")


class _WallEvent:
    """CPU stand-in for torch.cuda.Event, so the harness can be tested without a GPU."""

    def record(self) -> None:
        self.t = time.perf_counter()

    def elapsed_time(self, other: "_WallEvent") -> float:
        return (other.t - self.t) * 1e3


@dataclass
class StageTimer:
    """Wraps a pipeline's stage functions with CUDA events. Call `measure(fn)` per image."""

    pipe: Any
    count_forwards: bool = False
    _events: dict[str, list] = field(default_factory=dict)
    _forwards: int = 0

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self._cuda = torch.cuda.is_available()
        self._wrap(self.pipe, "encode_prompt", "text_encode")
        self._wrap(self.pipe.vae, "encode", "vae_encode")
        self._wrap(self.pipe.vae, "decode", "vae_decode")
        if self.count_forwards:  # a Python counter would break torch.compile graphs, so only when not compiling
            self.pipe.transformer.register_forward_pre_hook(self._count)

    def _count(self, *_: Any) -> None:
        self._forwards += 1

    def _event(self, name: str) -> None:
        e = self._torch.cuda.Event(enable_timing=True) if self._cuda else _WallEvent()
        e.record()
        self._events.setdefault(name, []).append(e)

    def _wrap(self, obj: Any, attr: str, stage: str) -> None:
        fn = getattr(obj, attr)

        def timed(*args: Any, **kwargs: Any) -> Any:
            self._event(f"{stage}_start")
            out = fn(*args, **kwargs)
            self._event(f"{stage}_end")
            return out

        setattr(obj, attr, timed)

    def _sample_device(self) -> None:
        """Whole-GPU used memory (cudaMemGetInfo), which also sees allocations outside PyTorch's allocator, such as
        Nunchaku's 4-bit kernels' own buffers. Sampled at every step end and after VAE decode; no sync needed."""
        if self._cuda:
            free, total = self._torch.cuda.mem_get_info()
            self._device_peak = max(self._device_peak, total - free)

    def step_callback(self, pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
        self._event("step_end")
        self._sample_device()
        return {}

    def _ms(self, a: str, b: str, i: int = 0, j: int = 0) -> float:
        ea, eb = self._events.get(a), self._events.get(b)
        return ea[i].elapsed_time(eb[j]) if ea and eb else 0.0

    def measure(self, call: Callable[..., Any], **kwargs: Any) -> tuple[Any, dict]:
        """Run `call(callback_on_step_end=..., **kwargs)` and return (output, timing dict in seconds)."""
        torch = self._torch
        self._events, self._forwards, self._device_peak = {}, 0, 0
        sync = torch.cuda.synchronize if self._cuda else (lambda: None)
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()
        sync()
        t0 = time.perf_counter()
        out = call(callback_on_step_end=self.step_callback, **kwargs)
        sync()
        total = time.perf_counter() - t0
        self._sample_device()
        steps = self._events.get("step_end", [])
        loop_start = "vae_encode_end" if "vae_encode_end" in self._events else "text_encode_end"
        step_ms = [self._ms(loop_start, "step_end", -1, 0)] if steps else []
        step_ms += [steps[k - 1].elapsed_time(steps[k]) for k in range(1, len(steps))]
        timing = {
            "total": total,
            "text_encode": self._ms("text_encode_start", "text_encode_end") / 1e3,
            "vae_encode": self._ms("vae_encode_start", "vae_encode_end") / 1e3,
            "transformer": self._ms(loop_start, "vae_decode_start", -1, 0) / 1e3,
            "vae_decode": self._ms("vae_decode_start", "vae_decode_end") / 1e3,
            "steps": len(steps),
            "step_s": [round(s / 1e3, 4) for s in step_ms],
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if self._cuda else 0.0,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if self._cuda else 0.0,
            "peak_device_used_gb": self._device_peak / 2**30,
        }
        if self.count_forwards:
            timing["transformer_forwards"] = self._forwards
        return out, timing


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows: list[dict]) -> dict:
    out: dict[str, Any] = {"images": len(rows)}
    for s in STAGES:
        xs = [r["timing"][s] for r in rows]
        out[s] = {"median": statistics.median(xs), "p90": percentile(xs, 0.9)} if xs else None
    out["peak_vram_gb"] = max((r["timing"]["peak_vram_gb"] for r in rows), default=None)
    dev = [r["timing"]["peak_device_used_gb"] for r in rows if "peak_device_used_gb" in r["timing"]]
    out["peak_device_used_gb"] = max(dev) if dev else None  # configs 0-3 predate this field
    return out


def main(paths: list[str]) -> None:
    for p in paths:
        rows = [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
        print(p, json.dumps(summarize(rows), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
