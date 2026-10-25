"""FP8 (E4M3) weights and activations for the transformer's linear layers, via torchao.

  1. Fuse the LoRA into the base weights first: quantization works on plain nn.Linear, and an unfused PEFT LoRA
     would either stay in bf16 (slow extra matmuls) or be quantized separately (worse precision).
  2. quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(PerRow())): each weight row gets its own scale
     once; each activation row gets a scale computed on the fly every call ("dynamic"). The matmul runs on FP8
     tensor cores (torch._scaled_mm) and accumulates in higher precision.
  3. Skip layers where FP8 buys nothing or costs most quality: the input/output projections and the timestep /
     guidance / text embedders (small, run once per step, and sensitive), and any layer whose dimensions are not
     multiples of 16 (FP8 GEMMs need them).
torch.compile (config 2 inherits it from config 1) fuses the per-call activation quantization into neighbouring
kernels; without it the quantization overhead eats much of the gain. See fast/TEACHING.md, "FP8".
"""
from __future__ import annotations

from typing import Any

import torch

SKIP = ("x_embedder", "context_embedder", "proj_out", "norm_out", "time_text_embed")


def _quantizable(module: torch.nn.Module, fqn: str) -> bool:
    return (isinstance(module, torch.nn.Linear) and not any(s in fqn for s in SKIP)
            and module.in_features % 16 == 0 and module.out_features % 16 == 0)


def apply(pipe: Any, granularity: str = "row") -> Any:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, PerTensor, quantize_

    if getattr(pipe, "get_active_adapters", None) and pipe.get_active_adapters():
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()
    gran = PerRow() if granularity == "row" else PerTensor()
    quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig(granularity=gran), filter_fn=_quantizable)
    return pipe
