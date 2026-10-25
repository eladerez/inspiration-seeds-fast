"""LoRA loading for configs with more than one LoRA (5, 6) or a Nunchaku transformer (3, 4, 7).

Two backends, two mechanisms (see fast/TEACHING.md, "LoRA stacking"):
  diffusers  each LoRA becomes a named PEFT adapter; set_adapters() mixes them with their weights at run time.
             Optional `fuse` merges them into the base weights once (W += sum_i w_i * B_i A_i), removing the extra
             low-rank matmuls from every step.
  nunchaku   the transformer is SVDQuant-quantized; LoRAs are converted to Nunchaku's format and run in its fused
             low-rank branch. Several LoRAs are first composed into one (compose_lora), then strength is set to 1
             because the per-LoRA weights are already folded in.
"""
from __future__ import annotations

from typing import Any


def _hf_file(lora: dict) -> str:
    """Nunchaku takes 'user/repo/file.safetensors' or a local file path."""
    return f"{lora['path']}/{lora['weight_name']}" if lora.get("weight_name") else lora["path"]


def load(pipe: Any, loras: list[dict], backend: str, fuse: bool = False) -> Any:
    if backend == "diffusers":
        names = []
        for i, lora in enumerate(loras):
            name = f"lora{i}"
            pipe.load_lora_weights(lora["path"], weight_name=lora.get("weight_name"), adapter_name=name)
            names.append(name)
        pipe.set_adapters(names, adapter_weights=[lora.get("weight", 1.0) for lora in loras])
        if fuse:
            pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
            pipe.unload_lora_weights()  # drops the now-redundant adapter modules; fused weights stay
        return pipe
    if backend == "nunchaku":
        transformer = pipe.transformer
        if len(loras) == 1:
            transformer.update_lora_params(_hf_file(loras[0]))
            transformer.set_lora_strength(loras[0].get("weight", 1.0))
        else:
            from nunchaku.lora.flux.compose import compose_lora

            composed = compose_lora([(_hf_file(lora), lora.get("weight", 1.0)) for lora in loras])
            transformer.update_lora_params(composed)
            transformer.set_lora_strength(1.0)
        return pipe
    raise ValueError(f"unknown backend {backend}")
