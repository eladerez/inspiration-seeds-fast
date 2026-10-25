"""Build the FluxKontextPipeline a config describes. The speedups live in fast/ and are called here:
  fast/fp8.py         apply(pipe, **options) -> pipe          (config key `fp8`)
  fast/step_cache.py  apply(pipe, **options) -> pipe          (config key `step_cache`)
  fast/lora_stack.py  load(pipe, loras, backend, fuse) -> pipe (more than one LoRA, or any LoRA on Nunchaku)
If a config needs one that fails to import, the run fails loudly."""
from __future__ import annotations

import importlib
from typing import Any

import torch

KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"


def _fast(module: str) -> Any:
    return importlib.import_module(f"fast.{module}")


def load_text_cache(blob: bytes, device: str) -> dict[str, torch.Tensor]:
    import io

    d = torch.load(io.BytesIO(blob), map_location=device, weights_only=True)
    return {"prompt_embeds": d["prompt_embeds"], "pooled_prompt_embeds": d["pooled_prompt_embeds"]}


def build(cfg: dict, device: str = "cuda", pipeline_cls: Any = None) -> Any:
    """Return a ready pipeline. `pipeline_cls` lets tests pass a tiny pipeline factory instead of from_pretrained."""
    dtype = getattr(torch, cfg["dtype"])
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "revision": cfg.get("revision")}
    if cfg["text_embedding"] == "cached":  # never load CLIP or T5
        kwargs.update(text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None)
    if cfg["backend"] == "nunchaku":
        from nunchaku import NunchakuFluxTransformer2dModel

        kwargs["transformer"] = NunchakuFluxTransformer2dModel.from_pretrained(cfg["nunchaku_weights"])
    elif cfg["backend"] != "diffusers":
        raise ValueError(f"unknown backend {cfg['backend']}")

    if pipeline_cls is None:
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(cfg.get("model", KONTEXT), **kwargs)
    else:
        pipe = pipeline_cls(**kwargs)

    loras = cfg.get("loras") or []
    if len(loras) > 1 or (loras and cfg["backend"] == "nunchaku"):
        pipe = _fast("lora_stack").load(pipe, loras, cfg["backend"], fuse=cfg.get("lora_fuse", False))
    elif loras:
        _load_single_lora(pipe, loras[0])

    pipe.to(device)
    if cfg.get("fp8"):
        pipe = _fast("fp8").apply(pipe, **cfg["fp8"])
    if cfg.get("step_cache"):
        pipe = _fast("step_cache").apply(pipe, **cfg["step_cache"])
    if cfg.get("compile"):
        pipe.transformer.compile(**cfg["compile"])
    return pipe


def _load_single_lora(pipe: Any, lora: dict) -> None:
    """Same as upstream generate.py: a HF repo id or a local path, adapter 'default', weight 1.0 unless set."""
    pipe.load_lora_weights(lora["path"], weight_name=lora.get("weight_name"), adapter_name="default")
    if lora.get("weight", 1.0) != 1.0:
        pipe.set_adapters(["default"], adapter_weights=[lora["weight"]])
