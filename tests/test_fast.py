"""CPU tests of fast/ on the tiny random pipeline: LoRA stacking (and fusing), step caching, FP8 layer selection."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from fast import fp8, lora_stack, step_cache
from tests.test_runner import tiny_pipeline


def make_lora(pipe, out_dir, seed: int) -> None:
    from peft import LoraConfig, get_peft_model_state_dict

    torch.manual_seed(seed)
    t = pipe.transformer
    t.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian",
                             target_modules=["to_q", "to_k", "to_v", "to_out.0"]), adapter_name=f"tmp{seed}")
    with torch.no_grad():  # PEFT starts lora_B at zero (a no-op LoRA); make it a real change
        for n, p in t.named_parameters():
            if f"lora_B.tmp{seed}" in n:
                p.normal_(0, 0.5)
    sd = get_peft_model_state_dict(t, adapter_name=f"tmp{seed}")
    t.delete_adapters(f"tmp{seed}")
    type(pipe).save_lora_weights(out_dir, transformer_lora_layers=sd, weight_name="lora.safetensors")


def render(pipe) -> np.ndarray:
    img = Image.new("RGB", (64, 64), (120, 30, 200))
    out = pipe(image=img, prompt="x", num_inference_steps=4, guidance_scale=4.0, height=64, width=64,
               generator=torch.Generator().manual_seed(0), output_type="np", max_area=64 * 64, _auto_resize=False)
    return out.images[0]


@pytest.fixture
def loras(tmp_path):
    base = tiny_pipeline()
    specs = []
    for seed, w in ((1, 1.0), (2, 0.5)):
        d = tmp_path / f"lora{seed}"
        make_lora(base, d, seed)
        specs.append({"path": str(d), "weight_name": "lora.safetensors", "weight": w})
    return specs


def test_stacked_loras_change_output_and_fuse_matches(loras):
    plain = render(tiny_pipeline())
    stacked = render(lora_stack.load(tiny_pipeline(), loras, "diffusers"))
    fused = render(lora_stack.load(tiny_pipeline(), loras, "diffusers", fuse=True))
    assert np.abs(stacked - plain).max() > 1e-3          # the LoRAs do something
    assert np.abs(fused - stacked).max() < 1e-4          # fusing is the same function


def test_stack_weights_matter(loras):
    a = render(lora_stack.load(tiny_pipeline(), loras, "diffusers"))
    loras[1]["weight"] = 0.0
    b = render(lora_stack.load(tiny_pipeline(), loras, "diffusers"))
    assert np.abs(a - b).max() > 1e-4


def test_step_cache_zero_threshold_is_exact_and_large_threshold_differs():
    plain = render(tiny_pipeline())
    exact = render(step_cache.apply(tiny_pipeline(), threshold=0.0))
    skipping = render(step_cache.apply(tiny_pipeline(), threshold=1e6))
    assert np.abs(exact - plain).max() < 1e-5
    assert np.abs(skipping - plain).max() > 1e-4


def test_fp8_skips_embedders_and_projections():
    t = tiny_pipeline().transformer
    picked = {n for n, m in t.named_modules() if fp8._quantizable(m, n)}
    assert picked and not any(s in n for n in picked for s in fp8.SKIP)
    assert any("attn.to_q" in n for n in picked)
