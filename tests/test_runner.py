"""CPU tests of the runner with a tiny random FluxKontextPipeline (same classes as the real model, toy sizes).
Run: uv run --group gen pytest -q tests"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from runner import run_config
from runner.store import LocalStore


def tiny_pipeline(**kwargs):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxKontextPipeline, FluxTransformer2DModel
    from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

    torch.manual_seed(0)
    transformer = FluxTransformer2DModel(patch_size=1, in_channels=4, num_layers=1, num_single_layers=1,
                                         attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                         pooled_projection_dim=32, axes_dims_rope=[4, 4, 8])
    vae = AutoencoderKL(sample_size=32, in_channels=3, out_channels=3, block_out_channels=(4, 4, 4, 4),
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        layers_per_block=1, latent_channels=1, norm_num_groups=1, use_quant_conv=False,
                        use_post_quant_conv=False, shift_factor=0.0609, scaling_factor=1.5035)
    parts = {"transformer": transformer, "vae": vae, "scheduler": FlowMatchEulerDiscreteScheduler(),
             "image_encoder": None, "feature_extractor": None}
    if "text_encoder" in kwargs:  # cached embedding: no CLIP, no T5
        parts.update(text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None)
    else:
        clip = CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-5,
                              num_attention_heads=4, num_hidden_layers=5, pad_token_id=1, vocab_size=1000,
                              hidden_act="gelu", projection_dim=32)
        parts.update(text_encoder=CLIPTextModel(clip),
                     text_encoder_2=T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5"),
                     tokenizer=CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip"),
                     tokenizer_2=AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5"))
    return FluxKontextPipeline(**parts)


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    d = tmp_path / "bench"
    (d / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("a", "b", "c"):
        Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)).save(d / "images" / f"{name}.png")
    pairs = [{"pair_id": f"{i:03d}_{x}_x_{y}", "image_1": f"{x}.png", "image_2": f"{y}.png"}
             for i, (x, y) in enumerate([("a", "b"), ("b", "c"), ("a", "c")])]
    (d / "pairs.json").write_text(json.dumps(pairs))
    return d


def config(tmp_path: Path, cfg_id: str, **over) -> Path:
    cfg = {"id": cfg_id, "dtype": "float32", "guidance_scale": 4.0, "seeds": [1, 2], "backend": "diffusers",
           "num_inference_steps": 2, "text_embedding": "encode", "compile": None, "loras": []}
    cfg.update(over)
    p = tmp_path / f"{cfg_id}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def run(cfg: Path, store: LocalStore, bench: Path, **kw):
    return run_config.run(cfg, store, bench, device="cpu", pipeline_cls=tiny_pipeline, **kw)


def test_full_run_writes_every_image_and_timing(tmp_path, bench):
    store = LocalStore(tmp_path / "store")
    res = run(config(tmp_path, "0"), store, bench)
    assert res == {"generated": 6, "timing_lines": 6}
    assert len(store.list("outputs/0/")) == 6
    rows = [json.loads(l) for l in store.get("logs/0/timing.jsonl").decode().splitlines()]
    t = rows[0]["timing"]
    assert t["steps"] == 2 and len(t["step_s"]) == 2 and t["transformer_forwards"] == 2
    assert t["total"] >= t["transformer"] > 0 and t["text_encode"] > 0 and t["vae_decode"] > 0
    env = json.loads(store.get("logs/0/env.json"))
    assert env["config"] == "0" and env["diffusers"]


def test_resume_only_generates_missing(tmp_path, bench):
    store = LocalStore(tmp_path / "store")
    cfg = config(tmp_path, "0")
    run(cfg, store, bench)
    (tmp_path / "store/outputs/0/001_b_x_c_seed2.png").unlink()
    assert run(cfg, store, bench)["generated"] == 1


def test_smoke_is_two_pairs_one_seed(tmp_path, bench):
    store = LocalStore(tmp_path / "store")
    assert run(config(tmp_path, "0"), store, bench, smoke=True)["generated"] == 2
    assert run(config(tmp_path, "0"), store, bench)["generated"] == 4  # the full run reuses the smoke images


def test_changed_config_refuses(tmp_path, bench):
    store = LocalStore(tmp_path / "store")
    run(config(tmp_path, "0"), store, bench, smoke=True)
    with pytest.raises(SystemExit, match="changed"):
        run(config(tmp_path, "0", guidance_scale=3.5), store, bench)


def test_cached_embedding_matches_encoding(tmp_path, bench):
    store = LocalStore(tmp_path / "store")
    run(config(tmp_path, "0"), store, bench, smoke=True, text_cache=True)
    run(config(tmp_path, "1", text_embedding="cached"), store, bench, smoke=True)
    a = np.asarray(Image.open(tmp_path / "store/outputs/0/000_a_x_b_seed1.png"))
    b = np.asarray(Image.open(tmp_path / "store/outputs/1/000_a_x_b_seed1.png"))
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
    t = json.loads(store.get("logs/1/timing/000_a_x_b_seed1.json"))["timing"]
    assert t["text_encode"] < 0.05
