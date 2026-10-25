"""Scoring logic without network or GPU: DreamSim bookkeeping with a fake embedder, Gemini word count and classes."""
from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from eval import eval_dreamsim as ed
from eval.eval_gemini import classify, word_count
from runner.store import LocalStore


def png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))  # (1, 3): the image's mean colour


def preprocess(img: Image.Image) -> torch.Tensor:
    return torch.tensor(list(img.getdata()), dtype=torch.float32).T.reshape(1, 3, 8, 8) / 255 + 0.01


def test_fidelity_and_diversity(tmp_path):
    store = LocalStore(tmp_path)
    for s, c in ((1, (255, 0, 0)), (2, (0, 255, 0))):
        store.put(f"outputs/0/000_a_x_b_seed{s}.png", png(c))
    store.put("outputs/1/000_a_x_b_seed1.png", png((255, 0, 0)))  # identical to config 0, seed 1
    store.put("outputs/1/000_a_x_b_seed2.png", png((255, 0, 0)))  # differs from config 0, seed 2
    res = ed.evaluate(store, "1", "0", FakeModel(), preprocess, "cpu")
    assert res["per_image_fidelity"]["000_a_x_b_seed1"] == pytest.approx(0, abs=1e-6)
    assert res["per_image_fidelity"]["000_a_x_b_seed2"] > 0.5
    assert res["per_pair_diversity"]["000_a_x_b"] == pytest.approx(0, abs=1e-6)  # both seeds identical: no diversity
    base = ed.evaluate(store, "0", "0", FakeModel(), preprocess, "cpu")
    assert base["fidelity"] is None and base["per_pair_diversity"]["000_a_x_b"] > 0.5


def test_word_count_ignores_bullets():
    assert word_count("* Use the shell texture\n* Form a lamp.") == 7
    assert word_count("*\n* copy <image2>") == 2


@pytest.mark.parametrize("text,cls", [
    ("* copy <image2>", "copy"),
    ("* Copy image 2.", "copy"),
    ("* Create a vertically split image.", "split"),
    ("* Place the cupcakes from image1 into the scene from image2.", "insertion"),
    ("* Use the texture of the frosting from image1 to create pink flower petals.", "other"),
])
def test_classify(text, cls):
    assert classify(text) == cls
