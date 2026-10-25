"""Input canvas, identical to upstream generate.py (github.com/kfirgoldberg/InspirationSeeds, MIT):
1024x1024 white canvas, image 1 resized to 512x512 at top-left, image 2 at bottom-right."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

PROMPT = (
    "combine the element in the top left with the element in the bottom right to create a single object inspired by "
    "both of them"
)


def load_rgb(path: Path) -> Image.Image:
    """Same as diffusers.utils.load_image for a local file: EXIF orientation applied, RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def create_input_grid(img1: Image.Image, img2: Image.Image) -> Image.Image:
    grid = Image.new("RGB", (1024, 1024), "white")
    grid.paste(img1.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
    grid.paste(img2.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS), (512, 512))
    return grid
