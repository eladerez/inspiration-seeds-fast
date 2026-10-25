"""Render the LinkedIn post images (PNG, 1200 px wide, at most 4:5) from the same files as the pages:
  1_same_blend.png  one paper pair (Fig. 1's cupcake x crystal), seed 1: baseline and three speedups
  2_seeds.png       the pair with the largest diversity loss: four seeds for baseline, 4.9x and 7.5x
  3_results.png     seconds per image for every config, with its creativity and diversity change vs baseline

Fonts: IBM Plex Sans and Newsreader (Google Fonts, OFL) in data/fonts/; DejaVu Sans if they are missing.
Usage: uv run analysis/make_post_images.py [--out data/post]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from make_page import BENCH, BUCKET, NAMES, ROOT

W = 1200
BG, SURFACE, INK, INK2, LINE = "#f4f5f7", "#ffffff", "#14171c", "#545b66", "#dde1e7"
ACCENT, BASE = "#2a78d6", "#9aa2ad"
FONTS = ROOT / "data/fonts"
BLEND_PAIR = "046_food1_x_nature2"  # the paper's Fig. 1
SEEDS_PAIR = "015_fashion6_x_sea7"  # largest seed-diversity loss for config 7
BLEND_CONFIGS = ["0", "3", "4", "7"]
SEED_CONFIGS = ["0", "4", "7"]
SHORT = {"0": "Original", "3": "4-bit (NVFP4)", "4": "4-bit + step cache", "7": "4-bit + 8-step LoRA"}


def font(size: int, weight: str = "Regular", display: bool = False) -> ImageFont.FreeTypeFont:
    path = FONTS / ("Newsreader.ttf" if display else "IBMPlexSans.ttf")
    try:
        f = ImageFont.truetype(str(path), size)
        f.set_variation_by_name(weight)
        return f
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def tile(path: Path, size: int) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)


def header(d: ImageDraw.ImageDraw, title: str, sub: str, m: int = 60) -> int:
    d.text((m, 52), title, font=font(54, "SemiBold", display=True), fill=INK)
    d.text((m, 124), sub, font=font(26), fill=INK2)
    return 186


def footer(d: ImageDraw.ImageDraw, h: int, lines: list[str], m: int = 60) -> None:
    y = h - 30 - 30 * len(lines)
    d.line((m, y - 16, W - m, y - 16), fill=LINE, width=2)
    for k, line in enumerate(lines):
        d.text((m, y + 30 * k), line, font=font(21), fill=INK2)


def inputs_row(img: Image.Image, d: ImageDraw.ImageDraw, pair: dict, y: int, size: int, m: int = 60) -> int:
    x = m
    for k, name in enumerate((pair["image_1"], pair["image_2"])):
        img.paste(tile(BENCH / "images" / name, size), (x, y))
        x += size
        if k == 0:
            d.text((x + 22, y + size // 2), "+", font=font(44), fill=INK2, anchor="mm")
            x += 44
    d.text((x + 28, y + size // 2), "Two input images", font=font(24), fill=INK2, anchor="lm")
    return y + size + 34


def same_blend(results: dict, pairs: dict, out: Path) -> None:
    size, gap, cap = 450, 40, 62
    m = (W - 2 * size - gap) // 2
    h = 1500
    img = Image.new("RGB", (W, h), BG)
    d = ImageDraw.Draw(img)
    y = header(d, "Inspiration Seeds, 7.5× faster", "Same input pair, same seed, one RTX PRO 6000.", m)
    y = inputs_row(img, d, pairs[BLEND_PAIR], y, 130, m)
    base = results["0"]["s_per_image_median"]
    for i, c in enumerate(BLEND_CONFIGS):
        x = m + (i % 2) * (size + gap)
        yy = y + (i // 2) * (size + cap)
        if c == "0":
            d.rectangle((x - 4, yy - 4, x + size + 3, yy + size + 3), fill=BASE)
        img.paste(tile(BUCKET / f"outputs/{c}/{BLEND_PAIR}_seed1.png", size), (x, yy))
        s = results[c]["s_per_image_median"]
        d.text((x, yy + size + 12), SHORT[c], font=font(25, "SemiBold"), fill=INK)
        d.text((x + size, yy + size + 12), f"{s:.1f} s · {base / s:.1f}×", font=font(25), fill=INK2, anchor="ra")
    footer(d, h, ["Input pair from the paper's Fig. 1. Times: median of 396 images per configuration."], m)
    img.save(out / "1_same_blend.png", optimize=True)


def seeds(results: dict, pairs: dict, out: Path) -> None:
    size, gap, cap = 258, 16, 58
    h = 186 + 120 + 34 + 3 * (cap + size + 26) + 110
    img = Image.new("RGB", (W, h), BG)
    d = ImageDraw.Draw(img)
    y = header(d, "Faster, but fewer ideas per pair", "Four seeds of one pair. Few-step LoRAs make the seeds agree.")
    y = inputs_row(img, d, pairs[SEEDS_PAIR], y, 120)
    pdiv = {c: json.loads((BUCKET / f"eval/{c}/dreamsim.json").read_text())["per_pair_diversity"][SEEDS_PAIR]
            for c in SEED_CONFIGS}
    base = results["0"]["s_per_image_median"]
    for c in SEED_CONFIGS:
        s = results[c]["s_per_image_median"]
        speed = "" if c == "0" else f" · {base / s:.1f}×"
        d.text((60, y + 10), f"{SHORT[c]}", font=font(26, "SemiBold"), fill=INK)
        d.text((W - 60, y + 10), f"{s:.1f} s{speed} · seeds {pdiv[c]:.2f} apart", font=font(24), fill=INK2, anchor="ra")
        y += cap
        for k in range(4):
            img.paste(tile(BUCKET / f"outputs/{c}/{SEEDS_PAIR}_seed{k + 1}.png", size), (60 + k * (size + gap), y))
        y += size + 26
    div = {c: results[c]["diversity"] for c in ("0", "4", "7")}
    footer(d, h, [f"All 99 pairs, mean seed diversity (DreamSim): {div['0']:.3f} original · {div['4']:.3f} at 4.9× · "
                  f"{div['7']:.3f} at 7.5×", "Input pair from the paper's Figs. 10 and 22."])
    img.save(out / "2_seeds.png", optimize=True)


def change(r: dict, base_val: float) -> str:
    lo, hi = r["ci95"]
    if lo <= 0 <= hi:
        return "no clear change"
    return f"{r['mean_diff'] / base_val * 100:+.1f}%".replace("-", "−")


def results_chart(results: dict, paired: dict, out: Path) -> None:
    rows = list(NAMES)
    row_h, top = 96, 300
    h = top + row_h * len(rows) + 140
    img = Image.new("RGB", (W, h), BG)
    d = ImageDraw.Draw(img)
    header(d, "Speed vs creativity, 8 configurations", "Every config on one RTX PRO 6000, on the paper's full benchmark (99 pairs × 4 seeds).")
    bar_x, bar_w = 60, 440
    col1, col2 = 820, W - 60
    xmax = 25.0
    d.text((bar_x, top - 62), "Seconds per image", font=font(22, "SemiBold"), fill=INK2)
    d.text((col1, top - 62), "Creativity", font=font(22, "SemiBold"), fill=INK2, anchor="ra")
    d.text((col2, top - 62), "Seed diversity", font=font(22, "SemiBold"), fill=INK2, anchor="ra")
    for t in (0, 5, 10, 15, 20, 25):
        x = bar_x + bar_w * t / xmax
        d.line((x, top - 20, x, top + row_h * len(rows) - 20), fill=LINE, width=1)
        d.text((x, top + row_h * len(rows) - 10), f"{t} s", font=font(19), fill=INK2, anchor="ma")
    base = results["0"]
    for i, c in enumerate(rows):
        y = top + i * row_h
        r = results[c]
        s = r["s_per_image_median"]
        d.text((bar_x, y - 12), f"{c} · {NAMES[c][0]}", font=font(23, "SemiBold"), fill=INK)
        x1 = bar_x + bar_w * s / xmax
        d.rounded_rectangle((bar_x, y + 22, x1, y + 46), radius=4, fill=BASE if c == "0" else ACCENT)
        d.text((x1 + 12, y + 34), f"{s:.2f} s · {base['s_per_image_median'] / s:.2f}×", font=font(22), fill=INK, anchor="lm")
        words = "baseline 51.5 words" if c == "0" else change(paired[c]["words"], base["words_mean"])
        div = f"baseline {base['diversity']:.3f}" if c == "0" else change(paired[c]["diversity"], base["diversity"])
        d.text((col1, y + 34), words, font=font(22), fill=INK, anchor="rm")
        d.text((col2, y + 34), div, font=font(22), fill=INK, anchor="rm")
    footer(d, h, ["Creativity: the paper's metric (Gemini description length). Changes vs the original over the same 99",
                  "pairs; \"no clear change\": the 95% interval includes zero."])
    img.save(out / "3_results.png", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/post")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {r["config"]: r for r in json.loads((ROOT / "analysis/results.json").read_text())}
    paired = json.loads((ROOT / "analysis/paired.json").read_text())
    pairs = {p["pair_id"]: p for p in json.loads((BENCH / "pairs.json").read_text())}
    same_blend(results, pairs, args.out)
    seeds(results, pairs, args.out)
    results_chart(results, paired, args.out)
    for p in sorted(args.out.glob("*.png")):
        with Image.open(p) as im:
            print(f"{p} {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main()
