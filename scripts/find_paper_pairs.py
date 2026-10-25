"""Find the benchmark pairs that appear in the paper's figures and write analysis/paper_pairs.json
({pair_id: [figure numbers]}). The pages show these pairs; every statistic still uses all 99.

Each image placed in the paper PDF (poppler's pdftohtml -xml gives position and file) is matched to the 41 benchmark
images by correlating 24x24 thumbnails (match: correlation > 0.9). In the figures a pair's two inputs sit next to
each other (stacked or side by side), so two matched images that touch form a pair; each image joins one pair at
most, so the bottom input of one row never pairs with the top input of the next. Fig. 1 arranges its pairs around
one shared cupcake image instead, so its pairs are listed by hand (checked against the figure).

Usage: uv run scripts/find_paper_pairs.py --pdf third_party/InspirationSeeds/docs/InspirationSeeds.pdf
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
FIG1 = ["009_fashion5_x_food1", "046_food1_x_nature2", "052_food1_x_sea4", "087_arch1_x_food1"]


def thumb(path: Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = np.asarray(img.resize((24, 24), Image.Resampling.BOX), dtype=np.float32).ravel()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n else a


def touching(a: tuple, b: tuple) -> bool:
    ta, la, wa, ha, _ = a
    tb, lb, wb, hb, _ = b
    stacked = abs(la - lb) <= 4 and 0 <= tb - (ta + ha) <= 0.35 * ha
    side = abs(ta - tb) <= 4 and 0 <= lb - (la + wa) <= 0.35 * wa
    return stacked or side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--bench", type=Path, default=ROOT / "data/benchmark")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/paper_pairs.json")
    args = ap.parse_args()

    images = sorted((args.bench / "images").iterdir())
    names = [p.stem for p in images]
    bench = np.stack([thumb(p) for p in images])
    pairs = json.loads((args.bench / "pairs.json").read_text())
    pair_of = {frozenset((p["image_1"].rsplit(".", 1)[0], p["image_2"].rsplit(".", 1)[0])): p["pair_id"] for p in pairs}

    found: dict[str, set[int]] = {pid: {1} for pid in FIG1}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftohtml", "-xml", "-q", str(args.pdf), f"{tmp}/doc"], check=True, capture_output=True)
        xml = Path(f"{tmp}/doc.xml").read_text()
        for page in re.finditer(r'<page number="\d+".*?</page>', xml, re.S):
            body = page.group(0)
            figs = [int(n) for n in re.findall(r"Fig\. (\d+)\.", re.sub(r"<[^>]+>", "", body))]
            hits = []
            for m in re.finditer(r'<image top="(-?\d+)" left="(-?\d+)" width="(\d+)" height="(\d+)" src="([^"]+)"', body):
                t, l, w, h = map(int, m.groups()[:4])
                if w < 30:
                    continue
                score = bench @ thumb(Path(m.group(5)))
                if score.max() > 0.9:
                    hits.append((t, l, w, h, names[int(score.argmax())]))
            used: set[tuple] = set()
            for a in sorted(hits):  # top to bottom, left to right
                if a in used:
                    continue
                b = next((b for b in sorted(hits) if b != a and b not in used and touching(a, b)), None)
                if b is None:
                    continue
                used.update((a, b))
                pid = pair_of.get(frozenset((a[4], b[4])))
                if pid and figs:
                    found.setdefault(pid, set()).add(figs[-1] if figs[-1] != 11 else figs[0])  # p. 9: Figs 10, 11
    out = {pid: sorted(f) for pid, f in sorted(found.items())}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
