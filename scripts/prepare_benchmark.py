"""Unpack the public Inspiration Seeds benchmark (HF: kfirgold99/Inspiration-Seeds-Benchmark) into
data/benchmark/images/<name>.<ext> (41 images) and data/benchmark/pairs.json (99 pairs, parquet order).

Usage: uv run scripts/prepare_benchmark.py [--parquet data/benchmark_raw/test.parquet]
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/kfirgold99/Inspiration-Seeds-Benchmark/resolve/main/data/test-00000-of-00001.parquet"
ROOT = Path(__file__).resolve().parents[1]


def pair_id(index: int, name_1: str, name_2: str) -> str:
    return f"{index:03d}_{name_1}_x_{name_2}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=ROOT / "data/benchmark_raw/test.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/benchmark")
    args = ap.parse_args()
    if not args.parquet.exists():
        args.parquet.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, args.parquet)

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    pairs, files = [], {}
    for i, row in enumerate(pq.read_table(args.parquet).to_pylist()):
        names = []
        for k in ("1", "2"):
            name, img = row[f"image_{k}_name"], row[f"image_{k}"]
            ext = Path(img["path"] or "x.png").suffix.lower() or ".png"
            fname = f"{name}{ext}"
            digest = hashlib.sha256(img["bytes"]).hexdigest()
            if files.setdefault(fname, digest) != digest:
                raise ValueError(f"{fname} appears with two different contents")
            (images_dir / fname).write_bytes(img["bytes"])
            names.append((name, fname, row[f"category_{k}"]))
        pairs.append({"pair_id": pair_id(i, names[0][0], names[1][0]),
                      "image_1": names[0][1], "image_2": names[1][1],
                      "category_1": names[0][2], "category_2": names[1][2]})
    (args.out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"{len(pairs)} pairs, {len(files)} images -> {args.out}")


if __name__ == "__main__":
    main()
