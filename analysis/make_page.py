"""Build docs/compare.html, comparing the baseline with every speedup config on real images, plus one
docs/pairs/<pair_id>.json for each benchmark pair that appears in the paper's figures (analysis/paper_pairs.json, from
scripts/find_paper_pairs.py), fetched when that pair is picked; every statistic still covers all 99 pairs. All numbers come from analysis/results.json, analysis/paired.json and the eval files in the local copy of
the bucket (data/bucket); images are downscaled JPEGs embedded as data URIs.

Usage: uv run analysis/make_page.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import statistics
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BUCKET = ROOT / "data/bucket"
BENCH = ROOT / "data/benchmark"
OUT = ROOT / "docs/compare.html"
PAIRS_DIR = ROOT / "docs/pairs"  # published next to the page as pairs/<pair_id>.json
PAPER_PAIRS = ROOT / "analysis/paper_pairs.json"  # pair_id -> figures of the paper that show it

NAMES = {  # every config ran on the same RTX PRO 6000
    "0": ("Baseline", "Upstream settings, bf16, 28 steps"),
    "1": ("Cached prompt + compile", "T5 never loaded; torch.compile"),
    "2": ("+ FP8", "FP8 weights and activations"),
    "3": ("NVFP4", "Nunchaku 4-bit transformer"),
    "4": ("NVFP4 + step cache", "First-block cache, threshold 0.12"),
    "5": ("Turbo-Alpha 8-step", "Few-step LoRA, bf16"),
    "6": ("Hyper-FLUX 8-step", "Few-step LoRA, bf16"),
    "7": ("NVFP4 + Turbo-Alpha", "4-bit and 8 steps"),
}
SEED_VIEW = ["0", "4", "7"]


STANDALONE_HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
PREVIEW_DIR = ROOT / "data/artifact"  # artifact previews: same body, no document head (the viewer adds its own)


def write_page(name: str, body: str, preview_links: dict[str, str] | None = None) -> Path:
    """docs/<name> gets a real document head for GitHub Pages; data/artifact/<name> is the bare body for previews."""
    out = ROOT / "docs" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(STANDALONE_HEAD + body)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old, new in (preview_links or {}).items():
        body = body.replace(old, new)
    (PREVIEW_DIR / name).write_text(body)
    return out


def jpeg(path: Path, size: int, quality: int = 80) -> str:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def rates(cfg: str) -> dict[str, float]:
    runs = [json.loads((BUCKET / f"eval/{cfg}/gemini_run{k}.json").read_text())["summary"] for k in (1, 2)]
    return {k: statistics.fmean(r[f"rate_{k}"] for r in runs) * 100 for k in ("copy", "insertion", "split")}


def choose_pairs(pairs: list[dict]) -> list[tuple[str, str]]:
    """Among the given pairs: the one where config 7 lost the most seed diversity, the one where it gained the most, and
    four typical pairs (drop closest to the median) from four common category combinations."""
    div = {c: json.loads((BUCKET / f"eval/{c}/dreamsim.json").read_text())["per_pair_diversity"] for c in ("0", "7")}
    drop = {p["pair_id"]: div["0"][p["pair_id"]] - div["7"][p["pair_id"]] for p in pairs}
    med = statistics.median(drop.values())
    chosen = [(max(drop, key=drop.get), "Largest diversity loss"), (min(drop, key=drop.get), "Diversity gained")]
    for combo in ("arch/food", "arch/sea", "fashion/food", "nature/sea"):
        cands = [p["pair_id"] for p in pairs if f"{p['category_1']}/{p['category_2']}" == combo
                 and p["pair_id"] not in {c for c, _ in chosen}]
        if cands:
            chosen.append((min(cands, key=lambda q: abs(drop[q] - med)), "Typical pair"))
    return chosen[2:] + chosen[:2]


# ---------------------------------------------------------------- charts (inline SVG, one scale each)
def speed_chart(rows: list[dict]) -> str:
    w, row_h, left, right = 640, 30, 190, 120
    h = row_h * len(rows) + 34
    xmax = 26.0
    x = lambda v: left + (w - left - right) * v / xmax
    base = rows[0]["s_per_image_median"]
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Seconds per image for each configuration">']
    for t in (0, 5, 10, 15, 20, 25):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="4" y2="{h - 26}"/>'
                   f'<text class="tick" x="{x(t):.1f}" y="{h - 10}" text-anchor="middle">{t} s</text>')
    for i, r in enumerate(rows):
        y = 6 + i * row_h
        v = r["s_per_image_median"]
        cls = "bar base" if r["config"] == "0" else "bar"
        name, _ = NAMES[r["config"]]
        rel = f"{base / v:.2f}×"
        tip = f'{r["config"]} · {name}: {v:.2f} s/image, {rel}'
        out.append(f'<text class="lab" x="{left - 10}" y="{y + 15}" text-anchor="end">{r["config"]} · {html.escape(name)}</text>')
        out.append(f'<rect class="{cls}" x="{left}" y="{y + 3}" width="{x(v) - left:.1f}" height="{row_h - 10}" rx="3" '
                   f'data-tip="{html.escape(tip)}"/>')
        out.append(f'<text class="val" x="{x(v) + 6:.1f}" y="{y + 15}">{v:.2f} s · {rel}</text>')
    out.append("</svg>")
    return "".join(out)


def forest(paired: dict, metric: str, title: str, unit_fmt, base_val: float, span: float) -> str:
    cfgs = [c for c in NAMES if c != "0"]
    w, row_h, left, right = 380, 28, 150, 20
    h = row_h * len(cfgs) + 40
    x = lambda v: left + (w - left - right) * (v + span) / (2 * span)
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for t in (-span, -span / 2, 0, span / 2, span):
        out.append(f'<line class="{"zero" if t == 0 else "grid"}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="4" y2="{h - 28}"/>'
                   f'<text class="tick" x="{x(t):.1f}" y="{h - 12}" text-anchor="middle">{unit_fmt(t)}</text>')
    for i, c in enumerate(cfgs):
        r = paired[c][metric]
        y = 6 + i * row_h + row_h / 2
        lo, hi, m = r["ci95"][0], r["ci95"][1], r["mean_diff"]
        sig = lo > 0 or hi < 0
        tip = (f'{c} · {NAMES[c][0]}: {unit_fmt(m)} ({m / base_val * 100:+.1f}%), 95% CI {unit_fmt(lo)} to '
               f'{unit_fmt(hi)}; lower in {r["share_lower"] * 100:.0f}% of pairs')
        out.append(f'<text class="lab" x="{left - 10}" y="{y + 4}" text-anchor="end">{c} · {html.escape(NAMES[c][0])}</text>')
        out.append(f'<line class="ci" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y}" y2="{y}"/>')
        out.append(f'<circle class="{"pt sig" if sig else "pt"}" cx="{x(m):.1f}" cy="{y}" r="5" data-tip="{html.escape(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    results = {r["config"]: r for r in json.loads((ROOT / "analysis/results.json").read_text())}
    paired = json.loads((ROOT / "analysis/paired.json").read_text())
    pairs = json.loads((BENCH / "pairs.json").read_text())
    figs = json.loads(PAPER_PAIRS.read_text())
    pairs = [p for p in pairs if p["pair_id"] in figs]  # the page shows the paper's pairs
    rows = [results[c] for c in NAMES]
    base = results["0"]

    chosen = dict(choose_pairs(pairs))
    words = {}  # (config, "pair_seedN") -> mean words over both Gemini runs
    for c in NAMES:
        runs = [json.loads((BUCKET / f"eval/{c}/gemini_run{k}.json").read_text())["items"] for k in (1, 2)]
        words[c] = {name: statistics.fmean(r[name]["words"] for r in runs) for name in runs[0]}
    pdiv = {c: json.loads((BUCKET / f"eval/{c}/dreamsim.json").read_text())["per_pair_diversity"] for c in NAMES}
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    for old in PAIRS_DIR.glob("*.json"):
        old.unlink()
    index = []
    for p in pairs:  # one file per pair, fetched by the page only when that pair is picked
        pid = p["pair_id"]
        data = {"in1": jpeg(BENCH / "images" / p["image_1"], 220), "in2": jpeg(BENCH / "images" / p["image_2"], 220),
                "outs": {c: jpeg(BUCKET / f"outputs/{c}/{pid}_seed1.png", 320, 78) for c in NAMES},
                "seeds": {c: [None] + [jpeg(BUCKET / f"outputs/{c}/{pid}_seed{s}.png", 220, 76) for s in (2, 3, 4)]
                          for c in SEED_VIEW},
                "words": {c: round(words[c][f"{pid}_seed1"], 1) for c in NAMES},
                "pdiv": {c: round(pdiv[c][pid], 3) for c in NAMES}}
        (PAIRS_DIR / f"{pid}.json").write_text(json.dumps(data))
        index.append({"id": pid, "label": f'{p["image_1"].rsplit(".", 1)[0]} × {p["image_2"].rsplit(".", 1)[0]}',
                      "cat": f'{p["category_1"]} × {p["category_2"]}', "why": chosen.get(pid), "figs": figs[pid]})

    # results table
    trs = []
    for r in rows:
        c = r["config"]
        rt = rates(c)
        fid = "—" if c == "0" else f'{r["fidelity"]:.3f}'
        vram = "not measured" if r["peak_vram_gb"] is None else f'{r["peak_vram_gb"]:.1f}'
        hl = ' class="hl"' if c in ("4", "7") else ""
        speedup = f'{base["s_per_image_median"] / r["s_per_image_median"]:.2f}×'
        trs.append(
            f'<tr{hl}><td>{c}</td><td>{html.escape(NAMES[c][0])}'
            f'<span class="sub">{html.escape(NAMES[c][1])}</span></td>'
            f'<td class="n">{r["s_per_image_median"]:.2f}</td><td class="n">{speedup}</td>'
            f'<td class="n">{vram}</td><td class="n">{r["words_mean"]:.1f}</td>'
            f'<td class="n">{rt["copy"]:.1f} / {rt["insertion"]:.1f} / {rt["split"]:.1f}</td>'
            f'<td class="n">{r["diversity"]:.3f}</td><td class="n">{fid}</td></tr>')

    page_json = json.dumps({"pairs": index, "showcase": list(chosen), "names": NAMES, "seedView": SEED_VIEW,
                            "speed": {r["config"]: r["s_per_image_median"] for r in rows},
                            "words": {r["config"]: r["words_mean"] for r in rows},
                            "div": {r["config"]: r["diversity"] for r in rows}})
    cost = (ROOT / "logs/cost.md").read_text().split("Running total: $")[1].split()[0]

    tpl = (ROOT / "analysis/compare_template.html").read_text()
    page = (tpl.replace("{{SPEED_CHART}}", speed_chart(rows))
               .replace("{{WORDS_CHART}}", forest(paired, "words", "Change in description length vs baseline",
                                                 lambda v: f"{v:+.1f}".replace("+0.0", "0"), base["words_mean"], 3.0))
               .replace("{{DIV_CHART}}", forest(paired, "diversity", "Change in seed diversity vs baseline",
                                               lambda v: f"{v:+.3f}".replace("+0.000", "0"), base["diversity"], 0.05))
               .replace("{{TABLE_ROWS}}", "".join(trs))
               .replace("{{COST}}", cost)
               .replace("{{N_PAIRS}}", str(len(index)))
               .replace("{{DATA}}", page_json))
    write_page(OUT.name, page)
    total = sum(f.stat().st_size for f in PAIRS_DIR.glob("*.json"))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) and {len(index)} pair files ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
