"""Build docs/index.html, the project page (GitHub Pages landing page). Charts and numbers come from the same files
as docs/compare.html (analysis/results.json, analysis/paired.json, data/bucket eval files); images are downscaled
outputs embedded as data URIs. The pair viewer stays in docs/compare.html.

Usage: uv run analysis/make_project_page.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from make_page import BENCH, BUCKET, NAMES, ROOT, forest, jpeg, rates, speed_chart, write_page

OUT = ROOT / "docs/index.html"
COMPARE_ARTIFACT = "https://claude.ai/artifact/C4hLmC2nD5Jodv57SXwUEJ"  # preview only; GitHub Pages uses compare.html
TEASER_PAIR = "046_food1_x_nature2"  # the cupcake and crystal pair from the paper's Fig. 1
TEASER_CONFIGS = ["0", "3", "4", "7"]
SEEDS_PAIR = "015_fashion6_x_sea7"  # largest seed-diversity loss for config 7 across the 99 pairs


def main() -> None:
    results = {r["config"]: r for r in json.loads((ROOT / "analysis/results.json").read_text())}
    paired = json.loads((ROOT / "analysis/paired.json").read_text())
    pairs = {p["pair_id"]: p for p in json.loads((BENCH / "pairs.json").read_text())}
    rows = [results[c] for c in NAMES]
    base = results["0"]
    pdiv = {c: json.loads((BUCKET / f"eval/{c}/dreamsim.json").read_text())["per_pair_diversity"] for c in NAMES}

    tp = pairs[TEASER_PAIR]
    teaser = [f'<figure><img src="{jpeg(BENCH / "images" / tp["image_1"], 300)}" alt="Input 1"><figcaption><b>Input 1</b></figcaption></figure>',
              f'<figure><img src="{jpeg(BENCH / "images" / tp["image_2"], 300)}" alt="Input 2"><figcaption><b>Input 2</b></figcaption></figure>']
    for c in TEASER_CONFIGS:
        s = results[c]["s_per_image_median"]
        cls = ' class="base"' if c == "0" else ""
        teaser.append(f'<figure{cls}><img src="{jpeg(BUCKET / f"outputs/{c}/{TEASER_PAIR}_seed1.png", 420)}" '
                      f'alt="{html.escape(NAMES[c][0])}"><figcaption><b>{html.escape(NAMES[c][0])}</b><br>'
                      f'{s:.2f} s · {base["s_per_image_median"] / s:.1f}×</figcaption></figure>')

    seed_rows = []
    for c in ("0", "7"):
        imgs = "".join(f'<img src="{jpeg(BUCKET / f"outputs/{c}/{SEEDS_PAIR}_seed{k}.png", 300)}" alt="{html.escape(NAMES[c][0])}, seed {k}">'
                       for k in (1, 2, 3, 4))
        seed_rows.append(f'<div class="seedrow"><div class="who"><b>{c} · {html.escape(NAMES[c][0])}</b>'
                         f'<span class="muted">seeds {pdiv[c][SEEDS_PAIR]:.3f} apart</span></div>{imgs}</div>')
    sp = pairs[SEEDS_PAIR]
    seeds_inputs = (f'<img src="{jpeg(BENCH / "images" / sp["image_1"], 200)}" alt="Input 1">'
                    f'<img src="{jpeg(BENCH / "images" / sp["image_2"], 200)}" alt="Input 2">')

    trs = []
    for r in rows:
        c = r["config"]
        rt = rates(c)
        speedup = f'{base["s_per_image_median"] / r["s_per_image_median"]:.2f}×'
        vram = "not measured" if r["peak_vram_gb"] is None else f'{r["peak_vram_gb"]:.1f}'
        fid = "—" if c == "0" else f'{r["fidelity"]:.3f}'
        hl = ' class="hl"' if c in ("3", "4", "7") else ""
        trs.append(f'<tr{hl}><td>{c}</td><td>{html.escape(NAMES[c][0])}<span class="sub">{html.escape(NAMES[c][1])}</span></td>'
                   f'<td class="n">{r["s_per_image_median"]:.2f}</td><td class="n">{speedup}</td><td class="n">{vram}</td>'
                   f'<td class="n">{r["words_mean"]:.1f}</td><td class="n">{rt["copy"]:.1f} / {rt["insertion"]:.1f} / {rt["split"]:.1f}</td>'
                   f'<td class="n">{r["diversity"]:.3f}</td><td class="n">{fid}</td></tr>')

    cost = (ROOT / "logs/cost.md").read_text().split("Running total: $")[1].split()[0]
    shared = (ROOT / "analysis/compare_template.html").read_text()  # one look for both pages
    shared = shared[shared.index('<link rel="preconnect"'):shared.index("</style>") + len("</style>")]
    tpl = (ROOT / "analysis/project_template.html").read_text().replace("{{SHARED_STYLE}}", shared)
    page = (tpl.replace("{{TEASER}}", "".join(teaser))
               .replace("{{SPEED_CHART}}", speed_chart(rows))
               .replace("{{WORDS_CHART}}", forest(paired, "words", "Change in description length vs baseline",
                                                 lambda v: f"{v:+.1f}".replace("+0.0", "0"), base["words_mean"], 3.0))
               .replace("{{DIV_CHART}}", forest(paired, "diversity", "Change in seed diversity vs baseline",
                                               lambda v: f"{v:+.3f}".replace("+0.000", "0"), base["diversity"], 0.05))
               .replace("{{SEEDS_INPUTS}}", seeds_inputs)
               .replace("{{SEED_ROWS}}", "".join(seed_rows))
               .replace("{{SEEDS_LABEL}}", f'{sp["image_1"].rsplit(".", 1)[0]} × {sp["image_2"].rsplit(".", 1)[0]}')
               .replace("{{TABLE_ROWS}}", "".join(trs))
               .replace("{{COST}}", cost))
    write_page(OUT.name, page, {'href="compare.html"': f'href="{COMPARE_ARTIFACT}"'})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
