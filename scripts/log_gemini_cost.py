"""Add one logs/cost.md row per Gemini scoring run (eval/<c>/gemini_run<k>.json) not logged yet, priced from the
token counts each run recorded. Rates: Gemini 3.8 Flash as listed by third-party price trackers on 24 Oct 2026
($0.75 / M input, $3.75 / M output incl. thinking), not yet checked against Google's own page.

Usage: uv run scripts/log_gemini_cost.py --store gs://inspiration-seeds-speed
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.store import open_store  # noqa: E402

IN_PER_M, OUT_PER_M = 0.75, 3.75


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    args = ap.parse_args()
    store = open_store(args.store)
    path = ROOT / "logs/cost.md"
    text = path.read_text()
    for key in store.list("eval/"):
        m = re.match(r"eval/([^/]+)/gemini_run(\d)\.json$", key)
        if not m:
            continue
        name = f"gemini-{m[1]}-run{m[2]}"
        if f"| {name} |" in text:
            continue
        s = json.loads(store.get(key))["summary"]
        if s["n"] < 396:
            continue  # log only finished runs
        cost = s["prompt_tokens"] / 1e6 * IN_PER_M + s["output_tokens"] / 1e6 * OUT_PER_M
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = f"| {name} | gemini-3.8-flash ({s['n']} calls) | API | — | {now} | — | — | {cost:.2f} |\n"
        total = float(re.search(r"Running total: \$([\d.]+)", text)[1]) + cost
        head, _ = text.split("\nRunning total:")
        text = head.rstrip("\n") + "\n" + row + f"\nRunning total: ${total:.2f}\n"
        print(f"logged {name}: ${cost:.2f}")
    path.write_text(text)


if __name__ == "__main__":
    main()
