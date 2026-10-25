"""Description complexity (paper appendix F): a Gemini Flash model describes how to recreate each output from its two
inputs; the word count of that description is the creativity proxy (paper: 54.8 +/- 12.5 words for Inspiration
Seeds, with Gemini 2.5 Flash; we use Gemini 3.8 Flash, so compare to our own config 0, not to 54.8). Run twice per config (--run 1, --run 2) to see the run-to-run spread.

Inputs follow the paper's setup for its own method: the two original input images and the output, as three images,
with the appendix F prompt verbatim. Runs on the laptop / Jetson, not on a GPU VM.

Copy / insertion / split: the paper reports these rates but does not give the rule. `classify()` below is OUR
keyword rule over Gemini's description, so those three rates are not directly comparable to the paper's table.

The API key is read from $GEMINI_API_KEY, or from Secret Manager (gemini-api-key) at runtime; it is never printed
or written anywhere. Token usage is recorded per call, so the real cost can be computed from the results.

Usage: uv run --group eval eval/eval_gemini.py --config 0 --run 1 --store gs://inspiration-seeds-speed
           [--bench-dir data/benchmark] [--limit 10] [--workers 8]
Writes eval/<config>/gemini_run<k>.json (resumable: items already in that file are skipped).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.eval_dreamsim import outputs  # noqa: E402
from runner.store import Store, open_store  # noqa: E402

MODEL = "gemini-3.8-flash"  # the paper used gemini-2.5-flash, closed to new API users by Oct 2026
PROMPT = """The first two images inspired the third. Describe briefly how you would recreate the output using only the two inputs.
Use short bullet points, not paragraphs. Maximum 5 bullets total, but you do not have to use them all.
Notes:
* Use "*" to denote bullets. Your answer should include only bullet points, no free text.
* Be concise when possible.
* If the output image is very similar to one of the inputs you can just say "copy <image1>/<image2>" accordingly.
* Examples of instructions you can use: "place object from <image1> in the scene from <image2>", "copy <image1>", "copy <image2>", "use the object from <image1> and the texture from <image2>". These are just examples, you can write your own instructions."""

COPY = re.compile(r"^\W*copy\b", re.I | re.M)
INSERT = re.compile(r"\b(place|insert|paste|put)\b.{0,60}\b(in|into|onto|on)\b.{0,40}\b(scene|background)\b", re.I)
SPLIT = re.compile(r"\b(split|side[- ]by[- ]side|grid|two halves|left half|right half|top half|bottom half)\b", re.I)


def word_count(text: str) -> int:
    """Words in the description, bullet markers not counted."""
    return len([w for w in text.replace("*", " ").split() if re.search(r"\w", w)])


def classify(text: str) -> str:
    if COPY.search(text):
        return "copy"
    if SPLIT.search(text):
        return "split"
    if INSERT.search(text):
        return "insertion"
    return "other"


def api_key() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    gcloud = "gcloud" if subprocess.run(["which", "gcloud"], capture_output=True).returncode == 0 \
        else str(Path.home() / "google-cloud-sdk/bin/gcloud")
    return subprocess.run([gcloud, "secrets", "versions", "access", "latest", "--secret=gemini-api-key",
                           "--project=inspiration-seed"], capture_output=True, text=True, check=True).stdout.strip()


def mime(name: str) -> str:
    return "image/png" if name.endswith(".png") else "image/jpeg"


def describe(client, parts: list[tuple[bytes, str]], retries: int = 5) -> dict:
    from google.genai import types

    contents = [types.Part.from_bytes(data=b, mime_type=m) for b, m in parts] + [PROMPT]
    for attempt in range(retries):
        try:
            r = client.models.generate_content(model=MODEL, contents=contents)
            u = r.usage_metadata
            return {"text": r.text or "", "prompt_tokens": u.prompt_token_count,
                    "output_tokens": u.candidates_token_count, "thought_tokens": u.thoughts_token_count or 0}
        except Exception as e:  # rate limits and transient errors: back off and retry
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1}: {type(e).__name__}", flush=True)
            time.sleep(2 ** attempt * 5)
    raise RuntimeError("unreachable")


def summary(items: dict) -> dict:
    wc = [v["words"] for v in items.values()]
    cls = [v["class"] for v in items.values()]
    return {"n": len(wc), "words_mean": statistics.fmean(wc) if wc else None,
            "words_std": statistics.pstdev(wc) if wc else None,
            **{f"rate_{c}": cls.count(c) / len(cls) if cls else None for c in ("copy", "insertion", "split")},
            "prompt_tokens": sum(v["prompt_tokens"] for v in items.values()),
            "output_tokens": sum(v["output_tokens"] + v["thought_tokens"] for v in items.values())}


def run(store: Store, cfg: str, k: int, bench_dir: Path, limit: int | None, workers: int) -> dict:
    from google import genai

    key = f"eval/{cfg}/gemini_run{k}.json"
    done = json.loads(store.get(key))["items"] if store.exists(key) else {}
    pairs = {p["pair_id"]: p for p in json.loads((bench_dir / "pairs.json").read_text())}
    todo = [(p, s, k_) for (p, s), k_ in sorted(outputs(store, cfg).items()) if f"{p}_seed{s}" not in done]
    todo = todo[:limit] if limit else todo
    client = genai.Client(api_key=api_key())

    def one(job: tuple) -> tuple[str, dict]:
        pair_id, seed, out_key = job
        p = pairs[pair_id]
        parts = [((bench_dir / "images" / p["image_1"]).read_bytes(), mime(p["image_1"])),
                 ((bench_dir / "images" / p["image_2"]).read_bytes(), mime(p["image_2"])),
                 (store.get(out_key), "image/png")]
        r = describe(client, parts)
        return f"{pair_id}_seed{seed}", {**r, "words": word_count(r["text"]), "class": classify(r["text"])}

    with ThreadPoolExecutor(workers) as pool:
        for n, (name, item) in enumerate(pool.map(one, todo), 1):
            done[name] = item
            if n % 20 == 0 or n == len(todo):  # save progress as we go
                store.put(key, json.dumps({"model": MODEL, "run": k, "summary": summary(done), "items": done},
                                          indent=1).encode())
                print(f"{n}/{len(todo)}", flush=True)
    return summary(done)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", type=int, required=True, choices=(1, 2))
    ap.add_argument("--store", required=True)
    ap.add_argument("--bench-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/benchmark")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(run(open_store(args.store), args.config, args.run, args.bench_dir, args.limit, args.workers)))


if __name__ == "__main__":
    main()
