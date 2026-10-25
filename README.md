# Inspiration Seeds, Faster

How much faster can [Inspiration Seeds](https://arxiv.org/abs/2602.08615) (FLUX.1 Kontext [dev] + a LoRA) run before it
stops producing non-literal, diverse blends? Eight configurations (configs 0–7: the baseline and seven speedups), each run on the paper's full benchmark
(99 pairs × 4 seeds = 396 images), scored for speed, creativity (the paper's description-length metric) and seed
diversity.

**Project page:** [docs/index.html](docs/index.html) · **The paper's pairs side by side:** [docs/compare.html](docs/compare.html)

## Results

Every config ran on the same GPU, one RTX PRO 6000 (Blackwell), so every speedup is measured against a baseline on
that GPU. Median of 396 images per config. The paper's ~34 s per image was measured on an L40S, a different GPU:
compare our speedups with our own baseline (20.91 s), not with the paper's time.

| # | Configuration | s/image | Speedup | Peak GB | Words | Diversity | Distance to baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Baseline: upstream settings, bf16, 28 steps | 20.91 | 1.00× | 34.2 | 51.5 | 0.358 | — |
| 1 | Cached prompt embedding + torch.compile | 15.87 | 1.32× | 25.0 | 51.2 | 0.356 | 0.020 |
| 2 | 1 + FP8 (torchao) | 11.89 | 1.76× | 15.4 | 51.2 | 0.357 | 0.036 |
| 3 | **Nunchaku NVFP4 (4-bit)** | 8.69 | **2.41×** | not measured | 51.4 | 0.369 | 0.074 |
| 4 | **NVFP4 + first-block step cache** | 4.24 | **4.93×** | 11.7 | 50.1 | 0.347 | 0.102 |
| 5 | Turbo-Alpha 8-step LoRA (bf16) | 7.06 | 2.96× | 27.3 | 50.8 | 0.326 | 0.219 |
| 6 | Hyper-FLUX 8-step LoRA (bf16) | 7.08 | 2.95× | 27.3 | 50.7 | 0.335 | 0.252 |
| 7 | **NVFP4 + Turbo-Alpha 8-step** | 2.77 | **7.54×** | 12.7 | 51.0 | 0.332 | 0.223 |

- **Words:** mean description length from two Gemini scoring runs (higher = more complex, non-literal blend).
- **Diversity:** mean DreamSim distance between the 4 seeds of a pair (higher = more different ideas per pair).
- **Distance to baseline:** DreamSim between each output and the baseline's output for the same pair and seed; two
  different baseline seeds are 0.358 apart.
- Paired tests over the same 99 pairs (95% bootstrap intervals) are in [analysis/paired.json](analysis/paired.json);
  full write-up in [analysis/conclusions.md](analysis/conclusions.md). All numbers come from files produced by the
  scripts in this repo.

### What we found

1. **No speedup made the blends more literal.** Description length moved by at most 1.4 words of 51.5; copy,
   insertion and split stayed rare.
2. **Cheaper steps are free.** FP8 and NVFP4 cost neither creativity nor diversity (NVFP4's seed diversity even
   rose 3%).
3. **Fewer steps cost diversity.** 8-step LoRAs keep each blend non-literal but make the seeds agree more
   (−6% to −9%), with or without 4-bit weights.

## Use it: faster generate.py

`generate_fast.py` is a drop-in for the original `generate.py` with the measured presets:

```bash
uv run --group gen --group nunchaku generate_fast.py \
    --image1 a.jpg --image2 b.jpg --preset nvfp4-cache --num-seeds 4
```

| Preset | Config | GPU | Trade-off |
|---|---|---|---|
| `nvfp4` | 3 | Blackwell | 2.4×, no loss in creativity or diversity |
| `nvfp4-cache` | 4 | Blackwell | 4.9×, −2.8% words, −3% diversity |
| `nvfp4-turbo` | 7 | Blackwell | 7.5×, −7% diversity |

FLUX.1 Kontext [dev] is released under a non-commercial license: accept it on Hugging Face and set `HF_TOKEN`.

## Reproduce the benchmark

Environment: Python 3.11 with [uv](https://docs.astral.sh/uv/). The GPU runs used Google Cloud spot VMs
(g4-standard-48 with an RTX PRO 6000) and the `common-cu129-ubuntu-2404-nvidia-580` Deep
Learning VM image.

```bash
uv sync --group gen --group eval                       # local tools and CPU tests
uv run pytest -q tests                                 # 16 CPU tests on a tiny random FluxKontextPipeline
uv run scripts/prepare_benchmark.py                    # 99 pairs / 41 images from the public HF benchmark
uv run scripts/find_paper_pairs.py --pdf <paper.pdf>   # which of them appear in the paper's figures
bash infra/setup_gcp.sh                                # once: secrets, runner service account, bucket access
infra/chain.sh 3 0 1 2 3 4 5 6 7                       # spot VM per config: generate, DreamSim, teardown, Gemini scoring
uv run analysis/plots.py --store gs://<bucket>         # results table
uv run analysis/paired.py --store gs://<bucket>        # paired statistics
uv run analysis/make_page.py && uv run analysis/make_project_page.py   # docs/ pages (from a local copy of the bucket)
```

Project, bucket and service-account names are set at the top of the `infra/` scripts; change them for your own
project. Every VM is spot, deletes itself, has a maximum run duration, and runs one at a time; `infra/check_teardown.sh`
removes stopped VMs and logs the cost. Config 0 must run first with `--write-text-cache` (it saves the prompt
embedding that later configs load).

## Repository

| Path | What |
|---|---|
| `runner/` | resumable batch runner, configs 0–7, canvas identical to upstream, storage (GCS or local) |
| `fast/` | the speedups: FP8 (`fp8.py`), step cache (`step_cache.py`), LoRA stacking (`lora_stack.py`); [TEACHING.md](fast/TEACHING.md) explains each |
| `bench/` | per-stage CUDA-event timing and peak memory |
| `eval/` | DreamSim fidelity and diversity; Gemini description complexity (paper appendix F prompt) |
| `analysis/` | result tables, paired statistics, conclusions, page builders |
| `infra/` | Google Cloud launch, startup, watch and teardown scripts |
| `docs/` | project page and the pair viewer for the 29 pairs shown in the paper (GitHub Pages) |
| `logs/cost.md` | every VM launch and Gemini run behind these results, with its cost (total $49.45) |

## Limits

- The paper scored creativity with Gemini 2.5 Flash, which is closed to new API users; we used Gemini 3.8 Flash, so
  word counts are compared with our own baseline (51.5), not the paper's 54.8. No human evaluation.
- Copy / insertion / split use our own keyword rule; the paper does not publish its rule.
- Config 3's peak memory was not recorded correctly (PyTorch's counter misses Nunchaku's allocations; fixed from
  config 4 on).
- One model, one benchmark, one GPU type (RTX PRO 6000), one step-cache threshold (0.12).

## Credits and licenses

- Inspiration Seeds: Kfir Goldberg, Elad Richardson, Yael Vinker ([paper](https://arxiv.org/abs/2602.08615),
  [code](https://github.com/kfirgoldberg/InspirationSeeds), MIT). The canvas and prompt code in `runner/canvas.py`
  follows their `generate.py`.
- FLUX.1 Kontext [dev]: Black Forest Labs, non-commercial license. No weights are included here; they are downloaded
  at run time.
- [Nunchaku / SVDQuant](https://github.com/nunchaku-tech/nunchaku), FLUX.1 Turbo-Alpha (Alimama Creative), Hyper-FLUX
  (ByteDance), [DreamSim](https://github.com/ssundaram21/dreamsim), torchao, diffusers.
- The images in `docs/` are downscaled benchmark inputs (Inspiration Seeds benchmark, MIT) and outputs generated for
  this study.
- This repository's code: MIT, see [LICENSE](LICENSE).

This is an independent, non-commercial research benchmark.
