# Conclusions: how much faster can Inspiration Seeds run before it stops being creative?

Written with Claude (Anthropic) at Elad Erez's request, 24 Oct 2026.
Every number comes from `analysis/results.json` and `analysis/paired.json` (built by `analysis/plots.py` and
`analysis/paired.py` from the bucket) or `logs/cost.md`.

## The short answer

**7.5× faster (20.91 s → 2.77 s per image) with no measurable loss in creativity, and 4.9× faster with creativity
and diversity almost intact.** The speedups that make each step cheaper (FP8, 4-bit NVFP4) leave the blends alone.
The ones that cut the number of steps (8-step LoRAs) keep the blends just as non-literal, but make the four seeds of a
pair agree with each other more: seed diversity drops by 6–9%, a loss that image-quality metrics would not show.

## Setup

- Inspiration Seeds = FLUX.1 Kontext [dev] + the authors' LoRA, the paper's fixed prompt and 2×2 canvas, 28 steps,
  guidance 4.0. Benchmark: the paper's 99 pairs × 4 seeds = 396 images per config, same seeds everywhere.
- One RTX PRO 6000 (Blackwell, 96 GB), spot VM on Google Cloud. Timing excludes 2 warm-up images; median of 396.
- **Creativity:** Gemini describes how to recreate each output from its two inputs (the paper's appendix F prompt,
  verbatim); the word count is the score. Two independent scoring runs per config.
- **Diversity:** mean pairwise DreamSim distance between the 4 seeds of a pair (higher = more varied).
- **Fidelity:** DreamSim distance between each output and the baseline's output for the same pair and seed.
- **Significance:** paired over the same 99 pairs, 95% bootstrap interval of the difference to config 0.

## Results

| # | Config | s/image | Speedup | Words (mean of 2 runs) | Words vs 0, 95% CI | Diversity | Diversity vs 0, 95% CI | Fidelity to 0 |
|---|---|---:|---:|---:|---|---:|---|---:|
| 0 | Baseline (upstream settings, bf16) | 20.91 | 1.00× | 51.5 | — | 0.358 | — | — |
| 1 | Cached prompt embedding + torch.compile | 15.87 | 1.32× | 51.2 | −0.3 [−1.1, +0.4] | 0.356 | −0.002 [−0.005, +0.001] | 0.020 |
| 2 | 1 + FP8 | 11.89 | 1.76× | 51.2 | −0.3 [−1.0, +0.4] | 0.357 | −0.002 [−0.006, +0.003] | 0.036 |
| 3 | Nunchaku NVFP4 (4-bit) | 8.69 | 2.41× | 51.4 | −0.1 [−0.9, +0.7] | 0.369 | **+0.011** [+0.005, +0.016] | 0.074 |
| 4 | 3 + first-block step cache | 4.24 | 4.93× | 50.1 | **−1.4** [−2.3, −0.5] | 0.347 | **−0.011** [−0.018, −0.004] | 0.102 |
| 5 | Turbo-Alpha 8-step LoRA (bf16) | 7.06 | 2.96× | 50.8 | −0.7 [−1.7, +0.3] | 0.326 | **−0.032** [−0.044, −0.020] | 0.219 |
| 6 | Hyper-FLUX 8-step LoRA (bf16) | 7.08 | 2.95× | 50.7 | −0.8 [−1.8, +0.2] | 0.335 | **−0.023** [−0.037, −0.009] | 0.252 |
| 7 | NVFP4 + Turbo-Alpha 8-step | **2.77** | **7.54×** | 51.0 | −0.5 [−1.5, +0.5] | 0.332 | **−0.026** [−0.038, −0.014] | 0.223 |

Bold = the 95% interval excludes zero. The run-to-run spread of the word count is 0.1–0.6 words. For scale on
fidelity: two different seeds of the same pair in the baseline are 0.358 apart.

Copy / insertion / split rates (our keyword rule, mean of 2 runs) stay low everywhere: copy 0.6–1.6%,
insertion 3.0–5.8%, split 1.8–3.5%. The baseline is 1.1 / 5.4 / 2.8%.

## What we learned

1. **No speedup made the blends more literal.** The word count never fell by more than 1.4 words (2.8%), far inside
   the study's 10% gate, and copy/insertion/split stayed rare (at most 1.6% / 5.8% / 3.5% of outputs, against
   the baseline's 1.1% / 5.4% / 2.8%). By the paper's own creativity measure, every config is
   as creative as the original.

2. **Making each step cheaper is free; skipping or shortening steps is not.** FP8 and NVFP4 cost neither creativity
   nor diversity (NVFP4 is even slightly *more* diverse: +0.011, +3%, 95% interval above zero). Their fidelity drift (0.04–0.07) is a fraction of
   the seed-to-seed distance: the images change in detail, not in idea.

3. **Few-step LoRAs cost diversity, not creativity.** Turbo-Alpha and Hyper-FLUX keep the word count but make the
   four seeds more alike (−6% to −9%, lower in 58–71% of pairs) and drift far from the baseline (0.22–0.25). Each
   blend is still non-literal; you just get fewer different ideas per pair. For a tool built for exploration, that is
   the cost that matters, and it is invisible to speed or image-quality numbers.

4. **The losses don't compound.** Adding NVFP4 to Turbo-Alpha (config 7) leaves fidelity (0.223 vs 0.219) and
   diversity (0.332 vs 0.326) where Turbo-Alpha alone put them. The diversity cost comes from the 8-step LoRA, not
   from the 4-bit weights.

5. **Step caching is the cheapest speedup for what it costs.** On top of NVFP4 it doubles the speed (8.69 → 4.24 s)
   for −2.8% words and −3% diversity: both statistically real, both small.

6. **Where the time went:** in the baseline, the transformer takes 20.65 of 20.91 s (98.8%). Text encoding and VAE
   decoding are negligible, so caching the prompt embedding saves memory (T5 is never loaded), not time.

## Recommendation

- **Fastest without giving up idea diversity: config 4** (NVFP4 + step cache), **4.93×**, 4.24 s per image.
- **Fastest overall: config 7** (NVFP4 + Turbo-Alpha), **7.54×**, 2.77 s per image, if a ~7% drop in seed diversity
  is acceptable.

Strictly by the study's gate (words within ~10% of config 0 and diversity "not clearly lower"), config 4 is the
answer: its diversity drop is statistically real but small (−3%), while config 7's is larger (−7%) and lower in 68%
of pairs. Which one is right depends on whether you value one fast blend or several different ones.

## Limits

- **Scorer:** the paper used Gemini 2.5 Flash, which is closed to new users; we used Gemini 3.8 Flash. Compare
  configs to our config 0 (51.5 words), not to the paper's 54.8. No human check of the word count was done.
- **Copy / insertion / split** use our own keyword rule; the paper does not publish its rule.
- **Peak VRAM** is PyTorch's allocator peak for configs 0–2, and whole-GPU used memory for configs 4–7 (PyTorch's
  counter misses Nunchaku's own allocations); config 3 has no valid figure. Measured: 34.2 GB (0), 25.0 GB (1),
  15.4 GB (2), 11.7 GB (4), 27.3 GB (5, 6), 12.7 GB (7).
- **Scope:** one model, one 99-pair benchmark, one GPU type, one threshold for the step cache (0.12), guidance 4.0
  kept for the turbo LoRAs.
- **Cloud cost of these runs:** $49.45 logged in `logs/cost.md` (GPU at published spot + 25%, Gemini at third-party listed
  rates). All results are copied to the Jetson (`data/bucket/`, checksums verified); cloud work ended 24 Oct 2026.
