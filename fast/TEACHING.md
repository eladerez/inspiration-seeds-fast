# fast/ — what each speedup does, why, and what it can break

The question of this study: each speedup makes Inspiration Seeds faster, but does it keep the blends non-literal
(Gemini description length) and different across seeds (DreamSim diversity)? This file explains every choice in
`fast/`. Numbers marked "measured" come from `results/` or the bucket; everything else is
reasoning you should be able to reproduce on a whiteboard.

## Where the time goes (measured, config 0 smoke, RTX PRO 6000)

| Stage | Seconds per image |
|---|---:|
| Text encoders (CLIP + T5) | 0.03 |
| VAE encode of the 1024² canvas | 0.07 |
| Transformer, 28 steps × 1 forward | 20.61 |
| VAE decode | 0.13 |
| Total | 20.87 |

So 98.8% is the transformer. Caching the text embedding (config 1) saves almost no *time* — its value is
memory (T5-XXL is ~9 GB in bf16) and start-up. Every real gain has to cut either the **cost of one transformer
forward** (compile, FP8, NVFP4) or the **number of forwards** (step caching, 8-step LoRAs).

Why each forward is expensive: Kontext concatenates the 4096 tokens of the image being generated with the 4096
tokens of the input canvas (plus 512 text tokens), so attention runs over ~8.7k tokens through 19 double-stream and
38 single-stream blocks. Guidance is distilled into the model (guidance 4.0 is an input, not classifier-free
guidance), which is why it is one forward per step, not two (measured: 28 forwards for 28 steps).

## LoRA stacking — `lora_stack.py` (configs 3, 5, 6, 7)

A LoRA adds `ΔW = B·A` (rank r) to a weight `W`. Two ways to run it:

- **Unfused (PEFT adapters):** every linear computes `W·x + w·B·(A·x)`. Several LoRAs are just several such branches,
  each with its own weight `w_i` (`set_adapters`). Cheap to switch, but each LoRA adds two small matmuls to every
  targeted layer on every step.
- **Fused:** compute `W' = W + Σ w_i·B_i·A_i` once and drop the branches. Same function (test: `fused == stacked`
  within 1e-4), zero per-step overhead. Config 0 is unfused because upstream `generate.py` is; `lora_fuse: true`
  turns fusing on for a config.

**Stacking a turbo LoRA with the Seeds LoRA (configs 5, 6)** is the risky part. Both LoRAs were trained on the same
base weights, each assuming the other is absent. Their deltas add linearly, but the model is not linear: the turbo
LoRA was trained to make 8 steps land where 28 steps of the *base* model land, not of base + Seeds. Expect the Seeds
"style" of combination to weaken. That is precisely what the word count and diversity should show. The Hyper-FLUX
scale 0.125 is its model card's value; Turbo-Alpha is used at 1.0.

**Nunchaku:** its transformer is quantized, so LoRAs can't be added to `W`. Nunchaku runs the LoRA in its own
low-rank branch, fused into the same kernel as the low-rank correction SVDQuant already uses. Several LoRAs are
merged first (`compose_lora` concatenates the A's and the weighted B's into one larger-rank LoRA), then strength 1
is set because the weights are already folded in.

Questions to check your understanding: Why isn't stacking two LoRAs guaranteed to work? Why is fusing exact in FP32 but not exactly
exact in bf16? Why can't you fuse a LoRA into an NVFP4 weight?

## Step caching — `step_cache.py` (config 4)

Between neighbouring denoising steps, the transformer's output changes slowly, especially in the middle and late
steps. First-block cache exploits that: at each step run only the **first** block, and compare its output residual
with the previous step's (relative L1 distance). If the change is below `threshold`, skip the remaining 56 blocks
and reuse the residual they produced last time.

- `threshold` is the only knob: 0 = exact (test verifies it), higher = more steps skipped = faster and blurrier.
  0.12 is Nunchaku's default for FLUX.
- **What it can break for this study:** skipped steps are, by construction, steps where the model "agrees with
  itself". Diversity between seeds comes mostly from early steps (layout), which are rarely skipped, so diversity
  may survive; fine detail and texture transfer (late steps) is what can degrade, and Gemini's descriptions of
  texture transfer are exactly the long, non-literal ones.
- **A diffusers bug worked around here:** the cache hooks require a "cache context" that `FluxPipeline` opens and
  `FluxKontextPipeline` (diffusers 0.42) does not, so the hooks raised an error. `apply()` wraps the transformer
  forward in that context. The pipeline resets the cache after every call, so images don't leak into each other.

## FP8 — `fp8.py` (config 2)

The RTX PRO 6000 (Blackwell) has FP8 tensor cores with about twice the bf16 throughput. FP8 E4M3 has 3 mantissa
bits, so every tensor needs a scale:

- **Weights:** one scale per output row, computed once (`PerRow`). Per-row matters because rows differ in
  magnitude; one scale for the whole matrix would crush small rows to zero.
- **Activations:** one scale per row, computed on the fly every call ("dynamic"). Activations have outliers that
  change with the input, so static scales would clip.
- The matmul runs in FP8 and accumulates in higher precision (`torch._scaled_mm`).

Choices you should be able to justify:
1. **Fuse the LoRA first.** Quantization works on plain `nn.Linear`. An unfused LoRA would stay in bf16 (two extra
   matmuls per layer, eating the gain), and quantizing it separately loses its small delta in rounding.
2. **Skip the embedders and input/output projections.** They are small (little time to win) and sit where errors
   propagate everywhere (the timestep/guidance embedding conditions every block).
3. **Keep torch.compile on** (inherited from config 1). Without it, the per-call activation quantization is a
   separate kernel launch with a full read/write of the activation, which can cancel the matmul speedup.

What it can break: small, systematic errors in every block. FP8 usually costs little in single-image FID-style
metrics; whether it shifts the *kind* of combination (word count, copy/insertion/split) is the open question.

## NVFP4 via Nunchaku — configs 3, 4

SVDQuant (Nunchaku) quantizes weights *and* activations to 4 bits. It first moves activation outliers into the
weights (smoothing), then splits each weight into a small low-rank part kept in 16-bit (absorbs the outliers) plus a
4-bit residual. NVFP4 is Blackwell's 4-bit float format with a scale per 16 values.

## What "faster" means here

`bench/bench.py` times each stage with CUDA events after two untimed warm-up images (torch.compile and autotuning
happen there), so medians exclude compilation. Peak VRAM is `max_memory_allocated` per image.
