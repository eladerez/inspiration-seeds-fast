"""Step caching (first-block cache): skip most of the transformer on steps where its output would barely change.

At each step, run only the first transformer block and compare its output residual with the previous step's
(relative L1). If the change is below `threshold`, reuse the cached residual of all remaining blocks instead of
computing them. See fast/TEACHING.md, "Step caching".

  diffusers transformer  diffusers.hooks FirstBlockCacheConfig(threshold)
  Nunchaku transformer   nunchaku.caching.diffusers_adapters.apply_cache_on_pipe(residual_diff_threshold)
Both implement the same idea; the threshold is the speed/quality knob (higher = more steps skipped).
"""
from __future__ import annotations

from typing import Any


def apply(pipe: Any, threshold: float = 0.12) -> Any:
    if type(pipe.transformer).__module__.startswith("nunchaku"):
        from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe

        return apply_cache_on_pipe(pipe, residual_diff_threshold=threshold)
    from diffusers.hooks import FirstBlockCacheConfig

    transformer = pipe.transformer
    transformer.enable_cache(FirstBlockCacheConfig(threshold=threshold))
    # FluxKontextPipeline (diffusers 0.42) never opens a cache context around its transformer call, which the cache
    # hooks require (FluxPipeline does: `with self.transformer.cache_context("cond")`). Open it here instead. The
    # pipeline resets the cache state after every call (maybe_free_model_hooks), so images don't leak into each other.
    hooked = transformer.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        with transformer.cache_context("cond"):
            return hooked(*args, **kwargs)

    transformer.forward = forward
    return pipe
