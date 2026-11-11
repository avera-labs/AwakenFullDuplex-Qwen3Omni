"""Fast thinker: CUDA-graphed single-frame backbone forward for the real-time loop. Collapses the bs=1
kernel-launch overhead (measured about 51 -> 10 ms/frame on one data-center GPU).

The win requires making the 30B MoE **shape-stable** so torch.compile(reduce-overhead) can cudagraph
it without per-frame recompiles: force the `batched_mm` experts impl (all experts dense + mask; same
math as the default grouped_mm, different bf16 rounding, so re-validate outputs when changing it).

Two extra gotchas this helper handles:
  * StaticCache's `cumulative_length` counter is created on CPU and only moves to GPU on the FIRST
    forward — so we PREFILL EAGER (lazy-init → GPU) BEFORE compiling, else cudagraph is skipped.
  * We pass NO position_ids/cache_position per frame — the model derives them from the cache's own
    counter, so dynamo sees no value-varying inputs (no recompiles).

Used by the engine's compiled path (BatchedCompiledThinker) and as a standalone B=1 helper:

    from duplexaudio.inference.thinker.fast_thinker import CompiledThinker
    ct = CompiledThinker(model, max_cache_len=2048)   # sets batched_mm; frozen model unchanged otherwise
    ct.reset()                                         # per conversation
    for e in prompt_embeds: ct.prefill_step(e)         # eager prompt prefill (also lazy-inits + then compiles)
    ct.compile()                                       # build the compiled fn (call once after first prefill)
    h = ct.step(fused_emb)                             # per frame -> (1, hidden); advances cache internally
"""
from __future__ import annotations
import torch


def enable_batched_mm(model) -> None:
    """Force the shape-stable `batched_mm` experts implementation on every MoE sub-module's config.
    Same math as grouped_mm (unselected experts masked out), different bf16 rounding."""
    cfgs = [getattr(model, "config", None), getattr(model.backbone, "config", None)]
    for m in model.backbone.modules():
        cfgs.append(getattr(m, "config", None))
    for c in cfgs:
        if c is not None:
            c._experts_implementation = "batched_mm"
            c._experts_implementation_internal = "batched_mm"


class CompiledThinker:
    """Wraps the frozen thinker backbone with a StaticCache + torch.compile(reduce-overhead) so each frame's
    forward replays a CUDA graph. Manages the cache internally (advances via its own counter)."""

    def __init__(self, model, max_cache_len: int = 2048, compile_mode: str = "reduce-overhead",
                 batched_mm: bool = True):
        if batched_mm:
            enable_batched_mm(model)
        self.model = model
        self.bb = model.backbone
        self.max_cache_len = int(max_cache_len)
        self.compile_mode = compile_mode
        self.dev = next(self.bb.parameters()).device
        self.dtype = next(self.bb.parameters()).dtype
        self.cache = None
        self.cbb = None                                  # compiled backbone (built after first prefill)

    # ---- cache lifecycle -----------------------------------------------------------------
    def _new_cache(self):
        from transformers.cache_utils import StaticCache
        for kw in (dict(config=self.bb.config, max_cache_len=self.max_cache_len,
                        device=self.dev, dtype=self.dtype),
                   dict(config=self.bb.config, max_cache_len=self.max_cache_len)):
            try:
                return StaticCache(**kw)
            except TypeError:
                continue
        raise RuntimeError("could not construct StaticCache")

    def reset(self):
        """New conversation. Prefer an IN-PLACE StaticCache reset (keeps tensor addresses -> the
        captured cudagraph stays valid -> no expensive re-record); only allocate fresh the first time."""
        if self.cache is not None and hasattr(self.cache, "reset"):
            try:
                self.cache.reset()          # zero in place; cudagraph remains valid -> fast reset
                return
            except Exception:
                pass
        self.cache = self._new_cache()

    # ---- forwards ------------------------------------------------------------------------
    @torch.no_grad()
    def prefill_step(self, e):
        """EAGER prompt/prefill frame (uncompiled) — lazy-inits the StaticCache onto the GPU. Run the
        whole prompt through this BEFORE `.compile()`. Returns last hidden (1, hidden)."""
        if self.cache is None:
            self.reset()
        o = self.bb(inputs_embeds=e, past_key_values=self.cache, use_cache=True)
        return o.last_hidden_state[:, -1, :]

    def compile(self):
        """Build the compiled backbone. Call ONCE after at least one prefill_step (so the cache's
        counter is already on GPU). First few `.step()` calls then compile + capture the graph (slow),
        thereafter ~9.7 ms/frame."""
        if self.cbb is None:
            self.cbb = torch.compile(self.bb, mode=self.compile_mode)

    @torch.no_grad()
    def step(self, e):
        """Per-frame compiled forward. `e` = fused (1,1,hidden) embedding. Advances the cache
        internally (its own counter). Returns last hidden (1, hidden)."""
        o = self.cbb(inputs_embeds=e, past_key_values=self.cache, use_cache=True)
        return o.last_hidden_state[:, -1, :]

    @torch.no_grad()
    def warmup(self, n: int = 24):
        """Run n compiled frames on VARIED real token embeddings to fully settle reduce-overhead's
        cudagraph tree, so the first real frame is already fast. Zeros/degenerate or a single repeated
        input doesn't settle it (measured: 200-440 ms/frame spillover); varied real embeddings do."""
        V = self.bb.config.vocab_size
        for _ in range(n):
            wemb = self.bb.embed_tokens(torch.randint(0, V, (1,), device=self.dev)).unsqueeze(0)
            self.step(wemb)
        torch.cuda.synchronize()


class BatchedCompiledThinker:
    """Batched (B>=1) generalization of CompiledThinker: batched StaticCache + torch.compile
    (reduce-overhead) cudagraph over the per-frame forward. Handles RAGGED (variable-length) prompts via
    a FIXED-WIDTH key-padding attention_mask (hides the left-pad tokens) — NOT per-row position_ids:
    RoPE attention depends on relative position (i-j), so a constant per-row left-pad offset is invisible
    to attention. The mask is CONSTANT during decode (pad slots known at prefill, decode slots real) ->
    cudagraph-clean. Validated (B=5, ragged prompts): eager StaticCache+mask == BatchedDecoder 100% argmax;
    compiled == 98% (bf16 near-tie), 2.4x, no per-frame recompile. dense-MoE (install_dense_moe) is the
    shape-stability prereq.

        ct = BatchedCompiledThinker(model, max_cache_len=2048)
        ct.prefill(ids_BxL, pad_mask_BxL, embed)   # eager left-pad prefill (lazy-inits cache)
        ct.compile(); ct.warmup()                  # capture cudagraph at this B
        h = ct.step(fused_Bx1xH)                    # per frame -> (B, hidden)
    """

    def __init__(self, model, max_cache_len: int = 2048, compile_mode: str = "reduce-overhead",
                 batched_mm: bool = True):
        if batched_mm:
            enable_batched_mm(model)
        self.model = model
        self.bb = model.backbone
        self.max_cache_len = int(max_cache_len)
        self.compile_mode = compile_mode
        self.dev = next(self.bb.parameters()).device
        self.dtype = next(self.bb.parameters()).dtype
        self.cache = None
        self.cbb = None
        self.B = 0
        self.full_mask = None            # (B, max_cache_len) fixed key-padding mask

    def _new_cache(self):
        from transformers.cache_utils import StaticCache
        for kw in (dict(config=self.bb.config, max_cache_len=self.max_cache_len, max_batch_size=self.B,
                        device=self.dev, dtype=self.dtype),
                   dict(config=self.bb.config, max_cache_len=self.max_cache_len, max_batch_size=self.B)):
            try:
                return StaticCache(**kw)
            except TypeError:
                continue
        raise RuntimeError("could not construct StaticCache with a batch dim")

    def reset(self):
        if self.cache is not None and hasattr(self.cache, "reset"):
            try:
                self.cache.reset()
                return
            except Exception:
                pass
        self.cache = self._new_cache()

    @torch.no_grad()
    def prefill(self, ids, pad_mask, embed, micro_batch=None):
        """EAGER left-pad prefill. ids/pad_mask: (B, Lmax) (left-padded, pad_mask 1=real). Builds the
        fixed (B, max_cache_len) key-padding mask (prompt region = pad_mask, decode region = 1) and lazy-
        inits the StaticCache onto the GPU. Call BEFORE compile().
        micro_batch=None (default): one forward over all B rows (the prefill transient scales with B —
        ~30 GiB/row for a ~600-token persona on the 30B MoE). micro_batch=m: see `_prefill_rows`."""
        self.B = ids.shape[0]
        self.cache = self._new_cache()
        Lmax = ids.shape[1]
        self.full_mask = torch.ones((self.B, self.max_cache_len), dtype=torch.long, device=self.dev)
        self.full_mask[:, :Lmax] = pad_mask
        if micro_batch:
            self._prefill_rows(ids, pad_mask, embed, int(micro_batch))
            return
        self.bb(inputs_embeds=embed(ids), attention_mask=self.full_mask[:, :Lmax],
                past_key_values=self.cache, use_cache=True)

    def _tmp_cache(self, batch, max_cache_len):
        from transformers.cache_utils import StaticCache
        for kw in (dict(config=self.bb.config, max_cache_len=max_cache_len, max_batch_size=batch,
                        device=self.dev, dtype=self.dtype),
                   dict(config=self.bb.config, max_cache_len=max_cache_len, max_batch_size=batch)):
            try:
                return StaticCache(**kw)
            except TypeError:
                continue
        raise RuntimeError("could not construct StaticCache with a batch dim")

    @torch.no_grad()
    def _prefill_rows(self, ids, pad_mask, embed, micro_batch):
        """Fill the batched StaticCache (fresh or just reset) with the prompt WITHOUT one forward over all B
        rows: identical rows (same ids + pad mask, e.g. every conversation shares one persona) are computed ONCE,
        and the distinct rows go through the backbone `micro_batch` at a time into a small temporary
        StaticCache whose K/V are then copied into their rows of the big cache (index_put_, in place ->
        tensor addresses kept, so a captured cudagraph stays valid). Peak prefill memory is bounded by
        `micro_batch` rows instead of B. Same math as the one-shot prefill: a row's K/V depend only on that
        row (bf16 batch-shape rounding aside; tests/test_fast_thinker_prefill.py)."""
        B, Lmax = ids.shape
        groups, seen = [], {}                                # [(representative row, [rows sharing it])]
        for b in range(B):
            k = (tuple(ids[b].tolist()), tuple(pad_mask[b].tolist()))
            if k in seen:
                groups[seen[k]][1].append(b)
            else:
                seen[k] = len(groups)
                groups.append((b, [b]))
        for g0 in range(0, len(groups), micro_batch):
            gs = groups[g0:g0 + micro_batch]
            reps = torch.tensor([r for r, _ in gs], device=self.dev)
            tmp = self._tmp_cache(len(gs), Lmax)
            self.bb(inputs_embeds=embed(ids[reps]), attention_mask=self.full_mask[reps, :Lmax],
                    past_key_values=tmp, use_cache=True)
            if len(tmp.layers) != len(self.cache.layers):
                raise RuntimeError("micro-batched prefill: cache layer count mismatch")
            for big, small in zip(self.cache.layers, tmp.layers):
                if type(big).__name__ != "StaticLayer" or not hasattr(big, "cumulative_length"):
                    raise RuntimeError(f"micro-batched prefill needs StaticLayer caches, got {type(big).__name__}")
                if not big.is_initialized:                   # first prefill: allocate the (B, …) backing tensors
                    H, Dk, Dv = small.keys.shape[1], small.keys.shape[-1], small.values.shape[-1]
                    big.lazy_initialization(small.keys.new_empty((B, H, 0, Dk)), small.values.new_empty((B, H, 0, Dv)))
                for j, (_, rows) in enumerate(gs):
                    idx = torch.tensor(rows, device=self.dev)
                    big.keys[idx, :, :Lmax] = small.keys[j:j + 1, :, :Lmax].expand(len(rows), -1, -1, -1)
                    big.values[idx, :, :Lmax] = small.values[j:j + 1, :, :Lmax].expand(len(rows), -1, -1, -1)
            del tmp
        for big in self.cache.layers:
            big.cumulative_length.fill_(Lmax)                # in place (static address)

    def compile(self):
        if self.cbb is None:
            self.cbb = torch.compile(self.bb, mode=self.compile_mode)

    @torch.no_grad()
    def reprefill(self, ids, pad_mask, embed, micro_batch=None):
        """Start a fresh context on the SAME cache + compiled wrapper (server re-prime):
        StaticCache.reset() zeroes keys/values and the cumulative length IN PLACE (addresses kept, so the
        captured cudagraph stays valid), the fixed mask is rewritten in place, then an eager prefill.
        Also purges warmup()'s random frames. Same batch size only. micro_batch: as in prefill()."""
        if self.cache is None or self.full_mask is None:
            raise RuntimeError("reprefill() needs a prior prefill()")
        if ids.shape[0] != self.B:
            raise ValueError(f"reprefill batch {ids.shape[0]} != compiled batch {self.B}")
        Lmax = ids.shape[1]
        if Lmax >= self.max_cache_len:
            raise ValueError(f"prompt length {Lmax} >= max_cache_len {self.max_cache_len}")
        self.cache.reset()
        self.full_mask.fill_(1)
        self.full_mask[:, :Lmax] = pad_mask
        if micro_batch:
            self._prefill_rows(ids, pad_mask, embed, int(micro_batch))
            return
        self.bb(inputs_embeds=embed(ids), attention_mask=self.full_mask[:, :Lmax],
                past_key_values=self.cache, use_cache=True)

    @torch.no_grad()
    def step(self, e):
        """Per-frame compiled forward. e: (B,1,H). Fixed-width mask -> shape-stable cudagraph. -> (B,H)."""
        bb = self.cbb if self.cbb is not None else self.bb
        o = bb(inputs_embeds=e, attention_mask=self.full_mask,
               past_key_values=self.cache, use_cache=True)
        return o.last_hidden_state[:, -1, :]

    @torch.no_grad()
    def warmup(self, n: int = 20):
        V = self.bb.config.vocab_size
        for _ in range(n):
            self.step(self.bb.embed_tokens(torch.randint(0, V, (self.B, 1), device=self.dev)))
        torch.cuda.synchronize()
