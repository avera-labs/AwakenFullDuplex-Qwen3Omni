"""OmniThinkerEngine: the thinker's inference engine (frame-by-frame, KV-cached, turn-taking state machine).

Per 80 ms frame the thinker reads the user's (8,) Mimi codes fused with the agent token it emitted last frame and
emits the next agent token. The turn-taking state machine (`state_machine.EventFeed` + `apply_state_machine`) masks
every sample to the legal moves, so callers get LEGAL fed tokens + kinds from `step_masked()` and never re-implement
the grammar. `step()` is the raw primitive (content logits + hidden state, no sampling or mask).

    eng = OmniThinkerEngine.load(CKPT)                     # checkpoints/thinker/model.pt: LoRA + adapters
    eng.prime([persona_0, persona_1, ...])                 # variable-length prompts -> left-padded batch + SM
    for f in range(nframes):
        out = eng.step_masked(user_codes_B8, in_user=flags_B, temperature=0.3)
        # out.fed (B,) legal token / out.kinds / out.sampled / out.silent
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from duplexaudio.inference.thinker.state_machine import EventFeed, apply_state_machine

FRAME_HZ = 12.5
MIMI_SR = 24000


# ── contract types ────────────────────────────────────────────────────────────────────────────
@dataclass
class ThinkerSpec:
    pad_id: int
    epad_id: int
    sil_start_id: int
    sil_end_id: int
    pause_ids: tuple
    hidden: int
    vocab: int
    frame_hz: float = FRAME_HZ
    sr: int = MIMI_SR


@dataclass
class BatchedThinkerStepOut:
    content_logits: torch.Tensor         # (B, V)  raw agent_head(hidden)
    hidden: torch.Tensor                 # (B, H)


@dataclass
class MaskedStepOut:
    """One frame of state-machine-MASKED decode for a whole batch (B rows; B=1 for the single-stream
    engine). fed = the EventFeed-LEGAL token to embed next frame (already fed back internally)."""
    fed: torch.Tensor                    # (B,) long — legal token embedded next frame
    kinds: list                          # len B — content|pause|ss|se|se_premature|discard|fallback|pend
    sampled: torch.Tensor                # (B,) long — raw sampled token (== fed on pend / frame-0 rows)
    silent: torch.Tensor                 # (B,) bool — each row's EventFeed.silent (consumer bookkeeping)
    content_logits: torch.Tensor | None  # (B,V) only when return_logits=True (debug/research)


def _mb(micro_batch):
    """kwargs for BatchedCompiledThinker.prefill/reprefill: pass micro_batch only when set (the default call
    stays exactly the pre-existing one)."""
    return {"micro_batch": int(micro_batch)} if micro_batch else {}


def _as_row_list(x, B):
    """Normalize a per-row arg (None / scalar / tensor / list) to a length-B python list (or None)."""
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * B


# ── the state-machine-mask mixin (shared by single + batched engines) ───────────────────────────
class _StateMachineMixin:
    """Owns the per-row EventFeed grammar + `step_masked`. The engine implements `_sm_B()` (row count)
    and `_forward_logits(fed_list, user_codes) -> (B,V)`; everything else (sampling, mask, feedback) is
    here, ONE implementation, so single-stream and batched decode can never drift."""

    def _init_state_machine(self, *, events_in_input=True, free_turn=False,
                            start_silent=True, force_frame0_ss=True):
        """Called by prime() AFTER prefill: build one EventFeed per row + the frame/history/fed state.
        events_in_input/free_turn -> EventFeed flags. start_silent: the user speaks first.
        force_frame0_ss: emit SS at frame 0 without sampling (the thinker training default)."""
        B = self._sm_B()
        pad, epad = self.spec.pad_id, self.spec.epad_id
        SS = self.spec.sil_start_id
        PAUSE = {p: i for i, p in enumerate(self.spec.pause_ids)}
        self._feeds = [EventFeed(pad, epad, PAUSE, SS, self.spec.sil_end_id,
                                 events_in_input=events_in_input, free_turn=free_turn) for _ in range(B)]
        if start_silent:
            for fe in self._feeds:
                fe.silent = True
        self._force_frame0_ss = bool(force_frame0_ss)
        self._frame = 0
        self._hist = [[] for _ in range(B)]
        self._fed = [SS if force_frame0_ss else pad for _ in range(B)]   # frame-0 fed token

    @torch.no_grad()
    def step_masked(self, user_codes, *, in_user, bargein=None, temperature=0.0,
                    silence_temperature=None, speech_temperature=None,
                    ss_bias=None, se_bias=None, rep_control=None, return_logits=False,
                    generators=None) -> MaskedStepOut:
        """Forward (with the previous frame's LEGAL fed tokens) -> sample content head -> per-row
        EventFeed -> return the legal fed tokens. The consumer NEVER touches EventFeed.

        user_codes : (B,8) [(8,) accepted for B=1] mimi codes this frame.
        in_user    : (B,) bool (or scalar for B=1) — user still speaking (agent listening window)?
        bargein    : (B,) bool or None — may speak over the user? (None -> all False).
        temperature: 0 = argmax. Single-temperature fallback for both states.
        silence_temperature / speech_temperature : float or None — STATE-DEPENDENT temperature. A row in
            SILENCE (yielded; legal={SE,PAD}) samples with silence_temperature, in SPEECH (holding floor)
            with speech_temperature; each None -> `temperature` (default = old single-temp behaviour). Use
            HIGH silence + LOW speech to break out of stuck silence while keeping content stable.
        ss_bias/se_bias : (B,) or None — added to SS/SE logit before sampling (consumer gates the value).
        rep_control: dict or None — {rep_penalty, no_repeat_ngram, rep_window}.
        generators : (B,) torch.Generator or None — per-row sampling RNG (see apply_state_machine);
            None = the global torch RNG (default, unchanged behaviour).
        """
        if not hasattr(self, "_feeds"):
            raise RuntimeError("call prime() before step_masked()")
        cl = self._forward_logits(self._fed, user_codes)            # (B, V)
        B = cl.shape[0]
        SS = self.spec.sil_start_id
        if self._force_frame0_ss and self._frame == 0:
            fed, kinds, sampled = [SS] * B, ["ss"] * B, [SS] * B    # frame-0 seed, no sampling
        else:
            fed, kinds, sampled = apply_state_machine(
                content_logits=cl, feeds=self._feeds, hist=self._hist, spec=self.spec,
                in_user=_as_row_list(in_user, B), bargein=_as_row_list(bargein, B),
                temperature=temperature, silence_temperature=silence_temperature,
                speech_temperature=speech_temperature, ss_bias=_as_row_list(ss_bias, B),
                se_bias=_as_row_list(se_bias, B), rep_control=rep_control, generators=generators)
        self._fed = fed
        self._frame += 1
        return MaskedStepOut(
            fed=torch.tensor(fed, dtype=torch.long, device=self.dev),
            kinds=kinds,
            sampled=torch.tensor(sampled, dtype=torch.long, device=self.dev),
            silent=torch.tensor([fe.silent for fe in self._feeds], dtype=torch.bool),
            content_logits=cl if return_logits else None)


# ── the unified Omni thinker engine (B>=1; eager or compiled) ────────────────────────────────────
class OmniThinkerEngine(_StateMachineMixin):
    """Unified thinker engine: ONE class for real-time (B=1) and batched offline decode (B>1).
    Sole KV path chosen by `compile`: eager `BatchedDecoder` (DynamicCache, ragged) or
    `BatchedCompiledThinker` (batched StaticCache + reduce-overhead cudagraph). compile is default ON
    (about 2.4x faster than eager, argmax-identical modulo bf16). Ragged prompt lengths are handled by a fixed
    key-padding mask (RoPE relative invariance, no per-row position_ids). dense-MoE by default (shape stability).
    `step()` = RAW logits; `step_masked()` (mixin) = the state-machine decode.

        eng = OmniThinkerEngine.load(CKPT)
        eng.prime([persona_0, ...])                          # compile=True default; str ok (B=1)
        for f in range(nframes):
            out = eng.step_masked(user_codes_B8, in_user=flags_B, temperature=0.3)
    """

    def __init__(self, model, tok, raw_spec, *, device="cuda", compile_default=True):
        self.model = model
        self.tok = tok
        self.dev = torch.device(device)
        self._raw_spec = raw_spec
        self.spec = ThinkerSpec(
            pad_id=raw_spec.pad_id, epad_id=raw_spec.epad_id,
            sil_start_id=raw_spec.sil_start_id, sil_end_id=raw_spec.sil_end_id,
            pause_ids=tuple(raw_spec.pause_ids),
            hidden=model.backbone.config.hidden_size,
            vocab=model.agent_head.weight.shape[0])
        self._emb = model.backbone.embed_tokens
        self._dec = None                 # eager BatchedDecoder (compile=False)
        self._ct = None                  # BatchedCompiledThinker (compile=True)
        self.compiled = False
        self._compile_default = bool(compile_default)
        self.B = 0

    def _sm_B(self):
        return self.B

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_or_deltas, *, device="cuda", dense_moe=True, compile=True, lora_rank=None):
        """Load a trainer-line thinker checkpoint (dual-channel: LoRA + user-audio front end + fusion + the 17
        reserved-row embeddings/head rows) on top of the Qwen3-Omni base. `lora_rank` (opt-in): allocate the LoRA
        at this rank so later swap_deltas() can load any checkpoint of rank <= lora_rank in place (lower-rank
        checkpoints are zero-padded, exact)."""
        from duplexaudio.models.thinker.processing_thinker import load_spec
        tok, spec = load_spec()
        from duplexaudio.inference.thinker.trainer_line_loader import is_trainer_line, build_trainer_line_model
        if not is_trainer_line(ckpt_or_deltas):
            raise ValueError(f"{ckpt_or_deltas}: not a trainer-line thinker checkpoint (expected 'pin' + 'lora.*')")
        model = build_trainer_line_model(ckpt_or_deltas, device=device, lora_rank=lora_rank)
        if dense_moe:                                        # default: dense-MoE (fast small-T decode)
            from duplexaudio.models.thinker.dense_moe import install_dense_moe
            install_dense_moe(getattr(model, "backbone", model))
        return cls(model, tok, spec, device=device, compile_default=compile)

    # ---- lifecycle ----------------------------------------------------------------------------
    def reset(self):
        self._dec = None
        if self._ct is not None:
            try:
                self._ct.reset()
            except Exception:
                self._ct = None

    @torch.no_grad()
    def prime(self, prompts, *, compile=None, max_cache_len=2048, purge_warmup=False, reuse_compiled=False,
              prefill_micro_batch=None, **sm_kwargs):
        """Prefill prompt(s) + init the state machine. `prompts`: str (B=1) or list. `compile` None ->
        the load-time default (True). Compiled path builds+warms a cudagraph (~56s one-time); eager on
        compile=False (or auto-fallback if a compile prime raises).

        `purge_warmup` (opt-in): the compiled path's warmup() runs 20 random-token frames that ADVANCE the
        StaticCache, so by default they stay attended between the prompt and frame 0. True -> reset the cache
        after warmup and re-prefill the prompt (clean context, same cudagraph). The realtime engine always purges.
        `reuse_compiled` (opt-in, implies purge): if a compiled thinker with the same B and max_cache_len
        already exists, re-prefill it IN PLACE instead of rebuilding wrapper+cache+compile per prime — keeps
        a long-running server off dynamo's recompile limit (and makes re-priming ~1 s).
        `prefill_micro_batch` (opt-in, compiled path): fill the batched cache `m` distinct rows at a time and
        identical prompts once (BatchedCompiledThinker._prefill_rows) -> the prefill transient no longer
        scales with B. None = the one-shot B-row prefill (default, unchanged)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        id_lists = [self.tok.encode(p, add_special_tokens=True) if isinstance(p, str) else list(p)
                    for p in prompts]
        B = len(id_lists)
        want_compile = self._compile_default if compile is None else bool(compile)
        if want_compile:
            try:
                # ★ decide reuse WITHOUT holding a local reference to the old compiled thinker: a live ref
                # across _prime_compiled() keeps the previous StaticCache + compiled wrapper alive while the
                # new one prefills (the ~30 GiB/row batched_mm transient) -> OOM at the 2nd per-chunk prime.
                can_reuse = (reuse_compiled and self._ct is not None and self._ct.cbb is not None
                             and self._ct.B == B and self._ct.max_cache_len == int(max_cache_len))
                if can_reuse:
                    ids, pmask = self._padded_ids(id_lists)
                    self._ct.reprefill(ids, pmask, self._emb, **_mb(prefill_micro_batch))
                    self._dec = None
                else:
                    self._prime_compiled(id_lists, max_cache_len, purge_warmup=purge_warmup or reuse_compiled,
                                         micro_batch=prefill_micro_batch)
                self.compiled = True
            except Exception as e:                           # never silently slow: log + eager fallback
                print(f"[engine] compiled prime failed ({type(e).__name__}: {e}); eager fallback", flush=True)
                self._prime_eager(id_lists)
                self.compiled = False
        else:
            self._prime_eager(id_lists)
            self.compiled = False
        self.B = B
        self._init_state_machine(**sm_kwargs)

    def _prime_eager(self, id_lists):
        from duplexaudio.inference.decode import BatchedDecoder
        self._ct = None
        self._dec = BatchedDecoder(self.model.backbone, self.dev, self.spec.pad_id)
        self._dec.prefill(id_lists, self._emb)

    def _padded_ids(self, id_lists):
        B = len(id_lists)
        Lmax = max(len(x) for x in id_lists)
        ids = torch.full((B, Lmax), self.spec.pad_id, dtype=torch.long, device=self.dev)
        pmask = torch.zeros((B, Lmax), dtype=torch.long, device=self.dev)
        for b, x in enumerate(id_lists):                     # left-pad (BatchedDecoder convention)
            ids[b, Lmax - len(x):] = torch.tensor(x, dtype=torch.long, device=self.dev)
            pmask[b, Lmax - len(x):] = 1
        return ids, pmask

    def _prime_compiled(self, id_lists, max_cache_len, purge_warmup=False, micro_batch=None):
        from duplexaudio.inference.thinker.fast_thinker import BatchedCompiledThinker
        ids, pmask = self._padded_ids(id_lists)
        self._dec = None
        self._ct = None                                       # free the previous cache/wrapper BEFORE building anew
        self._ct = BatchedCompiledThinker(self.model, max_cache_len=max_cache_len)
        self._ct.prefill(ids, pmask, self._emb, **_mb(micro_batch))   # eager left-pad prefill (lazy-inits cache)
        self._ct.compile()
        self._ct.warmup()                                     # advances the cache by 20 random frames
        if purge_warmup:
            self._ct.reprefill(ids, pmask, self._emb, **_mb(micro_batch))   # reset in place + re-prefill -> clean context

    def cache_frames_used(self) -> int:
        """Tokens currently in the compiled thinker's StaticCache (prompt + frames). 0 if eager/unprimed."""
        if self._ct is None or self._ct.cache is None:
            return 0
        return int(self._ct.cache.get_seq_length())

    # ---- the step -----------------------------------------------------------------------------
    def _forward(self, fed_tokens, user_codes):
        if self._dec is None and self._ct is None:
            raise RuntimeError("call prime() before step()")
        fed = torch.as_tensor(fed_tokens, dtype=torch.long, device=self.dev).view(self.B)
        uc = torch.as_tensor(user_codes, dtype=torch.long, device=self.dev).view(self.B, 8)
        u_emb = self.model.codes_to_uemb(uc)                          # (B, H)
        e = self.model.fuse(u_emb.view(self.B, 1, -1), fed.view(self.B, 1))   # (B, 1, H)
        return self._ct.step(e) if self.compiled else self._dec.step(e)       # (B, H)

    @torch.no_grad()
    def step(self, fed_tokens, user_codes) -> BatchedThinkerStepOut:
        """RAW primitive: (B,) fed tokens + (B,8) codes -> {content_logits:(B,V), hidden:(B,H)}. No mask."""
        h = self._forward(fed_tokens, user_codes)
        return BatchedThinkerStepOut(content_logits=self.model.agent_head(h).float(), hidden=h)

    def _forward_logits(self, fed_list, user_codes):         # for step_masked
        h = self._forward(fed_list, user_codes)
        return self.model.agent_head(h).float()              # (B, V)

    # ---- hot delta swap -----------------------------------------------------------------------
    def swap_deltas(self, path):
        """Copy another trainer-line checkpoint's LoRA/proj/head/codebook deltas IN PLACE (addresses preserved ->
        the compiled cudagraph stays valid; validates before copying). The caller must reset()+prime() a fresh
        conversation afterwards. Returns {rank, alloc_rank, n_lora}."""
        from duplexaudio.inference.thinker.trainer_line_loader import load_trainer_line_into
        return load_trainer_line_into(self.model, path)
