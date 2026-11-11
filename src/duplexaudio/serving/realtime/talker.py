"""B=1 streaming talker (voice prompt + train-layout prefix): one LMGenX step per live 80 ms
frame.

The class issues the same LMGenX calls in the same order as a whole-stream (batch) talker decode, one dialogue step
per live frame, so for the same fed stream and seed the codes are byte-identical to the batch decode. The batch
schedule at B=1, P = vp_frames + vp_silence_frames prefix steps, L = P + T + tail:

  LMGenX(legacy, temp, top_k, top_p, temp_text 0.7); set_force_buffers(1); torch.manual_seed(seed); streaming(1)
  s <  P : force ch0[s]; step(user SINE col, depformer_replace_tokens = train-layout col s)      -> 1 text draw
  s >= P : force ch0[min(s, L-1)]; cb_k forced to agent_prefix[k, s - delay_k] while s - delay_k < P; step(zeros)
           -> 1 text + 8 audio draws (forced codebooks are drawn, then replaced)
  step s returns undelayed frame s-1; frames with s-1 >= P are the dialogue codes (8, T + tail).

Live, tick k is step P + k and returns frame k-1 (frame -1 = the last spacer frame, silence in the train layout). The
prefix tensors come from talker_inputs / prefix_columns themselves, so the prefix is the reference one by construction.

RNG: the talker OWNS the device's default generator (moshi's sampler draws from it). Everything else in the
live process samples elsewhere (the thinker from a private torch.Generator). A tripwire compares the default
generator's state before each step with the state the previous step left; any foreign draw in between is recorded
(and raises when strict), because it would silently move every later talker sample.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

DEP_Q = 8


def default_generator(device) -> torch.Generator:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()]
    return torch.default_generator


def vp_prefix(cfg: dict, spec, *, vp_codes, vp_text) -> dict:
    """The voice-prompt prefix, built by the reference builder (talker_inputs): ch0 = vp
    transcript over the R vp frames (negative -> PAD) then PAD over the spacer; agent = vp codes then the
    spacer-marker codes; user = PersonaPlex sine."""
    from duplexaudio.inference.talker.inputs import talker_inputs
    if cfg.get("ch0_encoding") != "thinker_stream":
        raise ValueError("the streaming talker serves ch0_encoding='thinker_stream' talkers only")
    if not cfg.get("zero_user_emb", False):
        raise ValueError("dialogue user codes are zeros: only zero_user_emb talkers are defined")
    x = talker_inputs([], cfg, spec, vp_codes=np.asarray(vp_codes), vp_text=vp_text, canon=None, tail=0)
    return {"ch0": x["ch0"], "agent_prefix": x["agent_prefix"], "user_prefix": x["user_prefix"], "P": int(x["P"]),
            "R": int(x["R"])}


class RngTripwireError(RuntimeError):
    pass


class StreamingTalker:
    """One B=1 LMGenX session per conversation. `session="persistent"` keeps one LMGenX + streaming context for the
    process and resets it per conversation (CUDA graphs captured once, at warm-up); "fresh" builds a new LMGenX per
    conversation, exactly as a batch decode does per call.

    The open streaming context holds the LM's streaming state: anything else that streams the same LM (e.g. a batch
    reference decode) must run after shutdown(), never interleaved."""

    def __init__(self, lm, prefix: dict, spec, *, sampler: dict, device="cuda", session="persistent", strict_rng=True):
        """`prefix` = vp_prefix(...) (talker_inputs' prefix tensors): {"ch0": (P,), "agent_prefix": (8, P),
        "user_prefix": (8, P), "P": P}."""
        from duplexaudio.inference.talker.inputs import prefix_columns
        if session not in ("persistent", "fresh"):
            raise ValueError(f"session {session!r}")
        if sampler.get("noise", "legacy") != "legacy":
            raise ValueError("only the legacy sampler (moshi sample_token on the default generator) is served")
        self.lm, self.spec = lm, spec
        self.dev = torch.device(device)
        self.session_mode, self.strict_rng = session, bool(strict_rng)
        self.sampler = {"temp": float(sampler["temp"]), "top_k": int(sampler["top_k"]), "top_p": float(sampler["top_p"])}
        self.pad = int(spec.pad_id)
        self.dq = int(lm.dep_q)
        if self.dq != DEP_Q:
            raise ValueError(f"dep_q {self.dq} != {DEP_Q}")
        self.agent_delays = [int(d) for d in list(lm.delays)[1:1 + self.dq]]
        self.P = int(prefix["P"])
        if self.P <= 0:
            raise ValueError("the streaming talker needs a voice-prompt prefix (vp_frames > 0)")
        self.ch0_prefix = [int(x) for x in prefix["ch0"][:self.P].tolist()]
        ap = prefix["agent_prefix"].long()
        self.agent_prefix = ap.to(self.dev)                                                          # (8, P)
        self.cols = prefix_columns(ap, self.agent_delays, "train").unsqueeze(0).to(self.dev)          # (1, 8, P)
        self.user_pre = prefix["user_prefix"].long().unsqueeze(0).to(self.dev)                        # (1, 8, P)
        self.user0 = torch.zeros(1, DEP_Q, 1, dtype=torch.long, device=self.dev)
        self.force_text = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.gen = None
        self._stack = None
        self.s = None
        self.sessions_started = 0
        self.gen_builds = 0                  # LMGenX objects built (each one captures its own CUDA graphs)
        self.forced_ch0 = []                 # the ch0 actually forced on each dialogue step of this conversation
        self._rng_after = None
        self.rng_violations = 0
        self.rng_first_violation = None

    # ── LMGenX ───────────────────────────────────────────────────────────────────────────────────────────────
    def _on_text(self, tok):
        tok.copy_(self.force_text)

    def _new_gen(self):
        from duplexaudio.inference.talker.crn_sampler import LMGenX
        gen = LMGenX(self.lm, noise="legacy", use_sampling=True, temp=self.sampler["temp"], top_k=self.sampler["top_k"],
                     top_p=self.sampler["top_p"], temp_text=0.7, on_text_hook=self._on_text)
        gen.set_force_buffers(1, self.dev)              # before streaming(): the depformer graph holds these buffers
        self.gen_builds += 1
        return gen

    def _close_stack(self):
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    # ── RNG tripwire ─────────────────────────────────────────────────────────────────────────────────────────
    def _check_rng(self):
        if self._rng_after is None:
            return
        now = default_generator(self.dev).get_state()
        if not torch.equal(now, self._rng_after):
            self.rng_violations += 1
            if self.rng_first_violation is None:
                self.rng_first_violation = self.s
            if self.strict_rng:
                raise RngTripwireError(f"default generator moved between talker steps (before step {self.s}): "
                                       "another component drew from the talker's RNG")

    def _mark_rng(self):
        self._rng_after = default_generator(self.dev).get_state()

    def rng_summary(self) -> dict:
        return {"violations": self.rng_violations, "first_step": self.rng_first_violation}

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────────────────
    def _open(self):
        self._close_stack()
        self.gen = self._new_gen()
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(self.gen.streaming(1))

    def _prefix_steps(self):
        for s in range(self.P):
            self.force_text.fill_(self.ch0_prefix[s])
            self.gen.force_mask.zero_()
            self.gen.step(self.user_pre[:, :, s:s + 1], depformer_replace_tokens=self.cols[:, :, s:s + 1])

    @torch.no_grad()
    def _precapture(self, dialogue_steps: int = 2):
        """Run one throwaway conversation on the open session (prefix + a few PAD dialogue steps) so moshi captures
        the forward_text and depformer CUDA graphs now, then reset in place. Draws from the default generator BEFORE
        the conversation's seed, so it never shifts the seeded stream."""
        self.s = 0
        self._prefix_steps()
        self.s = self.P
        self._rng_after = None
        for _ in range(int(dialogue_steps)):
            self.step(self.pad)
        self.gen.reset_streaming()
        self.s = None

    @torch.no_grad()
    def start(self, seed: int):
        """Session start (as in the batch decode): seed the default generator, (re)open the streaming session, run the P prefix
        steps. After this, each step() is one live frame. A persistent session that was shut down is rebuilt AND
        pre-captured here (never on a live tick)."""
        if self.session_mode == "fresh":
            self._open()                                 # as the batch decode: LMGenX + force buffers, seed, streaming
            torch.manual_seed(int(seed))
        else:
            if self.gen is None:
                self._open()
                self._precapture()
            self.gen.reset_streaming()                   # in place: offsets / KV ends zeroed, CUDA graphs kept
            torch.manual_seed(int(seed))
        self.rng_violations, self.rng_first_violation = 0, None
        self.forced_ch0 = []
        self.sessions_started += 1
        self._prefix_steps()
        self.s = self.P
        self._mark_rng()

    @torch.no_grad()
    def step(self, ch0_id: int):
        """One live frame with talker ch0 `ch0_id` (Ch0Feeder output). Returns (frame, codes): the undelayed (8,)
        codes of dialogue frame `frame` = k-1 for live tick k (frame -1 is the last spacer frame)."""
        if self.s is None:
            raise RuntimeError("call start() before step()")
        self._check_rng()
        s = self.s
        self.force_text.fill_(int(ch0_id))
        self.forced_ch0.append(int(ch0_id))
        self.gen.force_mask.zero_()
        for k, d in enumerate(self.agent_delays):        # delayed codebooks still carry prefix frames (train layout)
            if s - d < self.P:
                self.gen.force_mask[:, k] = True
                self.gen.force_vals[:, k] = self.agent_prefix[k, s - d]
        out = self.gen.step(self.user0)
        self._mark_rng()
        self.s = s + 1
        if out is None:                                  # only step 0 of a session returns None; s >= P >= 1 here
            raise RuntimeError("LMGenX returned no frame on a dialogue step")
        return s - 1 - self.P, out[0, 1:1 + self.dq, 0].clamp(0, 2047).clone()

    @torch.no_grad()
    def warmup(self, seed: int = 0, dialogue_steps: int = 2):
        """Before any live conversation: compile moshi's lazily compiled kernels in the talker-first order of the
        reference decode process and capture the session's CUDA graphs (forward_text at prefix step 1, the depformer at
        dialogue step 1). Persistent: the captured session is kept for every conversation. Fresh: the warm session
        is dropped; each conversation then pays its own capture on its first two ticks (documented cost)."""
        self.start(seed)
        for _ in range(int(dialogue_steps)):
            self.step(self.pad)
        self.end()
        if self.session_mode == "fresh":
            self.shutdown()

    def end(self):
        """Close the conversation (the persistent session stays open for the next start())."""
        self.s = None
        self._rng_after = None
        if self.session_mode == "fresh":
            self._close_stack()
            self.gen = None

    def shutdown(self):
        self.s = None
        self._close_stack()
        self.gen = None

    # ── parity driver ────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def run_frozen(self, ch0_ids, seed: int, tail: int = 25):
        """The batch decode of one frozen stream, streamed: start(seed), one step per ch0 id, `tail` PAD steps and the
        final re-feed step (the batch decode's s = L uses ch0[L-1]). Returns (8, T + tail) codes on CPU."""
        self.start(seed)
        seq = [int(x) for x in ch0_ids] + [self.pad] * int(tail)
        if not seq:
            raise ValueError("empty stream")
        frames = []
        for c in seq + [seq[-1]]:
            j, codes = self.step(c)
            if j >= 0:
                frames.append(codes)
        self.end()
        return torch.stack(frames, dim=1).cpu()
