"""The thinker side of the live loop: OmniThinkerEngine.step_masked driven exactly like the offline (batch) decode.

Offline reference: prime(persona string, events_in_input / force_frame0_ss / start_silent / free_turn, purge_warmup,
reuse_compiled, prefill_micro_batch 1), then one step_masked per frame with a PER-ROW torch.Generator seeded
row_sampling_seed(name, seed). The live loop keeps those arguments verbatim (se_bias as a one-element list, ss_bias
None unless nonzero), so the fed stream equals the offline B=1 stream for the same user codes.

The private generator also keeps the thinker off the device's default generator, which the talker owns.
"""
from __future__ import annotations

import zlib

import torch

PRIME_KW = dict(events_in_input=True, force_frame0_ss=True, start_silent=True, free_turn=True)


def row_sampling_seed(name: str, seed: int) -> int:
    """The per-conversation (per-row) sampling seed of the thinker."""
    return (zlib.crc32(f"sample:{name}".encode()) ^ int(seed)) & 0xFFFFFFFF


def read_persona(path) -> str:
    """The persona file VERBATIM: no strip, UTF-8 (the trailing '<|im_end|>\\n' is part of the trained prompt)."""
    with open(path, encoding="utf-8") as f:
        return f.read()


class ThinkerRunner:
    def __init__(self, tprof, *, device="cuda"):
        self.p = tprof
        self.dev = torch.device(device)
        self.eng = None
        self.persona = None
        self.prompt_len = None
        self._g = None
        self.frames = 0
        self._prev_silent = True
        self.arm_info = {}

    @property
    def spec(self):
        return self.eng.spec

    @property
    def tok(self):
        return self.eng.tok

    def load(self, ckpt_path):
        from duplexaudio.inference.thinker.engine import OmniThinkerEngine
        self.eng = OmniThinkerEngine.load(str(ckpt_path), device=str(self.dev), compile=True,
                                          lora_rank=self.p.lora_alloc_rank)
        return self.eng

    @torch.no_grad()
    def arm(self, persona_path):
        """Clean context for the next conversation: (re)prefill the verbatim persona on the reused compiled wrapper.
        The first call builds + warms the compiled thinker (purge_warmup re-prefills, so the 20 random warmup frames
        never stay in the cache); later calls re-prefill in place (~1 s). Refuses an eager engine and an unpurged cache."""
        self.persona = read_persona(persona_path)
        self.eng.prime([self.persona], **PRIME_KW, compile=True, max_cache_len=int(self.p.max_cache_len),
                       purge_warmup=True, reuse_compiled=True, prefill_micro_batch=self.p.prefill_micro_batch)
        if not self.eng.compiled:
            raise RuntimeError("thinker fell back to eager: refusing to serve a non-real-time, non-reference engine")
        self.prompt_len = len(self.eng.tok.encode(self.persona, add_special_tokens=True))
        used = self.eng.cache_frames_used()
        if used != self.prompt_len:
            raise RuntimeError(f"thinker cache holds {used} tokens after arm, expected the {self.prompt_len}-token "
                               "persona only (warmup frames not purged?)")
        self._g = None
        self.frames = 0
        self.arm_info = {"prompt_len": self.prompt_len, "cache_at_start": used, "purge_verified": True,
                         "max_cache_len": int(self.p.max_cache_len)}
        return self.arm_info

    def begin(self, name: str, seed: int) -> int:
        """Seed this conversation's private sampling generator (the per-row rule). Call right after arm()."""
        if self.prompt_len is None:
            raise RuntimeError("arm() before begin()")
        s = row_sampling_seed(name, seed)
        self._g = torch.Generator(device=self.dev).manual_seed(s)
        self.frames = 0
        self._prev_silent = True                         # the agent starts silent (start_silent)
        return s

    def step_kwargs(self, prev_silent: bool = True) -> dict:
        """step_masked kwargs except user_codes / generators, for a B=1 row with no VAD terms: the SE bias applies
        while the agent's last state is silent, the SS bias every frame and only if nonzero."""
        d = self.p.decode
        ss = float(d["ss_bias"])
        return dict(in_user=[False], temperature=float(d["temperature"]),
                    silence_temperature=float(d["silence_temperature"]),
                    speech_temperature=float(d["speech_temperature"]),
                    se_bias=[float(d["se_bias"]) if prev_silent else 0.0], ss_bias=[ss] if ss else None,
                    rep_control=dict(d["rep_control"]))

    @torch.no_grad()
    def step(self, codes8):
        """One frame: (8,) user codes -> (fed id, kind, sampled id). The fed id is the agent's frame token (frame 0 is
        the forced SS); it is fed back at the next frame by the engine and goes to the talker now."""
        if self._g is None:
            raise RuntimeError("begin() before step()")
        uc = torch.as_tensor(codes8, dtype=torch.long, device=self.dev).view(1, 8)
        out = self.eng.step_masked(uc, generators=[self._g], **self.step_kwargs(self._prev_silent))
        self._prev_silent = bool(out.silent[0])
        self.frames += 1
        return int(out.fed[0]), out.kinds[0], int(out.sampled[0])

    def frames_left(self) -> int:
        return int(self.p.max_cache_len) - int(self.prompt_len) - self.frames
