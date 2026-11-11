"""One live conversation: the per-frame duplex loop, clock-agnostic.

Per input frame k (80 ms):  user frame (+ room-tone floor) -> streaming Mimi encode -> thinker step (private RNG)
-> Ch0Feeder (C2u canon, pause -> PAD) -> talker step (default RNG) -> agent codes of frame k-1 -> streaming decode.
Every frame forwards both models, pauses included; the user channel is never muted; no frame is dropped or inserted.
In parity mode the caller injects (8,) user codes instead of audio (to compare fed streams at the codes level).

The same object serves the WebSocket server and offline use. Its methods touch CUDA and run
on the engine's single GPU thread (RealtimeEngine.run / submit). Only the engine's ACTIVE session may tick / flush /
finalize (the engine's models carry one conversation's state at a time). Lifecycle: open -> flushed -> finalized;
an exception inside a tick marks the session failed (it can still be finalized, never ticked again).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from duplexaudio.serving.realtime.frontend import SPF, SR

COMPONENTS = ("encode", "thinker", "talker", "decode")
OPEN, FLUSHED, FAILED, FINALIZED = "open", "flushed", "failed", "finalized"


class SessionCapReached(RuntimeError):
    pass


class SessionStateError(RuntimeError):
    pass


@dataclass
class TickOut:
    k: int                      # input frame consumed this tick
    fed: int                    # the thinker's frame token (fed back next frame; voiced by the talker)
    kind: str                   # content | pause | ss | se | pad | pend
    sampled: int
    text: str                   # decoded token for content frames, else ""
    agent_frame: int            # k - 1 (frame -1 = the last voice-prompt spacer frame; decoded, not played)
    pcm: np.ndarray             # int16 (1920,) of agent_frame
    talker_silent: bool         # agent codes == Mimi's silence codes
    tick_ms: float              # host wall time of the whole tick
    frames_left: int            # ticks left before the session cap


@dataclass
class _Rec:
    user: list = field(default_factory=list)          # float32 frames as the encoder heard them (floor included)
    user_codes: list = field(default_factory=list)    # (8,) per frame
    fed: list = field(default_factory=list)
    kinds: list = field(default_factory=list)
    sampled: list = field(default_factory=list)
    ch0: list = field(default_factory=list)           # what the feeder produced for the talker
    agent_pcm: dict = field(default_factory=dict)     # frame -> int16
    agent_codes: dict = field(default_factory=dict)   # frame -> (8,)
    tick_ms: list = field(default_factory=list)
    comp_ms: dict = field(default_factory=lambda: {c: [] for c in COMPONENTS})
    errors: list = field(default_factory=list)


class _Timer:
    """Per-component GPU timings with CUDA events, resolved one tick later (by then every event is complete, so
    reading them never blocks the live tick); host perf_counter on CPU."""

    def __init__(self, device, sink: dict):
        self.cuda = torch.device(device).type == "cuda"
        self.sink = sink
        self.pending = None

    def mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def push(self, marks):
        self.resolve()
        self.pending = marks

    def resolve(self):
        if self.pending is None:
            return
        m = self.pending
        self.pending = None
        if self.cuda:
            m[-1].synchronize()
        for c, a, b in zip(COMPONENTS, m[:-1], m[1:]):
            self.sink[c].append(float(a.elapsed_time(b)) if self.cuda else (b - a) * 1000.0)


def _pct(x) -> dict:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2), "p90": round(float(np.percentile(a, 90)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "max": round(float(a.max()), 2)}


def _rtf_p90(tick_ms) -> float | None:
    a = np.asarray(tick_ms, dtype=float)
    return round(float(np.percentile(a, 90)) / 80.0, 3) if a.size else None


class DuplexSession:
    def __init__(self, eng, *, name: str, seeds: dict, floor, rec_dir: Path | None, cap_frames: int):
        self.eng = eng
        self.name, self.seeds, self.floor = name, dict(seeds), floor
        self.rec_dir = Path(rec_dir) if rec_dir is not None else None
        if self.rec_dir is not None and self.rec_dir.exists() and any(self.rec_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite the non-empty recording dir {self.rec_dir}")
        self.cap_frames = int(cap_frames)
        self.k = 0
        self.rec = _Rec()
        self.timer = _Timer(eng.device, self.rec.comp_ms)
        self.state = OPEN
        self.ended_reason = None
        self.meta = None
        self.n_audio = self.n_codes = 0
        self.flushed_frames = 0
        self.t_start = time.time()
        self.extra_meta: dict = {}                     # added to meta.json by finalize (e.g. the server's input_check)
        self.sil = eng.silence_codes

    # ── guards ───────────────────────────────────────────────────────────────────────────────────────────────
    def _guard(self, *states):
        require = getattr(self.eng, "require_gpu_thread", None)
        if require is not None:
            require()
        if getattr(self.eng, "_active", self) is not self:
            raise SessionStateError(f"session {self.name!r} is not the engine's active session")
        if self.state not in states:
            raise SessionStateError(f"session {self.name!r} is {self.state}; this call needs {states}")

    # ── the tick ─────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def tick(self, frame: np.ndarray | None = None, *, codes=None) -> TickOut:
        """One 80 ms frame. Live: `frame` = float32 (1920,) user audio at 24 kHz (the floor is added here). Parity:
        `codes` = the (8,) user codes to inject (front end bypassed)."""
        self._guard(OPEN)
        if self.k >= self.cap_frames:
            raise SessionCapReached(f"session cap of {self.cap_frames} frames reached")
        if (frame is None) == (codes is None):
            raise ValueError("pass exactly one of frame / codes")
        if codes is None:
            heard = np.asarray(frame, dtype=np.float32).reshape(-1)
            if heard.size != SPF:
                raise ValueError(f"a tick takes one {SPF}-sample frame, got {heard.size}")
        e, r = self.eng, self.rec
        try:
            self.timer.resolve()                           # last tick's component times (complete by now)
            t0 = time.perf_counter()
            m0 = self.timer.mark()
            if codes is None:
                heard = (heard + self.floor.take(SPF)).astype(np.float32)
                c8 = e.encoder.encode(heard)
                r.user.append(heard)
                self.n_audio += 1
            else:
                c8 = torch.as_tensor(codes, dtype=torch.long, device=e.device).view(8)
                self.n_codes += 1
            r.user_codes.append(c8.tolist())
            m1 = self.timer.mark()
            fed, kind, sampled = e.thinker.step(c8)
            r.fed.append(fed); r.kinds.append(kind); r.sampled.append(sampled)
            m2 = self.timer.mark()
            ch0 = e.feeder(fed)
            r.ch0.append(ch0)
            j, agent = e.talker.step(ch0)
            m3 = self.timer.mark()
            pcm = e.decoder.decode(agent)                  # .cpu() inside: the tick's single device sync
            m4 = self.timer.mark()
        except Exception as ex:
            r.errors.append({"frame": self.k, "type": type(ex).__name__, "msg": str(ex)[:300]})
            self.state = FAILED
            raise
        ms = (time.perf_counter() - t0) * 1000.0
        agent_l = agent.tolist()
        r.agent_codes[j] = agent_l
        r.agent_pcm[j] = pcm
        r.tick_ms.append(ms)
        self.timer.push((m0, m1, m2, m3, m4))
        k = self.k
        self.k += 1
        text = e.thinker.tok.decode([fed]) if kind == "content" else ""
        return TickOut(k=k, fed=fed, kind=kind, sampled=sampled, text=text, agent_frame=j, pcm=pcm,
                       talker_silent=agent_l == self.sil, tick_ms=ms, frames_left=self.cap_frames - self.k)

    @torch.no_grad()
    def flush(self, n: int | None = None) -> list:
        """After the input stops (once): voice the last words the talker was given. The talker alone steps on PAD for
        `n` frames (default: the profile's end_flush_frames = text lead + 1); the thinker's context is closed."""
        self._guard(OPEN)
        e = self.eng
        n = e.prof.session.end_flush_frames if n is None else int(n)
        outs = []
        try:
            for _ in range(n):
                j, agent = e.talker.step(e.spec.pad_id)
                pcm = e.decoder.decode(agent)
                self.rec.agent_codes[j] = agent.tolist()
                self.rec.agent_pcm[j] = pcm
                self.flushed_frames += 1
                outs.append((j, pcm, agent.tolist() == self.sil))
        except Exception as ex:
            self.rec.errors.append({"frame": f"flush+{self.flushed_frames}", "type": type(ex).__name__,
                                    "msg": str(ex)[:300]})
            self.state = FAILED
            raise
        self.state = FLUSHED
        return outs

    # ── end of conversation ──────────────────────────────────────────────────────────────────────────────────
    def finalize(self, reason: str) -> dict:
        """Legality + talker-input gates on the recorded stream, recordings and the full protocol metadata.
        Idempotent. The recordings are written even if a gate raises (the error is recorded instead)."""
        if self.state == FINALIZED:
            return self.meta
        self._guard(OPEN, FLUSHED, FAILED)
        from duplexaudio.inference.thinker.state_machine import validate_stream
        e, r = self.eng, self.rec
        self.ended_reason = reason
        meta = {"sid": self.name, "reason": reason, "state_at_end": self.state, "frames": self.k,
                "seconds": round(self.k * SPF / SR, 2), "flushed_frames": self.flushed_frames,
                "input_frames": {"audio": self.n_audio, "codes": self.n_codes}, "errors": list(r.errors)}
        try:
            self.timer.resolve()
            e.talker.end()
            meta.update(e.session_provenance())
            fl = dict(self.floor.describe())
            fl["applied_frames"] = self.n_audio                        # codes-injected ticks never get the floor
            meta.update({
                "seeds": self.seeds, "floor": fl, "cap_frames": self.cap_frames,
                "rng": {"thinker": "private torch.Generator (row_sampling_seed)",
                        "talker": "device default generator (torch.manual_seed)",
                        "talker_tripwire": e.talker.rng_summary()},
                "arm": dict(e.thinker.arm_info),
                # flat keys read by offline drivers
                "purge_warmup": bool(e.thinker.arm_info.get("purge_verified")),
                "cache_at_start": e.thinker.arm_info.get("cache_at_start"),
                "external_vad": False, "vad_frames": 0,
                "tick_ms": _pct(r.tick_ms), "tick_ms_ex_first5": _pct(r.tick_ms[5:]),
                "component_ms": {c: _pct(v) for c, v in r.comp_ms.items()},
                # the first 5 ticks reported on their own (every tick is in timing.json)
                "tick_ms_first5": [round(x, 2) for x in r.tick_ms[:5]],
                "component_ms_first5": {c: [round(x, 2) for x in v[:5]] for c, v in r.comp_ms.items()},
                "rtf_p90": _rtf_p90(r.tick_ms[5:]),                     # ticks[5:]: the first ticks include warm-up
                "rtf_p90_all": _rtf_p90(r.tick_ms),
                "rec_dir": str(self.rec_dir) if self.rec_dir is not None else None,
                "wall_sec": round(time.time() - self.t_start, 2)})
            meta.update(self.extra_meta)
            viol = validate_stream(r.fed, e.spec, start_silent=False, require_frame0_ss=True) if r.fed else []
            meta["validate_violations"] = len(viol)
            meta["validate_first"] = [list(map(str, v)) for v in viol[:3]]
            meta["talker_input_gate"] = (e.talker_input_gate(self.name, r.fed, e.talker.forced_ch0) if r.fed
                                         else {"verdict": "EMPTY"})
        except Exception as ex:
            meta["gate_error"] = {"type": type(ex).__name__, "msg": str(ex)[:300]}
        finally:
            self.meta = meta
            self.state = FINALIZED
            if getattr(e, "_active", None) is self:
                e._active = None
            if self.rec_dir is not None:
                self._write(meta)
        return meta

    def _write(self, meta: dict):
        import soundfile as sf
        r = self.rec
        d = self.rec_dir
        d.mkdir(parents=True, exist_ok=True)
        user = np.concatenate(r.user) if r.user else np.zeros(0, np.float32)
        n_agent = (max(r.agent_pcm) + 1) if r.agent_pcm else 0
        agent = np.zeros(max(n_agent, 0) * SPF, dtype=np.float32)
        for j, a in r.agent_pcm.items():
            if j >= 0:
                agent[j * SPF:(j + 1) * SPF] = a.astype(np.float32) / 32768.0
        n = max(user.size, agent.size)
        st = np.zeros((n, 2), dtype=np.float32)
        st[:user.size, 0] = user
        st[:agent.size, 1] = agent
        sf.write(d / "user.wav", user, SR)
        sf.write(d / "agent.wav", agent, SR)
        sf.write(d / "stereo.wav", st, SR)
        frames = sorted(j for j in r.agent_codes if j >= 0)
        np.savez_compressed(d / "codes.npz",
                            user_codes=np.asarray(r.user_codes, dtype=np.int16).reshape(-1, 8),
                            agent_codes=np.asarray([r.agent_codes[j] for j in frames], dtype=np.int16).reshape(-1, 8),
                            agent_frames=np.asarray(frames, dtype=np.int32))
        json.dump({"fed": r.fed, "kinds": r.kinds, "sampled": r.sampled, "ch0": r.ch0},
                  open(d / "stream.json", "w"))
        json.dump({"tick_ms": [round(x, 3) for x in r.tick_ms],      # host wall time per tick
                   "component_ms": {c: [round(x, 3) for x in v] for c, v in r.comp_ms.items()}},   # CUDA events
                  open(d / "timing.json", "w"))
        json.dump(meta, open(d / "meta.json", "w"), indent=1, default=str)
