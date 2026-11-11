"""User-audio front end: 80 ms framing and the training-style room-tone floor.

The user channel is ALWAYS the real, continuous mic (or wav) audio, agent speaking or not: no VAD, no muting, no
digital-silence substitution. A room-tone floor at the checkpoint's evaluated range is laid under it, so quiet frames
look like the user channel the thinker was trained and measured on. The floor draw is deterministic per conversation
seed (tests pin it).
"""
from __future__ import annotations

import numpy as np

SR = 24000
SPF = 1920                       # samples per 80 ms frame @ 24 kHz (Mimi 12.5 Hz)


class FrameAssembler:
    """Collects PCM chunks of any size and emits complete 1920-sample float32 frames, in order (no drop, no insert)."""

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self.samples_in = 0

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self.samples_in = 0

    def push_int16(self, data: bytes) -> list:
        return self.push(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0)

    def push(self, x: np.ndarray) -> list:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        self.samples_in += x.size
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x.copy()
        n = self._buf.size // SPF
        frames = [self._buf[i * SPF:(i + 1) * SPF].copy() for i in range(n)]
        self._buf = self._buf[n * SPF:]
        return frames


class RoomToneFloor:
    """One training-style room-tone draw per conversation (duplexaudio.inference.room_tone.colored_noise): a band is chosen
    by weight from `bands` [(lo_db, hi_db, weight), ...] (the choice is drawn even for one band), then a `seconds`-long
    texture is read cyclically. Empty `bands` = off (zeros)."""

    def __init__(self, bands, seed: int, sr: int = SR, seconds: int = 60):
        self.off = not bands
        self.band, self.level_db, self._pos = None, None, 0
        if self.off:
            self._buf = np.zeros(SPF, dtype=np.float32)
            return
        from duplexaudio.inference.room_tone import colored_noise
        rng = np.random.default_rng(int(seed))
        w = np.array([b[2] for b in bands], dtype=float)
        lo, hi, _ = bands[int(rng.choice(len(bands), p=w / w.sum()))]
        self.band = (float(lo), float(hi))
        self._buf = colored_noise(sr * seconds, rng, lo, hi, sr=sr).astype(np.float32)
        self.level_db = round(20 * float(np.log10(np.sqrt((self._buf ** 2).mean()) + 1e-12)), 1)

    def take(self, n: int) -> np.ndarray:
        if self.off:
            return np.zeros(n, dtype=np.float32)
        idx = (self._pos + np.arange(n)) % self._buf.size
        self._pos = int((self._pos + n) % self._buf.size)
        return self._buf[idx]

    def describe(self) -> dict:
        return {"off": self.off, "band": self.band, "level_db": self.level_db}


def floor_seed(name: str, seed: int) -> int:
    """Per-conversation floor seed, a stream distinct from the thinker's sampling seed."""
    import zlib
    return (zlib.crc32(f"floor:{name}".encode()) ^ int(seed)) & 0x7FFFFFFF
