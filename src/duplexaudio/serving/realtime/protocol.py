"""Wire format between the realtime server and its clients (browser, headless client).

Upstream (client -> server):
  binary  raw little-endian int16 mono PCM at 24 kHz (any chunk size; the browser sends 20 ms)
  text    JSON control: {"type": "start", "seed"?: int} | {"type": "stop"} | {"type": "ping", "t": ...}

Downstream (server -> client):
  binary  one message per tick: LCA2 header + int16 PCM of agent frame `agent_frame` (1920 samples, or none for
          frame -1, the voice-prompt spacer, which is never played). Tick k consumed input frame k and emits agent
          frame k-1 (the talker's acoustic delay); flush frames after `stop` carry tick = the last tick.
  text    JSON: status / ready / text / metrics / warn / error / ended / pong

LCA2 header: magic, tick, agent frame, flags, tick time, queue depth, sample count, fed token id, kind code and a
per-session sequence number.
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = b"LCA2"
DOWN_HDR = struct.Struct("<4sIiIfHHiBxxxI")   # magic, tick, agent_frame, flags, tick_ms, queue_frames, n_samples,
                                               # fed, kind, pad, seq
FLAG_HAS_AUDIO = 1
FLAG_TALKER_SILENT = 2
FLAG_FLUSH = 4
KINDS = ("content", "pause", "ss", "se", "pad", "pend")
KIND_CODE = {k: i for i, k in enumerate(KINDS)}
KIND_OTHER = 255


def pack_down(*, tick: int, agent_frame: int, flags: int, tick_ms: float, queue_frames: int, fed: int, kind: str,
              seq: int, pcm_i16: np.ndarray | None) -> bytes:
    n = 0 if pcm_i16 is None else int(pcm_i16.size)
    hdr = DOWN_HDR.pack(MAGIC, int(tick), int(agent_frame), int(flags), float(tick_ms), min(int(queue_frames), 65535),
                        n, int(fed), KIND_CODE.get(kind, KIND_OTHER), int(seq))
    return hdr if pcm_i16 is None else hdr + np.asarray(pcm_i16, dtype="<i2").tobytes()


def unpack_down(buf: bytes):
    magic, tick, agent_frame, flags, tick_ms, queue_frames, n, fed, kind, seq = DOWN_HDR.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("bad frame magic")
    pcm = np.frombuffer(buf, dtype="<i2", count=n, offset=DOWN_HDR.size) if n else None
    return {"tick": tick, "agent_frame": agent_frame, "flags": flags, "tick_ms": tick_ms, "queue_frames": queue_frames,
            "n": n, "fed": fed, "kind": KINDS[kind] if kind < len(KINDS) else "other", "seq": seq}, pcm


class TextDelta:
    """The agent's text, streamed token by token without breaking multi-byte characters: within one agent turn the
    whole turn is re-decoded and only the new suffix is emitted; a decode ending in U+FFFD (a character split across
    tokens) is held back until it completes. `turn_end()` (on SS) closes the turn."""

    def __init__(self, tok):
        self.tok = tok
        self.ids: list[int] = []
        self.sent = ""

    def push(self, tid: int) -> str:
        self.ids.append(int(tid))
        text = self.tok.decode(self.ids)
        if text.endswith("�") or not text.startswith(self.sent):
            return ""
        delta, self.sent = text[len(self.sent):], text
        return delta

    def turn_end(self) -> str:
        text = self.tok.decode(self.ids) if self.ids else ""
        rest = text[len(self.sent):] if text.startswith(self.sent) else ""
        self.ids, self.sent = [], ""
        return rest.replace("�", "")
