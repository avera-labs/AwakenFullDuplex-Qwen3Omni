"""Thinker fed token -> talker ch0 token, one frame at a time (the dialogue rule of the served convention).

The batch builder (inference.talker.inputs.talker_inputs) builds ch0 from a whole frozen stream. Its canonicalizer is
causal with O(1) state, so this token-at-a-time twin gives the same output (the engine re-checks it on every finished
session). The order is the builder's: canonicalize first, then pause ids -> PAD.
"""
from __future__ import annotations

CANON_MODES = ("c1", "c2", "c2u", "c1+c3", "c2+c3", "c2u+c3")


class StreamingCanon:
    """Token-at-a-time C1 / C2 / C2u (+C3) canonicalizer over a canon table (inference.talker.canon_table).

    State: an SE since the last content token (C2), any special since the last content token (C2u), whether the
    previous content token was mapped to PAD or ',' (C2u; C1 PAD/dash or C3's PAD), the previous ORIGINAL id (C3)."""

    def __init__(self, table, mode, spec):
        if mode not in CANON_MODES:
            raise ValueError(f"canon mode {mode!r} not in {CANON_MODES}")
        self.base, self.c3 = mode.split("+")[0], mode.endswith("+c3")
        self.pad, self.epad, self.se = int(spec.pad_id), int(spec.epad_id), int(spec.sil_end_id)
        self.specials = {self.pad, self.epad, int(spec.sil_start_id), self.se} | {int(p) for p in spec.pause_ids}
        self.c1, self.space, self.punct = table["c1"], table["space"], table["punct"]
        self.reset()

    def reset(self):
        self.seen_se = False
        self.special_since = False
        self.prev_pad_or_dash = False
        self.prev = None

    def push(self, t: int) -> int:
        t = int(t)
        if t in self.specials:
            self.special_since = True
            self.seen_se |= t == self.se
            self.prev = t
            return t
        trig = (self.seen_se if self.base == "c2" else
                (self.special_since or self.prev_pad_or_dash) if self.base == "c2u" else False)
        new, pad_or_dash = t, False
        if self.c3 and self.prev == self.epad and t in self.punct:
            new, pad_or_dash = self.pad, True
        elif t in self.c1:
            kind, new = self.c1[t]
            pad_or_dash = kind in ("PAD", "dash")
        elif trig and t in self.space:
            new = self.space[t]
        self.seen_se = self.special_since = False
        self.prev_pad_or_dash = pad_or_dash
        self.prev = t
        return int(new)


class Ch0Feeder:
    """fed id -> talker ch0 id: canon (optional) THEN pause -> PAD (the batch builder's dialogue rule).

    A fed stream never carries pause ids (the state machine feeds pause_0 as EPAD and pause_n as PAD); one showing up
    means sampled ids were fed by mistake, so it raises instead of silently voicing a different stream."""

    def __init__(self, spec, canon: StreamingCanon | None = None):
        self.pad = int(spec.pad_id)
        self.pauses = {int(p) for p in spec.pause_ids}
        self.canon = canon

    def reset(self):
        if self.canon is not None:
            self.canon.reset()

    def __call__(self, fed_id: int) -> int:
        t = int(fed_id)
        if t in self.pauses:
            raise ValueError(f"pause id {t} in a FED stream (sampled ids passed by mistake?)")
        if self.canon is not None:
            t = self.canon.push(t)
        return self.pad if t in self.pauses else t
