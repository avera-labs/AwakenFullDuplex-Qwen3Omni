"""Duplex special tokens of the thinker's agent-text stream (12.5 Hz frame grid, one token per 80 ms frame).

Per frame the agent row carries one of:

  * PAD        silence / no new word token this frame;
  * EPAD       word-onset marker, written into the frame immediately before a word's first token;
  * a content token (the word's Qwen subwords, one per frame from the word's onset frame);
  * pause_m    a declared pause: m silent frames, then the next onset (pause_0 = the next word starts on the very next
               frame);
  * SS / SE    silence start / silence end events: the agent yields / takes the floor.

The tokens are appended to the Qwen3-Omni tokenizer in a fixed order; the order fixes their ids, so changing it
invalidates every checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

DUPLEX_PAD = "<|duplex_pad|>"
DUPLEX_EPAD = "<|duplex_epad|>"
DUPLEX_SILENCE = "<|duplex_silence|>"
DUPLEX_UT = "<|duplex_ut|>"
DUPLEX_AT = "<|duplex_at|>"
DUPLEX_UPAD = "<|duplex_user_pad|>"
DUPLEX_APAD = "<|duplex_agent_pad|>"
DUPLEX_PAUSE_LONG = "<|duplex_pause_long|>"
DUPLEX_SIL_START = "<|duplex_sil_start|>"      # SS
DUPLEX_SIL_END = "<|duplex_sil_end|>"          # SE

# Non-vocab sentinels (never embedded).
ZERO_TOKEN_ID = -1
UNGENERATED_TOKEN_ID = -2

PAUSE_MAX_DEFAULT = 20


def pause_token_names(pause_max: int, include_long: bool = True) -> list[str]:
    """`pause_0 .. pause_<pause_max>`, optionally followed by the 'longer than pause_max' bucket."""
    names = [f"<|duplex_pause_{n}|>" for n in range(0, pause_max + 1)]
    return names + [DUPLEX_PAUSE_LONG] if include_long else names


@dataclass(frozen=True)
class FormatSpec:
    """Which special tokens a vocabulary registers."""

    silence: bool = False
    utat: bool = False
    splitpad: bool = False
    pause_token: bool = False
    pause_max: int = PAUSE_MAX_DEFAULT
    pause_long: bool = True
    silence_events: bool = False


def added_special_tokens(fmt: FormatSpec) -> list[str]:
    """Special tokens to register on the Qwen tokenizer, in their load-bearing order."""
    toks = [DUPLEX_PAD, DUPLEX_EPAD]
    if fmt.silence:
        toks.append(DUPLEX_SILENCE)
    if fmt.utat:
        toks += [DUPLEX_UT, DUPLEX_AT]
    if fmt.splitpad:
        toks += [DUPLEX_UPAD, DUPLEX_APAD]
    if fmt.pause_token:
        toks += pause_token_names(fmt.pause_max, fmt.pause_long)
    if fmt.silence_events:
        toks += [DUPLEX_SIL_START, DUPLEX_SIL_END]
    return toks


@dataclass(frozen=True)
class DuplexSpecialTokens:
    """Resolved special-token ids for a loaded tokenizer (built once after `add_special_tokens`)."""

    pad_id: int
    epad_id: int
    silence_id: int | None = None
    ut_id: int | None = None
    at_id: int | None = None
    upad_id: int | None = None
    apad_id: int | None = None
    pause_ids: tuple[int, ...] | None = None     # [pause_0 .. pause_<pause_max>(, long)]
    zero_id: int = ZERO_TOKEN_ID
    ungenerated_id: int = UNGENERATED_TOKEN_ID
    pause_max: int = PAUSE_MAX_DEFAULT
    pause_long: bool = True                      # whether pause_ids ends with the '>pause_max' bucket
    sil_start_id: int | None = None
    sil_end_id: int | None = None

    def pause_id(self, n: int) -> int:
        """Token id for a gap of n frames (n > pause_max -> the long bucket, or pause_max without one)."""
        assert self.pause_ids is not None, "pause tokens are not registered (FormatSpec.pause_token)"
        want = self.pause_max + (2 if self.pause_long else 1)
        assert len(self.pause_ids) == want, (
            f"pause_ids has {len(self.pause_ids)} entries but pause_max={self.pause_max} "
            f"pause_long={self.pause_long} implies {want}: the tokenizer and the FormatSpec disagree")
        return self.pause_ids[min(n, self.pause_max + (1 if self.pause_long else 0))]

    @property
    def pause_set(self) -> set[int]:
        return set(self.pause_ids) if self.pause_ids else set()

    @classmethod
    def from_tokenizer(cls, tokenizer, fmt: FormatSpec) -> "DuplexSpecialTokens":
        pad = tokenizer.convert_tokens_to_ids(DUPLEX_PAD)
        epad = tokenizer.convert_tokens_to_ids(DUPLEX_EPAD)
        unk = getattr(tokenizer, "unk_token_id", None)
        if pad is None or epad is None or pad == unk or epad == unk:
            raise ValueError("duplex special tokens are not registered on the tokenizer "
                             "(add added_special_tokens(fmt) first)")

        def _opt(tokstr):
            i = tokenizer.convert_tokens_to_ids(tokstr)
            return int(i) if (i is not None and i != unk) else None

        pids = [_opt(t) for t in pause_token_names(fmt.pause_max, fmt.pause_long)]
        pause_ids = tuple(pids) if all(i is not None for i in pids) else None
        return cls(pad_id=int(pad), epad_id=int(epad), silence_id=_opt(DUPLEX_SILENCE),
                   ut_id=_opt(DUPLEX_UT), at_id=_opt(DUPLEX_AT),
                   upad_id=_opt(DUPLEX_UPAD), apad_id=_opt(DUPLEX_APAD),
                   pause_ids=pause_ids, pause_max=fmt.pause_max, pause_long=fmt.pause_long,
                   sil_start_id=_opt(DUPLEX_SIL_START), sil_end_id=_opt(DUPLEX_SIL_END))
