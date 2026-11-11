"""Thinker text format: the Qwen3-Omni thinker tokenizer plus the duplex special tokens on a 12.5 Hz frame grid.

* 12.5 Hz = Mimi's native frame rate: the user channel is one embedding per Mimi frame, no resampling.
* pause_max = 12 (pause_0 .. pause_12, no long bucket) plus the SS/SE silence events: 17 added tokens, which fit in
  the embedding table's unused rows (no resize).
"""

from __future__ import annotations

from duplexaudio.models.thinker.tokens import DuplexSpecialTokens, FormatSpec, added_special_tokens

FRAME_RATE_HZ = 12.5
OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MIMI_HZ = 12.5

PAUSE_MAX = 12
OMNILINE_FORMAT = FormatSpec(pause_token=True, pause_max=PAUSE_MAX, pause_long=False, silence_events=True)


def load_omni_tokenizer(fmt: FormatSpec = OMNILINE_FORMAT):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(OMNI_MODEL_ID)
    tok.add_special_tokens({"additional_special_tokens": added_special_tokens(fmt)})
    spec = DuplexSpecialTokens.from_tokenizer(tok, fmt)
    return tok, spec


def load_spec():
    """Tokenizer + resolved special-token ids of the shipped thinker (asserts the SS/SE and pause tokens exist)."""
    tok, spec = load_omni_tokenizer(OMNILINE_FORMAT)
    assert spec.sil_start_id and spec.sil_end_id, "SS/SE not registered"
    assert spec.pause_ids and len(spec.pause_ids) == PAUSE_MAX + 1
    return tok, spec
