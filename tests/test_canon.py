"""CPU: the token-at-a-time talker ch0 feeder (StreamingCanon + Ch0Feeder) equals the batch builder
(inference.talker.inputs: canonicalize, then pause ids -> PAD) on random fed streams over a synthetic canon table."""
from types import SimpleNamespace

import numpy as np
import pytest

from duplexaudio.inference.talker.inputs import CANON_MODES, canonicalize, talker_inputs
from duplexaudio.serving.realtime.canon import Ch0Feeder, StreamingCanon

SPEC = SimpleNamespace(pad_id=3, epad_id=4, sil_start_id=5, sil_end_id=6, pause_ids=(90, 91, 92))
TABLE = {"c1": {20: ("curly", 21), 22: ("dash", 23), 24: ("PAD", 3)},
         "space": {30: 31, 32: 33, 20: 34},
         "punct": {40, 41}}
SPECIALS = [3, 4, 5, 6]
CONTENT = [10, 11, 12, 20, 22, 24, 30, 32, 40, 41]


def _fed_stream(rng, n):
    """A fed stream never carries pause ids (the state machine expands them); mix specials and content."""
    return [int(rng.choice(SPECIALS)) if rng.random() < 0.35 else int(rng.choice(CONTENT)) for _ in range(n)]


@pytest.mark.parametrize("mode", CANON_MODES)
def test_streaming_canon_equals_batch_canonicalize(mode):
    rng = np.random.default_rng(0)
    for _ in range(50):
        ids = _fed_stream(rng, 60)
        want, _ = canonicalize(ids, TABLE, mode, SPEC)
        sc = StreamingCanon(TABLE, mode, SPEC)
        assert [sc.push(t) for t in ids] == want


def test_feeder_equals_the_builder_dialogue_ch0():
    rng = np.random.default_rng(1)
    cfg = {"ch0_encoding": "thinker_stream", "vp_frames": 0, "vp_silence_frames": 0}
    for _ in range(20):
        ids = _fed_stream(rng, 40)
        x = talker_inputs(ids, cfg, SPEC, vp_codes=None, vp_text=None, canon=TABLE, canon_mode="c2u")
        feeder = Ch0Feeder(SPEC, StreamingCanon(TABLE, "c2u", SPEC))
        assert [feeder(t) for t in ids] == x["ch0"][x["P"]:].tolist()


def test_feeder_refuses_pause_ids():
    feeder = Ch0Feeder(SPEC)
    with pytest.raises(ValueError):
        feeder(90)
