"""CPU: framing and the room-tone floor. The floor draw is pinned by frozen digests (same seed, same floor)."""
import hashlib

import numpy as np
import pytest

from duplexaudio.serving.realtime.engine import session_seeds
from duplexaudio.serving.realtime.frontend import SPF, FrameAssembler, RoomToneFloor, floor_seed

BANDS = [[(-78.0, -63.0, 1.0)], [(-78.0, -63.0, 0.867), (-75.0, -45.0, 0.133)]]


def test_frame_assembler_chunking_is_lossless():
    x = np.random.default_rng(0).standard_normal(SPF * 7 + 333).astype(np.float32)
    whole = FrameAssembler().push(x)
    fa, chunked, i = FrameAssembler(), [], 0
    for n in [480] * 10 + [17, 1, 1919, 4000] + [480] * 40:
        chunked += fa.push(x[i:i + n])
        i += n
    assert len(whole) == len(chunked) == 7 and all(np.array_equal(a, b) for a, b in zip(whole, chunked))
    pcm = (np.clip(x[:SPF], -1, 1) * 32767).astype("<i2")
    assert np.array_equal(FrameAssembler().push_int16(pcm.tobytes())[0], pcm.astype(np.float32) / 32768.0)


def _take(floor, n_chunks=5):
    return np.concatenate([floor.take(SPF) for _ in range(n_chunks)] + [floor.take(777)])


FLOOR_DIGESTS = {   # frozen reference digests; the take pattern wraps the 60 s buffer
    (0, 0): ((-78.0, -63.0), -74.0, "7a8e8373e2aecab7258187405741db97"),
    (0, 1234): ((-78.0, -63.0), -72.3, "72a66c2ca6353114a8fa49b203fa1ba5"),
    (1, 0): ((-78.0, -63.0), -74.0, "7a8e8373e2aecab7258187405741db97"),
    (1, 1234): ((-75.0, -45.0), -63.6, "14350ecc590915301dfb427372e675ab"),
}


@pytest.mark.parametrize("key", sorted(FLOOR_DIGESTS))
def test_room_tone_floor_frozen_digest(key):
    bi, seed = key
    band, level, digest = FLOOR_DIGESTS[key]
    f = RoomToneFloor(BANDS[bi], seed)
    x = np.concatenate([_take(f), f.take(24000 * 60 - 5 * SPF - 777 + 5000)])    # crosses the 60 s buffer's end
    assert (f.band, f.level_db) == (band, level)
    assert hashlib.md5(x.tobytes()).hexdigest() == digest


def test_floor_off_is_zeros():
    f = RoomToneFloor([], 5)
    assert f.off and not _take(f).any() and f.describe() == {"off": True, "band": None, "level_db": None}


def test_session_seeds_are_distinct_streams():
    s = session_seeds("live-1", 0)
    assert len({s["thinker"], s["talker"], s["floor"]}) == 3 and s["floor"] == floor_seed("live-1", 0)
    assert session_seeds("live-2", 0)["thinker"] != s["thinker"] and session_seeds("live-1", 1)["talker"] != s["talker"]

