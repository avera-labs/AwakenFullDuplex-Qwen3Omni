"""CPU: the room-tone floor generator is deterministic per seed and lands in its level range."""
import numpy as np

from duplexaudio.inference.room_tone import colored_noise


def test_deterministic_per_seed():
    a = colored_noise(24000, np.random.default_rng(7), -75.0, -45.0)
    b = colored_noise(24000, np.random.default_rng(7), -75.0, -45.0)
    assert np.array_equal(a, b) and a.dtype == np.float32


def test_level_within_range_and_never_silent():
    for seed in range(20):
        y = colored_noise(24000 * 2, np.random.default_rng(seed), -75.0, -45.0)
        db = 20 * np.log10(np.sqrt((y.astype(np.float64) ** 2).mean()))
        assert -75.5 <= db <= -44.5
        assert np.abs(y).max() > 0
