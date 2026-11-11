"""Room-tone noise floor laid under the user channel, of the same kind the thinker was trained with."""
from __future__ import annotations

import numpy as np


def colored_noise(n, rng, lo_db, hi_db, sr=24000):
    """n samples of a low-level room-tone draw, perceptually diverse across draws.

    Each draw picks a random level in [lo_db, hi_db] dBFS (full-band RMS), a random 1/f^beta spectral slope (beta 0 =
    white ... 2.5 = brown / rumble) and, sometimes, a low-frequency mains-hum tone, so textures range from HVAC rumble
    to bright hiss. `rng` is a numpy Generator; the draw order is part of the contract (same seed, same floor)."""
    if n < 8:
        y = rng.standard_normal(n).astype(np.float32)
        level_db = float(rng.uniform(lo_db, hi_db))
        return y * 10.0 ** (level_db / 20.0)
    level_db = float(rng.uniform(lo_db, hi_db))
    level = 10.0 ** (level_db / 20.0)                           # random RMS level (dBFS)
    beta = float(rng.uniform(0.0, 2.5))                         # spectral slope: white -> pink -> brown
    L = min(n, sr * 4)                                          # colour a <= 4 s texture, then tile (room tone is
    W = np.fft.rfft(rng.standard_normal(L))                    # stationary: tiling is imperceptible, keeps FFT cheap)
    freqs = np.fft.rfftfreq(L, 1.0 / sr).copy()
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    shaped = W / (freqs ** (beta / 2.0))
    tex = np.fft.irfft(shaped, L).astype(np.float32)
    y = np.tile(tex, int(np.ceil(n / L)))[:n]
    y = y / (np.sqrt((y ** 2).mean()) + 1e-9)                  # unit RMS
    if rng.random() < 0.3:                                      # sometimes add mains hum
        f0 = float(rng.choice([50.0, 60.0, 100.0, 120.0]))
        t = np.arange(n, dtype=np.float32) / sr
        hum = np.sin(2 * np.pi * f0 * t).astype(np.float32) * float(rng.uniform(0.15, 0.6))
        y = y + hum
        y = y / (np.sqrt((y ** 2).mean()) + 1e-9)
    return (y * level).astype(np.float32)
