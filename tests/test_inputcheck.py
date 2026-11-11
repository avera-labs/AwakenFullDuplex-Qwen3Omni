"""CPU: the live mic checks (serving/realtime/inputcheck.py) measure echo and the mic background on synthetic signals
shaped like real failure cases (echo at 426 ms, -2.6 dB, inverted; a -45 dBFS idle mic; a user who talks through the
first seconds over a quiet floor). Measurement only: nothing here changes the user channel."""
import numpy as np

from duplexaudio.serving.realtime.inputcheck import SPF, SR, InputMonitor, dbfs


def _speechy(seconds, seed=0):
    """Agent-like signal: 0.6 s noise bursts (lightly smoothed) with 0.4 s gaps."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR)).astype(np.float32)
    x = np.convolve(x, [0.5, 0.5], mode="same").astype(np.float32)
    env = (np.arange(x.size) % SR) < int(0.6 * SR)
    return (x * env * 0.14).astype(np.float32)


def _noise(seconds, level_dbfs, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR)).astype(np.float32)
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (level_dbfs / 20)).astype(np.float32)


def _run(mon, mic, agent, results=None):
    """Feed like the server: tick k consumes mic frame k and emits agent frame k-1."""
    out = []
    for k in range(mic.size // SPF):
        j = k - 1
        pcm = None if j < 0 else np.clip(np.round(agent[j * SPF:(j + 1) * SPF] * 32767), -32768, 32767).astype(np.int16)
        mon.push(k, mic[k * SPF:(k + 1) * SPF], j, pcm)
        if mon.due():
            res = InputMonitor.analyze(mon.snapshot(), max_lag_s=mon.max_lag_s)
            if results is not None:
                results.append(res)
            out += mon.update(res)
    return out


def test_echo_is_detected_with_its_lag_and_gain():
    agent = _speechy(8.0)
    lag = int(0.426 * SR)
    mic = _noise(8.0, -45.0)
    mic[lag:] += -0.74 * agent[:-lag]                       # observed echo shape: inverted, -2.6 dB, 426 ms late
    mon = InputMonitor()
    warns = _run(mon, mic, agent)
    assert [w["kind"] for w in warns].count("echo") == 1      # one warning per conversation
    e = mon.summary()["echo"]
    assert e["warned"] and e["max_abs_r"] > 0.5 and abs(e["lag_ms"] - 426) <= 5 and abs(e["gain_db"] + 2.6) < 1.0


def test_independent_mic_is_not_echo():
    mon = InputMonitor()
    warns = _run(mon, _noise(8.0, -30.0, seed=7), _speechy(8.0))
    assert "echo" not in [w["kind"] for w in warns] and mon.summary()["echo"]["max_abs_r"] < 0.1


def test_no_echo_check_without_agent_speech():
    mon = InputMonitor()
    mic = _noise(4.0, -70.0)
    _run(mon, mic, np.zeros_like(mic))
    assert mon.summary()["echo"]["checks"] == 0


def test_mic_background_warning():
    """A -45 dBFS idle mic warns once, at 10.0 s (the first check with 125 frames), not before; -75 dBFS never does."""
    assert _run(InputMonitor(), _noise(9.9, -45.0), np.zeros(int(9.9 * SR), np.float32)) == []
    loud = InputMonitor()
    warns = _run(loud, _noise(12.0, -45.0), np.zeros(int(12.0 * SR), np.float32))
    bg = loud.summary()["background"]
    assert [w["kind"] for w in warns] == ["background"] and abs(bg["warn_dbfs"] + 45) < 1.5 and bg["warn_at_s"] == 10.0
    assert warns[0]["measured"]["frames"] == 125 and "-45" in warns[0]["msg"]
    quiet = InputMonitor()
    assert _run(quiet, _noise(12.0, -75.0), np.zeros(int(12.0 * SR), np.float32)) == []
    s = quiet.summary()
    assert s["frames"] == 150 and abs(s["mic_dbfs_p50"] + 75) < 1.5 and s["warnings"] == []


def test_talking_through_the_first_seconds_over_a_quiet_floor_does_not_warn():
    """A short-window rule false-alarms here (it warned at 2.0 s even on quiet sessions): the user talks through the
    first 6 s, listens for 3 s, talks again, over a -70 dBFS floor. The window p10 at 2 s
    is the speech level; the conversation's p5 from 10 s on is the floor."""
    mic = _noise(14.0, -70.0, seed=2)
    mic[: int(6.0 * SR)] += _noise(6.0, -25.0, seed=3)
    mic[int(9.0 * SR): int(13.0 * SR)] += _noise(4.0, -25.0, seed=4)
    mon, res = InputMonitor(), []
    warns = _run(mon, mic, np.zeros_like(mic), res)
    assert res[0]["frames_seen"] == 25 and res[0]["bg_p10_dbfs"] > -30          # what the first version judged
    s = mon.summary()
    assert warns == [] and not s["background"]["warned"] and s["mic_dbfs_p5"] < -65 and s["frames"] == 175


def test_agent_frames_outside_the_window_are_dropped():
    mon = InputMonitor(window_s=1.0)
    z = np.zeros(SPF, np.float32)
    for k in range(200):
        mon.push(k, z, k - 1, np.zeros(SPF, np.int16))
    assert len(mon.agent) <= mon.n + 17 and len(mon.mic) == mon.n and len(mon.levels) == 200
    assert dbfs(np.zeros(10)) < -170
