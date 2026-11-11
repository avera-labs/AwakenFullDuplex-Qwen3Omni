"""Live mic checks for the demo server: MEASUREMENT ONLY. The user channel is never altered, gated or muted.

Two failure modes that only a live mic causes:
  echo        the agent's own voice comes back into the mic (e.g. laptop speakers without echo cancellation; one
              observed case: cross-correlation |r| 0.70 at 426 ms, gain -2.6 dB) and the thinker yields (SS) shortly
              after its echo arrives, so every sentence gets cut after about a second;
  background  a raw mic idling around -45 dBFS sits 30 dB above the training floor (-78..-63 dBFS). Mimi codes those
              frames as activity, so the thinker never takes the turn.
The server feeds every input frame and every agent frame here (O(1) appends on the event loop), runs `analyze` off
the event loop every `every_frames` frames, and sends the user one warning per problem; `summary` goes into the
session's meta.json (`input_check`). The background warning looks at the 5th percentile of every frame so far, from
10 s on: a percentile of a short window is the speech level whenever the user talks through it (see `update`).

Alignment: the mic frame k and the agent frame j are placed on one model-time axis (k * 80 ms, j * 80 ms), as in the
recordings; an echo then shows up as the mic lagging the agent by the browser playback path (~0.2-0.5 s).
"""
from __future__ import annotations

from collections import deque

import numpy as np

SR, SPF = 24000, 1920
AUDIBLE = 0.01                  # agent frame RMS that counts as speech


def dbfs(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-9))


class InputMonitor:
    def __init__(self, *, window_s: float = 6.0, every_frames: int = 25, max_lag_s: float = 1.0,
                 echo_r_min: float = 0.3, echo_gain_db_min: float = -20.0, echo_min_agent_s: float = 1.0,
                 bg_warn_dbfs: float = -55.0, bg_pct: float = 5.0, bg_min_frames: int = 125):
        self.n = int(round(window_s * SR / SPF))
        self.every, self.max_lag_s = int(every_frames), float(max_lag_s)
        self.echo_r_min, self.echo_gain_db_min, self.echo_min_agent_s = echo_r_min, echo_gain_db_min, echo_min_agent_s
        self.bg_warn_dbfs, self.bg_pct, self.bg_min_frames = bg_warn_dbfs, float(bg_pct), int(bg_min_frames)
        self.mic: deque = deque(maxlen=self.n)            # (k, frame float32 (1920,)) of the last window
        self.agent: dict = {}                              # j -> agent frame float32, last window only
        self.levels: list = []                             # every mic frame's dBFS (the whole conversation)
        self.k = -1
        self.echo = {"checks": 0, "max_abs_r": 0.0, "lag_ms": None, "gain_db": None, "warned": False}
        self.bg = {"warned": False, "warn_dbfs": None, "warn_at_s": None}
        self.warnings: list = []

    # ── feeding (event loop, O(1)) ──────────────────────────────────────────────────────────────────────
    def push(self, k: int, mic_frame: np.ndarray, agent_frame: int, agent_pcm_i16) -> None:
        f = np.asarray(mic_frame, np.float32).reshape(-1)
        self.k = int(k)
        self.mic.append((self.k, f))
        self.levels.append(dbfs(f))
        if agent_pcm_i16 is not None and agent_frame >= 0:
            self.agent[int(agent_frame)] = np.asarray(agent_pcm_i16, np.float32) / 32768.0
        lo = self.k - self.n - 16
        for j in [j for j in self.agent if j < lo]:
            del self.agent[j]

    def due(self) -> bool:
        return self.k >= 0 and (self.k + 1) % self.every == 0

    def snapshot(self) -> dict:
        """Copies for an off-loop `analyze`: the mic window and the agent frames on the same model-time axis."""
        if not self.mic:
            return {"mic": np.zeros(0, np.float32), "agent": np.zeros(0, np.float32), "levels": []}
        k0 = self.mic[0][0]
        mic = np.concatenate([f for _, f in self.mic])
        agent = np.zeros_like(mic)
        for j, a in self.agent.items():
            if k0 <= j < k0 + len(self.mic):
                agent[(j - k0) * SPF:(j - k0 + 1) * SPF] = a
        return {"mic": mic, "agent": agent, "levels": list(self.levels[-self.n:]), "frames_seen": len(self.levels)}

    # ── analysis (any thread) ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def analyze(snap: dict, *, max_lag_s: float = 1.0) -> dict:
        """Echo: normalized cross-correlation of the agent signal with the mic at lags 0..max_lag (mic lagging), at
        8 kHz; gain = least-squares scale of the lagged agent signal in the mic over the agent's audible samples.
        `bg_p10_dbfs`, the 10th percentile of the window's mic frame levels, is information only: it is the speech level
        whenever the user talks through most of the window. The background warning is decided in `update`."""
        from scipy.signal import fftconvolve, resample_poly
        mic, agent = snap["mic"], snap["agent"]
        lv = np.asarray(snap["levels"], float)
        out = {"bg_p10_dbfs": round(float(np.percentile(lv, 10)), 1) if lv.size else None,
               "frames_seen": snap.get("frames_seen", 0)}
        n_audible = int(sum(np.sqrt(np.mean(agent[i:i + SPF] ** 2)) > AUDIBLE for i in range(0, agent.size, SPF)))
        out["agent_audible_s"] = round(n_audible * SPF / SR, 2)
        if n_audible == 0 or mic.size < SR:
            return out
        m8, a8 = resample_poly(mic.astype(np.float64), 1, 3), resample_poly(agent.astype(np.float64), 1, 3)
        n = min(m8.size, a8.size)
        m8, a8 = m8[:n], a8[:n]
        c = fftconvolve(m8, a8[::-1], mode="full")[n - 1:n - 1 + int(max_lag_s * 8000)]
        norm = np.sqrt((m8 ** 2).sum() * (a8 ** 2).sum()) + 1e-12
        lag = int(np.argmax(np.abs(c)))
        r = float(c[lag] / norm)
        sh = lag * 3                                         # back to 24 kHz samples
        aa = np.zeros_like(mic, dtype=np.float64)
        aa[sh:] = agent[:mic.size - sh]
        act = np.abs(aa) > AUDIBLE
        g = float((mic[act] * aa[act]).sum() / ((aa[act] ** 2).sum() + 1e-12)) if act.any() else 0.0
        out.update({"echo_r": round(r, 3), "echo_lag_ms": round(lag / 8.0, 1),
                    "echo_gain_db": round(20.0 * np.log10(abs(g) + 1e-9), 1)})
        return out

    # ── decisions (event loop) ──────────────────────────────────────────────────────────────────────────
    def update(self, res: dict) -> list:
        """Fold one analysis in; return the NEW warnings ({kind, msg, measured}) to send (one per problem per
        conversation). Echo: from `res`. Background: the `bg_pct` (5th) percentile of every mic frame level so far,
        once there are `bg_min_frames` (10 s) of them; by then the user has paused or listened, so the low tail is the
        idle level. (A short-window percentile from 2 s on measures the speech level while the user talks and warns on
        quiet sessions too.)"""
        new = []
        if "echo_r" in res and res["agent_audible_s"] >= self.echo_min_agent_s:
            e = self.echo
            e["checks"] += 1
            if abs(res["echo_r"]) > e["max_abs_r"]:
                e.update(max_abs_r=abs(res["echo_r"]), lag_ms=res["echo_lag_ms"], gain_db=res["echo_gain_db"])
            if (not e["warned"] and abs(res["echo_r"]) >= self.echo_r_min
                    and res["echo_gain_db"] >= self.echo_gain_db_min):
                e["warned"] = True
                new.append({"kind": "echo", "msg": (
                    f"Echo detected: the agent's voice is coming back into the mic (delay about "
                    f"{res['echo_lag_ms']:.0f} ms, attenuated by only {-res['echo_gain_db']:.0f} dB), so the model "
                    f"takes it for you interrupting and stops. Use headphones, or choose the 'noise suppression + "
                    f"echo cancellation' input and start again."),
                    "measured": {k: res[k] for k in ("echo_r", "echo_lag_ms", "echo_gain_db", "agent_audible_s")}})
        if not self.bg["warned"] and len(self.levels) >= self.bg_min_frames:
            bg = round(float(np.percentile(self.levels, self.bg_pct)), 1)
            if bg > self.bg_warn_dbfs:
                self.bg.update(warned=True, warn_dbfs=bg, warn_at_s=round(len(self.levels) * SPF / SR, 2))
                new.append({"kind": "background", "msg": (
                    f"High mic background noise: about {bg:.0f} dBFS during pauses, while the model was trained on "
                    f"silence at -78..-63 dBFS. It will think you are still talking and not take its turn. Lower the "
                    f"system mic input gain, or use a headset mic close to your mouth."),
                    "measured": {"pct": self.bg_pct, "dbfs": bg, "frames": len(self.levels)}})
        self.warnings += new
        return new

    def summary(self) -> dict:
        lv = np.asarray(self.levels, float)
        return {"frames": int(lv.size),
                **{f"mic_dbfs_p{q}": round(float(np.percentile(lv, q)), 1) if lv.size else None for q in (5, 10, 50, 90)},
                "echo": dict(self.echo), "background": dict(self.bg),
                "warnings": [w["kind"] for w in self.warnings],
                "thresholds": {"echo_r_min": self.echo_r_min, "echo_gain_db_min": self.echo_gain_db_min,
                               "bg_warn_dbfs": self.bg_warn_dbfs, "bg_pct": self.bg_pct,
                               "bg_min_frames": self.bg_min_frames}}
