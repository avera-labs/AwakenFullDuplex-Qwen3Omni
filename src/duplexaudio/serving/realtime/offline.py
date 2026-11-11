"""awaken-offline: a recorded user track through the engine in one process, without the server.

    awaken-offline --model ./pretrained_models/awaken-omni --input user.wav --out out/ [--profile default] [--seed N]
                   [--tail-s 3.0]
    # (= python -m duplexaudio.serving.realtime.offline ...); --model as for awaken-server

The wav (any rate, mono or stereo: read and resampled to 24 kHz mono like awaken-client) is cut into 80 ms frames of
1920 samples (the last one zero-padded) and followed by --tail-s seconds of silence, so the agent can finish its turn.
Every frame is one tick of the same causal per-frame loop the live server runs, here as fast as the engine goes (so
the timings are not a live measurement). Then the talker's end flush, and the session is finalized into --out, a new
or empty directory that may not lie inside the model folder:

    stereo.wav       L the user as the model heard it (room-tone floor included), R the agent
    transcript.txt   the agent's text, one line per turn: "[start-end] text", in seconds of input (when the thinker
                     wrote it; the voice follows a few frames later)
    user.wav  agent.wav  codes.npz  stream.json  timing.json  meta.json      the session's own recordings
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from duplexaudio.serving.realtime.frontend import SPF, SR
from duplexaudio.serving.realtime.server import add_model_args, use_model

FRAME_S = SPF / SR                                         # 0.08 s per tick


def frames_of(x: np.ndarray, tail_s: float) -> np.ndarray:
    """(n, 1920) float32: the track zero-padded to whole frames, then ceil(tail_s / 0.08 s) frames of silence."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n_in = -(-x.size // SPF)
    n_tail = math.ceil(round(float(tail_s) / FRAME_S, 6))
    y = np.zeros((n_in + n_tail) * SPF, dtype=np.float32)
    y[:x.size] = x
    return y.reshape(-1, SPF)


def transcript(outs, tok) -> list[str]:
    """The agent's text from the ticks, one line per turn ("[start-end] text"): a turn is the content tokens up to the
    thinker's SS (start of silence); each turn is decoded whole, so a character split across tokens stays intact."""
    turns, cur = [], None
    for o in outs:
        if o.kind == "content":
            if cur is None:
                cur = [o.k, o.k, []]
                turns.append(cur)
            cur[1] = o.k
            cur[2].append(int(o.fed))
        elif o.kind == "ss":
            cur = None
    lines = []
    for k0, k1, ids in turns:
        text = tok.decode(ids).replace("�", "").strip()
        if text:
            lines.append(f"[{k0 * FRAME_S:.2f}-{(k1 + 1) * FRAME_S:.2f}] {text}")
    return lines


def run_offline(eng, x: np.ndarray, *, name: str, out, seed=None, tail_s: float = 3.0):
    """Tick every frame of `x` (24 kHz mono float32) plus the silent tail on a loaded engine, flush the talker, and
    finalize the session into `out` (its recordings) next to transcript.txt. Stops early at the session cap. A failed
    tick still finalizes (the recordings so far, the error in meta.json) and then raises. Returns (meta, lines)."""
    out = Path(out)
    frames = frames_of(x, tail_s)
    sess = eng.run(eng.new_session, name, seed, rec_dir=out)
    n_in = -(-int(np.asarray(x).size) // SPF)
    sess.extra_meta["offline"] = {"input_frames": n_in, "tail_frames": len(frames) - n_in, "tail_s": float(tail_s)}
    outs, reason = [], "eof"
    try:
        for f in frames:
            o = eng.run(sess.tick, f)
            outs.append(o)
            if o.frames_left <= 0 and len(outs) < len(frames):
                reason = "cap"
                print(f"[offline] session cap of {sess.cap_frames} frames ({sess.cap_frames * FRAME_S:.1f} s) "
                      f"reached: the last {len(frames) - len(outs)} frames were not fed", flush=True)
                break
        eng.run(sess.flush)
    except KeyboardInterrupt:
        reason = "interrupted"
        raise
    except Exception:
        reason = "error"
        raise
    finally:
        meta = eng.run(sess.finalize, reason)
        lines = transcript(outs, eng.thinker.tok)
        out.mkdir(parents=True, exist_ok=True)
        (out / "transcript.txt").write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
    return meta, lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a recorded user track through the model offline (no server) and "
                                 "write stereo.wav, transcript.txt and the session's recordings into --out.")
    add_model_args(ap)
    ap.add_argument("--input", required=True, help="the user's audio file (any rate, mono or stereo)")
    ap.add_argument("--out", required=True, help="output directory, new or empty (not inside the model folder)")
    ap.add_argument("--seed", type=int, default=None, help="base seed (default: the profile's session seed)")
    ap.add_argument("--tail-s", type=float, default=3.0, help="seconds of silence after the input (default 3.0)")
    return ap


def main(argv=None):
    from duplexaudio.common import paths
    from duplexaudio.serving.realtime.client import load_wav_24k
    from duplexaudio.serving.realtime.engine import RealtimeEngine
    from duplexaudio.serving.realtime.profile import load_profile
    ap = build_parser()
    a = ap.parse_args(argv)
    src = Path(a.input).expanduser()
    out = Path(a.out).expanduser().resolve()
    if not src.is_file():
        ap.error(f"--input {a.input}: no such file")
    if not math.isfinite(a.tail_s) or a.tail_s < 0:
        ap.error("--tail-s must be a finite number of seconds >= 0")
    if out.exists() and not (out.is_dir() and not any(out.iterdir())):
        ap.error(f"--out {a.out}: want a new or empty directory (recordings are never overwritten)")
    try:
        x = load_wav_24k(src)                            # before use_model: a repo id would download first
    except Exception as e:
        ap.error(f"--input {a.input}: not a readable audio file ({type(e).__name__}: {e})")
    root = use_model(ap, a.model)                        # before the profile: its data/ checkpoints/ resolve there
    model_dirs = (root,) if a.model is not None else (paths.CHECKPOINTS_DIR, paths.DATA_DIR)
    if any(out.is_relative_to(d) for d in model_dirs):
        ap.error(f"--out {a.out} is inside the model folder {root}: nothing is ever written there")
    prof = load_profile(a.profile)
    eng = RealtimeEngine(prof)
    try:
        print(f"[offline] {src.name}: {x.size / SR:.1f} s; loading profile {prof.name} (md5 pins, weights, compile "
              f"+ warm-up: a few minutes)", flush=True)
        eng.run(eng.load)
        meta, lines = run_offline(eng, x, name=src.stem, out=out, seed=a.seed, tail_s=a.tail_s)
    finally:
        eng.gpu.shutdown(wait=True)
    for ln in lines:
        print(ln)
    print(f"[offline] {meta.get('frames')} frames ({meta.get('seconds')} s, {meta.get('reason')}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
