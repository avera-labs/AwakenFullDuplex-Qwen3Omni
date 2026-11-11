"""Headless client for the realtime demo server: streams a wav over the WebSocket at real-time pace (20 ms chunks,
like the browser) and records what comes back. WS-path RTF and listenable recordings without a human at a mic.

    awaken-client --wav in.wav --out DIR [--url ws://127.0.0.1:8998/ws]
    # (= python -m duplexaudio.serving.realtime.client ...); --url defaults to awaken-server's default port, 8998

Writes DIR/{stereo.wav (L user as sent, R agent placed at agent_frame*1920), agent.wav, client.json}. client.json has
per-tick timings: the server's tick_ms and queue depth, and in->out = receipt of tick k - sending of input frame k's
last chunk (the server-side pipeline seen from the client, network included).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np

from duplexaudio.serving.realtime.protocol import FLAG_FLUSH, unpack_down

SR, SPF, CHUNK = 24000, 1920, 480
DEFAULT_URL = "ws://127.0.0.1:8998/ws"                   # awaken-server's default port


def load_wav_24k(path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = x.mean(axis=1) if x.ndim > 1 else x
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return np.asarray(x, dtype=np.float32)


async def stream(url: str, x: np.ndarray, *, realtime: bool = True, tail_sec: float = 3.0, seed=None,
                 ready_timeout: float = 600.0) -> dict:
    import websockets
    x = np.concatenate([x, np.zeros(int(tail_sec * SR), np.float32)])
    res = {"frames": {}, "ticks": [], "text": [], "events": [], "meta": None, "ended": None}
    async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
        deadline = time.time() + ready_timeout
        while True:                                       # wait until the engine is loaded
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - time.time())))
            if m.get("type") == "error":
                raise RuntimeError(m.get("msg"))
            if m.get("type") == "status" and m.get("state") == "ready":
                break
        await ws.send(json.dumps({"type": "start", **({"seed": int(seed)} if seed is not None else {})}))
        while True:
            m = json.loads(await ws.recv())
            if m.get("type") == "error":
                raise RuntimeError(m.get("msg"))
            if m.get("type") == "ready":
                res["ready"] = m
                break
        sent_t: dict[int, float] = {}

        async def receiver():
            while True:
                m = await ws.recv()
                now = time.perf_counter()
                if isinstance(m, bytes):
                    h, pcm = unpack_down(m)
                    if pcm is not None and h["agent_frame"] >= 0:
                        res["frames"][h["agent_frame"]] = pcm.copy()
                    if not h["flags"] & FLAG_FLUSH:
                        t = sent_t.get(h["tick"])
                        res["ticks"].append({"tick": h["tick"], "tick_ms": round(h["tick_ms"], 2),
                                             "queue": h["queue_frames"], "kind": h["kind"],
                                             "in_out_ms": None if t is None else round((now - t) * 1000.0, 2)})
                else:
                    j = json.loads(m)
                    if j["type"] == "text":
                        res["text"].append(j)
                    elif j["type"] == "ended":
                        res["meta"], res["ended"] = j.get("meta"), j
                        return
                    elif j["type"] in ("warn", "error", "event"):
                        res["events"].append(j)

        rx = asyncio.create_task(receiver())
        t0 = time.perf_counter()
        for i in range(0, len(x) - len(x) % CHUNK, CHUNK):
            if realtime:
                await asyncio.sleep(max(0.0, t0 + i / SR - time.perf_counter()))
            pcm = np.clip(np.round(x[i:i + CHUNK] * 32767.0), -32768, 32767).astype("<i2")
            await ws.send(pcm.tobytes())
            if (i + CHUNK) % SPF == 0:
                sent_t[(i + CHUNK) // SPF - 1] = time.perf_counter()
            if rx.done():
                break
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(rx, timeout=120)
    res["user"] = x
    return res


def write(res: dict, out: Path):
    import soundfile as sf
    out.mkdir(parents=True, exist_ok=True)
    n = (max(res["frames"]) + 1) if res["frames"] else 0
    agent = np.zeros(n * SPF, np.float32)
    for j, pcm in res["frames"].items():
        agent[j * SPF:(j + 1) * SPF] = pcm.astype(np.float32) / 32768.0
    user = res["user"]
    m = max(user.size, agent.size)
    st = np.zeros((m, 2), np.float32)
    st[:user.size, 0], st[:agent.size, 1] = user, agent
    sf.write(out / "stereo.wav", st, SR)
    sf.write(out / "agent.wav", agent, SR)
    io = [t["in_out_ms"] for t in res["ticks"] if t["in_out_ms"] is not None]
    tk = [t["tick_ms"] for t in res["ticks"]]
    summary = {"ticks": len(res["ticks"]), "agent_frames": n, "server_meta": res["meta"],
               "tick_ms_p50_p90_p99": [round(float(np.percentile(tk, q)), 2) for q in (50, 90, 99)] if tk else None,
               "in_out_ms_p50_p90_p99": [round(float(np.percentile(io, q)), 2) for q in (50, 90, 99)] if io else None,
               "max_queue": max((t["queue"] for t in res["ticks"]), default=0),
               "text": "".join(t.get("delta", "") for t in res["text"]), "events": res["events"]}
    json.dump({"summary": summary, "ticks": res["ticks"]}, open(out / "client.json", "w"), indent=1, default=str)
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Headless client for the realtime demo server: streams a wav at "
                                 "real-time pace and records what comes back.")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"the server's WebSocket (default {DEFAULT_URL})")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tail-sec", type=float, default=3.0, help="silence after the wav (the agent finishes speaking)")
    ap.add_argument("--fast", action="store_true", help="send as fast as possible instead of real time (not a live "
                    "measurement: the server queue fills up)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    res = asyncio.run(stream(a.url, load_wav_24k(a.wav), realtime=not a.fast, tail_sec=a.tail_sec, seed=a.seed))
    s = write(res, Path(a.out))
    print(json.dumps({k: s[k] for k in ("ticks", "agent_frames", "tick_ms_p50_p90_p99", "in_out_ms_p50_p90_p99",
                                        "max_queue")}), flush=True)


if __name__ == "__main__":
    main()
