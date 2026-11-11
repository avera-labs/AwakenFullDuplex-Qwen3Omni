"""Realtime full-duplex demo server: one client and one conversation at a time.

The MIC SAMPLE CLOCK is the master clock: every complete 80 ms input frame is processed as soon as it arrives (no
wall-clock tick, nothing inserted, nothing dropped) and the agent frame it produces is sent immediately. Every CUDA
call runs on the engine's single GPU thread. The models are pinned by one profile (serving/realtime/profiles/).

    awaken-server --model ./pretrained_models/awaken-omni [--profile default] [--port 8998]
    # (= python -m duplexaudio.serving.realtime ...); --model: the downloaded model repo (checkpoints/ + data/) or its
    # Hugging Face repo id; without it, checkpoints/ and data/ under $DUPLEX_RUNTIME_ROOT, else the repo root (neither
    # there: a usage error); --port 0 = a random free port in 20500-20599
    # remote GPU box: ssh -L 8998:localhost:8998 <box>, then open http://localhost:8998
    # (browsers only give the mic to localhost or https pages)
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import json
import random
import socket
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket          # module level: FastAPI resolves the (postponed)
from fastapi.responses import FileResponse                     # annotations from this module's globals
from fastapi.staticfiles import StaticFiles

from duplexaudio.serving.realtime.frontend import SPF, SR, FrameAssembler
from duplexaudio.serving.realtime.inputcheck import InputMonitor
from duplexaudio.serving.realtime.protocol import FLAG_FLUSH, FLAG_HAS_AUDIO, FLAG_TALKER_SILENT, TextDelta, pack_down

STATIC = Path(__file__).resolve().parent / "static"
DOWNLOADS = ("stereo.wav", "user.wav", "agent.wav", "meta.json", "stream.json", "timing.json")
DEFAULT_PORT = 8998
MODEL_HINT = "pass --model ./pretrained_models/awaken-omni (or --model openmartdev/AwakenFullDuplex-Qwen3Omni)"
WARN_FRAMES_LEFT = 375                                   # 30 s
BACKLOG_WARN_FRAMES = 25                                 # 2 s behind real time: tell the client (once per episode)
BACKLOG_MAX_FRAMES = 125                                 # 10 s behind: not real time any more, end the conversation


class Rolling:
    def __init__(self, maxlen: int = 375):               # 30 s of ticks
        self.x = collections.deque(maxlen=maxlen)

    def add(self, v: float):
        self.x.append(float(v))

    def snapshot(self) -> dict:
        if not self.x:
            return {"n": 0}
        a = np.asarray(self.x)
        return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 1),
                "p90": round(float(np.percentile(a, 90)), 1), "p99": round(float(np.percentile(a, 99)), 1),
                "max": round(float(a.max()), 1)}


class Hub:
    """Process-wide state: the engine and its load status, the single connected client, the conversation."""

    def __init__(self, eng, *, idle_sec: float = 5.0, exit_after_idle_min: float = 0.0):
        self.eng = eng
        self.idle_sec, self.exit_after_idle_min = float(idle_sec), float(exit_after_idle_min)
        self.state, self.msg = "loading", "loading models"
        self.client = None
        self.conv = None
        self.recordings: dict[str, str] = {}
        self.last_activity = time.time()
        self.loop = None
        self.server = None

    async def gpu(self, fn, *a, **k):
        return await asyncio.wrap_future(self.eng.submit(fn, *a, **k))

    def status(self) -> dict:
        p = self.eng.prof
        return {"type": "status", "state": self.state, "msg": self.msg, "profile": p.name,
                "thinker": p.thinker.ckpt, "talker": p.talker.ckpt, "live": self.conv is not None}

    async def send_status(self):
        if self.client is not None:
            await _send_json(self.client, self.status())


async def _send_json(ws, obj) -> bool:
    try:
        await ws.send_text(json.dumps(obj, default=str))
        return True
    except Exception:
        return False                                    # the client went away: never fatal for the conversation


async def _send_bytes(ws, b: bytes) -> bool:
    try:
        await ws.send_bytes(b)
        return True
    except Exception:
        return False


class Conversation:
    """One conversation on one socket: frames in, ticks on the GPU thread, agent frames + text out."""

    def __init__(self, hub: Hub, ws, sess):
        self.hub, self.ws, self.sess = hub, ws, sess
        self.sid = sess.name
        self.queue: asyncio.Queue = asyncio.Queue()
        self.assembler = FrameAssembler()
        self.text = TextDelta(hub.eng.thinker.tok)
        self.tick_ms, self.in_out_ms = Rolling(), Rolling()
        self.seq = 0
        self.max_backlog, self.backlog_warned = 0, False
        self.monitor = InputMonitor()                  # echo / mic-background checks: measurement only
        self.monitor_error = None                      # why the checks turned themselves off, if they did
        self.checks: set = set()                       # running _check_input tasks (held so they are not collected)
        self.stop_reason = None
        self.last_audio = time.time()
        self.ended = asyncio.Event()
        self.worker = asyncio.create_task(self._tick_loop())
        self.metrics = asyncio.create_task(self._metrics_loop())

    def on_audio(self, data: bytes):
        if self.stop_reason is not None:
            return
        now = time.perf_counter()
        for f in self.assembler.push_int16(data):
            self.queue.put_nowait((f, now))
        self.last_audio = time.time()

    def _feed_monitor(self, out, frame):
        """Feed the input checks on the event loop. Never fatal: the first failure turns them off for this
        conversation (recorded in its meta)."""
        if self.monitor is None:
            return
        try:
            self.monitor.push(out.k, frame, out.agent_frame, out.pcm)
            if self.monitor.due():
                task = asyncio.create_task(self._check_input(self.monitor, self.monitor.snapshot()))
                self.checks.add(task)
                task.add_done_callback(self.checks.discard)
        except Exception as e:
            self.monitor, self.monitor_error = None, f"{type(e).__name__}: {e}"
            print(f"[realtime] {self.sid}: input checks off: {self.monitor_error}", flush=True)

    def _input_check_summary(self) -> dict:
        try:
            return self.monitor.summary() if self.monitor is not None else {"off": self.monitor_error}
        except Exception as e:
            return {"off": f"{type(e).__name__}: {e}"}

    async def _check_input(self, mon: InputMonitor, snap: dict):
        """Echo / mic-background analysis off the event loop; new warnings go to the client and the server log.
        Never fatal: a failed check is logged and the conversation goes on."""
        try:
            res = await asyncio.get_running_loop().run_in_executor(
                None, lambda: InputMonitor.analyze(snap, max_lag_s=mon.max_lag_s))
            for w in mon.update(res):
                print(f"[realtime] {self.sid}: input warning {w['kind']}: {w['measured']} (window {res})", flush=True)
                await _send_json(self.ws, {"type": "warn", "kind": w["kind"], "msg": w["msg"]})
        except Exception as e:
            print(f"[realtime] {self.sid}: input check failed: {type(e).__name__}: {e}", flush=True)

    async def _send_frame(self, out, flags_extra=0, tick=None):
        pcm = out.pcm if out.agent_frame >= 0 else None                    # frame -1 (spacer) is never played
        flags = (FLAG_HAS_AUDIO if pcm is not None else 0) | (FLAG_TALKER_SILENT if out.talker_silent else 0) | flags_extra
        b = pack_down(tick=out.k if tick is None else tick, agent_frame=out.agent_frame, flags=flags,
                      tick_ms=out.tick_ms, queue_frames=self.queue.qsize(), fed=out.fed, kind=out.kind, seq=self.seq,
                      pcm_i16=pcm)
        self.seq += 1
        await _send_bytes(self.ws, b)

    async def _tick_loop(self):
        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                frame, t_in = item
                out = await self.hub.gpu(self.sess.tick, frame)
                self.tick_ms.add(out.tick_ms)
                self._feed_monitor(out, frame)
                await self._send_frame(out)
                q = self.queue.qsize()
                self.max_backlog = max(self.max_backlog, q)
                if q >= BACKLOG_MAX_FRAMES:
                    await _send_json(self.ws, {"type": "error", "msg": f"{q * 0.08:.1f} s behind real time: ending"})
                    asyncio.create_task(self.end("backlog"))
                    break
                if q >= BACKLOG_WARN_FRAMES and not self.backlog_warned:
                    self.backlog_warned = True
                    await _send_json(self.ws, {"type": "warn", "msg": f"falling behind real time ({q * 0.08:.1f} s)"})
                elif q < BACKLOG_WARN_FRAMES // 2:
                    self.backlog_warned = False
                self.in_out_ms.add((time.perf_counter() - t_in) * 1000.0)
                if out.kind == "content":
                    delta = self.text.push(out.fed)
                    if delta:
                        await _send_json(self.ws, {"type": "text", "k": out.k, "delta": delta})
                elif out.kind == "ss":
                    rest = self.text.turn_end()
                    await _send_json(self.ws, {"type": "text", "k": out.k, "delta": rest, "turn_end": True})
                elif out.kind == "se":
                    await _send_json(self.ws, {"type": "event", "k": out.k, "kind": "se"})
                if out.frames_left == WARN_FRAMES_LEFT:
                    await _send_json(self.ws, {"type": "warn", "msg": "30 s left in this conversation"})
                if out.frames_left <= 0:
                    asyncio.create_task(self.end("cap"))
                    break
        except Exception as e:
            if self.stop_reason is None:
                traceback.print_exc()
                await _send_json(self.ws, {"type": "error", "msg": f"tick failed: {type(e).__name__}: {e}"})
                asyncio.create_task(self.end("error"))

    async def _metrics_loop(self):
        while self.stop_reason is None:
            await asyncio.sleep(1.0)
            if self.stop_reason is not None:
                break
            await _send_json(self.ws, {"type": "metrics", "tick_ms": self.tick_ms.snapshot(),
                                       "in_to_out_ms": self.in_out_ms.snapshot(), "backlog_frames": self.queue.qsize(),
                                       "max_backlog_frames": self.max_backlog,
                                       "frames": self.sess.k, "cap_frames": self.sess.cap_frames})
            if time.time() - self.last_audio > self.hub.idle_sec:
                asyncio.create_task(self.end("idle"))
                break

    async def end(self, reason: str):
        """Idempotent: stop taking audio, process what already arrived, flush the talker tail, finalize, report."""
        if self.stop_reason is not None:
            await self.ended.wait()
            return
        self.stop_reason = reason
        try:
            self.queue.put_nowait(None)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.worker, timeout=60)
            if self.sess.state == "open":
                with contextlib.suppress(Exception):
                    last = self.sess.k - 1
                    for j, pcm, sil in await self.hub.gpu(self.sess.flush):
                        b = pack_down(tick=last, agent_frame=j, flags=FLAG_HAS_AUDIO | FLAG_FLUSH |
                                      (FLAG_TALKER_SILENT if sil else 0), tick_ms=0.0, queue_frames=0, fed=-1,
                                      kind="pad", seq=self.seq, pcm_i16=pcm)
                        self.seq += 1
                        await _send_bytes(self.ws, b)
            self.sess.extra_meta["input_check"] = self._input_check_summary()
            meta = await self.hub.gpu(self.sess.finalize, reason)
            self.hub.recordings[self.sid] = meta.get("rec_dir")
            keep = ("sid", "reason", "frames", "seconds", "tick_ms", "tick_ms_ex_first5", "tick_ms_first5",
                    "component_ms", "component_ms_first5", "rtf_p90",
                    "validate_violations", "talker_input_gate", "floor", "seeds", "errors", "profile", "profile_md5",
                    "input_check")
            await _send_json(self.ws, {"type": "ended", "reason": reason, "sid": self.sid,
                                       "meta": {k: meta.get(k) for k in keep}, "max_backlog_frames": self.max_backlog,
                                       "downloads": [f"download/{self.sid}/{n}" for n in DOWNLOADS]})
        finally:
            self.metrics.cancel()
            self.hub.conv = None
            self.hub.last_activity = time.time()
            self.ended.set()


def build_app(hub: Hub, *, load: bool = True):
    app = FastAPI()
    app.state.hub = hub

    @app.on_event("startup")
    async def _startup():
        hub.loop = asyncio.get_running_loop()
        if load:
            fut = hub.eng.submit(hub.eng.load)

            def _done(f):
                err = f.exception()
                hub.state, hub.msg = ("error", f"load failed: {err}") if err else ("ready", "ready")
                if err:
                    traceback.print_exception(err)
                hub.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(hub.send_status()))
            fut.add_done_callback(_done)
        if hub.exit_after_idle_min > 0:
            asyncio.create_task(_idle_exit(hub))

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status():
        return hub.status()

    @app.get("/download/{sid}/{name}")
    async def download(sid: str, name: str):
        d = hub.recordings.get(sid)
        if name not in DOWNLOADS or d is None:
            raise HTTPException(404)
        p = Path(d) / name
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, filename=f"{sid}_{name}")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        if hub.client is not None:
            await _send_json(ws, {"type": "error", "msg": "another client is connected: one conversation at a time"})
            await ws.close(code=4409)
            return
        hub.client = ws
        hub.last_activity = time.time()
        try:
            await _send_json(ws, hub.status())
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if hub.conv is not None and hub.conv.ws is ws:
                        hub.conv.on_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    await _on_control(hub, ws, json.loads(msg["text"]))
        finally:
            if hub.conv is not None and hub.conv.ws is ws:
                await hub.conv.end("disconnect")
            hub.client = None
            hub.last_activity = time.time()

    return app


def client_audio(x, _depth: int = 0):
    """The browser's mic report (source, requested and applied getUserMedia settings) made safe for meta.json:
    nested dicts of plain scalars only, at most 40 keys per level, strings cut to 200 characters."""
    if isinstance(x, bool) or x is None or isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        return x[:200]
    if isinstance(x, dict) and _depth < 2:
        return {str(k)[:60]: client_audio(v, _depth + 1) for k, v in list(x.items())[:40]
                if isinstance(v, (dict, str, int, float, bool)) or v is None}
    return None


async def _on_control(hub: Hub, ws, m: dict):
    t = m.get("type")
    if t == "ping":
        await _send_json(ws, {"type": "pong", "t": m.get("t"), "server_t": time.time()})
    elif t == "start":
        if hub.conv is not None:
            await _send_json(ws, {"type": "error", "msg": "a conversation is already running"})
            return
        if hub.state != "ready":
            await _send_json(ws, {"type": "error", "msg": f"engine not ready ({hub.state}: {hub.msg})"})
            return
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        seed = m.get("seed")
        hub.state = "arming"
        await hub.send_status()
        try:
            sess = await hub.gpu(hub.eng.new_session, sid, None if seed is None else int(seed))
        except Exception as e:
            hub.state, hub.msg = ("ready", "ready") if hub.eng.ready else ("error", f"engine out of service: {e}")
            print(f"[realtime] start failed: {type(e).__name__}: {e} (engine ready={hub.eng.ready})", flush=True)
            await _send_json(ws, {"type": "error", "msg": f"start failed: {type(e).__name__}: {e}"})
            await hub.send_status()
            return
        hub.state = "ready"
        sess.extra_meta["client_audio"] = client_audio(m.get("audio"))     # the mic constraints the browser applied
        hub.conv = Conversation(hub, ws, sess)
        p = hub.eng.prof
        await _send_json(ws, {"type": "ready", "sid": sid, "sr": SR, "frame_samples": SPF, "profile": p.name,
                              "cap_frames": sess.cap_frames, "floor": sess.floor.describe(),
                              "model_lag_ms": 80 * int(p.session.end_flush_frames), "seeds": sess.seeds})
    elif t == "stop":
        if hub.conv is not None and hub.conv.ws is ws:
            await hub.conv.end("stopped")
    else:
        await _send_json(ws, {"type": "error", "msg": f"unknown message type {t!r}"})


async def _idle_exit(hub: Hub):
    while True:
        await asyncio.sleep(30)
        idle_min = (time.time() - hub.last_activity) / 60.0
        if hub.conv is None and hub.client is None and idle_min >= hub.exit_after_idle_min and hub.server is not None:
            print(f"[realtime] idle for {idle_min:.1f} min: exiting to free the GPU", flush=True)
            hub.server.should_exit = True
            return


def pick_port(lo: int = 20500, hi: int = 20599) -> int:
    cands = list(range(lo, hi + 1))
    random.shuffle(cands)
    for p in cands:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {lo}-{hi}")


def add_model_args(ap: argparse.ArgumentParser):
    """--model and --profile, shared by awaken-server and awaken-offline (resolved by use_model)."""
    ap.add_argument("--model", default=None, metavar="DIR_OR_REPO_ID", help="the model files: a folder holding "
                    "checkpoints/ and data/ (the downloaded model repo) or a Hugging Face repo id to fetch them from "
                    "(default: checkpoints/ and data/ under $DUPLEX_RUNTIME_ROOT, else the repo root); recordings are "
                    "never written there")
    ap.add_argument("--profile", default="default", help="profiles/<name>.yaml or a path (default: default)")


def use_model(ap: argparse.ArgumentParser, model):
    """Point the profile's checkpoints/ and data/ at --model, before the profile loads. Without --model the default
    model root is used, and a root with neither checkpoints/ nor data/ is a usage error (exit 2) instead of a server
    that starts and then fails its load. Returns the model folder."""
    from duplexaudio.common import paths
    if model is None:
        if not any(d.is_dir() for d in (paths.CHECKPOINTS_DIR, paths.DATA_DIR)):
            ap.error(f"no model files: neither checkpoints/ nor data/ under {paths.MODEL_ROOT}; {MODEL_HINT}")
        return paths.MODEL_ROOT
    try:
        root = paths.set_model_dir(model)
    except FileNotFoundError as e:
        ap.error(str(e))
    print(f"[realtime] model files from {root}", flush=True)
    return root


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_model_args(ap)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}; 0 = a random free port "
                    "in 20500-20599")
    ap.add_argument("--host", default="127.0.0.1", help="loopback only by default: there is NO auth and every "
                    "conversation (mic audio) is recorded; reach a remote box with `ssh -L`")
    ap.add_argument("--allow-non-loopback", action="store_true", help="required to bind anything but 127.0.0.1/::1")
    ap.add_argument("--idle-sec", type=float, default=5.0, help="end a conversation after this long without audio")
    ap.add_argument("--exit-after-idle-min", type=float, default=0.0, help="exit (free the GPU) after this many "
                    "minutes with no client; 0 = never")
    ap.add_argument("--rec-root", default=None, help="recordings root (default: logs/demo/sessions under "
                    "$DUPLEX_RUNTIME_ROOT, else the repo root)")
    return ap


def main(argv=None):
    import uvicorn

    from duplexaudio.serving.realtime.engine import RealtimeEngine
    from duplexaudio.serving.realtime.profile import load_profile
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.host not in ("127.0.0.1", "::1", "localhost") and not a.allow_non_loopback:
        ap.error(f"refusing to bind {a.host}: no auth + mic recording; use ssh -L, or pass --allow-non-loopback")
    use_model(ap, a.model)                               # before the profile: its data/ checkpoints/ resolve there
    prof = load_profile(a.profile)
    eng = RealtimeEngine(prof, rec_root=a.rec_root)
    hub = Hub(eng, idle_sec=a.idle_sec, exit_after_idle_min=a.exit_after_idle_min)
    port = a.port or pick_port()
    server = uvicorn.Server(uvicorn.Config(build_app(hub), host=a.host, port=port, ws_ping_interval=20,
                                           ws_ping_timeout=20, log_level="info"))
    hub.server = server
    print(f"[realtime] profile {prof.name} on http://{a.host}:{port}/", flush=True)
    server.run()


if __name__ == "__main__":
    main()
