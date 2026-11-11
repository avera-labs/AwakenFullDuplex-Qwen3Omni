"""CPU: the realtime server's WebSocket protocol and conversation lifecycle against a fake engine (no models).

start -> ready -> mic chunks -> one LCA2 frame per tick (frame -1 carries no audio) + text deltas -> stop -> talker
flush frames -> ended; one client at a time; the idle watchdog; the session cap; the download whitelist; the wire
format (and the browser's header size); multi-byte-safe text deltas; the --model option (set before the profile, and
a usage error without any model files); the default port 8998 (--port 0: a random free port) and the client's URL."""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from duplexaudio.common import paths
from duplexaudio.serving.realtime import profile as profile_mod
from duplexaudio.serving.realtime import server as srv
from duplexaudio.serving.realtime.frontend import SPF, RoomToneFloor
from duplexaudio.serving.realtime.profile import load_profile
from duplexaudio.serving.realtime.protocol import (DOWN_HDR, FLAG_FLUSH, FLAG_HAS_AUDIO, TextDelta, pack_down,
                                                   unpack_down)
from duplexaudio.serving.realtime.session import TickOut

KINDS = ["ss", "pad", "se", "content", "content", "content", "ss", "pad", "pad", "pad", "pad", "pad"]
FED = [1, 3, 2, 104, 105, 106, 1, 3, 3, 3, 3, 3]


class _Tok:
    def decode(self, ids):
        return "".join(chr(0x3B1 + (i - 100)) if i >= 100 else "" for i in ids)     # Greek: multi-byte UTF-8


class _Sess:
    def __init__(self, name, cap):
        self.name, self.cap_frames, self.k, self.state = name, cap, 0, "open"
        self.floor, self.seeds = RoomToneFloor([], 0), {"base": 0}
        self.flushed = self.finalized = None
        self.extra_meta = {}

    def tick(self, frame):
        assert frame.shape == (SPF,)
        k = self.k
        self.k += 1
        pcm = np.full(SPF, k, dtype=np.int16)
        return TickOut(k=k, fed=FED[k % len(FED)], kind=KINDS[k % len(KINDS)], sampled=0, text="", agent_frame=k - 1,
                       pcm=pcm, talker_silent=False, tick_ms=1.5, frames_left=self.cap_frames - self.k)

    def flush(self):
        self.state, self.flushed = "flushed", True
        return [(self.k - 1 + i, np.zeros(SPF, np.int16), True) for i in range(4)]

    def finalize(self, reason):
        self.state, self.finalized = "finalized", reason
        return {"sid": self.name, "reason": reason, "frames": self.k, "seconds": self.k * 0.08, "rec_dir": None,
                "tick_ms": {"p90": 1.5}, "validate_violations": 0, "talker_input_gate": {"verdict": "GREEN"}}


class _Eng:
    def __init__(self, cap=50):
        self.prof = load_profile("default")
        self.gpu = ThreadPoolExecutor(1)
        self.thinker = SimpleNamespace(tok=_Tok())
        self.cap, self.sessions, self.ready = cap, [], True

    def submit(self, fn, *a, **k):
        return self.gpu.submit(fn, *a, **k)

    def new_session(self, name, seed=None):
        s = _Sess(name, self.cap)
        self.sessions.append(s)
        return s


def _client(cap=50, idle_sec=5.0):
    from fastapi.testclient import TestClient
    eng = _Eng(cap)
    hub = srv.Hub(eng, idle_sec=idle_sec)
    hub.state = "ready"
    return TestClient(srv.build_app(hub, load=False)), hub, eng


def _recv_until(ws, typ, limit=200):
    got_bin, got_json = [], []
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            got_bin.append(unpack_down(m["bytes"]))
        else:
            j = json.loads(m["text"])
            got_json.append(j)
            if j["type"] == typ:
                return got_bin, got_json
    raise AssertionError(f"no {typ!r} within {limit} messages: {got_json[-3:]}")


def _chunks(n_frames):
    pcm = (np.arange(n_frames * SPF) % 7).astype("<i2")
    return [pcm[i:i + 480].tobytes() for i in range(0, pcm.size, 480)]


def test_conversation_round_trip():
    tc, hub, eng = _client()
    with tc, tc.websocket_connect("/ws") as ws:
        st = ws.receive_json()
        assert st["type"] == "status" and st["state"] == "ready" and st["profile"] == "default"
        ws.send_text(json.dumps({"type": "start", "seed": 3}))
        _, js = _recv_until(ws, "ready")
        assert js[-1]["frame_samples"] == SPF and js[-1]["cap_frames"] == 50
        for c in _chunks(7):
            ws.send_bytes(c)
        bins, early = [], []
        while len(bins) < 7:
            m = ws.receive()
            if m.get("bytes") is not None:
                bins.append(unpack_down(m["bytes"]))
            else:
                early.append(json.loads(m["text"]))
        hdrs = [h for h, _ in bins]
        assert [h["tick"] for h in hdrs] == list(range(7)) and [h["agent_frame"] for h in hdrs] == list(range(-1, 6))
        assert bins[0][1] is None and not hdrs[0]["flags"] & FLAG_HAS_AUDIO          # frame -1: never played
        assert all(h["flags"] & FLAG_HAS_AUDIO for h in hdrs[1:]) and bins[3][1][0] == 3
        assert [h["kind"] for h in hdrs] == KINDS[:7] and [h["seq"] for h in hdrs] == list(range(7))
        ws.send_text(json.dumps({"type": "stop"}))
        b, js = _recv_until(ws, "ended")
        flush = [h for h, _ in b if h["flags"] & FLAG_FLUSH]
        assert [h["agent_frame"] for h in flush] == [6, 7, 8, 9]
        texts = [j for j in early + js if j["type"] == "text"]
        assert "".join(t["delta"] for t in texts) == "εζη" and any(t.get("turn_end") for t in texts)
        end = js[-1]
        assert end["reason"] == "stopped" and end["meta"]["validate_violations"] == 0
        assert end["downloads"][0].startswith(f"download/{end['sid']}/")
    s = eng.sessions[0]
    assert s.flushed and s.finalized == "stopped" and hub.conv is None and hub.client is None


def test_one_client_at_a_time():
    tc, hub, _ = _client()
    with tc, tc.websocket_connect("/ws") as a:
        a.receive_json()
        with tc.websocket_connect("/ws") as b:
            m = b.receive_json()
            assert m["type"] == "error" and "another client" in m["msg"]


def test_idle_watchdog_and_cap():
    tc, hub, eng = _client(idle_sec=0.2)
    with tc, tc.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start"}))
        _recv_until(ws, "ready")
        _, js = _recv_until(ws, "ended")                       # no audio: the watchdog ends it (checked every 1 s)
        assert js[-1]["reason"] == "idle"
    tc, hub, eng = _client(cap=3)
    with tc, tc.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start"}))
        _recv_until(ws, "ready")
        for c in _chunks(3):
            ws.send_bytes(c)
        _, js = _recv_until(ws, "ended")
        assert js[-1]["reason"] == "cap" and eng.sessions[0].k == 3


def test_start_refused_when_not_ready_and_downloads_whitelisted():
    tc, hub, _ = _client()
    hub.state = "loading"
    with tc, tc.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start"}))
        m = ws.receive_json()
        assert m["type"] == "error" and "not ready" in m["msg"]
        assert tc.get("/download/nope/stereo.wav").status_code == 404
        hub.recordings["x"] = "/tmp"
        assert tc.get("/download/x/secret.txt").status_code == 404
        assert tc.get("/").status_code == 200 and tc.get("/static/app.js").status_code == 200


def test_wire_format_and_browser_header_size():
    pcm = np.arange(SPF, dtype=np.int16)
    b = pack_down(tick=5, agent_frame=4, flags=3, tick_ms=62.5, queue_frames=2, fed=151676, kind="content", seq=9,
                  pcm_i16=pcm)
    h, p = unpack_down(b)
    assert h == {"tick": 5, "agent_frame": 4, "flags": 3, "tick_ms": 62.5, "queue_frames": 2, "n": SPF, "fed": 151676,
                 "kind": "content", "seq": 9} and np.array_equal(p, pcm)
    js = (srv.STATIC / "app.js").read_text()
    assert int(re.search(r"const HDR = (\d+);", js).group(1)) == DOWN_HDR.size == 36
    assert "getInt32(8, true)" in js and "getUint32(12, true)" in js and "getUint16(22, true)" in js


def test_page_explains_the_secure_context_requirement():
    """The mic and the audio worklets exist only on https or http://localhost: elsewhere the page says how to connect
    (an ssh tunnel to localhost) and keeps Start disabled instead of failing with a JS error."""
    js = (srv.STATIC / "app.js").read_text()
    html = (srv.STATIC / "index.html").read_text()
    assert 'id="secureWarn" class="warnbox" hidden' in html
    assert "window.isSecureContext && navigator.mediaDevices" in js and "ssh -L ${port}:localhost:${port}" in js
    assert '|| !!insecure;' in js and 'if (insecure) { log(insecure, "e"); return; }' in js
    assert js.index("const insecure = insecureReason();") < js.rindex("connect();")

def test_start_records_the_browser_mic_settings_and_input_check(monkeypatch):
    """The page sends the getUserMedia settings it actually got; they land in the session meta with the server's
    echo / mic-background summary (measurement only), and an echo finding reaches the client as a typed warning."""
    from duplexaudio.serving.realtime.inputcheck import InputMonitor
    monkeypatch.setattr(InputMonitor, "analyze", staticmethod(lambda snap, max_lag_s=1.0: {
        "bg_p10_dbfs": -70.0, "frames_seen": 30, "agent_audible_s": 2.0, "echo_r": -0.7, "echo_lag_ms": 426.0,
        "echo_gain_db": -2.6}))
    tc, hub, eng = _client()
    audio = {"source": "mic", "requested": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": False},
             "applied": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": False,
                         "sampleRate": 48000, "deviceId": "x" * 500, "nested": {"too": {"deep": 1}}}}
    with tc, tc.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start", "audio": audio}))
        _recv_until(ws, "ready")
        for c in _chunks(30):
            ws.send_bytes(c)
        _, js = _recv_until(ws, "warn", limit=400)
        assert js[-1]["kind"] == "echo" and "Echo detected" in js[-1]["msg"]
        ws.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws, "ended", limit=400)
    m = eng.sessions[-1].extra_meta
    ca = m["client_audio"]
    assert ca["requested"]["noiseSuppression"] is True and ca["applied"]["autoGainControl"] is False
    assert len(ca["applied"]["deviceId"]) == 200 and ca["applied"]["nested"] is None     # 2 levels: audio -> applied
    ic = m["input_check"]
    assert ic["echo"]["warned"] and ic["warnings"] == ["echo"] and ic["frames"] >= 25


def test_input_checks_are_never_fatal(monkeypatch):
    """A failing input monitor turns itself off for the conversation: every frame is still ticked, the conversation
    ends normally and its meta records why the checks are off."""
    from duplexaudio.serving.realtime.inputcheck import InputMonitor

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(InputMonitor, "push", boom)
    tc, hub, eng = _client()
    with tc, tc.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start"}))
        _recv_until(ws, "ready")
        for c in _chunks(5):
            ws.send_bytes(c)
        ws.send_text(json.dumps({"type": "stop"}))
        _, js = _recv_until(ws, "ended", limit=400)
    assert js[-1]["reason"] == "stopped" and js[-1]["meta"]["frames"] == 5
    assert eng.sessions[-1].extra_meta["input_check"] == {"off": "RuntimeError: boom"}


def test_page_defaults_to_noise_suppression_and_echo_cancellation():
    js = (srv.STATIC / "app.js").read_text()
    html = (srv.STATIC / "index.html").read_text()
    assert 'name="out" value="processed" checked' in html and 'name="out" value="raw"' in html
    assert "{ echoCancellation: proc, noiseSuppression: proc, autoGainControl: false, channelCount: 1 }" in js
    assert 'value !== "raw"' in js and 'type: "start", audio: audioInfo' in js and "getSettings()" in js


def test_start_failure_is_reported_and_logged(capsys):
    """A refused start (e.g. the re-arm's memory check) reaches the client AND the server log; the engine stays in
    service and the next start works."""
    client, hub, eng = _client()
    real = eng.new_session

    def refuse(name, seed=None):
        raise RuntimeError("not enough free GPU memory to re-prefill the persona: 10.0 GB free on the device")
    eng.new_session = refuse
    with client, client.websocket_connect("/ws") as ws:
        _recv_until(ws, "status")
        ws.send_text(json.dumps({"type": "start"}))
        _, js = _recv_until(ws, "error")
        assert js[-1]["msg"].startswith("start failed: RuntimeError: not enough free GPU memory")
        assert "[realtime] start failed: RuntimeError: not enough free GPU memory" in capsys.readouterr().out
        eng.new_session = real
        ws.send_text(json.dumps({"type": "start"}))
        _recv_until(ws, "ready")


def test_text_delta_holds_split_characters():
    tok = SimpleNamespace(decode=lambda ids: bytes(ids).decode("utf-8", errors="replace"))
    td = TextDelta(tok)
    dash = "—".encode()                                   # a 3-byte character split across 3 tokens
    out = [td.push(b) for b in [ord("a"), dash[0], dash[1], dash[2], ord("b")]]
    assert out == ["a", "", "", "—", "b"] and td.turn_end() == "" and td.ids == []


def _slow(monkeypatch):
    real_tick = _Sess.tick

    def slow_tick(self, frame):
        import time as _t
        _t.sleep(0.01)
        return real_tick(self, frame)
    monkeypatch.setattr(_Sess, "tick", slow_tick)


def test_backlog_warning_once_per_episode(monkeypatch):
    """Frames arrive faster than the ticks run: one 'falling behind' warning for the episode, no forced end."""
    monkeypatch.setattr(srv, "BACKLOG_WARN_FRAMES", 4)
    monkeypatch.setattr(srv, "BACKLOG_MAX_FRAMES", 1000)
    _slow(monkeypatch)
    tc, hub, eng = _client()
    with tc, tc.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start"}))
        _recv_until(ws, "ready")
        for c in _chunks(20):                                    # 20 frames at once
            ws.send_bytes(c)
        msgs = []
        while sum(1 for m in msgs if m.get("bytes") is not None) < 20:
            msgs.append(ws.receive())
        ws.send_text(json.dumps({"type": "stop"}))
        _, js = _recv_until(ws, "ended")
        texts = [json.loads(m["text"]) for m in msgs if m.get("text") is not None] + js
        warns = [j for j in texts if j["type"] == "warn" and "behind" in j["msg"]]
        assert len(warns) == 1 and js[-1]["reason"] == "stopped" and js[-1]["max_backlog_frames"] >= 4


def test_backlog_hard_limit_ends_the_conversation(monkeypatch):
    monkeypatch.setattr(srv, "BACKLOG_MAX_FRAMES", 10)
    _slow(monkeypatch)
    tc, hub, eng = _client()
    with tc, tc.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start"}))
        _recv_until(ws, "ready")
        for c in _chunks(30):
            ws.send_bytes(c)
        _, js = _recv_until(ws, "ended", limit=400)
        assert js[-1]["reason"] == "backlog" and js[-1]["max_backlog_frames"] >= 10
        assert any(j["type"] == "error" and "behind real time" in j["msg"] for j in js)


def test_cli_model_option():
    """`awaken-server --model DIR_OR_REPO_ID`: optional (without it the runtime root's checkpoints/ data/ are used)."""
    ap = srv.build_parser()
    a = ap.parse_args(["--model", "./pretrained_models/awaken-omni", "--port", "20500"])
    assert (a.model, a.port, a.profile, a.host) == ("./pretrained_models/awaken-omni", 20500, "default", "127.0.0.1")
    assert ap.parse_args([]).model is None
    assert ap.parse_args([]).port == srv.DEFAULT_PORT == 8998 and ap.parse_args(["--port", "0"]).port == 0
    assert "--model DIR_OR_REPO_ID" in ap.format_help() and "0 = a random free port in 20500-20599" in ap.format_help()


def test_cli_sets_the_model_dir_before_the_profile_loads(tmp_path, monkeypatch, keep_model_dir):
    class Loaded(Exception):
        pass
    seen = []

    def fake_load_profile(name):
        seen.append((name, paths.MODEL_ROOT, profile_mod.repo_path("data/talker_canon.json")))
        raise Loaded                                          # stop before the engine and uvicorn
    monkeypatch.setattr(profile_mod, "load_profile", fake_load_profile)
    m = tmp_path / "awaken-omni"
    (m / "checkpoints").mkdir(parents=True)
    (m / "data").mkdir()
    with pytest.raises(Loaded):
        srv.main(["--model", str(m), "--port", "20500"])
    assert seen == [("default", m.resolve(), m.resolve() / "data/talker_canon.json")]
    monkeypatch.setattr(paths, "MODEL_ROOT", paths.RUNTIME_ROOT)
    with pytest.raises(SystemExit) as e:                      # no model files there: a usage error, nothing loaded
        srv.main(["--model", str(tmp_path / "nope")])
    assert e.value.code == 2 and len(seen) == 1 and paths.MODEL_ROOT == paths.RUNTIME_ROOT


def _model_folder(root):
    (root / "checkpoints").mkdir(parents=True)
    (root / "data").mkdir()
    return root


def _fake_uvicorn(monkeypatch):
    """main() up to uvicorn without a server: the engine and uvicorn are faked, the config passed is recorded."""
    from duplexaudio.serving.realtime import engine as engine_mod
    configs = []

    class Server:
        def __init__(self, config):
            self.config = config

        def run(self):
            configs.append(self.config)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(
        Config=lambda app, **kw: kw, Server=Server))
    monkeypatch.setattr(engine_mod, "RealtimeEngine", lambda prof, rec_root=None: SimpleNamespace(prof=prof))
    return configs


def test_cli_port_default_and_port_zero(tmp_path, monkeypatch, keep_model_dir):
    """The server listens on 8998 unless told otherwise; --port 0 keeps the random free port in 20500-20599."""
    configs = _fake_uvicorn(monkeypatch)
    picked = []
    monkeypatch.setattr(srv, "pick_port", lambda: picked.append(1) or 20577)
    m = _model_folder(tmp_path / "awaken-omni")
    srv.main(["--model", str(m)])
    assert configs[-1]["port"] == 8998 and configs[-1]["host"] == "127.0.0.1" and picked == []
    srv.main(["--model", str(m), "--port", "0"])
    assert configs[-1]["port"] == 20577 and picked == [1]
    srv.main(["--model", str(m), "--port", "9100"])
    assert configs[-1]["port"] == 9100 and picked == [1]
    assert 20500 <= srv.pick_port() <= 20599


def test_cli_without_model_files_is_a_usage_error(tmp_path, monkeypatch, capsys, keep_model_dir):
    """No --model and neither checkpoints/ nor data/ under the default model root: exit 2 with the --model hint,
    before any profile, engine or server; with model files there, the default root is used as before."""
    seen = []

    def fake_load_profile(name):
        seen.append(paths.MODEL_ROOT)
        raise KeyError("stop here")
    monkeypatch.setattr(profile_mod, "load_profile", fake_load_profile)
    empty = tmp_path / "runtime"
    empty.mkdir()
    for k, v in {"MODEL_ROOT": empty, "DATA_DIR": empty / "data", "CHECKPOINTS_DIR": empty / "checkpoints"}.items():
        monkeypatch.setattr(paths, k, v)
    with pytest.raises(SystemExit) as e:
        srv.main([])
    err = capsys.readouterr().err
    assert e.value.code == 2 and seen == []
    assert "no model files" in err and str(empty) in err
    assert "--model ./pretrained_models/awaken-omni" in err and "openmartdev/AwakenFullDuplex-Qwen3Omni" in err
    (empty / "data").mkdir()                                  # data/ alone is enough (as for set_model_dir)
    with pytest.raises(KeyError):
        srv.main([])
    assert seen == [empty]


def test_client_default_url_is_the_server_default_port():
    from duplexaudio.serving.realtime import client
    ap = client.build_parser()
    a = ap.parse_args(["--wav", "in.wav", "--out", "out"])
    assert a.url == client.DEFAULT_URL == f"ws://127.0.0.1:{srv.DEFAULT_PORT}/ws" == "ws://127.0.0.1:8998/ws"
    assert ap.parse_args(["--wav", "a", "--out", "b", "--url", "ws://127.0.0.1:20500/ws"]).url.endswith(":20500/ws")
