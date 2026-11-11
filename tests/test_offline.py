"""CPU: awaken-offline end to end against a fake engine driving the real DuplexSession (no models are loaded). The wav
is read and resampled like awaken-client's, cut into 1920-sample frames (the last zero-padded) and followed by the
silent tail; every frame is ticked, the talker flushed, the session finalized into --out (stereo.wav L user / R agent,
the session's recordings, transcript.txt per agent turn); nothing is written into the model folder; the session cap
stops the feed; bad --input / --out / missing model files are usage errors before anything loads."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from duplexaudio.serving.realtime import engine as engine_mod
from duplexaudio.serving.realtime import offline
from duplexaudio.serving.realtime.client import load_wav_24k
from duplexaudio.serving.realtime.frontend import SPF, SR, RoomToneFloor
from duplexaudio.serving.realtime.profile import load_profile
from duplexaudio.serving.realtime.session import DuplexSession

SS, SE, PAD, PAUSE = 1, 2, 3, 4
# tick k -> (fed, kind): one closed agent turn with a pause inside it, then one left open at the end of the input
SCRIPT = {0: (SS, "ss"), 2: (SE, "se"), 3: (104, "content"), 4: (105, "content"), 5: (PAUSE, "pause"),
          6: (106, "content"), 7: (SS, "ss"), 10: (SE, "se"), 11: (107, "content"), 12: (108, "content")}


class _Tok:
    def decode(self, ids):
        return " " + "".join(chr(0x3B1 + (i - 100)) for i in ids if i >= 100)     # Greek: multi-byte UTF-8


class _FakeEngine:
    """What DuplexSession and awaken-offline use of RealtimeEngine, with stand-in models: the encoder records the
    frames it hears, the talker answers frame k-1 with codes (k, ...), the decoder turns codes into a constant pcm."""
    instances = []

    def __init__(self, prof, *, cap=3000):
        self.prof, self.device, self.cap = prof, "cpu", cap
        self.silence_codes = [0] * 8
        self.spec = type("Spec", (), {"pad_id": PAD})()
        self.heard, self.calls, self.sessions = [], [], []
        self._active = None
        self.gpu = type("Gpu", (), {"shutdown": lambda _, wait=True: self.calls.append("shutdown")})()
        eng = self

        class Encoder:
            def encode(self, x):
                eng.heard.append(np.array(x))
                return torch.zeros(8, dtype=torch.long)

        class Thinker:
            tok, arm_info = _Tok(), {"purge_verified": True, "cache_at_start": 421}

            def step(self, c8):
                return (*SCRIPT.get(len(eng.heard) - 1, (PAD, "pad")), 0)

        class Talker:
            forced_ch0, j = [], -2

            def step(self, ch0):
                self.j += 1
                return self.j, torch.full((8,), self.j + 1, dtype=torch.long)

            def end(self):
                eng.calls.append("talker.end")

            def rng_summary(self):
                return {"violations": 0}

        self.encoder, self.thinker, self.talker = Encoder(), Thinker(), Talker()
        self.feeder = lambda fed: fed
        self.decoder = type("Dec", (), {"decode": lambda _, a: np.full(SPF, 50 * int(a[0]), np.int16)})()
        _FakeEngine.instances.append(self)

    def run(self, fn, *a, **k):
        return fn(*a, **k)

    def load(self):
        self.calls.append("load")

    def new_session(self, name, seed=None, *, rec_dir=None):
        self.calls.append(("new_session", name, seed, rec_dir))
        s = DuplexSession(self, name=name, seeds={"base": seed}, floor=RoomToneFloor([], 0), rec_dir=rec_dir,
                          cap_frames=self.cap)
        self._active = s
        self.sessions.append(s)
        return s

    def session_provenance(self):
        return {"profile": self.prof.name}

    def talker_input_gate(self, sid, fed, forced_ch0):
        return {"verdict": "GREEN"}


@pytest.fixture
def fake_engine(monkeypatch):
    from duplexaudio.inference.thinker import state_machine
    monkeypatch.setattr(state_machine, "validate_stream", lambda fed, spec, **kw: [])
    monkeypatch.setattr(engine_mod, "RealtimeEngine", _FakeEngine)
    _FakeEngine.instances = []
    return _FakeEngine


def _model_folder(root):
    for d in ("checkpoints/thinker", "data"):
        (root / d).mkdir(parents=True)
    (root / "data" / "talker_canon.json").write_text("{}")
    return root


def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def test_offline_end_to_end(tmp_path, fake_engine, keep_model_dir, capsys):
    m = _model_folder(tmp_path / "awaken-omni")
    before = _tree(m)
    wav = tmp_path / "user_turns.wav"                                     # 1 s of 16 kHz stereo: resampled + downmixed
    t = np.arange(16000) / 16000.0
    sf.write(wav, np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.1 * np.sin(2 * np.pi * 330 * t)], 1), 16000)
    out = tmp_path / "out"
    offline.main(["--model", str(m), "--input", str(wav), "--out", str(out), "--seed", "7", "--tail-s", "0.4"])

    eng = fake_engine.instances[-1]
    assert eng.calls[0] == "load" and eng.calls[1] == ("new_session", "user_turns", 7, out.resolve())
    assert eng.calls[-1] == "shutdown"
    x = load_wav_24k(wav)
    n_in = -(-x.size // SPF)                                              # 24000 samples -> 13 frames (last padded)
    assert x.size == 24000 and n_in == 13 and len(eng.heard) == n_in + 5   # + 0.4 s of tail = 5 frames
    assert all(f.shape == (SPF,) and f.dtype == np.float32 for f in eng.heard)
    fed = np.concatenate(eng.heard)
    assert np.allclose(fed[:x.size], x) and not fed[x.size:].any()        # the track, then zeros (floor off here)

    files = {p.name for p in out.iterdir()}
    assert files == {"stereo.wav", "transcript.txt", "user.wav", "agent.wav", "codes.npz", "stream.json",
                     "timing.json", "meta.json"}
    st, sr = sf.read(out / "stereo.wav", dtype="float32", always_2d=True)
    assert sr == SR and st.shape[1] == 2
    n_agent = len(eng.heard) - 1 + 4                                      # frames 0..k-1 from the ticks, + 4 flushed
    assert st.shape[0] == max(fed.size, n_agent * SPF)
    assert np.allclose(st[:fed.size, 0], fed, atol=1e-4)                  # L: the user as the model heard it
    for j in (0, 3, n_agent - 1):                                         # R: the agent, frame j at j * 1920
        assert np.allclose(st[j * SPF:(j + 1) * SPF, 1], 50 * (j + 1) / 32768.0, atol=1e-4), j
    assert (out / "transcript.txt").read_text(encoding="utf-8") == "[0.24-0.56] εζη\n[0.88-1.04] θι\n"
    meta = json.loads((out / "meta.json").read_text())
    assert (meta["reason"], meta["frames"], meta["flushed_frames"]) == ("eof", 18, 4)
    assert meta["offline"] == {"input_frames": 13, "tail_frames": 5, "tail_s": 0.4}
    assert "gate_error" not in meta and meta["talker_input_gate"]["verdict"] == "GREEN"
    assert json.loads((out / "stream.json").read_text())["kinds"][:4] == ["ss", "pad", "se", "content"]
    assert _tree(m) == before                                             # nothing written into the model folder
    printed = capsys.readouterr().out
    assert "[0.24-0.56] εζη" in printed and f"18 frames (1.44 s, eof) -> {out.resolve()}" in printed


def test_offline_stops_at_the_session_cap(tmp_path):
    prof = load_profile("default")
    eng = _FakeEngine(prof, cap=6)
    meta, lines = offline.run_offline(eng, np.zeros(10 * SPF, np.float32), name="capped", out=tmp_path / "o",
                                      tail_s=0.0)
    assert len(eng.heard) == 6 and meta["reason"] == "cap" and meta["frames"] == 6 and meta["flushed_frames"] == 4
    assert lines == ["[0.24-0.40] εζ"]                                    # the turn so far (content at ticks 3-4)
    assert (tmp_path / "o" / "transcript.txt").read_text(encoding="utf-8") == "[0.24-0.40] εζ\n"


def test_failed_tick_still_finalizes_the_recordings(tmp_path):
    eng = _FakeEngine(load_profile("default"))
    real = eng.thinker.step

    def step(c8):
        if len(eng.heard) == 4:
            raise RuntimeError("boom")
        return real(c8)
    eng.thinker.step = step
    with pytest.raises(RuntimeError, match="boom"):
        offline.run_offline(eng, np.zeros(8 * SPF, np.float32), name="x", out=tmp_path / "o", tail_s=0.0)
    meta = json.loads((tmp_path / "o" / "meta.json").read_text())
    assert meta["reason"] == "error" and meta["frames"] == 3 and meta["errors"][0]["msg"] == "boom"
    assert (tmp_path / "o" / "stereo.wav").exists() and (tmp_path / "o" / "transcript.txt").read_text() == ""


def test_frames_and_tail_rounding():
    assert offline.frames_of(np.ones(SPF + 1, np.float32), 0.0).shape == (2, SPF)
    assert offline.frames_of(np.ones(SPF, np.float32), 0.24).shape == (4, SPF)          # 0.24 s = 3 frames exactly
    assert offline.frames_of(np.ones(SPF, np.float32), 3.0).shape == (1 + 38, SPF)      # 37.5 frames -> 38
    f = offline.frames_of(np.ones(SPF + 1, np.float32), 0.08)
    assert f[0].all() and f[1, 0] == 1 and not f[1, 1:].any() and not f[2].any()


def test_usage_errors_before_anything_loads(tmp_path, fake_engine, monkeypatch, capsys, keep_model_dir):
    from duplexaudio.common import paths
    m = _model_folder(tmp_path / "awaken-omni")
    wav = tmp_path / "in.wav"
    sf.write(wav, np.zeros(2400, np.float32), SR)
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_text("x")
    (m / "checkpoints" / "runs").mkdir()                                  # an empty folder inside the model folder
    cases = [
        (["--model", str(m), "--input", str(tmp_path / "nope.wav"), "--out", str(tmp_path / "o1")], "no such file"),
        (["--model", str(m), "--input", str(wav), "--out", str(full)], "new or empty directory"),
        (["--model", str(m), "--input", str(wav), "--out", str(m / "out")], "inside the model folder"),
        (["--model", str(m), "--input", str(wav), "--out", str(m / "checkpoints" / "runs")], "inside the model folder"),
        (["--model", str(m), "--input", str(wav), "--out", str(tmp_path / "o2"), "--tail-s", "-1"], "--tail-s"),
        (["--model", str(tmp_path / "nope"), "--input", str(wav), "--out", str(tmp_path / "o3")], "model folder"),
        (["--model", str(m), "--input", str(wav), "--out", str(tmp_path / "o5"), "--tail-s", "nan"], "--tail-s"),
        (["--model", str(m), "--input", str(wav), "--out", str(tmp_path / "o6"), "--tail-s", "inf"], "--tail-s"),
        # the input is decoded before --model is resolved (a repo id would download first): a bad file wins
        (["--model", str(tmp_path / "nope"), "--input", str(bad), "--out", str(tmp_path / "o7")], "not a readable audio"),
    ]
    before = _tree(m)
    for argv, msg in cases:
        with pytest.raises(SystemExit) as e:
            offline.main(argv)
        assert e.value.code == 2 and msg in capsys.readouterr().err, argv
    empty = tmp_path / "runtime"                                          # no --model, no model files: the server's
    empty.mkdir()                                                         # fail-fast with the --model hint
    for k, v in {"MODEL_ROOT": empty, "DATA_DIR": empty / "data", "CHECKPOINTS_DIR": empty / "checkpoints"}.items():
        monkeypatch.setattr(paths, k, v)
    with pytest.raises(SystemExit) as e:
        offline.main(["--input", str(wav), "--out", str(tmp_path / "o4")])
    assert e.value.code == 2 and "--model ./pretrained_models/awaken-omni" in capsys.readouterr().err
    assert fake_engine.instances == [] and _tree(m) == before
    assert not any((tmp_path / f"o{i}").exists() for i in range(1, 8))
    assert {p.name for p in full.iterdir()} == {"keep.txt"}


def test_cli_help():
    h = offline.build_parser().format_help()
    for flag in ("--model DIR_OR_REPO_ID", "--input", "--out", "--profile", "--seed", "--tail-s"):
        assert flag in h
    a = offline.build_parser().parse_args(["--input", "a.wav", "--out", "o"])
    assert (a.profile, a.seed, a.tail_s, a.model) == ("default", None, 3.0, None)
