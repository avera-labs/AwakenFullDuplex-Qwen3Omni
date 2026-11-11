"""CPU: the model folder (paths.set_model_dir). checkpoints/ and data/ resolve in the downloaded model repo, logs/ (the
recordings) stay under the runtime root, configs/ under the repo; a folder without model files is refused; a repo id
downloads only checkpoints/ and data/ (huggingface_hub faked: no network); without set_model_dir nothing moves."""
import sys
import types

import pytest

from duplexaudio.common import paths
from duplexaudio.serving.realtime.engine import TALKER_TEXT_EMB, RealtimeEngine
from duplexaudio.serving.realtime.profile import check_md5s, load_profile, pinned_files, repo_path

REPO_ID = "openmartdev/AwakenFullDuplex-Qwen3Omni"


def _model_folder(root):
    for d in ("checkpoints/thinker", "checkpoints/talker", "data"):
        (root / d).mkdir(parents=True)
    return root


def test_default_layout_is_the_runtime_root():
    assert paths.MODEL_ROOT == paths.RUNTIME_ROOT
    assert (paths.DATA_DIR, paths.CHECKPOINTS_DIR, paths.LOGS_DIR) == tuple(paths.RUNTIME_ROOT / d for d in
                                                                             ("data", "checkpoints", "logs"))
    for p in ("checkpoints/thinker/model.pt", "data/talker_canon.json", "logs/demo/sessions"):
        assert repo_path(p) == paths.RUNTIME_ROOT / p
    assert repo_path("configs/talker.yaml") == paths.REPO_ROOT / "configs/talker.yaml"


def test_local_model_folder(tmp_path, monkeypatch, keep_model_dir):
    m = _model_folder(tmp_path / "pretrained_models" / "awaken-omni")
    monkeypatch.setenv("HOME", str(tmp_path))
    root = paths.set_model_dir("~/pretrained_models/awaken-omni")          # expanduser + resolve
    assert root == m.resolve() and root.is_absolute() and paths.MODEL_ROOT == root
    assert (paths.DATA_DIR, paths.CHECKPOINTS_DIR) == (root / "data", root / "checkpoints")
    prof = load_profile("default")
    for label, (p, _) in pinned_files(prof).items():
        want = (root if label != "talker.config" else paths.REPO_ROOT) / p
        assert repo_path(p) == want, label
    assert repo_path(TALKER_TEXT_EMB) == root / "data" / "thinker_text_emb.pt"
    # recordings: under the runtime root's logs/, never in the model folder
    assert paths.LOGS_DIR == paths.RUNTIME_ROOT / "logs" and repo_path("logs/x") == paths.RUNTIME_ROOT / "logs/x"
    eng = RealtimeEngine(prof, device="cpu", verify_md5=False)
    try:
        assert eng.rec_root == paths.RUNTIME_ROOT / "logs" / "demo" / "sessions"
        assert root not in eng.rec_root.parents
    finally:
        eng.gpu.shutdown(wait=True)
    # the pins are checked in the model folder: the tracked config still matches, the model files are looked up there
    (root / "checkpoints/thinker/persona.txt").write_text("not the released persona\n", encoding="utf-8")
    bad = dict(b.split(": ", 1) for b in check_md5s(prof))
    assert "talker.config" not in bad and "!= pinned" in bad["thinker.persona"]
    assert {k for k, v in bad.items() if "missing" in v} == set(pinned_files(prof)) - {"talker.config",
                                                                                      "thinker.persona"}
    for k, (p, _) in pinned_files(prof).items():                           # each says where it was looked for
        if k not in ("talker.config", "thinker.persona"):
            assert bad[k] == f"{p} missing (looked for {root / p})", k
    assert paths.set_model_dir(m) == root                                   # a Path works too


def test_refuses_a_folder_without_model_files(tmp_path, keep_model_dir):
    (tmp_path / "empty").mkdir()
    (tmp_path / "f.txt").write_text("x")
    for bad in (tmp_path / "empty", tmp_path / "f.txt", tmp_path / "nope", "./pretrained_models/not-there"):
        with pytest.raises(FileNotFoundError, match="model folder"):
            paths.set_model_dir(bad)
    assert paths.MODEL_ROOT == paths.RUNTIME_ROOT                           # a refused folder changes nothing
    (tmp_path / "only_data" / "data").mkdir(parents=True)                   # checkpoints/ or data/ is enough
    assert paths.set_model_dir(tmp_path / "only_data") == (tmp_path / "only_data").resolve()


def _fake_hub(monkeypatch, snapshot):
    calls = []

    def snapshot_download(repo_id, **kw):
        calls.append((repo_id, kw))
        if isinstance(snapshot, Exception):
            raise snapshot
        return str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_repo_id_downloads_checkpoints_and_data_only(tmp_path, monkeypatch, keep_model_dir):
    snap = _model_folder(tmp_path / "hub" / "snapshots" / "abc")
    calls = _fake_hub(monkeypatch, snap)
    monkeypatch.chdir(tmp_path)                                             # no local folder named like the repo id
    root = paths.set_model_dir(REPO_ID)
    assert calls == [(REPO_ID, {"allow_patterns": ["checkpoints/*", "data/*"]})]
    assert root == snap.resolve() and repo_path("data/talker_canon.json") == root / "data/talker_canon.json"
    assert repo_path("logs/demo") == paths.RUNTIME_ROOT / "logs/demo"


def test_repo_id_failures(tmp_path, monkeypatch, keep_model_dir):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hollow").mkdir()
    calls = _fake_hub(monkeypatch, tmp_path / "hollow")                     # a snapshot without the model files
    with pytest.raises(FileNotFoundError, match="neither checkpoints/ nor data/"):
        paths.set_model_dir(REPO_ID)
    calls = _fake_hub(monkeypatch, OSError("401 Client Error"))
    with pytest.raises(RuntimeError, match="downloading it as a Hugging Face repo id failed"):
        paths.set_model_dir(REPO_ID)
    assert len(calls) == 1 and paths.MODEL_ROOT == paths.RUNTIME_ROOT
    _model_folder(tmp_path / "openmartdev" / "AwakenFullDuplex-Qwen3Omni")  # a local folder wins over the repo id
    calls = _fake_hub(monkeypatch, OSError("must not download"))
    assert paths.set_model_dir(REPO_ID) == (tmp_path / REPO_ID).resolve() and calls == []
