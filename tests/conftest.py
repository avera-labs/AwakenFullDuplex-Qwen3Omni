import pytest

from duplexaudio.common import paths


@pytest.fixture
def keep_model_dir(monkeypatch):
    """paths.set_model_dir() rebinds module state: restore it after the test."""
    for k in ("MODEL_ROOT", "DATA_DIR", "CHECKPOINTS_DIR"):
        monkeypatch.setattr(paths, k, getattr(paths, k))
