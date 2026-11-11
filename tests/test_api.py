"""CPU: the top-level Python API. `from duplexaudio import RealtimeEngine, load_profile, set_model_dir` gives the same
objects as their modules; `import duplexaudio` alone imports neither torch nor the engine (checked in a fresh
interpreter); unknown names still raise AttributeError."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

PROBE = r"""
import json, sys
import duplexaudio
light = {"torch": "torch" in sys.modules, "engine": "duplexaudio.serving.realtime.engine" in sys.modules,
         "dir": [n for n in ("RealtimeEngine", "load_profile", "set_model_dir") if n in dir(duplexaudio)],
         "all": duplexaudio.__all__, "version": duplexaudio.__version__}
from duplexaudio import load_profile, set_model_dir
light["after_profile_api"] = "torch" in sys.modules
from duplexaudio import RealtimeEngine
light["after_engine"] = "torch" in sys.modules
print(json.dumps(light))
"""


def test_import_duplexaudio_does_not_import_torch(tmp_path):
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                         env={**os.environ, "PYTHONPATH": str(SRC)})
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["torch"] is False and got["engine"] is False
    assert got["dir"] == ["RealtimeEngine", "load_profile", "set_model_dir"]
    assert got["all"] == ["RealtimeEngine", "load_profile", "set_model_dir", "__version__"]
    assert got["version"] == "0.1.0"
    assert got["after_profile_api"] is False and got["after_engine"] is True


def test_top_level_names_are_the_module_objects():
    import duplexaudio
    from duplexaudio import RealtimeEngine, load_profile, set_model_dir
    from duplexaudio.common import paths
    from duplexaudio.serving.realtime import engine, profile
    assert RealtimeEngine is engine.RealtimeEngine and load_profile is profile.load_profile
    assert set_model_dir is paths.set_model_dir and duplexaudio.load_profile("default").name == "default"
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        duplexaudio.nope
    with pytest.raises(ImportError):
        exec("from duplexaudio import nope", {})
