"""Repo-relative paths (single source of truth).

Code, configs and the realtime profiles are tracked under REPO_ROOT. The session recordings (logs/) live under
RUNTIME_ROOT, which defaults to REPO_ROOT and can be moved with the DUPLEX_RUNTIME_ROOT environment variable. The model
files the profiles pin (checkpoints/, data/) live under MODEL_ROOT: RUNTIME_ROOT by default, or the model folder given
to set_model_dir() (the downloaded Hugging Face model repo), called before load_profile / RealtimeEngine. Use an
editable install (`uv sync`) so REPO_ROOT is the checkout.

    from duplexaudio.common import paths
    paths.set_model_dir("./pretrained_models/awaken-omni")    # optional; a repo id "org/name" downloads the files
    paths.CHECKPOINTS_DIR, paths.DATA_DIR                       # read as attributes: set_model_dir() moves them
"""
from __future__ import annotations

import os as _os
import re as _re
from pathlib import Path

# src/duplexaudio/common/paths.py -> parents[0]=common, [1]=duplexaudio, [2]=src, [3]=repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
CONFIGS_DIR = REPO_ROOT / "configs"

RUNTIME_ROOT = Path(_os.environ.get("DUPLEX_RUNTIME_ROOT", REPO_ROOT)).expanduser().resolve()   # absolute: survives chdir
LOGS_DIR = RUNTIME_ROOT / "logs"
MODEL_ROOT = RUNTIME_ROOT                        # checkpoints/ and data/; set_model_dir() moves it
DATA_DIR = MODEL_ROOT / "data"
CHECKPOINTS_DIR = MODEL_ROOT / "checkpoints"

MODEL_DIRS = ("checkpoints", "data")             # a model folder's layout (= the Hugging Face model repo's)
MODEL_ALLOW_PATTERNS = [f"{d}/*" for d in MODEL_DIRS]      # what a repo-id download fetches (not the card, assets/)
_REPO_ID = _re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")     # org/name; ./x, ~/x and /x are always paths


def set_model_dir(dir_or_repo_id) -> Path:
    """Serve checkpoints/ and data/ from one model folder: a local directory (the downloaded Hugging Face model repo)
    or a Hugging Face repo id "org/name", whose checkpoints/ and data/ are downloaded into the Hugging Face cache.
    Session recordings stay under RUNTIME_ROOT/logs, never in the model folder. Returns the absolute model folder."""
    global MODEL_ROOT, DATA_DIR, CHECKPOINTS_DIR
    root = _model_folder(dir_or_repo_id)
    MODEL_ROOT, DATA_DIR, CHECKPOINTS_DIR = root, root / "data", root / "checkpoints"
    return root


def _model_folder(dir_or_repo_id) -> Path:
    s = str(dir_or_repo_id)
    p = Path(s).expanduser()
    if p.is_dir():
        return _checked(p.resolve(), s)
    if p.exists() or not _REPO_ID.fullmatch(s):
        raise FileNotFoundError(f"model folder {s}: no such directory (and not a Hugging Face repo id org/name)")
    from huggingface_hub import snapshot_download
    try:
        got = snapshot_download(s, allow_patterns=MODEL_ALLOW_PATTERNS)
    except Exception as e:
        raise RuntimeError(f"model {s}: no such local folder, and downloading it as a Hugging Face repo id failed "
                           f"({type(e).__name__}: {e})") from e
    return _checked(Path(got).resolve(), s)


def _checked(root: Path, what: str) -> Path:
    if not any((root / d).is_dir() for d in MODEL_DIRS):
        raise FileNotFoundError(f"model folder {what}: neither checkpoints/ nor data/ inside {root} (want the "
                                f"downloaded model repo)")
    return root


__all__ = ["REPO_ROOT", "SRC_DIR", "CONFIGS_DIR", "RUNTIME_ROOT", "LOGS_DIR", "MODEL_ROOT", "DATA_DIR",
           "CHECKPOINTS_DIR", "MODEL_DIRS", "MODEL_ALLOW_PATTERNS", "set_model_dir"]
