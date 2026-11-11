"""Realtime profiles: ONE YAML file per deployment, every numeric axis explicit and md5-pinned.

A profile is the whole decode convention of the live engine (the decode must match the one the models were
evaluated with). The loader refuses missing and unknown keys and there are no dataclass defaults, so a value can
never drift in silently from code. Swapping a model (another thinker or talker checkpoint) = another YAML file, never
a code edit.

Shipped profiles live next to this module (`profiles/*.yaml`); `load_profile("default")` resolves a bare name.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from duplexaudio.common import paths

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

THINKER_KEYS = {"ckpt", "ckpt_md5", "persona", "persona_md5", "lora_alloc_rank", "max_cache_len", "prefill_micro_batch",
                "decode"}
DECODE_KEYS = {"temperature", "silence_temperature", "speech_temperature", "rep_control", "ss_bias", "se_bias"}
REP_KEYS = {"rep_penalty", "no_repeat_ngram", "rep_window"}
TALKER_KEYS = {"ckpt", "ckpt_md5", "config", "config_md5", "vp_codes", "vp_codes_md5", "vp_text", "vp_text_md5",
               "layout", "canon_table", "canon_md5", "canon_mode", "sampler", "session"}
SAMPLER_KEYS = {"noise", "temp", "top_k", "top_p"}
FRONTEND_KEYS = {"floor_bands", "mimi_ring_capacity", "mimi_cuda_graph"}
SESSION_KEYS = {"cap_frames", "seed", "end_flush_frames"}
TOP_KEYS = {"name", "description", "thinker", "talker", "frontend", "session"}

TALKER_SESSIONS = ("persistent", "fresh")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class ThinkerProfile:
    ckpt: str
    ckpt_md5: str
    persona: str
    persona_md5: str
    lora_alloc_rank: int | None          # None = the checkpoint's native rank
    max_cache_len: int
    prefill_micro_batch: int | None      # 1 = micro-batch-1 persona prefill (as the offline fast path); None = one-shot
    decode: dict                         # step_masked sampling settings (DECODE_KEYS)


@dataclass(frozen=True)
class TalkerProfile:
    ckpt: str
    ckpt_md5: str
    config: str
    config_md5: str
    vp_codes: str
    vp_codes_md5: str
    vp_text: str
    vp_text_md5: str
    layout: str                          # vp-prefix layout; the served convention is "train"
    canon_table: str
    canon_md5: str
    canon_mode: str
    sampler: dict                        # LMGenX settings (SAMPLER_KEYS)
    session: str                         # "persistent" (one LMGenX, reset per conversation) | "fresh"


@dataclass(frozen=True)
class FrontendProfile:
    floor_bands: list                    # [[lo_db, hi_db, weight], ...] room-tone floor under the user channel; [] = off
    mimi_ring_capacity: int              # encoder-transformer ring KV capacity (moshi default 250)
    mimi_cuda_graph: bool                # moshi CUDA graphs for the 1-frame Mimi encode/decode calls


@dataclass(frozen=True)
class SessionProfile:
    cap_frames: int                      # requested hard cap; the engine lowers it to what the caches can hold
    seed: int                            # base seed; per-session seeds are derived from (seed, session name)
    end_flush_frames: int                # PAD frames the talker voices after the input stops (text lead + 1)


@dataclass(frozen=True)
class RealtimeProfile:
    name: str
    description: str
    thinker: ThinkerProfile
    talker: TalkerProfile
    frontend: FrontendProfile
    session: SessionProfile
    path: str
    md5: str                             # md5 of the profile file bytes
    overrides: tuple = ()                # ("thinker.max_cache_len=2048", ...) applied by with_overrides()

    def to_json(self) -> dict:
        return asdict(self)


def _exact(d, keys, where):
    if not isinstance(d, dict):
        raise ProfileError(f"{where}: expected a mapping")
    missing, extra = keys - set(d), set(d) - keys
    if missing or extra:
        raise ProfileError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")
    return d


_INT, _FLOAT, _STR, _BOOL, _OPT_INT = "int", "float", "str", "bool", "int|None"
TYPES = {
    "thinker": {"ckpt": _STR, "ckpt_md5": _STR, "persona": _STR, "persona_md5": _STR, "lora_alloc_rank": _OPT_INT,
                "max_cache_len": _INT, "prefill_micro_batch": _OPT_INT},
    "thinker.decode": {"temperature": _FLOAT, "silence_temperature": _FLOAT, "speech_temperature": _FLOAT,
                       "ss_bias": _FLOAT, "se_bias": _FLOAT},
    "thinker.decode.rep_control": {"rep_penalty": _FLOAT, "no_repeat_ngram": _INT, "rep_window": _INT},
    "talker": {k: _STR for k in TALKER_KEYS - {"sampler"}},
    "talker.sampler": {"noise": _STR, "temp": _FLOAT, "top_k": _INT, "top_p": _FLOAT},
    "frontend": {"mimi_ring_capacity": _INT, "mimi_cuda_graph": _BOOL},
    "session": {"cap_frames": _INT, "seed": _INT, "end_flush_frames": _INT},
}


def _typed(v, t, where):
    ok = {"int": isinstance(v, int) and not isinstance(v, bool),
          "float": isinstance(v, (int, float)) and not isinstance(v, bool),
          "str": isinstance(v, str), "bool": isinstance(v, bool),
          "int|None": v is None or (isinstance(v, int) and not isinstance(v, bool))}[t]
    if not ok:
        raise ProfileError(f"{where}: {v!r} is not {t}")


def _check_types(sections: dict):
    for where, d in sections.items():
        for k, t in TYPES[where].items():
            _typed(d[k], t, f"{where}.{k}")


def resolve_profile_path(name_or_path) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    q = PROFILES_DIR / f"{name_or_path}.yaml"
    if q.exists():
        return q
    raise ProfileError(f"no profile {name_or_path!r} (looked for a file and for {q})")


def load_profile(name_or_path) -> RealtimeProfile:
    path = resolve_profile_path(name_or_path)
    raw = path.read_bytes()
    d = _exact(yaml.safe_load(raw), TOP_KEYS, path.name)
    th = _exact(d["thinker"], THINKER_KEYS, "thinker")
    dec = _exact(th["decode"], DECODE_KEYS, "thinker.decode")
    _exact(dec["rep_control"], REP_KEYS, "thinker.decode.rep_control")
    tk = _exact(d["talker"], TALKER_KEYS, "talker")
    _exact(tk["sampler"], SAMPLER_KEYS, "talker.sampler")
    fe = _exact(d["frontend"], FRONTEND_KEYS, "frontend")
    se = _exact(d["session"], SESSION_KEYS, "session")
    _check_types({"thinker": th, "thinker.decode": dec, "thinker.decode.rep_control": dec["rep_control"], "talker": tk,
                  "talker.sampler": tk["sampler"], "frontend": fe, "session": se})
    if not isinstance(fe["floor_bands"], list):
        raise ProfileError("frontend.floor_bands must be a list")
    if tk["layout"] != "train":
        raise ProfileError("talker.layout: realtime profiles serve the train-layout voice-prompt prefix only")
    if tk["session"] not in TALKER_SESSIONS:
        raise ProfileError(f"talker.session must be one of {TALKER_SESSIONS}")
    if tk["sampler"]["noise"] != "legacy":
        raise ProfileError("talker.sampler.noise: the realtime talker serves the 'legacy' sampler only")
    for b in fe["floor_bands"]:
        if len(b) != 3 or not b[0] <= b[1] or b[2] <= 0:
            raise ProfileError(f"frontend.floor_bands: bad band {b} (want [lo_db, hi_db, weight>0])")
    if int(fe["mimi_ring_capacity"]) < 250:
        raise ProfileError("frontend.mimi_ring_capacity below Mimi's 250-step context drops in-window keys")
    return RealtimeProfile(
        name=d["name"], description=d["description"],
        thinker=ThinkerProfile(**th), talker=TalkerProfile(**tk), frontend=FrontendProfile(**fe),
        session=SessionProfile(**se), path=str(path), md5=hashlib.md5(raw).hexdigest())


def md5_file(path, chunk=1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


MODEL_DIRS = set(paths.MODEL_DIRS)              # data/ checkpoints/
RUNTIME_DIRS = MODEL_DIRS | {"logs"}


def repo_path(p) -> Path:
    """Profile paths are repo-relative. The model files (data/ checkpoints/) resolve under paths.MODEL_ROOT (the
    folder given to paths.set_model_dir(); the runtime root by default), logs/ under paths.RUNTIME_ROOT (the
    DUPLEX_RUNTIME_ROOT env var; the repo root by default), tracked dirs (configs/ ...) under paths.REPO_ROOT."""
    p = Path(p)
    if p.is_absolute():
        return p
    head = p.parts[0] if p.parts else None
    root = paths.MODEL_ROOT if head in MODEL_DIRS else paths.RUNTIME_ROOT if head in RUNTIME_DIRS else paths.REPO_ROOT
    return root / p


def pinned_files(prof: RealtimeProfile) -> dict:
    """{label: (path, expected md5)} for every file the profile pins."""
    t, k = prof.thinker, prof.talker
    return {"thinker.ckpt": (t.ckpt, t.ckpt_md5), "thinker.persona": (t.persona, t.persona_md5),
            "talker.ckpt": (k.ckpt, k.ckpt_md5), "talker.config": (k.config, k.config_md5),
            "talker.vp_codes": (k.vp_codes, k.vp_codes_md5), "talker.vp_text": (k.vp_text, k.vp_text_md5),
            "talker.canon_table": (k.canon_table, k.canon_md5)}


def check_md5s(prof: RealtimeProfile, skip=()) -> list:
    """Mismatches as human-readable strings ([] = all pinned files match). `skip` labels are not hashed. A missing
    file names the path it was looked for at: relative profile paths resolve as repo_path() says (data/ and
    checkpoints/ in the model folder), never against the working directory."""
    bad = []
    for label, (p, want) in pinned_files(prof).items():
        if label in skip:
            continue
        rp = repo_path(p)
        if not rp.exists():
            bad.append(f"{label}: {p} missing" + ("" if str(rp) == str(p) else f" (looked for {rp})"))
            continue
        got = md5_file(rp)
        if got != want:
            bad.append(f"{label}: {p} md5 {got} != pinned {want}")
    return bad


def profile_json(prof: RealtimeProfile) -> str:
    """The realized settings (file location and file md5 excluded), canonical JSON."""
    d = prof.to_json()
    for k in ("path", "md5"):
        d.pop(k)
    return json.dumps(d, sort_keys=True)


def realized_md5(prof: RealtimeProfile) -> str:
    """md5 of the settings actually in force: differs from the file's md5 as soon as anything is overridden."""
    return hashlib.md5(profile_json(prof).encode()).hexdigest()


def with_overrides(prof: RealtimeProfile, **dotted) -> RealtimeProfile:
    """A copy with dotted-key overrides, recorded in `overrides`: with_overrides(p, **{"thinker.max_cache_len": 2048})."""
    from dataclasses import replace
    out = prof
    for key, val in dotted.items():
        section, _, fld = key.partition(".")
        sub = getattr(out, section)
        if not hasattr(sub, fld):
            raise ProfileError(f"unknown override {key!r}")
        out = replace(out, **{section: replace(sub, **{fld: val})})
    return replace(out, overrides=tuple(prof.overrides) + tuple(f"{k}={v}" for k, v in dotted.items()))
