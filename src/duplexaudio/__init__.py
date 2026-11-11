"""duplexaudio (the awaken-omni distribution): Awaken Full Duplex based on Qwen3-Omni, thinker-talker full-duplex
spoken dialogue (inference and the realtime demo).

Quick start::

    from duplexaudio import RealtimeEngine, load_profile, set_model_dir

    set_model_dir("./pretrained_models/awaken-omni")    # or the HF repo id; before load_profile / RealtimeEngine
    eng = RealtimeEngine(load_profile("default"))
    eng.run(eng.load)                                   # md5 pins, weights, compile + warm-up (a few minutes)
    sess = eng.run(eng.new_session, "demo")
    out = eng.run(sess.tick, frame)                     # per float32 (1920,) frame: 80 ms of 24 kHz mono user audio
    eng.run(sess.flush)
    meta = eng.run(sess.finalize, "stop")

See ``awaken-server`` for live use and ``awaken-offline`` for a recorded track. The names are imported on first use,
so ``import duplexaudio`` alone does not load torch.
"""

__version__ = "0.1.0"

_LAZY = {"RealtimeEngine": "duplexaudio.serving.realtime.engine",
         "load_profile": "duplexaudio.serving.realtime.profile",
         "set_model_dir": "duplexaudio.common.paths"}

__all__ = ["RealtimeEngine", "load_profile", "set_model_dir", "__version__"]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
