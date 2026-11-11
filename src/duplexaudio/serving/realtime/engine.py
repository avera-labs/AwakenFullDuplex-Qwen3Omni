"""RealtimeEngine: owns the pinned models (one profile per process) and the ONE thread every CUDA call runs on.

    eng = RealtimeEngine(load_profile("default"))
    eng.run(eng.load)                                     # weights, first compiled prime, warm-up (~3-4 min)
    sess = eng.run(eng.new_session, "my-sid")             # re-arm (~1 s) + seeds + talker prefix
    out = eng.run(sess.tick, frame)                       # per 80 ms frame
    meta = eng.run(sess.finalize, "stop")

Process rules (each one is a parity or real-time requirement):
- every CUDA call runs on the engine's single worker thread (compiled cudagraph trees and moshi's streaming state are
  thread-bound); entry points check it;
- one thinker configuration per process: the compiled thinker is built once and re-prefilled per conversation;
  dynamo's recompile limit fails loudly instead of silently decoding eager;
- warm-up before `ready`: first compiled prime, the talker's CUDA graphs and lazily compiled kernels (talker first,
  as in the reference talker decode), two encoder and two decoder calls, then gc.freeze();
- the talker owns the device default generator; nothing else may draw from it (StreamingTalker trips on it);
- one active conversation at a time: the models carry its state.
"""
from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from duplexaudio.common import paths
from duplexaudio.serving.realtime.canon import Ch0Feeder, StreamingCanon
from duplexaudio.serving.realtime.frontend import RoomToneFloor, floor_seed
from duplexaudio.serving.realtime.profile import RealtimeProfile, check_md5s, pinned_files, realized_md5, repo_path
from duplexaudio.serving.realtime.session import DuplexSession

PREFILL_GB_PER_TOKEN = 0.0755        # transformers batched_mm expert-weight gather during the persona prefill (B=1)
PREFILL_MARGIN_GB = 4.0
TALKER_TEXT_EMB = "data/thinker_text_emb.pt"   # the talker's text-embedding table (model folder; not md5-pinned)


def resolve_device(device) -> torch.device:
    """An explicit device. For CUDA: make it the thread's current device (torch.cuda.set_device needs an INDEX, not a
    bare 'cuda'), so every implicit 'cuda' (the talker loader, CUDA events, moshi's capture streams) is this one."""
    device = torch.device(device)
    if device.type != "cuda":
        return device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(int(idx))
    return torch.device("cuda", int(idx))


def session_seeds(name: str, seed: int) -> dict:
    """Per-conversation seeds, each its own stream: thinker sampling (the per-row rule), talker, floor."""
    from duplexaudio.serving.realtime.thinker import row_sampling_seed
    return {"base": int(seed), "thinker": row_sampling_seed(name, seed),
            "talker": (zlib.crc32(f"talker:{name}".encode()) ^ int(seed)) & 0x7FFFFFFF,
            "floor": floor_seed(name, seed)}


def session_cap(prof: RealtimeProfile, *, talker_context: int, prefix_frames: int, prompt_len: int) -> dict:
    """How many live frames one conversation may take: the requested cap, the talker's context (the voice prompt must
    stay inside its KV window through the end flush) and the thinker's static cache (prompt + frames + 2 spare)."""
    return {"requested": int(prof.session.cap_frames),
            "talker_context": int(talker_context) - int(prefix_frames) - int(prof.session.end_flush_frames),
            "thinker_cache": int(prof.thinker.max_cache_len) - int(prompt_len) - 2}


def talker_input_gate(sid, fed, forced_ch0, *, cfg, spec, vp_codes, vp_text, canon, canon_mode) -> dict:
    """CPU check on a finished stream: the ch0 the talker was actually forced with, step by step (streaming
    canonicalizer + feeder), == the reference builder's dialogue ch0 for the same fed stream (talker_inputs)."""
    from duplexaudio.inference.talker.inputs import talker_inputs
    x = talker_inputs(list(fed), cfg, spec, vp_codes=vp_codes, vp_text=vp_text, canon=canon, canon_mode=canon_mode)
    want = x["ch0"][x["P"]:].tolist()
    same = want == [int(c) for c in forced_ch0[:len(want)]]
    return {"verdict": "GREEN" if same else "RED", "builder_equal": same}


def git_state(root: Path) -> dict:
    """sha and dirty (modified tracked files) of the code checkout, recorded in every session's meta."""
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": None, "git_dirty": None}


class RealtimeEngine:
    def __init__(self, prof: RealtimeProfile, *, device="cuda", rec_root=None, verify_md5=True):
        self.prof = prof
        self.device = torch.device(device)
        self.rec_root = Path(rec_root) if rec_root is not None else paths.LOGS_DIR / "demo" / "sessions"
        self.verify_md5 = verify_md5
        self.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="realtime-gpu")
        self._gpu_thread = self.gpu.submit(threading.get_ident).result()
        self.thinker = self.talker = self.encoder = self.decoder = self.feeder = None
        self.talker_cfg = self.vp_codes = self.vp_text = self.canon_table = None
        self.silence_codes = None
        self.cap_limits = {}
        self.code_gate = {}
        self.md5_check = {"verified": False, "skipped": sorted(pinned_files(prof))}
        self.ring_capacities = {}
        self.arms = 0
        self.ready = False
        self._active = None

    # ── threading ────────────────────────────────────────────────────────────────────────────────────────────
    def submit(self, fn, *a, **k):
        return self.gpu.submit(fn, *a, **k)

    def run(self, fn, *a, **k):
        if threading.get_ident() == self._gpu_thread:   # already on the GPU thread: inline (a queued call would deadlock)
            return fn(*a, **k)
        return self.submit(fn, *a, **k).result()

    def require_gpu_thread(self):
        if threading.get_ident() != self._gpu_thread:
            raise RuntimeError("CUDA work off the engine's GPU thread: call it through RealtimeEngine.run/submit")

    @property
    def spec(self):
        return self.thinker.spec

    # ── load + warm-up (GPU thread) ──────────────────────────────────────────────────────────────────────────
    def load(self, *, skip_md5=()):
        from duplexaudio.common.utils.config import load_yaml
        from duplexaudio.inference.talker.canon_table import load_canon_table
        from duplexaudio.inference.talker.load import load_talker
        from duplexaudio.inference.talker.voice import VP_AGENT_SILENCE
        from duplexaudio.serving.realtime.codec import StreamingDecoder, StreamingEncoder
        from duplexaudio.serving.realtime.talker import StreamingTalker, vp_prefix
        from duplexaudio.serving.realtime.thinker import ThinkerRunner
        self.require_gpu_thread()
        p = self.prof
        self.device = resolve_device(self.device)
        if self.verify_md5:
            bad = check_md5s(p, skip=skip_md5)
            if bad:
                raise RuntimeError("profile pins do not match:\n  " + "\n  ".join(bad))
            self.md5_check = {"verified": True, "skipped": sorted(skip_md5)}
        if not repo_path(TALKER_TEXT_EMB).is_file():          # not pinned: fail here, not after the thinker loads
            raise RuntimeError(f"missing {repo_path(TALKER_TEXT_EMB)} (the talker's text-embedding table)")
        import torch._dynamo as dynamo                  # (`import torch._dynamo` would make `torch` local here)
        if hasattr(dynamo.config, "fail_on_recompile_limit_hit"):
            dynamo.config.fail_on_recompile_limit_hit = True
        self.code_gate = self.git_state()
        self.thinker = ThinkerRunner(p.thinker, device=self.device)
        self.thinker.load(repo_path(p.thinker.ckpt))
        lm = load_talker(repo_path(p.talker.ckpt), repo_path(p.talker.config),
                         thinker_emb_path=str(repo_path(TALKER_TEXT_EMB)))
        self.talker_cfg = load_yaml(str(repo_path(p.talker.config)))
        lead = int(self.talker_cfg.get("text_lead_frames", 3))
        if int(p.session.end_flush_frames) != lead + 1:
            raise RuntimeError(f"session.end_flush_frames {p.session.end_flush_frames} != text_lead_frames {lead} + 1")
        self.vp_codes = np.load(repo_path(p.talker.vp_codes))
        self.vp_text = np.load(repo_path(p.talker.vp_text))
        self.canon_table, canon_md5 = load_canon_table(repo_path(p.talker.canon_table))
        if canon_md5 != p.talker.canon_md5:
            raise RuntimeError(f"canon table md5 {canon_md5} != pinned {p.talker.canon_md5}")
        prefix = vp_prefix(self.talker_cfg, self.spec, vp_codes=self.vp_codes, vp_text=self.vp_text)
        self.talker = StreamingTalker(lm, prefix, self.spec, sampler=p.talker.sampler, device=self.device,
                                      session=p.talker.session)
        self.feeder = Ch0Feeder(self.spec, StreamingCanon(self.canon_table, p.talker.canon_mode, self.spec))
        self.encoder = StreamingEncoder(self.device, ring_capacity=p.frontend.mimi_ring_capacity,
                                        graphed=p.frontend.mimi_cuda_graph)
        self.decoder = StreamingDecoder(self.device, graphed=p.frontend.mimi_cuda_graph)
        self.silence_codes = [int(x) for x in VP_AGENT_SILENCE]
        self.warmup()
        return {"cap_limits": self.cap_limits, "arm": dict(self.thinker.arm_info), "rings": self.ring_capacities}

    def warmup(self):
        """Everything that compiles or captures runs here, before any live tick: the first compiled prime, then the
        talker (its lazily compiled kernels first, as in the reference decode process), then the two Mimi."""
        from duplexaudio.serving.realtime.codec import encoder_ring_capacities
        self.require_gpu_thread()
        p = self.prof
        self._arm()                                              # first compiled prime (+ purge re-prefill)
        self.talker.warmup(seed=0)
        z = np.zeros(1920, dtype=np.float32)
        for _ in range(2):
            self.encoder.encode(z)
            self.decoder.decode(torch.tensor(self.silence_codes, device=self.device))
        self.encoder.reset()
        self.decoder.reset()
        self.ring_capacities = {"encoder": sorted(set(encoder_ring_capacities(self.encoder.mimi))),
                                "decoder_encoder_side": sorted(set(encoder_ring_capacities(self.decoder.mimi))),
                                "talker_context": int(self.talker_cfg["moshi_lm"]["context"])}
        if self.ring_capacities["encoder"] != [int(p.frontend.mimi_ring_capacity)]:
            raise RuntimeError(f"encoder ring capacities {self.ring_capacities['encoder']} != profile's "
                               f"{p.frontend.mimi_ring_capacity}")
        self.cap_limits = session_cap(p, talker_context=self.ring_capacities["talker_context"],
                                      prefix_frames=self.talker.P, prompt_len=self.thinker.prompt_len)
        gc.collect()
        gc.freeze()
        self.ready = True

    def _arm(self):
        """Clean thinker context, with the two checks a re-prefill needs: the persona file still matches its pin, and
        the prefill's expert-weight gather fits in the device's free memory plus this process's cached blocks. Any
        failure takes the engine out of service (an engine.prime failure has already dropped the compiled thinker).
        A re-arm records its memory (`arm_info["mem_gb"]`, in each session's meta): reserved memory that does not
        grow across the re-prefill shows the transient was served from the cache."""
        p = self.prof.thinker
        persona = repo_path(p.persona)
        got = hashlib.md5(persona.read_bytes()).hexdigest()
        if got != p.persona_md5:
            raise RuntimeError(f"persona {p.persona} md5 {got} != pinned {p.persona_md5}")
        mem = None
        if self.device.type == "cuda" and self.thinker.prompt_len is not None:   # re-arms (the load-time one is exclusive)
            # The prefill's transient can reuse what this process's caching allocator already holds (the previous
            # prefill's blocks stay reserved), so count that too: device-free memory alone can refuse a re-arm that
            # fits (e.g. when another job shares the GPU).
            free = torch.cuda.mem_get_info(self.device)[0] / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            cached = reserved - torch.cuda.memory_allocated(self.device) / 1e9
            need = PREFILL_GB_PER_TOKEN * self.thinker.prompt_len + PREFILL_MARGIN_GB
            mem = {"free_gb": round(free, 1), "cached_gb": round(cached, 1), "need_gb": round(need, 1),
                   "reserved_gb_before": round(reserved, 1)}
            if free + cached < need:
                raise RuntimeError(f"not enough free GPU memory to re-prefill the persona: {free:.1f} GB free on the "
                                   f"device + {cached:.1f} GB cached by this process, ~{need:.1f} GB needed "
                                   f"(is another job on the GPU?)")
        try:
            info = self.thinker.arm(persona)
        except Exception:
            self.ready = False
            raise
        if mem is not None:
            mem["reserved_gb_after"] = round(torch.cuda.memory_reserved(self.device) / 1e9, 1)
            info["mem_gb"] = mem                          # info is the thinker's arm_info: the session meta's "arm"
        self.arms += 1
        return info

    # ── conversations (GPU thread) ───────────────────────────────────────────────────────────────────────────
    def cap_frames(self) -> int:
        return min(self.cap_limits.values())

    def new_session(self, name: str, seed: int | None = None, *, record=True, rec_dir=None) -> DuplexSession:
        """Clean context + per-conversation seeds + talker prefix (~1 s). Refuses while another conversation is
        active (finalize it first)."""
        self.require_gpu_thread()
        if not self.ready:
            raise RuntimeError("engine not ready (not loaded, or taken out of service by a failed re-arm)")
        if self._active is not None:
            raise RuntimeError(f"conversation {self._active.name!r} is still active: finalize it first")
        seed = self.prof.session.seed if seed is None else int(seed)
        seeds = session_seeds(name, seed)
        self._arm()
        seeds["thinker_row"] = self.thinker.begin(name, seed)
        floor = RoomToneFloor(self.prof.frontend.floor_bands, seed=seeds["floor"])
        self.encoder.reset()
        self.decoder.reset()
        self.feeder.reset()
        self.talker.start(seeds["talker"])                     # last: seeds the default generator, runs the prefix
        if self.device.type == "cuda":
            # The prefix is launched asynchronously; unsynced, its tail can run inside tick 0's wall time (the first
            # ticks measured 85-88 ms while every component stayed <= 28 ms). No model sees an extra frame.
            torch.cuda.synchronize(self.device)
        rd = (Path(rec_dir) if rec_dir is not None else self.rec_root / f"{name}_s{seed}") if record else None
        sess = DuplexSession(self, name=name, seeds=seeds, floor=floor, rec_dir=rd, cap_frames=self.cap_frames())
        self._active = sess
        return sess

    # ── gates and provenance ─────────────────────────────────────────────────────────────────────────────────
    def talker_input_gate(self, sid, fed, forced_ch0) -> dict:
        p = self.prof.talker
        return talker_input_gate(sid, fed, forced_ch0, cfg=self.talker_cfg, spec=self.spec, vp_codes=self.vp_codes,
                                 vp_text=self.vp_text, canon=self.canon_table, canon_mode=p.canon_mode)

    def session_provenance(self) -> dict:
        p = self.prof
        return {"profile": p.name, "profile_md5": realized_md5(p), "profile_file_md5": p.md5,
                "profile_overrides": list(p.overrides), "profile_realized": p.to_json(),
                "md5_check": dict(self.md5_check), "code_gate": dict(self.code_gate), "env": self.fingerprint(),
                "cap_limits": dict(self.cap_limits), "arms_in_process": self.arms,
                "talker": {"session": self.talker.session_mode, "gen_builds": self.talker.gen_builds,
                           "sessions_in_process": self.talker.sessions_started}}

    @staticmethod
    def git_state() -> dict:
        return git_state(paths.REPO_ROOT)

    def fingerprint(self) -> dict:
        import moshi
        import transformers
        fp = {"torch": torch.__version__, "moshi": getattr(moshi, "__version__", None),
              "transformers": transformers.__version__, "device": str(self.device),
              "tf32": {"cudnn": torch.backends.cudnn.allow_tf32, "matmul": torch.backends.cuda.matmul.allow_tf32},
              "env": {k: os.environ.get(k) for k in ("NO_TORCH_COMPILE", "NO_CUDA_GRAPH", "PYTORCH_CUDA_ALLOC_CONF",
                                                     "DUPLEX_RUNTIME_ROOT")},
              "model_root": str(paths.MODEL_ROOT), "rings": dict(self.ring_capacities)}
        if self.device.type == "cuda":
            fp["gpu"] = torch.cuda.get_device_name(self.device)
        try:
            from moshi.utils import compile as mc
            fp["moshi_flags"] = {k: getattr(mc, k) for k in ("_compile_disabled", "_disable_cuda_graph") if hasattr(mc, k)}
        except Exception:
            pass
        return fp
