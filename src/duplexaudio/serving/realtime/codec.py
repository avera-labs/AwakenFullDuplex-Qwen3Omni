"""Mimi for the live loop: an incremental streaming ENCODER (user channel) and a streaming DECODER (agent channel).

Each is its own Mimi instance: moshi's reset_streaming() resets an instance's encode and decode halves together, and a
Mimi that has ever streamed must never serve a whole-clip call (its 250-slot ring overflows inside one call). One
1920-sample frame per call, always, including while catching up after a stall: a multi-frame call changes both the
ring behaviour and the kernel shapes, so it would change the codes.

Front-end axes (profile `frontend`, all per instance, never process-global):
  mimi_ring_capacity  encoder-transformer ring KV capacity. moshi's default is 250 (= the 250-step context). Each call
                      writes 2 encoder steps before attending, so from Mimi frame 125 (10 s) on the first query of each call
                      loses its oldest in-window key; 251 restores the whole-clip / training encoder exactly (measured on
                      CPU fp32: 0 differing frames vs 3/173 at 250).
  mimi_cuda_graph     False = eager (moshi's no_cuda_graph() around each call); True = moshi graphs.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from duplexaudio.serving.realtime.frontend import SPF

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"


def load_mimi(device):
    """A fresh, never-streamed Mimi (8 codebooks), as the offline encode/decode uses."""
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    m = loaders.get_mimi(hf_hub_download(MIMI_REPO, MIMI_FILE), device=device).eval()
    if hasattr(m, "set_num_codebooks"):
        m.set_num_codebooks(8)
    return m


def _graph_ctx(graphed: bool):
    if graphed:
        return contextlib.nullcontext()
    from moshi.utils.compile import no_cuda_graph
    return no_cuda_graph()


def set_encoder_ring_capacity(mimi, capacity: int) -> int:
    """Replace the ring KV cache of every encoder-transformer attention module of THIS instance by one of `capacity`
    slots (context and mask unchanged). Call after streaming_forever(), before the first encode. Returns the count."""
    from moshi.modules.transformer import RingKVCache, StreamingMultiheadAttention
    n = 0
    for mod in mimi.encoder_transformer.modules():
        if isinstance(mod, StreamingMultiheadAttention):
            st = mod._streaming_state
            if st is None or st.kv_cache is None:
                continue
            old = st.kv_cache
            _, B, H, _, D = old.cache.shape
            st.kv_cache = RingKVCache(B, H, D, int(capacity), respect_exec_mask=old.respect_exec_mask,
                                      device=old.cache.device, dtype=old.cache.dtype)
            n += 1
    if n == 0:
        raise RuntimeError("no streaming encoder-transformer attention found to resize")
    return n


def encoder_ring_capacities(mimi) -> list:
    from moshi.modules.transformer import StreamingMultiheadAttention
    caps = []
    for mod in mimi.encoder_transformer.modules():
        if isinstance(mod, StreamingMultiheadAttention) and mod._streaming_state is not None:
            kv = mod._streaming_state.kv_cache
            if kv is not None:
                caps.append(int(kv.cache.shape[3]))
    return caps


class StreamingEncoder:
    """User audio -> (8,) Mimi codes per 80 ms frame, incremental."""

    def __init__(self, device="cuda", *, ring_capacity=250, graphed=False, mimi=None):
        self.dev = torch.device(device)
        self.graphed = bool(graphed)
        self.mimi = mimi if mimi is not None else load_mimi(self.dev)
        if self.mimi._streaming_state is not None:
            raise RuntimeError("the encoder needs its own never-streamed Mimi instance")
        self.mimi.streaming_forever(1)
        self.ring_capacity = int(ring_capacity)
        if self.ring_capacity != 250:
            set_encoder_ring_capacity(self.mimi, self.ring_capacity)

    def reset(self):
        self.mimi.reset_streaming()

    @torch.no_grad()
    def encode(self, frame: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).view(1, 1, -1)
        if x.shape[-1] != SPF:
            raise ValueError(f"encode takes exactly one {SPF}-sample frame, got {x.shape[-1]}")
        with _graph_ctx(self.graphed):
            c = self.mimi.encode(x.to(self.dev))
        c = c.audio_codes if hasattr(c, "audio_codes") else c
        if c.shape[-1] != 1:
            raise RuntimeError(f"streaming Mimi encode returned T={c.shape[-1]} for one frame")
        return c[0, :8, 0].long()                                       # (8,)


class StreamingDecoder:
    """Agent codes -> int16 PCM, one frame per call, gapless across the whole conversation (frame -1 included)."""

    def __init__(self, device="cuda", *, graphed=False, mimi=None):
        self.dev = torch.device(device)
        self.graphed = bool(graphed)
        self.mimi = mimi if mimi is not None else load_mimi(self.dev)
        if self.mimi._streaming_state is not None:
            raise RuntimeError("the decoder needs its own never-streamed Mimi instance")
        self.mimi.streaming_forever(1)
        self.card = int(getattr(self.mimi, "cardinality", 2048))

    def reset(self):
        self.mimi.reset_streaming()

    @torch.no_grad()
    def decode(self, codes8: torch.Tensor) -> np.ndarray:
        c = codes8.to(self.dev).long().view(1, 8, 1)
        if bool(((c < 0) | (c >= self.card)).any()):
            raise ValueError("agent codes out of the codebook range")
        with _graph_ctx(self.graphed):
            pcm = self.mimi.decode(c)
        x = pcm.detach().float().cpu().numpy().reshape(-1)
        return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
