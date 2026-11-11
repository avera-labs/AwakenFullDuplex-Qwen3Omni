"""Talker LM: a Moshi LMModel (moshiko architecture: depformer over 8 agent audio codebooks, per-codebook delays)
whose text input is the thinker's Qwen token stream.

The main transformer keeps moshiko's width (dim 4096) with fewer layers (set by the model config's `moshi_lm`
block). Both Moshi text tables (backbone `text_emb` -> dim, depformer `depformer_text_emb` -> depformer_dim) are
replaced by `ThinkerTextEmb`: an embedding over the thinker's vocabulary (duplex specials included) plus a linear
projection, so the thinker's agent tokens feed the talker with no conversion. Text is a forced input; the talker
samples only the 8 agent codebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from moshi.models import loaders
from moshi.models.lm import LMModel

DEP_Q = 8          # agent audio codebooks predicted by the depformer


def build_talker_lm(moshi_lm_cfg: dict, dtype, device="cuda") -> LMModel:
    """Construct an LMModel from moshiko's defaults overridden by the model config's `moshi_lm` block."""
    kw = dict(loaders._lm_kwargs)
    kw.update({k: v for k, v in (moshi_lm_cfg or {}).items()})
    for k in ("lora", "lora_rank", "lora_scaling", "depformer_causal", "conditioners", "fuser",
              "quantize", "demux_second_stream"):
        kw.pop(k, None)
    return LMModel(device=device, dtype=dtype, **kw)


class ThinkerTextEmb(nn.Module):
    """proj(embed(tok)) over the thinker's vocabulary. Maps Moshi's zero_token_id (-1) to a zero vector."""

    def __init__(self, thinker_emb_path: str, out_dim: int, freeze_embed: bool = True, dtype=torch.bfloat16):
        super().__init__()
        d = torch.load(thinker_emb_path, map_location="cpu")
        w = d["weight"]                                  # (vocab, 2048) float32
        self.vocab, self.in_dim = w.shape
        self.embed = nn.Embedding(self.vocab, self.in_dim)
        with torch.no_grad():
            self.embed.weight.copy_(w)
        self.embed.weight.requires_grad = not freeze_embed
        self.proj = nn.Linear(self.in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=1.0 / (self.in_dim ** 0.5))
        self.to(dtype)

    def forward(self, tok):                              # (...) long -> (..., out_dim)
        # CUDA-graph-capture safe: zero the embedding at token -1 with a mask multiply (no data-dependent branch).
        safe = tok.clamp(min=0)
        out = self.proj(self.embed(safe))
        return out * (tok >= 0).unsqueeze(-1).to(out.dtype)


def install_qwen_text(lm: LMModel, thinker_emb_path: str, dtype, freeze_embed: bool = False) -> None:
    """Replace both Moshi text embeddings with independent ThinkerTextEmb tables."""
    lm.text_emb = ThinkerTextEmb(thinker_emb_path, out_dim=lm.dim, freeze_embed=freeze_embed, dtype=dtype).cuda()
    if getattr(lm, "depformer_text_emb", None) is not None:
        dep_dim = int(lm.depformer_text_emb.weight.shape[1])
        lm.depformer_text_emb = ThinkerTextEmb(thinker_emb_path, out_dim=dep_dim, freeze_embed=freeze_embed,
                                               dtype=dtype).cuda()
