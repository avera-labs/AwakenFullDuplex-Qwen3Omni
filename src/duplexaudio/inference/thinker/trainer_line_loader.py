"""Thinker checkpoint loader for OmniThinkerEngine.

A trainer-line checkpoint (the dual-channel thinker, e.g. the shipped checkpoints/thinker/model.pt) stores only the
trained parts on top of the frozen Qwen3-Omni base: {pin, pout (the 17 reserved-row embeddings / head rows),
proj.weight (the user/agent fusion), user_codebook_emb / user_in_proj (the user-audio front end), lora.* (attention
LoRA)}.

This wraps such a checkpoint into the engine's model interface (backbone / embed_tokens / codes_to_uemb / fuse /
agent_head), so the engine and its EventFeed state machine drive it; the turn-taking grammar lives once, in
EventFeed.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from duplexaudio.models.thinker.frontend import FrontEnd
from duplexaudio.models.thinker.lora import inject_lora
from duplexaudio.models.thinker.processing_thinker import OMNI_MODEL_ID

ROW0, NROWS = 151676, 17


def _load_sd(ckpt):
    return ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location="cpu")


def is_trainer_line(ckpt) -> bool:
    """A trainer-line ckpt = reserved-row `pin` + LoRA deltas."""
    try:
        sd = _load_sd(ckpt)
    except Exception:
        return False
    return isinstance(sd, dict) and "pin" in sd and any(str(k).startswith("lora.") for k in sd)


class _AgentHead:
    """Callable content-logits head with a `.weight` (the engine reads agent_head.weight for the
    vocab size). Splices the 17 trained reserved rows (pout) into the frozen lm_head output."""
    def __init__(self, lm_head, pout):
        self._lm_head = lm_head
        self.pout = pout
        self.weight = lm_head.weight

    def __call__(self, h):
        base = self._lm_head(h)
        pl = h @ self.pout.to(h.dtype).t()
        return torch.cat([base[..., :ROW0], pl, base[..., ROW0 + NROWS:]], -1)


class _TrainerLineModel:
    """Adapts a trainer-line dual-channel model to the OmniThinkerEngine model interface."""
    def __init__(self, lm, fe, pin, pout, proj, dt, lora_rank=None):
        self._lm = lm                                     # kept for in-place swaps (load_trainer_line_into)
        self.lora_rank = lora_rank                        # ALLOCATED LoRA rank (>= every ckpt loaded into it)
        self.backbone = lm.model
        self.embed_tokens = lm.model.embed_tokens
        self.agent_head = _AgentHead(lm.lm_head, pout)
        self._fe = fe
        self._pin = pin
        self._proj = proj
        self._dt = dt

    def codes_to_uemb(self, uc):                          # (B,8) mimi codes -> (B,H)
        return self._fe(uc.long()).to(self._dt)

    def fuse(self, u_emb, fed):                           # u_emb (B,1,H), fed (B,1) -> (B,1,H)
        e = self.embed_tokens(fed)
        m = (fed >= ROW0) & (fed < ROW0 + NROWS)
        if m.any():
            e = e.clone(); e[m] = self._pin[fed[m] - ROW0].to(e.dtype)   # reserved rows via trained pin
        return self._proj(torch.cat([u_emb, e], -1).float()).to(u_emb.dtype)


_DENSE_KEYS = ("pin", "pout", "proj.weight", "user_codebook_emb.weight", "user_in_proj.weight", "user_in_proj.bias")


def ckpt_lora_rank(sd) -> int:
    """LoRA rank of a trainer-line state dict (rows of any `lora.*.A.weight`)."""
    return int(next(v.shape[0] for k, v in sd.items() if k.startswith("lora.") and ".A." in k))


@torch.no_grad()
def _copy_lora_padded(dst, src):
    """Copy a rank-r LoRA factor into a rank-R (R >= r) allocated parameter IN PLACE, zero-padding the rest.
    A: (r, in) -> rows [:r]; B: (out, r) -> cols [:r]. With alpha = 2*rank the scale alpha/r = 2 at ANY rank,
    so B_pad @ A_pad == B @ A exactly: a rank-16 ckpt in a rank-32 allocation computes the same delta."""
    src = src.to(dst.device, dst.dtype)
    if dst.shape == src.shape:
        dst.copy_(src); return
    if dst.shape[1] == src.shape[1] and dst.shape[0] > src.shape[0]:        # A factor
        dst.zero_(); dst[:src.shape[0]].copy_(src); return
    if dst.shape[0] == src.shape[0] and dst.shape[1] > src.shape[1]:        # B factor
        dst.zero_(); dst[:, :src.shape[1]].copy_(src); return
    raise ValueError(f"LoRA factor shape {tuple(src.shape)} cannot pad into {tuple(dst.shape)}")


def build_trainer_line_model(ckpt, *, device: str = "cuda", dtype=torch.bfloat16, lora_rank=None) -> _TrainerLineModel:
    """`lora_rank` (opt-in): ALLOCATE the injected LoRA at this rank (>= the ckpt's) and zero-pad the ckpt
    into it: numerically the same model (see _copy_lora_padded) but leaves room to later swap IN PLACE to a
    higher-rank ckpt via load_trainer_line_into. None = the ckpt's own rank (default)."""
    sd = _load_sd(ckpt)
    if torch.cuda.get_device_capability()[0] >= 10:
        from transformers.integrations import moe as _moe
        _moe._can_use_grouped_mm = lambda *x, **k: False
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    th = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(OMNI_MODEL_ID, dtype=dtype)
    lm = th.model.language_model if hasattr(th.model, "language_model") else th
    if not hasattr(lm, "lm_head"):
        lm = th
    lm.to(device).eval()
    for p in th.parameters():
        p.requires_grad_(False)
    hidden = lm.lm_head.weight.shape[1]

    fe = FrontEnd(hidden).to(device).to(dtype)
    fe.user_codebook_emb.weight.data.copy_(sd["user_codebook_emb.weight"].to(dtype))
    fe.user_in_proj.weight.data.copy_(sd["user_in_proj.weight"].to(dtype))
    fe.user_in_proj.bias.data.copy_(sd["user_in_proj.bias"].to(dtype))
    fe.eval()

    pin = sd["pin"].to(device).float()
    pout = sd["pout"].to(device).float()
    proj = nn.Linear(2 * hidden, hidden, bias=False).to(device).to(torch.float32)
    proj.weight.data.copy_(sd["proj.weight"].to(device).float())

    r = ckpt_lora_rank(sd)                                    # LoRA rank of this ckpt
    r_alloc = r if lora_rank is None else int(lora_rank)
    if r_alloc < r:
        raise ValueError(f"lora_rank={r_alloc} < ckpt rank {r}")
    inject_lora(lm.model, {"q_proj", "k_proj", "v_proj", "o_proj"}, r_alloc, r_alloc * 2, 0.05, dtype, device)
    named = dict(lm.named_parameters())
    for k, v in sd.items():
        if k.startswith("lora."):
            _copy_lora_padded(named[k[5:]].data, v)           # == plain copy when r_alloc == r
    lm.eval()   # re-assert eval AFTER inject_lora: the newly-inserted LoRALinear (nn.Dropout) modules
                # were not on the tree when .eval() first ran, so they'd otherwise keep dropout ON at inference
    return _TrainerLineModel(lm, fe, pin, pout, proj, dtype, lora_rank=r_alloc)


@torch.no_grad()
def load_trainer_line_into(model: _TrainerLineModel, ckpt) -> dict:
    """Swap a trainer-line ckpt into an ALREADY-BUILT model IN PLACE (no 30B base reload; every parameter
    keeps its address, so a captured cudagraph stays valid). VALIDATES everything before copying anything:
    the dense keys + exactly the injected LoRA set, matching shapes, ckpt rank <= the allocated rank.
    The caller must reset()+prime() a fresh conversation afterwards. Returns {rank, n_lora, alloc_rank}."""
    sd = _load_sd(ckpt)
    lm = model._lm
    named = dict(lm.named_parameters())
    want_lora = {k for k in named if (".A." in k or ".B." in k) and k.endswith(".weight")
                 and any(t in k for t in (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj."))}
    have_lora = {k[5:] for k in sd if k.startswith("lora.")}
    missing = [k for k in _DENSE_KEYS if k not in sd]
    if missing or have_lora != want_lora:
        raise ValueError(f"not a compatible trainer-line ckpt: missing dense {missing}, "
                         f"lora missing {len(want_lora - have_lora)} / unexpected {len(have_lora - want_lora)}")
    r = ckpt_lora_rank(sd)
    if model.lora_rank is not None and r > model.lora_rank:
        raise ValueError(f"ckpt LoRA rank {r} > allocated rank {model.lora_rank} (load the engine with lora_rank>={r})")
    fe = model._fe
    dense = {"pin": model._pin, "pout": model.agent_head.pout, "proj.weight": model._proj.weight,
             "user_codebook_emb.weight": fe.user_codebook_emb.weight,
             "user_in_proj.weight": fe.user_in_proj.weight, "user_in_proj.bias": fe.user_in_proj.bias}
    for k, dst in dense.items():
        if tuple(sd[k].shape) != tuple(dst.shape):
            raise ValueError(f"{k}: ckpt shape {tuple(sd[k].shape)} != model {tuple(dst.shape)}")
    for k in have_lora:                                       # shape check (pad-compatible) before any write
        dst, src = named[k], sd["lora." + k]
        ok = (dst.shape == src.shape or (dst.shape[1] == src.shape[1] and dst.shape[0] >= src.shape[0])
              or (dst.shape[0] == src.shape[0] and dst.shape[1] >= src.shape[1]))
        if not ok:
            raise ValueError(f"{k}: LoRA shape {tuple(src.shape)} incompatible with {tuple(dst.shape)}")
    ptrs = {k: t.data_ptr() for k, t in dense.items()}
    for k, dst in dense.items():
        dst.data.copy_(sd[k].to(dst.device, dst.dtype))
    for k in have_lora:
        _copy_lora_padded(named[k].data, sd["lora." + k])
    assert all(dense[k].data_ptr() == p for k, p in ptrs.items()), "swap moved a tensor (cudagraph would break)"
    lm.eval(); fe.eval()                                      # never let a swap re-enable LoRA dropout
    assert not any(m.training for m in lm.modules()), "a module is in training mode after swap"
    return {"rank": r, "alloc_rank": model.lora_rank, "n_lora": len(have_lora)}
