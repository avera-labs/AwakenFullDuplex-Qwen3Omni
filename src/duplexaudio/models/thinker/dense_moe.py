"""Dense-MoE forward for the Qwen3-Omni-30B-A3B thinker.

WHY: the model's native `Qwen3OmniMoeThinkerTextExperts.forward` is a Python `for expert in hit:` loop
(one `F.linear` per active expert, token gather/scatter, `index_add_`). At the 12.5 Hz decode scale the
per-frame batch is TINY (T = B·S ≈ a handful of tokens), so that loop launches ~top_k tiny kernels over
near-empty groups: pure overhead and poor GPU utilization, which matters for the 80 ms real-time frame budget.

WHAT: compute EVERY expert's SwiGLU on ALL tokens with two batched GEMMs (dense), then weight by the
router's top-k (zero elsewhere) and sum. One big dense kernel instead of top_k sequential small matmuls;
no python loop, no gather/scatter. Trades extra FLOPs (E/top_k ≈ 16× — all 128 experts vs top-8) for
kernel efficiency. That trade WINS only at small T (decode); at large T (prefill) the 16× FLOPs lose, so
we fall back to the native loop above a token threshold. Numerics match the native path (softmax-fp32→
topk→norm router unchanged; router scalar applied AFTER down-proj) modulo bf16 accumulation order.

USAGE: `install_dense_moe(model)` (idempotent, class-level monkeypatch). The thinker engine calls it by
default on load. `uninstall_dense_moe()` restores the native loop.
"""
from __future__ import annotations

import torch

_ORIG_EXPERTS_FORWARD = None          # saved native loop, for large-T fallback + uninstall
_DEFAULT_MAX_DENSE_TOKENS = 256       # T above this -> native loop (dense's E/k× FLOPs stop paying off)


def _dense_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Drop-in for Qwen3OmniMoeThinkerTextExperts.forward.
    hidden_states: (T, H); top_k_index/top_k_weights: (T, top_k). Returns (T, H)."""
    T = hidden_states.shape[0]
    max_dense = getattr(self, "_dense_max_tokens", _DEFAULT_MAX_DENSE_TOKENS)
    if _ORIG_EXPERTS_FORWARD is not None and T > max_dense:
        return _ORIG_EXPERTS_FORWARD(self, hidden_states, top_k_index, top_k_weights)   # prefill: loop cheaper

    E = self.num_experts
    # dense router matrix (T, E): scatter the top_k weights into their expert slots, 0 elsewhere
    W = torch.zeros(T, E, dtype=top_k_weights.dtype, device=hidden_states.device)
    W.scatter_(1, top_k_index, top_k_weights)

    # all experts on all tokens (F.linear(x, w) == x @ w.T, matching the native per-expert call)
    #   gate_up_proj: (E, 2I, H)  ->  gu[e,t,:] = x[t] @ gate_up_proj[e].T   -> (E, T, 2I)
    gu = torch.einsum("th,ejh->etj", hidden_states, self.gate_up_proj)
    gate, up = gu.chunk(2, dim=-1)                                   # (E, T, I) each
    h = self.act_fn(gate) * up                                      # SwiGLU, (E, T, I)
    #   down_proj: (E, H, I)      ->  y[e,t,:] = h[e,t] @ down_proj[e].T    -> (E, T, H)
    y = torch.einsum("eti,ehi->eth", h, self.down_proj)
    # combine: router weight applied AFTER down-proj (native order); sum over experts -> (T, H)
    out = torch.einsum("te,eth->th", W.to(y.dtype), y)
    return out.to(hidden_states.dtype)


def install_dense_moe(model=None, *, max_dense_tokens: int = _DEFAULT_MAX_DENSE_TOKENS):
    """Class-level monkeypatch: replace the native per-expert loop with the dense forward (with a large-T
    fallback to the loop). Idempotent. `model` is accepted+returned for call-site convenience but the patch
    is global (all Qwen3OmniMoeThinkerTextExperts instances, i.e. every layer of every loaded thinker)."""
    global _ORIG_EXPERTS_FORWARD
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextExperts)
    if getattr(Qwen3OmniMoeThinkerTextExperts.forward, "_is_dense_moe", False):
        Qwen3OmniMoeThinkerTextExperts._dense_max_tokens = max_dense_tokens
        return model
    _ORIG_EXPERTS_FORWARD = Qwen3OmniMoeThinkerTextExperts.forward
    _dense_experts_forward._is_dense_moe = True
    Qwen3OmniMoeThinkerTextExperts.forward = _dense_experts_forward
    Qwen3OmniMoeThinkerTextExperts._dense_max_tokens = max_dense_tokens
    return model


def uninstall_dense_moe():
    """Restore the native per-expert loop (for A/B / numeric-parity checks)."""
    global _ORIG_EXPERTS_FORWARD
    if _ORIG_EXPERTS_FORWARD is None:
        return
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextExperts)
    Qwen3OmniMoeThinkerTextExperts.forward = _ORIG_EXPERTS_FORWARD
    _ORIG_EXPERTS_FORWARD = None
