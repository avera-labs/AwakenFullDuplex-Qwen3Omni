"""Batched autoregressive decode over a HF-style causal backbone with KV cache.

It owns ONLY the transformer plumbing, nothing task-specific (the thinker engine's eager path uses it):

  * left-padded batched prefill of variable-length prompts (attention_mask + cumsum position_ids)
  * one batched forward PER STEP with incremental KV cache
  * correct position_ids for left-padded rows (base = true prompt length, advance by step index)

Everything task-specific stays in the CALLER: how the next input embedding is built (plain token
embed, or a fused multi-channel embedding), how the next token is chosen (greedy / sample / a state
machine), what gets recorded, and when a sequence stops. The caller drives the loop; the engine just
does prefill() once then step() per frame.

Equivalent to per-sequence decoding: with greedy, a B>1 batch reproduces each sequence's solo trajectory
frame-for-frame; the only residual is bf16/MoE logit noise that does not flip decisions.

--- Plain-token Omni batch generation (the common case) --------------------------------------------
    from duplexaudio.inference.decode import batched_generate
    tok = ...; model = ...                       # any HF causal LM with .model + .lm_head (or pass parts)
    outs = batched_generate(
        backbone=model.model, embed=model.model.embed_tokens, head=model.lm_head,
        prompts=[tok.encode(x) for x in texts], pad_id=tok.pad_token_id,
        max_new=128, temperature=0.8, stop_ids={tok.eos_token_id})
    for ids in outs: print(tok.decode(ids))

--- Custom per-frame control (duplex / fused channels / state machines) ----------------------------
    dec = BatchedDecoder(backbone, dev, pad_id)
    dec.prefill(prompts, embed)                  # one batched prefill
    for f in range(max_frames):
        e_in = build_inputs_embeds(f, last_tokens)   # (B,1,H) — caller's channel fusion
        h = dec.step(e_in)                           # (B,H) last hidden state
        logits = head(h)                             # caller owns head + sampling + state + stop
        last_tokens = choose(logits)
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch


class BatchedDecoder:
    """Owns prefill + per-step batched forward + KV cache + left-pad position bookkeeping.

    Stateless about tokens: the caller supplies inputs_embeds each step and consumes the returned
    hidden states however it likes. One instance decodes one batch; call prefill() then step()*.
    """

    def __init__(self, backbone, dev, pad_id: int):
        self.backbone = backbone
        self.dev = dev
        self.pad = pad_id
        self.kv = None
        self.amask: torch.Tensor | None = None
        self.base_pos: torch.Tensor | None = None   # (B,) true prompt length per row
        self.step_idx = 0
        self.B = 0

    @torch.no_grad()
    def prefill(self, prompts: Sequence[Sequence[int]], embed: Callable) -> int:
        """Left-pad `prompts` (a list of token-id lists), run the batched prefill, store the KV cache.
        `embed` maps a (B,L) LongTensor of ids to (B,L,H) embeddings. Returns batch size B."""
        B = len(prompts)
        Lmax = max(len(p) for p in prompts)
        pids = torch.full((B, Lmax), self.pad, dtype=torch.long, device=self.dev)
        amask = torch.zeros((B, Lmax), dtype=torch.long, device=self.dev)
        for b, p in enumerate(prompts):
            L = len(p)
            pids[b, Lmax - L:] = torch.tensor(p, dtype=torch.long, device=self.dev)
            amask[b, Lmax - L:] = 1
        self.base_pos = amask.sum(-1)                                  # real length ignores left pad
        pos = (amask.cumsum(-1) - 1).clamp(min=0)                      # left-pad rows start at pos 0
        o = self.backbone(inputs_embeds=embed(pids), position_ids=pos,
                          attention_mask=amask, use_cache=True)
        self.kv = o.past_key_values
        self.amask = amask
        self.B = B
        self.step_idx = 0
        return B

    @torch.no_grad()
    def step(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Advance one frame. `inputs_embeds` is (B,1,H). Returns the last hidden state (B,H).
        position_ids continue from each row's true prompt length so left-padded rows stay aligned."""
        if self.kv is None:
            raise RuntimeError("call prefill() before step()")
        self.amask = torch.cat(
            [self.amask, torch.ones((self.B, 1), dtype=torch.long, device=self.dev)], dim=1)
        pos = (self.base_pos + self.step_idx).unsqueeze(1)            # (B,1)
        o = self.backbone(inputs_embeds=inputs_embeds, position_ids=pos,
                          attention_mask=self.amask, past_key_values=self.kv, use_cache=True)
        self.kv = o.past_key_values
        self.step_idx += 1
        return o.last_hidden_state[:, -1, :]


@torch.no_grad()
def batched_generate(backbone, embed: Callable, head: Callable,
                     prompts: Sequence[Sequence[int]], pad_id: int,
                     max_new: int = 128, temperature: float = 0.0,
                     stop_ids: Iterable[int] | None = None,
                     seed: int | None = None) -> list[list[int]]:
    """Plain-token batched generation over any HF-style causal backbone.

    backbone: a module returning `.last_hidden_state` + `.past_key_values` given
              (inputs_embeds, position_ids, attention_mask, past_key_values, use_cache).
    embed:    (B,L) ids -> (B,L,H).   head: (B,H) -> (B,V) logits.
    prompts:  list of token-id lists (variable length; left-padded internally).
    temperature: 0 => greedy argmax; >0 => multinomial sampling.
    stop_ids: tokens that end a sequence; generation past a row's stop is discarded.
    Returns per-prompt lists of generated (new) token ids, stop token excluded.
    """
    dev = next(backbone.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    stop = set(stop_ids or ())
    dec = BatchedDecoder(backbone, dev, pad_id)
    B = dec.prefill(prompts, embed)
    last = torch.tensor([p[-1] for p in prompts], device=dev)         # last prompt tok feeds frame 0
    out: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    for _ in range(max_new):
        h = dec.step(embed(last.unsqueeze(1)))                        # (B,H)
        logits = head(h).float()
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
        else:
            nxt = logits.argmax(-1)
        last = nxt
        ids = nxt.tolist()
        for b in range(B):
            if done[b]:
                continue
            if ids[b] in stop:
                done[b] = True
            else:
                out[b].append(ids[b])
        if all(done):
            break
    return out
