"""Talker audio sampler with optional common random numbers (CRN).

``LMGenX(LMGen)`` overrides ``depformer_step`` only (text and the main transformer are moshi's). Noise modes:

  legacy : moshi's ``sample_token`` per codebook on torch's global RNG. With scalar settings this is byte-identical to
           moshi's ``LMGen`` (this is the mode the realtime talker serves).
  crn    : counter-based noise, independent of torch's RNG and of the batch composition. For row b, session step s,
           codebook k and vocab id v:
               x = Philox4x32-10(key = (seed, crc32(item_id_b)), counter = (rollout_b, s, k, v // 4))[v % 4]
               u_v = ((x >> 8) + 0.5) / 2^24,   E_v = -log(u_v)
           and the token is the exponential race argmax_v p_v / E_v over the sampler's support, p = softmax(logits/T_k).
           moshi's multinomial is the same race (so the distribution is the same), but there the noise is indexed by
           top-k RANK and drawn from a sequential RNG. Indexing it by VOCAB ID and by item couples two models' draws
           (Gumbel-max coupling): paired arms on the same item share their randomness, which is what CRN pairing needs.

Support per codebook: top-p if top_p > 0, else top-k if top_k > 0, else the full vocab (moshi's precedence). T, top_k and
top_p take a scalar or one value per codebook. temp <= 0 or use_sampling=False is greedy in both modes.
Forcing: ``set_force_buffers`` then ``force_mask`` / ``force_vals`` (B, dep_q). A forced codebook's token replaces the
sample AND conditions the next codebook (as teacher forcing did in training).
Sampler log-probs (``logprobs=True``): ``last_logprobs`` (B, dep_q) = log pi(token) of the latest step under the sampler
distribution (support-renormalised softmax(logits/T_k)), i.e. l_a/T_k - logsumexp_support(l/T_k). Greedy: 0 for the
argmax token, -inf for any other (forced) token. A forced codebook gets the log-prob the sampler would have given the
forced token (-inf outside the support). Steps written through depformer_replace_tokens (no depformer) are NaN.
Read it in on_audio_hook (called after the step's tokens are final).
Graph-safe: every per-step input is a fixed buffer updated in place (step counter, keys, force and log-prob buffers);
moshi CUDA-graphs ``depformer_step`` on CUDA (NO_CUDA_GRAPH=1 runs it eagerly). cfg_coef must be 1: this depformer
step has no CFG branch. set_force_buffers / set_crn are refused while streaming (the session's graph holds the buffers).
crn keys are (seed, item, rollout): a caller that re-voices the same item and rollout and wants FRESH noise (e.g. a
training loop's next iteration) must change the seed.
"""
from __future__ import annotations

import zlib

import torch
from moshi.models.lm import LMGen
from moshi.utils.sampling import sample_token

NOISE_MODES = ("legacy", "crn")
MASK32 = 0xFFFFFFFF
_M0, _M1 = 0xD2511F53, 0xCD9E8D57          # Philox4x32 multipliers
_W0, _W1 = 0x9E3779B9, 0xBB67AE85          # Weyl key increments


def _mulhilo32(a, m):
    """(hi, lo) 32-bit halves of a * m for uint32 values held in int64 (16-bit limbs: no int64 overflow)."""
    a0, a1 = a & 0xFFFF, a >> 16
    m0, m1 = m & 0xFFFF, m >> 16
    p00, p01, p10, p11 = a0 * m0, a0 * m1, a1 * m0, a1 * m1
    mid = (p00 >> 16) + (p01 & 0xFFFF) + (p10 & 0xFFFF)
    lo = ((mid & 0xFFFF) << 16) | (p00 & 0xFFFF)
    hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16)
    return hi, lo


def philox4x32(c0, c1, c2, c3, k0, k1, rounds=10):
    """Philox4x32-R (Salmon et al. 2011, Random123) on int64 tensors holding uint32 values (broadcastable).
    Returns the 4 output words."""
    for _ in range(rounds):
        hi0, lo0 = _mulhilo32(c0, _M0)
        hi1, lo1 = _mulhilo32(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + _W0) & MASK32
        k1 = (k1 + _W1) & MASK32
    return c0, c1, c2, c3


def crn_uniform(k0, k1, rollout, step, cb, V):
    """(B, V) float32 uniforms in (0, 1). k0 = seed, k1 = crc32(item id), rollout: (B,) int64; step: (B,) or (1,) int64;
    cb: codebook index (int). Each value depends only on its own (k0, k1, rollout, step, cb, v)."""
    nblk = (V + 3) // 4
    blk = torch.arange(nblk, device=k0.device, dtype=torch.long)[None, :]
    B = k0.shape[0]
    c0 = rollout.view(-1, 1).expand(B, nblk)
    c1 = step.view(-1, 1).expand(B, nblk)
    c2 = torch.full_like(c0, int(cb))
    x = torch.stack(philox4x32(c0, c1, c2, blk.expand(B, nblk), k0.view(-1, 1), k1.view(-1, 1)), dim=-1)
    x = x.reshape(B, nblk * 4)[:, :V]
    return ((x >> 8).float() + 0.5) * (2.0 ** -24)


def item_key(item_id):
    """The per-item CRN key: crc32 of the item id (str)."""
    return zlib.crc32(str(item_id).encode()) & MASK32


def support_mask(probs, top_k, top_p):
    """(B, V) bool support of the sampler for probs = softmax(logits / T), moshi's rules and precedence:
    top-p (sorted cumulative mass, a token is kept while the mass BEFORE it is <= p), else top-k (torch.topk), else all."""
    if top_p > 0.0:
        ps, pi = torch.sort(probs, dim=-1, descending=True)
        keep = (ps.cumsum(-1) - ps) <= top_p
        return torch.zeros_like(probs, dtype=torch.bool).scatter(-1, pi, keep)
    if top_k > 0:
        idx = torch.topk(probs, min(int(top_k), probs.shape[-1]), dim=-1).indices
        return torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, True)
    return torch.ones_like(probs, dtype=torch.bool)


def crn_sample(logits, u, temp, top_k, top_p):
    """Exponential-race sample with given uniforms. logits, u: (B, V). Returns (B,) long."""
    probs = torch.softmax(logits.float() / temp, dim=-1)
    sup = support_mask(probs, top_k, top_p)
    race = torch.where(sup, probs / (-torch.log(u)), torch.full_like(probs, -1.0))
    return race.argmax(dim=-1)


def sampler_logprob(logits, tok, temp, top_k, top_p):
    """log pi(tok) under the sampler distribution (support-renormalised softmax(logits/T)); (B,) float32."""
    lt = logits.float() / temp
    sup = support_mask(torch.softmax(lt, dim=-1), top_k, top_p)
    lp = torch.log_softmax(torch.where(sup, lt, torch.full_like(lt, float("-inf"))), dim=-1)
    return lp.gather(-1, tok.view(-1, 1)).view(-1)


def _per_cb(x, n, name, cast):
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"{name}: {len(x)} values for {n} codebooks")
        return [cast(v) for v in x]
    return [cast(x)] * n


class LMGenX(LMGen):
    """moshi LMGen with a CRN / legacy audio sampler, per-codebook settings, codebook forcing and sampler log-probs."""

    def __init__(self, lm_model, *, noise="legacy", temp=0.8, top_k=250, top_p=0.0, logprobs=False, cfg_coef=1.0,
                 **kw):
        if noise not in NOISE_MODES:
            raise ValueError(f"noise {noise!r} not in {NOISE_MODES}")
        if cfg_coef != 1.0:
            raise ValueError("LMGenX: cfg_coef must be 1 (its depformer step has no CFG branch)")
        dq = lm_model.dep_q
        temps, top_ks, top_ps = _per_cb(temp, dq, "temp", float), _per_cb(top_k, dq, "top_k", int), \
            _per_cb(top_p, dq, "top_p", float)
        super().__init__(lm_model, temp=temps[0], top_k=top_ks[0], cfg_coef=1.0, **kw)
        self.noise, self.temps, self.top_ks, self.top_ps = noise, temps, top_ks, top_ps
        self.logprobs = bool(logprobs)
        self.force_mask = self.force_vals = self.last_logprobs = None
        self._crn = None

    def _refuse_while_streaming(self, what):
        if self._streaming_state is not None:
            raise RuntimeError(f"{what} while streaming: set it before entering streaming()")

    def set_force_buffers(self, batch, device):
        self._refuse_while_streaming("set_force_buffers")
        self.force_mask = torch.zeros(batch, self.lm_model.dep_q, dtype=torch.bool, device=device)
        self.force_vals = torch.zeros(batch, self.lm_model.dep_q, dtype=torch.long, device=device)

    def set_crn(self, seed, item_ids, rollouts=None, device="cpu"):
        """Key the CRN noise for one session: seed (0 <= seed < 2^32), one item id per row, rollout index per row
        (default 0). Call before entering streaming; resets the step counter."""
        if self.noise != "crn":
            raise ValueError("set_crn needs noise='crn'")
        self._refuse_while_streaming("set_crn")
        if not 0 <= int(seed) <= MASK32:
            raise ValueError(f"seed {seed} outside [0, 2^32)")
        B = len(item_ids)
        rollouts = [0] * B if rollouts is None else list(rollouts)
        if len(rollouts) != B:
            raise ValueError("one rollout index per row")
        self._crn = {"k0": torch.full((B,), int(seed), dtype=torch.long, device=device),
                     "k1": torch.tensor([item_key(i) for i in item_ids], dtype=torch.long, device=device),
                     "r": torch.tensor([int(r) & MASK32 for r in rollouts], dtype=torch.long, device=device),
                     "step": torch.zeros(1, dtype=torch.long, device=device)}

    def _init_streaming_state(self, batch_size):
        if self.noise == "crn":
            if self._crn is None or self._crn["k0"].shape[0] != batch_size:
                raise RuntimeError("noise='crn': call set_crn(seed, item_ids, ...) with one id per row before streaming")
            self._crn["step"].zero_()
        if self.logprobs:
            self.last_logprobs = torch.zeros(batch_size, self.lm_model.dep_q, device=self.lm_model.device)
        return super()._init_streaming_state(batch_size)

    @torch.no_grad()
    def _step(self, input_tokens, depformer_replace_tokens=None):
        if self.logprobs and depformer_replace_tokens is not None:
            self.last_logprobs.fill_(float("nan"))         # no depformer this step: no sampler log-prob
        out = super()._step(input_tokens, depformer_replace_tokens)
        if self._crn is not None:
            self._crn["step"].add_(1)                      # in place: the CUDA graph keeps reading this buffer
        return out

    def depformer_step(self, text_token, transformer_out):
        lm = self.lm_model
        B, = text_token.shape
        prev, toks = text_token, []
        with lm.depformer.streaming(B):
            for k in range(lm.dep_q):
                logits = lm.forward_depformer(k, prev[:, None, None], transformer_out)   # (B, 1, 1, card)
                T, tk, tp = self.temps[k], self.top_ks[k], self.top_ps[k]
                greedy = not self.use_sampling or T <= 0.0
                if self.noise == "legacy" or greedy:
                    samp = sample_token(logits.float(), self.use_sampling, T, tk, tp)[:, 0, 0]
                else:
                    lg = logits[:, 0, 0]
                    c = self._crn
                    samp = crn_sample(lg, crn_uniform(c["k0"], c["k1"], c["r"], c["step"], k, lg.shape[-1]), T, tk, tp)
                nt = samp
                if self.force_mask is not None:
                    nt = torch.where(self.force_mask[:, k], self.force_vals[:, k], samp)
                if self.logprobs:
                    if greedy:
                        lp = torch.where(nt == samp, 0.0, float("-inf")).float()
                    else:
                        lp = sampler_logprob(logits[:, 0, 0], nt, T, tk, tp)
                    self.last_logprobs[:, k].copy_(lp)
                toks.append(nt)
                prev = nt
        return torch.stack(toks, dim=1)

    def describe(self):
        """Provenance string for protocol.json."""
        s = {"legacy": "legacy (moshi sample_token per codebook on torch's global RNG)",
             "crn": "crn (Philox4x32-10, key (seed, crc32(item id)), counter (rollout, step, codebook, vocab id // 4); "
                    "exponential race over vocab ids)"}[self.noise]
        return {"mode": s, "temp": self.temps, "top_k": self.top_ks, "top_p": self.top_ps}
