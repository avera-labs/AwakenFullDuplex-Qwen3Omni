"""`apply_state_machine` (what `engine.step_masked` calls) is MASK-THEN-SAMPLE: it removes the tokens illegal for
the current state BEFORE sampling, so the model samples the best LEGAL token and an illegal token is NEVER produced
(no sample-then-fallback-to-PAD). Legality:
    SPEECH (silent=False): legal = content | pause_m | SS ;   illegal = {PAD, EPAD, SE}
    SILENCE (silent=True): legal = {SE, PAD}              ;   illegal = {content, pause_m, SS, EPAD}
EPAD is illegal to SAMPLE in both states (it is only ever FED via `pending`). The user's turn is NOT force-masked:
`in_user`/`bargein` do not change the mask; yielding is the model's own learned SS decision. Pure CPU, no model."""
from __future__ import annotations

import torch

from duplexaudio.inference.thinker.engine import ThinkerSpec
from duplexaudio.inference.thinker.state_machine import EventFeed, apply_state_machine

PAD, EPAD = 151676, 151677
PAUSE_IDS = tuple(range(151678, 151691))          # 13 pause tokens (m = index)
SS, SE = 151691, 151692
VOCAB = 151693
CONTENT = 100
SPEC = ThinkerSpec(pad_id=PAD, epad_id=EPAD, sil_start_id=SS, sil_end_id=SE,
                   pause_ids=PAUSE_IDS, hidden=8, vocab=VOCAB)


def _onehot(tok: int) -> torch.Tensor:
    lg = torch.full((VOCAB,), -30.0)
    lg[tok] = 30.0
    return lg


def _new_feed() -> EventFeed:
    return EventFeed(PAD, EPAD, {p: i for i, p in enumerate(PAUSE_IDS)}, SS, SE, events_in_input=True)


def _step(feeds, hist, top_tok, in_user=False):
    """One frame: the model's top logit is on `top_tok`. Returns (fed, kind, sampled)."""
    fed, kinds, sampled = apply_state_machine(
        content_logits=_onehot(top_tok).unsqueeze(0), feeds=feeds, hist=hist, spec=SPEC,
        in_user=[in_user], temperature=0.0)
    return int(fed[0]), kinds[0], int(sampled[0])


# ---- legal tokens are produced verbatim + drive the right state transition ----------------------
def test_legal_tokens_produced_verbatim():
    f, h = [_new_feed()], [[]]
    assert _step(f, h, CONTENT) == (CONTENT, "content", CONTENT) and not f[0].silent
    assert _step(f, h, PAUSE_IDS[0])[0] == EPAD                       # pause_0 -> onset EPAD fed
    assert _step(f, h, SS)[:2] == (SS, "ss") and f[0].silent           # yield -> SILENCE
    assert _step(f, h, SE)[:2] == (SE, "se") and not f[0].silent       # resume -> SPEECH (+pending EPAD)
    assert _step(f, h, CONTENT)[:2] == (EPAD, "pend")                  # the queued onset, not sampled


# ---- SILENCE masks everything but {SE, PAD}: an illegal top token never appears ------------------
def test_silence_masks_to_se_or_pad():
    for top in (CONTENT, SS, PAUSE_IDS[2], EPAD):
        f, h = [_new_feed()], [[]]
        _step(f, h, SS)                                               # enter SILENCE
        fed, kind, sampled = _step(f, h, top)
        assert fed in (SE, PAD) and sampled in (SE, PAD), f"SILENCE produced illegal {sampled} (top {top})"
    # PAD keeps silence; SE resumes
    f, h = [_new_feed()], [[]]; _step(f, h, SS)
    assert _step(f, h, PAD)[:2] == (PAD, "pad") and f[0].silent
    assert _step(f, h, SE)[:2] == (SE, "se") and not f[0].silent


# ---- SPEECH masks out {PAD, EPAD, SE}: never fed back, never fallback-to-PAD ---------------------
def test_speech_masks_out_pad_epad_se():
    for top in (PAD, EPAD, SE):
        f, h = [_new_feed()], [[]]
        fed, kind, sampled = _step(f, h, top)
        assert sampled not in (PAD, EPAD, SE), f"SPEECH sampled illegal {sampled} (top {top})"
        assert kind == "content"                                     # picks a legal content token instead


# ---- in_user / bargein do NOT change the mask (no oracle force-mute) ----------------------------
def test_in_user_does_not_change_the_mask():
    for top in (CONTENT, SS):
        outs = []
        for iu in (False, True):
            f, h = [_new_feed()], [[]]
            outs.append(_step(f, h, top, in_user=iu))
        assert outs[0] == outs[1], f"in_user changed SPEECH output for {top}: {outs}"
    # SILENCE + SE while the user talks: the resume is ALLOWED (no premature-exit block)
    for iu in (False, True):
        f, h = [_new_feed()], [[]]; _step(f, h, SS)
        assert _step(f, h, SE, in_user=iu)[:2] == (SE, "se"), f"in_user={iu} wrongly blocked SE resume"


# ---- a pause(m) declaration still expands to PAD*(m-1)+EPAD via pending --------------------------
def test_pause_expansion_fed_row():
    feeds, hist, m = [_new_feed()], [[]], 3
    fed_all = [_step(feeds, hist, PAUSE_IDS[m])[0] for _ in range(m + 1)]
    assert fed_all == [PAD, PAD, PAD, EPAD]      # PAD (decl) + PAD*(m-1) + EPAD onset


# ---- state-dependent temperature: high in SILENCE (break out) / low in SPEECH (stable) ---------------
def _se_rate_in_silence(*, silence_temperature=None, speech_temperature=None, temperature=0.0, trials=400):
    """Enter SILENCE, then sample ONE frame `trials` times (fresh feed, seeded) with PAD>SE logits (both
    legal). Returns the SE (resume) rate — higher temperature flattens PAD>SE so SE fires more often."""
    se = 0
    for i in range(trials):
        torch.manual_seed(i)
        f, h = [_new_feed()], [[]]
        apply_state_machine(content_logits=_onehot(SS).unsqueeze(0), feeds=f, hist=h, spec=SPEC,
                            in_user=[False], temperature=0.0)            # enter SILENCE
        lg = torch.full((VOCAB,), -30.0); lg[PAD] = 1.0; lg[SE] = 0.0    # PAD > SE, both legal in SILENCE
        _, _, sampled = apply_state_machine(
            content_logits=lg.unsqueeze(0), feeds=f, hist=h, spec=SPEC, in_user=[False],
            temperature=temperature, silence_temperature=silence_temperature,
            speech_temperature=speech_temperature)
        se += int(sampled[0] == SE)
    return se / trials


def test_silence_temperature_breaks_out_of_silence():
    assert _se_rate_in_silence(silence_temperature=0.0) == 0.0          # argmax -> always PAD, stuck silent
    assert _se_rate_in_silence(silence_temperature=5.0) > 0.2           # high temp -> SE fires -> resumes


def test_silence_temp_falls_back_to_temperature_and_can_override():
    assert _se_rate_in_silence(temperature=0.0) == 0.0                  # None -> uses temperature (0)
    assert _se_rate_in_silence(temperature=5.0) > 0.2                   # None -> uses temperature (5)
    assert _se_rate_in_silence(temperature=5.0, silence_temperature=0.0) == 0.0  # explicit overrides


def test_speech_temp_is_independent_of_silence_temp():
    """In SPEECH, speech_temperature=0 is deterministic argmax content — a high silence_temperature (for
    the OTHER state) must not leak in."""
    for i in range(50):
        torch.manual_seed(i)
        f, h = [_new_feed()], [[]]
        lg = torch.full((VOCAB,), -30.0); lg[CONTENT] = 5.0; lg[CONTENT + 1] = 4.0
        _, _, sampled = apply_state_machine(
            content_logits=lg.unsqueeze(0), feeds=f, hist=h, spec=SPEC, in_user=[False],
            temperature=0.0, silence_temperature=9.0, speech_temperature=0.0)
        assert int(sampled[0]) == CONTENT      # argmax; SILENCE temp does not affect SPEECH sampling


# ---- per-row sampling RNG: a row's draws do not depend on its batch neighbours or their order -----
def test_per_row_generators_make_rows_order_independent():
    g = torch.Generator().manual_seed(0)
    B, T = 4, 12
    logits = [torch.randn(B, VOCAB, generator=g) * 3 for _ in range(T)]
    def run(perm):
        feeds, hist = [_new_feed() for _ in range(B)], [[] for _ in range(B)]
        gens = [torch.Generator().manual_seed(1000 + int(p)) for p in perm]           # seed = the row's identity
        out = []
        for t in range(T):
            fed, _, sampled = apply_state_machine(content_logits=logits[t][perm], feeds=feeds, hist=hist, spec=SPEC,
                                                  in_user=[False] * B, speech_temperature=1.0, silence_temperature=1.0,
                                                  generators=gens)
            out.append(sampled)
        return [[out[t][j] for t in range(T)] for j in range(B)]                      # per row, over frames
    perm = torch.tensor([2, 0, 3, 1])
    base, shuffled = run(torch.arange(B)), run(perm)
    for j, p in enumerate(perm.tolist()):
        assert shuffled[j] == base[p]
    assert base[0] != base[1]                                                         # rows really sampled (T>0)


def test_generators_none_keeps_global_rng_behaviour():
    lg = (torch.randn(2, VOCAB, generator=torch.Generator().manual_seed(3)) * 3)
    def once():
        torch.manual_seed(7)
        return apply_state_machine(content_logits=lg, feeds=[_new_feed(), _new_feed()], hist=[[], []], spec=SPEC,
                                   in_user=[False, False], speech_temperature=1.0)[2]
    assert once() == once()
