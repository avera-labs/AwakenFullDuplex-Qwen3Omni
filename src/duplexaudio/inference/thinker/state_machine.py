"""Turn-taking STATE MACHINE: the SS/SE legality automaton + the per-frame mask.

The thinker ALWAYS decodes through this state machine; callers use it via the engine's `step_masked` rather than
wiring their own EventFeed loop. A full-duplex model samples at every frame, and each draw decides at once whether
to speak, what to say and whether to pause, so it needs this external structure to keep its samples within the legal
moves.

Two pieces:
  * `EventFeed` — the 2-state automaton (SPEECH / SILENCE over `self.silent`). Given ONE sampled token
    id per frame + `(in_user, bargein)`, returns `(legal_fed_token, kind)`. Owns `silent` + a `pending`
    FIFO (multi-frame pause / post-SE onset). No logits, pure grammar; it mirrors the training-data
    construction of the agent stream exactly.
  * `apply_state_machine` / `_apply_rep` — the per-frame batched MASK the engine's `step_masked` calls:
    sample the content head (temperature + optional rep-control + optional SS/SE bias) and run each
    row's EventFeed. One implementation, shared by the single-stream and batched engines.
"""
from __future__ import annotations

import torch

IGNORE_INDEX = -100

# rep-control defaults
_REP_PENALTY, _NO_REPEAT_NGRAM, _REP_WINDOW = 1.3, 4, 24


class EventFeed:
    """Agent-input generator for the EVENT scheme: pause chunks (short) + SS/SE events (long).

    Per-frame contract (all the state machine needs / produces):
      INPUT  : one sampled token id `a` + `in_user` (user still speaking?) + `bargein` (may speak over?).
      OUTPUT : `(fed_id, kind)` — `fed_id` is the LEGAL token embedded next frame; `kind` is advisory
               bookkeeping ∈ {content, pause, ss, se, se_premature, discard, fallback}.
      STATE  : `self.silent` (inside an SS..SE silence) + `self.pending` (FIFO of pre-committed frames).
    Query `sampling_now()` first; if False, feed `next_pending()` and DON'T sample this frame.
    """

    def __init__(self, pad, epad, pause_frames: dict, ss: int, se: int,
                 events_in_input: bool = False, free_turn: bool = False):
        self.pad, self.epad, self.PAUSE, self.ss, self.se = pad, epad, pause_frames, ss, se
        self.pending: list[int] = []
        self.silent = False                          # inside an SS..SE long silence
        # events_in_input: training INPUTS carry SS/SE at their own frames, so a sampled event is fed
        # back VERBATIM (False: inputs never contain them -> PAD).
        self.events_in_input = events_in_input
        # free_turn: disable the two forced-mute paths so the model's own samples are
        # honored during the user's turn (turn-taking shaped only by SS/SE logit ramps, not oracle mute).
        self.free_turn = free_turn

    def sampling_now(self) -> bool:
        return not self.pending                      # silent mode still samples every frame

    def on_decision(self, a: int, in_user: bool, bargein: bool) -> tuple[int, str]:
        """Returns (input id fed for this frame, kind) — kind in
        content|pause|ss|se|se_premature|discard|fallback.

        events_in_input=False: SS/SE are LABEL-ONLY, never fed back raw (feeding one poisons the context with a
        symbol training inputs never contain). events_in_input=True (the shipped thinker): inputs DO carry them
        at their own frames, so an IN-GRAMMAR event is fed verbatim; off-grammar occurrences still degrade to PAD
        in both variants."""
        if self.silent:
            if a == self.se:
                if in_user and not bargein and not self.free_turn:
                    # PREMATURE exit: training silences span the user's speech, so an SE while the
                    # user is talking contradicts the data. Stay silent (caller counts these).
                    return self.pad, "se_premature"
                self.silent = False
                self.pending = [self.epad]           # "next frame is the onset"
                return self.se if self.events_in_input else self.pad, "se"
            return self.pad, "discard"               # interiors: everything but SE is noise
        if a == self.ss:
            self.silent = True
            return self.ss if self.events_in_input else self.pad, "ss"
        # FORCED MUTE while the user speaks (no barge-in / not free_turn): the agent's channel during a
        # user turn is pure silence in training (SS -> PAD interior -> SE). Everything sampled here
        # (content, a pause DECLARATION, a stray SE) is fed as PAD; only SS/SE (above) are honored.
        if in_user and not bargein and not self.free_turn:
            return self.pad, "discard"
        if a == self.se:
            return self.pad, "fallback"              # SE outside a silence: off-grammar, PAD frame
        if a in self.PAUSE:
            m = self.PAUSE[a]
            if m == 0:
                return self.epad, "pause"
            self.pending = [self.pad] * (m - 1) + [self.epad]
            return self.pad, "pause"
        if a in (self.pad, self.epad):
            return self.pad, "fallback"
        return a, "content"

    def legal_ids(self):
        """Tokens LEGAL to SAMPLE this frame given the current state: PURE SS/SE grammar (the user's turn is
        NOT force-masked; whether to yield is the model's own learned decision from the user-audio channel,
        not an oracle). Returns either:
          ('restrict', [legal ids]) — legal set is exactly this small list (SILENCE), or
          ('mask_out', [illegal special ids]) — legal set is 'all content + specials NOT listed' (SPEECH).
        The caller sets illegal logits to -inf and samples among the legal, so an illegal token is
        NEVER produced (no sample-then-fallback-to-PAD). EPAD is illegal to sample in BOTH states — it
        is only ever FED via `pending` (its training label is always IGNORE)."""
        if self.silent:                                   # SILENCE (yielded): only stay silent or resume
            return ("restrict", [self.se, self.pad])      # legal = {SE, PAD}
        # SPEECH (holds floor): content + pause_m + SS(yield) legal; PAD, EPAD, SE illegal
        return ("mask_out", [self.pad, self.epad, self.se])

    def commit(self, a: int) -> tuple[int, str]:
        """Update state for an ALREADY-LEGAL sampled token `a` (from legal_ids masking) and return
        (fed_id, kind). No fallback branch — every case here is reachable only by a legal token."""
        if a == self.ss:                                  # yield the floor
            self.silent = True
            return (self.ss if self.events_in_input else self.pad), "ss"
        if a == self.se:                                  # resume speaking
            self.silent = False
            self.pending = [self.epad]                    # next frame is the onset
            return (self.se if self.events_in_input else self.pad), "se"
        if a in self.PAUSE:                               # pause declaration (SPEECH only)
            m = self.PAUSE[a]
            if m == 0:
                return self.epad, "pause"
            self.pending = [self.pad] * (m - 1) + [self.epad]
            return self.pad, "pause"
        if a == self.pad:
            return self.pad, "pad"
        return a, "content"

    def next_pending(self) -> int:
        return self.pending.pop(0)


def _apply_rep(lg: torch.Tensor, hist: list, cfg) -> torch.Tensor:
    """In-place repetition control on a (V,) logit row given this row's content history.
    cfg: dict {rep_penalty, no_repeat_ngram, rep_window} (missing keys fall back to the reference
    defaults). Window penalty + n-gram block."""
    rep = float(cfg.get("rep_penalty", _REP_PENALTY))
    nrn = int(cfg.get("no_repeat_ngram", _NO_REPEAT_NGRAM))
    win = int(cfg.get("rep_window", _REP_WINDOW))
    if hist and rep != 1.0:
        idx = torch.tensor(sorted(set(hist[-win:])), device=lg.device)
        l = lg[idx]
        lg[idx] = torch.where(l > 0, l / rep, l * rep)
    if nrn > 1 and len(hist) >= nrn - 1:
        pre = tuple(hist[-(nrn - 1):])
        for i in range(len(hist) - (nrn - 1)):
            if tuple(hist[i:i + nrn - 1]) == pre:
                lg[hist[i + nrn - 1]] = float("-inf")
    return lg


def _sample(lg: torch.Tensor, temperature: float, generator: torch.Generator | None = None) -> int:
    if temperature <= 0:
        return int(lg.argmax(-1))
    return int(torch.multinomial(torch.softmax(lg / temperature, -1), 1, generator=generator))


def _sample_restricted(lg: torch.Tensor, ids: list[int], temperature: float,
                       generator: torch.Generator | None = None) -> int:
    """Sample ONLY among `ids` (the legal set for a restricted state, e.g. SILENCE={SE,PAD}). Equivalent
    to masking every other logit to -inf then sampling, but O(len(ids)) instead of O(V)."""
    sub = lg[ids]
    if temperature <= 0:
        return int(ids[int(sub.argmax(-1))])
    return int(ids[int(torch.multinomial(torch.softmax(sub / temperature, -1), 1, generator=generator))])


def apply_state_machine(*, content_logits, feeds, hist, spec, in_user,
                        bargein=None, temperature=0.0, silence_temperature=None,
                        speech_temperature=None, ss_bias=None, se_bias=None,
                        rep_control=None, generators=None):
    """One frame of the batched MASK: sample each row's content head, run its EventFeed, return the
    LEGAL fed tokens. Shared by the single-stream (B=1) and batched engines' `step_masked`.

    Args (per row b, all length-B sequences unless noted):
      content_logits : (B, V) raw agent_head logits this frame.
      feeds          : list[EventFeed] (engine-owned, one per row; state mutated in place).
      hist           : list[list[int]] per-row content history (for rep-control; appended in place).
      spec           : ThinkerSpec (pad/epad/ss/se/pause ids).
      in_user        : (B,) bool — is the user still speaking (agent listening window)?
      bargein        : (B,) bool or None — may the agent speak over the user? (None -> all False).
      temperature    : float — 0 = argmax. The single-temperature fallback for both states.
      silence_temperature / speech_temperature : float or None — STATE-DEPENDENT temperature.
                       When a row is in the SILENCE state (legal = {SE, PAD}) it samples with
                       `silence_temperature`; in the SPEECH state (legal = content+pause+SS) with
                       `speech_temperature`. Each None -> falls back to `temperature`, so the default
                       (both None) is byte-identical to the old single-temperature behaviour. Use a HIGH
                       silence_temperature to make SE (resume) more likely — break out of a stuck silence
                       — and a LOW speech_temperature to keep content stable + suppress stray SS yields.
      ss_bias/se_bias: (B,) or None — added to the SS / SE logit before the legality mask + sampling
                       (nudge how readily the model yields/resumes; an SS/SE illegal for the state is
                       still removed by the mask, so the bias only matters where the token is legal).
      rep_control    : dict or None — {rep_penalty, no_repeat_ngram, rep_window}.
      generators     : (B,) torch.Generator or None — per-row sampling RNG. None (default) = the global
                       torch RNG, shared by all rows, so a row's draws depend on its batch neighbours and
                       their order. Per-row generators (seeded from the row's own identity, e.g. seed + case
                       name) make each row's samples independent of batch composition/order.

    Decode is MASK-THEN-SAMPLE: for each row, the illegal tokens for its current state
    (EventFeed.legal_ids) are removed BEFORE sampling, so the model samples the highest-probability
    LEGAL token — never sampling then falling back to PAD (which threw away the model's probability
    mass and biased toward silence). `in_user`/`bargein` do not force-mask: whether to
    yield is the model's own learned SS decision from the user-audio channel, not an oracle.

    Returns (fed, kinds, sampled): fed = list[int] legal tokens to embed next frame; kinds = list[str];
    sampled = list[int] the (already-legal) sampled token (== fed except on pause/onset pending frames
    and on SS/SE where fed may be PAD when events_in_input is False).
    """
    SS, SE = spec.sil_start_id, spec.sil_end_id
    st = temperature if silence_temperature is None else silence_temperature   # SILENCE-state temp
    sp = temperature if speech_temperature is None else speech_temperature     # SPEECH-state temp
    B = len(feeds)
    fed, kinds, sampled = [], [], []
    for b in range(B):
        fe = feeds[b]
        if not fe.sampling_now():                      # pause / onset pre-committed frame: no sample
            t = fe.next_pending()
            fed.append(int(t)); kinds.append("pend"); sampled.append(int(t)); continue
        lg = content_logits[b]
        gen = generators[b] if generators is not None else None
        if rep_control or ss_bias is not None or se_bias is not None:
            lg = lg.clone()
        if rep_control:
            _apply_rep(lg, hist[b], rep_control)
        if ss_bias is not None:
            lg[SS] = lg[SS] + float(ss_bias[b])
        if se_bias is not None:
            lg[SE] = lg[SE] + float(se_bias[b])
        mode, ids = fe.legal_ids()                     # legality mask for THIS row's state
        if mode == "restrict":                         # SILENCE: sample only among {SE, PAD}
            a = _sample_restricted(lg, ids, st, gen)   # ← SILENCE-state temperature
        else:                                          # SPEECH: mask out {PAD, EPAD, SE}, sample the rest
            if not (rep_control or ss_bias is not None or se_bias is not None):
                lg = lg.clone()
            lg[ids] = float("-inf")
            a = _sample(lg, sp, gen)                   # ← SPEECH-state temperature
        f, kind = fe.commit(a)                          # a is legal -> pure state update, no fallback
        fed.append(int(f)); kinds.append(kind); sampled.append(int(a))
        if kind == "content":
            hist[b].append(int(a))
    return fed, kinds, sampled


def validate_stream(ids, spec, *, start_silent: bool = False, require_frame0_ss: bool = False):
    """Legality checker for a GENERATED agent token stream (the fed/output ids of one dialogue turn) —
    a quick gate anyone can run over what our checkpoint + decode produced. Walks the SS/SE turn-taking
    grammar and returns a list of `(frame, token, reason)` violations; EMPTY list == legal.

    Grammar (that of the training-data construction), tracking `silent` (inside an SS..SE yield):
      * SS  legal only in SPEECH (nested SS while already silent = illegal) -> enters SILENCE
      * SE  legal only in SILENCE (orphan SE with no open silence = illegal) -> exits to SPEECH
      * content / pause_m  legal only in SPEECH (emitting them while yielded = illegal)
      * PAD (silence interior / pause interior) and EPAD (onset) are filler/markers, legal in either state
    Note: this checks the SS/SE STRUCTURE (what matters for turn-taking + what the decode mask enforces),
    not the exact pause-expansion frame counts. `start_silent`: the state before frame 0 (a raw agent
    turn starts in SPEECH; frame-0-SS streams then yield on frame 0).

    `require_frame0_ss`: for checkpoints trained with `force_frame0_ss=True`, the agent
    ALWAYS opens in the listening state, so frame 0 MUST be ⟨SS⟩. That is a training-DISTRIBUTION
    invariant, not an SS/SE grammar rule, so the grammar walk alone can't see it — a decode that opens
    with a bare PAD is grammar-legal under start_silent=True yet off-distribution at frame 0. Pass
    require_frame0_ss=True when decoding such a checkpoint to assert ids[0] == SS and catch that drift."""
    SS, SE, PAD, EPAD = spec.sil_start_id, spec.sil_end_id, spec.pad_id, spec.epad_id
    pause = set(spec.pause_ids)
    silent = bool(start_silent)
    viol = []
    if require_frame0_ss and (len(ids) == 0 or int(ids[0]) != SS):
        got = "empty stream" if len(ids) == 0 else int(ids[0])
        viol.append((0, got, "frame-0 is not ⟨SS⟩ (force_frame0_ss checkpoint must open the agent stream with SS)"))
    for i, t in enumerate(ids):
        t = int(t)
        if t == SS:
            if silent:
                viol.append((i, t, "nested SS: already in a silence (SS while silent)"))
            silent = True
        elif t == SE:
            if not silent:
                viol.append((i, t, "orphan SE: no open silence to end (SE while speaking)"))
            silent = False
        elif t in (PAD, EPAD):
            continue                                     # filler / onset marker: legal in either state
        elif t in pause:
            if silent:
                viol.append((i, t, "pause declaration inside a silence"))
        else:                                            # content word
            if silent:
                viol.append((i, t, "content token inside a silence (speaking while yielded)"))
    return viol
