"""Talker inputs for a fed thinker stream (the reference builder the streaming talker reproduces step by step).

  ch0      [ voice-prompt transcript over the R prompt frames (negative -> PAD; all PAD without vp_text) | PAD over
             the spacer | the fed stream with pause ids -> PAD (after the optional canonicalizer) | `tail` PAD ]
  agent    [ prompt codes (Rall, 8) time-major -> (8, R), clamped to 0..2047 | spacer-marker codes ]  (prefix only)
  user     PersonaPlex user sine over the prefix; dialogue user codes are zeros (zero_user_emb talkers)

Canonicalizer (`canonicalize`): the C1 / C2 / C2u / C3 rules over a canon table (see canon_table.py), applied to a
whole fed stream. The prefix is written through `depformer_replace_tokens` in one of two layouts:
  infer : all 8 codebooks of prompt frame s in step s's column;
  train : the training layout under the model's codebook delays (cb_k of frame s - delay_k in step s).
"""
from __future__ import annotations

import numpy as np
import torch

from duplexaudio.inference.talker.voice import VP_USER_SINE, spacer_marker_codes

LAYOUTS = ("infer", "train")
CANON_MODES = ("c1", "c2", "c2u", "c1+c3", "c2+c3", "c2u+c3")


def canonicalize(ids, table, mode, spec):
    """Frame-preserving canonicalization of a fed id list (same length):
      C1  every token in table['c1'] -> its mapped id (kinds: curly / dash / PAD)
      C2  a token in table['space'] -> its spaced twin when it is the first content token after an SE
      C2u a token in table['space'] -> its spaced twin when it is the first content token after any special
          (PAD / EPAD / SS / SE / pause) or right after a token mapped to PAD or ',' (C1 PAD / dash, or C3's PAD)
      C3  (+c3) a token in table['punct'] right after an EPAD -> PAD
    table: as returned by canon_table.load_canon_table (c1: {id: (kind, new)}, space: {id: new}, punct: set)."""
    if mode not in CANON_MODES:
        raise ValueError(f"canon mode {mode!r} not in {CANON_MODES}")
    base, c3 = mode.split("+")[0], mode.endswith("+c3")
    pad, epad, se = int(spec.pad_id), int(spec.epad_id), int(spec.sil_end_id)
    specials = {pad, epad, int(spec.sil_start_id), se} | {int(p) for p in spec.pause_ids}
    c1, space, punct = table["c1"], table["space"], table["punct"]
    stats = {"curly": 0, "dash": 0, "PAD": 0, "space": 0, "c3_punct": 0}
    out = []
    seen_se_since_content = False          # C2: an SE since the last content token
    special_since_content = False          # C2u: any special since the last content token
    prev_c1_pad_or_dash = False            # C2u: the previous content token was C1-mapped to PAD or ','
    prev_orig = None
    for t in (int(x) for x in ids):
        if t in specials:
            out.append(t)
            special_since_content = True
            seen_se_since_content |= t == se
            prev_orig = t
            continue
        trig = seen_se_since_content if base == "c2" else \
            (special_since_content or prev_c1_pad_or_dash) if base == "c2u" else False
        new, c1_pad_or_dash = t, False
        if c3 and prev_orig == epad and t in punct:
            new = pad; stats["c3_punct"] += 1; c1_pad_or_dash = True
        elif t in c1:
            kind, new = c1[t]; stats[kind] += 1
            c1_pad_or_dash = kind in ("PAD", "dash")
        elif trig and t in space:
            new = space[t]; stats["space"] += 1
        out.append(int(new))
        seen_se_since_content = special_since_content = False
        prev_c1_pad_or_dash = c1_pad_or_dash
        prev_orig = t
    return out, stats


def talker_inputs(stream, cfg, spec, *, vp_codes, vp_text, canon=None, canon_mode=None, tail=0):
    """The talker inputs for one fed stream (see the module docstring). Returns
    {"ch0": LongTensor[P + T + tail], "agent_prefix": LongTensor[8, P], "user_prefix": LongTensor[8, P], "P", "R",
     "T", "starts_with_ss", "canon_stats"}."""
    if cfg.get("ch0_encoding") != "thinker_stream":
        raise ValueError("only ch0_encoding='thinker_stream' talkers are supported")
    ids = [int(x) for x in (stream.tolist() if hasattr(stream, "tolist") else stream)]
    stats = None
    if canon is not None:
        ids, stats = canonicalize(ids, canon, canon_mode, spec)
    pad = int(spec.pad_id)
    pauses = {int(p) for p in spec.pause_ids}
    dialogue = [pad if t in pauses else t for t in ids] + [pad] * int(tail)
    n_vp, n_sp = int(cfg.get("vp_frames", 0)), int(cfg.get("vp_silence_frames", 0))
    if n_vp > 0:
        vc = np.asarray(vp_codes)
        if vc.ndim != 2 or vc.shape[1] != 8:
            raise ValueError(f"vp_codes must be (Rall, 8) time-major; got {vc.shape}")
        R = min(n_vp, vc.shape[0])
        agent = torch.as_tensor(vc[:R].T.copy()).long().clamp(0, 2047)
        if n_sp:
            agent = torch.cat([agent, torch.tensor(spacer_marker_codes(cfg.get("spacer_marker")),
                                                   dtype=torch.long).view(8, 1).repeat(1, n_sp)], 1)
        ch0_pre = [pad] * (R + n_sp)
        if vp_text is not None:
            vt = [int(x) for x in np.asarray(vp_text)[:R]]
            ch0_pre[:len(vt)] = [pad if x < 0 else x for x in vt]
    else:
        R, n_sp, agent, ch0_pre = 0, 0, torch.zeros(8, 0, dtype=torch.long), []
    P = R + n_sp
    user = torch.tensor(VP_USER_SINE, dtype=torch.long).view(8, 1).repeat(1, P)
    return {"ch0": torch.tensor(ch0_pre + dialogue, dtype=torch.long), "agent_prefix": agent, "user_prefix": user,
            "P": P, "R": R, "T": len(ids), "starts_with_ss": bool(ids) and ids[0] == int(spec.sil_start_id),
            "canon_stats": stats}


def prefix_columns(agent_prefix, agent_delays, layout):
    """Per prefix step s the 8 codes written through depformer_replace_tokens. infer: frame s's codes. train: cb_k of
    frame s - delay_k (steps before a codebook's first frame repeat frame 0: LMGen feeds its initial token there)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r} not in {LAYOUTS}")
    if layout == "infer":
        return agent_prefix.clone()
    P = agent_prefix.shape[1]
    cols = torch.empty_like(agent_prefix)
    for k, d in enumerate(agent_delays):
        idx = torch.clamp(torch.arange(P) - int(d), min=0)
        cols[k] = agent_prefix[k, idx]
    return cols
