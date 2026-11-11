"""Talker ch0 canonicalizer table: maps thinker-stream tokens the talker never saw in training onto tokens it did
see, one token per frame (frame-preserving, no lookahead).

Why: the thinker writes text in its own layout (no-space utterance-initial tokens such as "Oh", curly quotes and
apostrophes, em-dashes) while the talker was trained on differently tokenized text (" Oh", ASCII). Words containing
tokens unseen in talker training are mis-voiced far more often.

The table is data (a JSON whose md5 is pinned by the serving profile), built from the talker's own training
ch0 token counts; this module only loads it. Rules (applied only to tokens with training count 0):
  C1  typography: curly ' " -> straight twin (if that twin is a single seen token); dash -> ","; standalone quote /
      colon / asterisk / parenthesis tokens -> PAD.
  C2  C1 + a no-space alphabetic token -> its single-token spaced twin (" Oh") when it is the first content token
      after SE (turn-initial).
  C2u C1 + the same space twin when it is the first content token after ANY special (SS/SE/EPAD/pause/PAD) or after a
      token C1 or C3 mapped to PAD or "," (utterance-initial).
  C3  (optional, combinable: "c2u+c3") a punctuation token right after an EPAD -> PAD.
"""
from __future__ import annotations

import hashlib
import json

MODES = ("c1", "c2", "c2u", "c1+c3", "c2+c3", "c2u+c3")


def load_canon_table(path):
    """Load a canon table JSON; returns (table, md5 of the file bytes)."""
    raw = open(path, "rb").read()
    tab = json.loads(raw)
    tab["c1"] = {int(k): (v[0], int(v[1])) for k, v in tab["c1"].items()}
    tab["space"] = {int(k): int(v) for k, v in tab["space"].items()}
    tab["punct"] = set(int(x) for x in tab["punct"])
    return tab, hashlib.md5(raw).hexdigest()
