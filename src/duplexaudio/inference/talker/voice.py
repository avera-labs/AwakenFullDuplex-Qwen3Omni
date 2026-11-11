"""Voice-prompt prefix codes of the talker (PersonaPlex-style voice conditioning).

The talker is conditioned on a voice prompt: for its R prefix frames the agent audio codebooks carry the prompt's
Mimi codes, the user channels carry fixed "sine" codes and ch0 carries the prompt's transcript (or PAD), followed by
a short spacer before the dialogue starts.

Encoding a NEW voice prompt from raw audio: if the clip does not open on silence, prepend a short segment of
silence to the wav before `mimi.encode`, so frame 0 lands on the codec's silence-onset code (cb0 = 1049). Almost all
training voice prompts start silent; a prompt whose frame 0 is speech is out of distribution for the talker's prefix
and tends to make it babble at the start.
"""
from __future__ import annotations

# User-channel codes over the voice-prompt prefix (PersonaPlex's fixed sine frame).
VP_USER_SINE = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
# Agent-audio silence codes (Mimi-encoded zero frame): the default prompt -> dialogue spacer.
VP_AGENT_SILENCE = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
VP_AGENT_ZERO = [0] * 8
# A fixed 440 Hz tone code tuple (pre-computed; inserted as tokens, never re-encoded in context, because Mimi is
# contextual): an explicit prompt -> dialogue boundary marker for talkers trained with it.
VP_AGENT_SINE440 = [1203, 1522, 1185, 99, 1095, 128, 1017, 916]
SPACER_MARKERS = {"silence": VP_AGENT_SILENCE, "sine440": VP_AGENT_SINE440, "zero": VP_AGENT_ZERO}


def spacer_marker_codes(name):
    """Fixed agent-audio codes for the prompt -> dialogue spacer: 'silence' (default), 'sine440' or 'zero'."""
    return SPACER_MARKERS.get(name or "silence", VP_AGENT_SILENCE)
