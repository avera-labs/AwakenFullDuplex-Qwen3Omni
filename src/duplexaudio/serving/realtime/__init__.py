"""Real-time full-duplex engine and demo server: the thinker and the talker inside one 80 ms frame.

profile.py  one YAML per deployment (profiles/*.yaml), every axis explicit and md5-pinned
engine.py   RealtimeEngine: models, the single GPU thread, warm-up, per-conversation seeds
session.py  DuplexSession: the clock-agnostic per-frame loop, recordings, protocol metadata
thinker.py  OmniThinkerEngine.step_masked driven like the offline decode (private RNG)
talker.py   StreamingTalker: the batch talker decode, one step per live frame (default RNG + tripwire)
canon.py    fed token -> talker ch0 (C2u canon, pause -> PAD), token at a time
codec.py    incremental Mimi encoder (user) and decoder (agent), separate instances
frontend.py framing and the training-style room-tone floor
server.py   FastAPI/WebSocket demo server (loopback by default) + the browser client in static/
client.py   headless client: stream a wav through a running server
offline.py  awaken-offline: a recorded user track through the engine, no server
"""
