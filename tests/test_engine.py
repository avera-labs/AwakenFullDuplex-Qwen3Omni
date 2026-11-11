"""CPU: RealtimeEngine's lifecycle with recording fakes (no model is loaded). The session cap derivation, the
single-GPU-thread contract (no deadlock on nested run, off-thread calls refused), the order of a conversation start
(talker seeded last), identical re-arms across conversations, one active conversation at a time, and a failed re-arm
taking the engine out of service."""
import hashlib
import threading
from types import SimpleNamespace

import pytest

from duplexaudio.serving.realtime.engine import RealtimeEngine, session_cap
from duplexaudio.serving.realtime.profile import load_profile, with_overrides


def test_session_cap():
    prof = load_profile("default")
    cap = session_cap(prof, talker_context=3000, prefix_frames=72, prompt_len=421)
    assert cap == {"requested": 3000, "talker_context": 2924, "thinker_cache": 3673} and min(cap.values()) == 2924
    small = with_overrides(prof, **{"thinker.max_cache_len": 2048})
    assert min(session_cap(small, talker_context=3000, prefix_frames=72, prompt_len=421).values()) == 2048 - 421 - 2


@pytest.fixture
def eng(tmp_path):
    persona = tmp_path / "persona_pinned.txt"                 # a stand-in persona, pinned by its own md5
    persona.write_text("<|im_start|>system\nYou are a test persona.<|im_end|>\n", encoding="utf-8")
    md5 = hashlib.md5(persona.read_bytes()).hexdigest()
    prof = with_overrides(load_profile("default"), **{"thinker.persona": str(persona), "thinker.persona_md5": md5})
    e = RealtimeEngine(prof, device="cpu", verify_md5=False)
    yield e
    e.gpu.shutdown(wait=True)


def test_single_gpu_thread_contract(eng):
    assert eng.run(threading.get_ident) == eng._gpu_thread != threading.get_ident()
    assert eng.run(lambda: eng.run(lambda: 5)) == 5                     # nested run is inline, not a deadlock
    with pytest.raises(RuntimeError):
        eng.require_gpu_thread()
    eng.run(eng.require_gpu_thread)
    with pytest.raises(RuntimeError):
        eng.new_session("x")                                              # off-thread entry point


def _fakes(eng, calls, fail_arm=False):
    class Thinker:
        prompt_len, frames = 421, 0
        arm_info = {"prompt_len": 421, "cache_at_start": 421, "purge_verified": True}
        tok = SimpleNamespace(decode=lambda ids: "")

        def arm(self, path):
            calls.append(("thinker.arm", str(path)))
            if fail_arm:
                raise RuntimeError("prime failed")
            return dict(self.arm_info)

        def begin(self, name, seed):
            calls.append(("thinker.begin", name, seed))
            return 1000 + seed

    class Talker:
        P, session_mode, gen_builds, sessions_started, forced_ch0 = 72, "persistent", 1, 0, []

        def start(self, seed):
            calls.append(("talker.start", seed))

        def end(self):
            calls.append(("talker.end",))

        def rng_summary(self):
            return {"violations": 0, "first_step": None}

    def resetter(tag):
        return SimpleNamespace(reset=lambda: calls.append((tag,)))

    eng.thinker, eng.talker = Thinker(), Talker()
    eng.encoder, eng.decoder, eng.feeder = resetter("encoder.reset"), resetter("decoder.reset"), resetter("feeder.reset")
    eng.thinker.spec = SimpleNamespace(pad_id=3)
    eng.silence_codes = [0] * 8
    eng.cap_limits = {"requested": 3000, "talker_context": 2924, "thinker_cache": 3673}
    eng.ready = True


def test_new_session_order_and_repeatability(eng):
    calls = []
    _fakes(eng, calls)
    a = eng.run(eng.new_session, "conv-a", 0, record=False)
    assert [c[0] for c in calls] == ["thinker.arm", "thinker.begin", "encoder.reset", "decoder.reset", "feeder.reset",
                                     "talker.start"]
    assert a.cap_frames == 2924 and eng._active is a and a.seeds["thinker_row"] == 1000
    with pytest.raises(RuntimeError):
        eng.run(eng.new_session, "conv-b", 0, record=False)               # one active conversation at a time
    eng.run(a.finalize, "stop")
    assert eng._active is None
    first = list(calls)
    calls.clear()
    b = eng.run(eng.new_session, "conv-a", 0, record=False)               # same name + seed -> same seeds, same arm
    assert calls[:2] == first[:2] and calls[-1] == first[5] and b.seeds == a.seeds
    eng.run(b.finalize, "stop")
    assert eng.arms == 2


def test_failed_rearm_takes_the_engine_out_of_service(eng):
    calls = []
    _fakes(eng, calls, fail_arm=True)
    with pytest.raises(RuntimeError, match="prime failed"):
        eng.run(eng.new_session, "x", record=False)
    assert eng.ready is False and eng._active is None
    with pytest.raises(RuntimeError, match="not ready"):
        eng.run(eng.new_session, "y", record=False)


def test_rearm_refuses_a_changed_persona(eng, tmp_path):
    calls = []
    _fakes(eng, calls)
    p = tmp_path / "persona.txt"
    p.write_text("not the pinned persona\n", encoding="utf-8")
    eng.prof = with_overrides(eng.prof, **{"thinker.persona": str(p)})
    with pytest.raises(RuntimeError, match="md5"):
        eng.run(eng.new_session, "x", record=False)
    assert calls == []
