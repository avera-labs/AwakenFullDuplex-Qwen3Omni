"""CPU: the live thinker driver (ThinkerRunner) drives a recording fake engine with the documented prime arguments,
the profile's decode settings verbatim and a private per-row generator."""
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from duplexaudio.serving.realtime.profile import load_profile
from duplexaudio.serving.realtime.thinker import PRIME_KW, ThinkerRunner, read_persona, row_sampling_seed


class _Tok:
    def encode(self, s, add_special_tokens=True):
        return [ord(c) % 997 for c in s]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


class _RecordingEngine:
    """Records prime / step_masked calls; samples its output with the generator it is handed (so a different
    generator or seed changes the fed stream, as it would on the real engine)."""

    def __init__(self, spec):
        self.spec, self.tok, self.compiled = spec, _Tok(), True
        self.calls, self._cache = [], 0
        self._probs = torch.softmax(torch.randn(50, generator=torch.Generator().manual_seed(3)), -1)

    def prime(self, prompts, **kw):
        kw = dict(kw)
        kw.setdefault("compile", None)
        self.calls.append(("prime", list(prompts), kw))
        self._cache, self._f = len(self.tok.encode(prompts[0])), 0

    def cache_frames_used(self):
        return self._cache

    def step_masked(self, user_codes, **kw):
        gens = kw.pop("generators")
        state = gens[0].get_state().clone()
        kw.pop("in_user")
        self.calls.append(("step", user_codes.tolist(), kw, state))
        a = int(torch.multinomial(self._probs, 1, generator=gens[0]))
        fed = self.spec.sil_start_id if self._f == 0 else 1000 + a
        self._f += 1
        return SimpleNamespace(fed=torch.tensor([fed]), kinds=["content" if self._f > 1 else "ss"],
                               sampled=torch.tensor([fed]), silent=torch.tensor([self._f == 1]))


PINNED = {("synthetic_user_interruption/150", 0): 4093883200, ("live-20261101-xyz", 12345): 849819948}


def test_row_sampling_seed_is_pinned():
    assert row_sampling_seed("synthetic_user_interruption/150", 0) == PINNED[("synthetic_user_interruption/150", 0)]
    assert row_sampling_seed("live-20261101-xyz", 12345) == PINNED[("live-20261101-xyz", 12345)]
    assert row_sampling_seed("a", 0) != row_sampling_seed("a", 1) != row_sampling_seed("b", 1)


def test_read_persona_is_verbatim(tmp_path):
    p = tmp_path / "persona.txt"
    raw = "<|im_start|>system\nYou are — calm.\n<|im_end|>\n"
    p.write_bytes(raw.encode("utf-8"))
    assert read_persona(p) == raw


@pytest.mark.parametrize("profile", ["default"])
def test_runner_drives_the_engine_with_the_profile_settings(tmp_path, profile):
    spec = SimpleNamespace(pad_id=3, epad_id=4, sil_start_id=1, sil_end_id=2, pause_ids=(90, 91))
    prof = load_profile(profile).thinker
    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("<|im_start|>system\nhi\n<|im_end|>\n", encoding="utf-8")
    eng = _RecordingEngine(spec)
    run = ThinkerRunner(prof, device="cpu")
    run.eng = eng
    info = run.arm(persona_file)
    assert info["purge_verified"] and info["prompt_len"] == len(eng.tok.encode(read_persona(persona_file)))
    seed = run.begin("conv", 0)
    assert seed == row_sampling_seed("conv", 0)
    fed = [run.step(torch.tensor([k % 7] * 8))[0] for k in range(20)]
    (_, prompts, kw) = eng.calls[0]
    assert prompts == [read_persona(persona_file)]
    assert {k: kw[k] for k in PRIME_KW} == PRIME_KW
    assert kw["purge_warmup"] is True and kw["reuse_compiled"] is True and kw["compile"] is True
    assert kw["max_cache_len"] == prof.max_cache_len and kw["prefill_micro_batch"] == prof.prefill_micro_batch
    steps = [c for c in eng.calls if c[0] == "step"]
    d = prof.decode
    for _, _, kw_s, _ in steps:
        assert kw_s["temperature"] == d["temperature"] and kw_s["speech_temperature"] == d["speech_temperature"]
        assert kw_s["silence_temperature"] == d["silence_temperature"] and kw_s["rep_control"] == d["rep_control"]
    # the private generator is seeded from (name, seed): same name + seed -> same fed stream
    eng2 = _RecordingEngine(spec)
    run2 = ThinkerRunner(prof, device="cpu")
    run2.eng = eng2
    run2.arm(persona_file)
    run2.begin("conv", 0)
    assert [run2.step(torch.tensor([k % 7] * 8))[0] for k in range(20)] == fed
    assert fed[0] == spec.sil_start_id


def test_se_and_ss_bias_kwargs():
    p = load_profile("default").thinker
    r = ThinkerRunner(dataclasses.replace(p, decode=dict(p.decode, se_bias=1.0)), device="cpu")
    assert r.step_kwargs(True)["se_bias"] == [1.0] and r.step_kwargs(False)["se_bias"] == [0.0]
    assert r.step_kwargs()["ss_bias"] is None
    p = dataclasses.replace(p, decode=dict(p.decode, ss_bias=0.5))
    assert ThinkerRunner(p, device="cpu").step_kwargs()["ss_bias"] == [0.5]


def test_runner_refuses_eager_and_unpurged(tmp_path):
    spec = SimpleNamespace(pad_id=3, epad_id=4, sil_start_id=1, sil_end_id=2, pause_ids=(90, 91))
    persona_file = tmp_path / "p.txt"
    persona_file.write_text("x\n", encoding="utf-8")
    run = ThinkerRunner(load_profile("default").thinker, device="cpu")
    run.eng = _RecordingEngine(spec)
    run.eng.compiled = False
    with pytest.raises(RuntimeError):
        run.arm(persona_file)
    run.eng = _RecordingEngine(spec)
    run.eng.cache_frames_used = lambda: 999
    with pytest.raises(RuntimeError):
        run.arm(persona_file)
    with pytest.raises(RuntimeError):
        ThinkerRunner(load_profile("default").thinker, device="cpu").begin("x", 0)
