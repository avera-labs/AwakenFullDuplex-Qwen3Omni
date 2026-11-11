"""CPU: the shipped realtime profile loads, pins the tracked talker config by md5, uses the one talker convention, and
the loader refuses malformed profiles. Runtime artifacts (checkpoints, voice prompt, canon table) are not needed."""
import pytest
import yaml

from duplexaudio.serving.realtime.profile import (PROFILES_DIR, ProfileError, check_md5s, load_profile, pinned_files,
                                                  realized_md5, repo_path, with_overrides)

PROFILES = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))
RUNTIME_LABELS = ("thinker.ckpt", "thinker.persona", "talker.ckpt", "talker.vp_codes", "talker.vp_text",
                  "talker.canon_table")


def test_shipped_profiles():
    assert set(PROFILES) == {"default"}


@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads_and_tracked_pins_match(name):
    prof = load_profile(name)
    assert prof.name == name and len(prof.md5) == 32
    assert set(pinned_files(prof)) == set(RUNTIME_LABELS) | {"talker.config"}
    assert check_md5s(prof, skip=RUNTIME_LABELS) == []                  # the tracked talker config matches its pin


@pytest.mark.parametrize("name", PROFILES)
def test_one_talker_convention(name):
    prof = load_profile(name)
    k = prof.talker
    assert k.layout == "train" and k.canon_mode == "c2u" and k.session == "persistent"
    assert k.sampler == {"noise": "legacy", "temp": 0.8, "top_k": 250, "top_p": 0.0}
    assert (k.vp_codes, k.vp_text, k.canon_table) == ("data/default_vp_codes.npy", "data/default_vp_text.npy",
                                                      "data/talker_canon.json")
    cfg = yaml.safe_load(open(repo_path(k.config)))
    assert prof.session.end_flush_frames == int(cfg["text_lead_frames"]) + 1
    assert cfg["ch0_encoding"] == "thinker_stream" and cfg["zero_user_emb"] is True
    assert int(cfg["text_lead_frames"]) == 3
    assert prof.frontend.floor_bands == [[-78.0, -63.0, 1.0]]


def _write(tmp_path, mutate):
    d = yaml.safe_load(open(PROFILES_DIR / "default.yaml"))
    mutate(d)
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


@pytest.mark.parametrize("mutate", [
    lambda d: d["thinker"].pop("max_cache_len"),
    lambda d: d["thinker"].__setitem__("extra", 1),
    lambda d: d["thinker"]["decode"]["rep_control"].pop("rep_window"),
    lambda d: d["talker"].__setitem__("layout", "infer"),
    lambda d: d["talker"].__setitem__("session", "shared"),
    lambda d: d["talker"]["sampler"].__setitem__("noise", "crn"),
    lambda d: d["frontend"].__setitem__("mimi_ring_capacity", 200),
    lambda d: d["frontend"].__setitem__("floor_bands", [[-60.0, -70.0, 1.0]]),
    lambda d: d.pop("session"),
    lambda d: d["session"].__setitem__("end_flush_frames", 4.0),
    lambda d: d["frontend"].__setitem__("mimi_cuda_graph", "false"),
    lambda d: d["thinker"].__setitem__("max_cache_len", "4096"),
    lambda d: d["talker"]["sampler"].__setitem__("top_k", 250.0),
])
def test_loader_refuses_bad_profiles(tmp_path, mutate):
    with pytest.raises(ProfileError):
        load_profile(_write(tmp_path, mutate))


def test_md5_mismatch_and_missing_files_are_reported(tmp_path):
    persona = tmp_path / "persona.txt"
    persona.write_text("x\n", encoding="utf-8")
    p = _write(tmp_path, lambda d: d["thinker"].update(persona=str(persona), persona_md5="0" * 32))
    bad = check_md5s(load_profile(p), skip=[lb for lb in RUNTIME_LABELS if lb != "thinker.persona"])
    assert len(bad) == 1 and bad[0].startswith("thinker.persona") and "!= pinned" in bad[0]
    q = _write(tmp_path, lambda d: d["thinker"].update(persona=str(tmp_path / "nope.txt")))
    bad = check_md5s(load_profile(q), skip=[lb for lb in RUNTIME_LABELS if lb != "thinker.persona"])
    assert bad == [f"thinker.persona: {tmp_path / 'nope.txt'} missing"]       # absolute: looked for where it says


def test_missing_relative_file_names_the_resolved_path(tmp_path):
    """A relative profile path is not resolved against the working directory: the message says where it was looked
    for (data/ and checkpoints/ in the model folder, other relative paths under the repository root)."""
    q = _write(tmp_path, lambda d: d["talker"].update(vp_codes="my_vp.npy"))
    skip = [lb for lb in RUNTIME_LABELS if lb != "talker.vp_codes"]
    assert check_md5s(load_profile(q), skip=skip) == [f"talker.vp_codes: my_vp.npy missing (looked for "
                                                     f"{repo_path('my_vp.npy')})"]
    q = _write(tmp_path, lambda d: d["talker"].update(vp_codes="data/my_vp.npy"))
    assert check_md5s(load_profile(q), skip=skip) == [f"talker.vp_codes: data/my_vp.npy missing (looked for "
                                                     f"{repo_path('data/my_vp.npy')})"]


def test_overrides_change_the_realized_md5_and_are_recorded():
    p = load_profile("default")
    q = with_overrides(p, **{"thinker.max_cache_len": 2048})
    assert q.thinker.max_cache_len == 2048 and q.overrides == ("thinker.max_cache_len=2048",)
    assert q.md5 == p.md5 and realized_md5(q) != realized_md5(p) and realized_md5(p) == realized_md5(load_profile(p.path))
    with pytest.raises(ProfileError):
        with_overrides(p, **{"thinker.nope": 1})
