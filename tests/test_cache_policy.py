"""Unit tests for Cappy's framework-independent reuse safety policy."""

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "cache_policy", Path(__file__).parents[1] / "cache_policy.py"
)
assert SPEC and SPEC.loader
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def choose(state, step, **changes):
    state.step_index = step
    return policy.decide(
        state=state, total_steps=8, threshold=0.10, max_consecutive_reuses=5,
        have_matching_residual=True, whole_change=0.01, audio_change=0.01,
        video_change=0.01, **changes,
    )


def test_first_and_last_steps_are_full():
    state = policy.BranchState()
    assert choose(state, 0).reason == "noHistory"
    assert choose(state, 7).reason == "cooldown"


def test_audio_is_a_separate_veto():
    state = policy.BranchState()
    decision = choose(state, 2, whole_change=0.02, audio_change=0.11)
    assert not decision.reuse
    assert decision.reason == "audioAboveThreshold"


def test_cap_forces_a_refresh_after_five_reuses():
    state = policy.BranchState()
    for step in range(1, 6):
        assert choose(state, step).reuse
    assert choose(state, 6).reason == "consecutiveCap"


def test_nonfinite_probe_never_reuses():
    state = policy.BranchState()
    assert choose(state, 3, audio_change=float("inf")).reason == "nonFinite"
