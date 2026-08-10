"""Framework-independent state machine for the Cappy H3 cache.

The policy is intentionally separate from the Comfy patch: its safety rules can
be tested without a 66 GB H3 checkpoint or a running Comfy server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class CacheDecision:
    reuse: bool
    reason: str
    whole_change: float | None
    audio_change: float | None
    video_change: float | None
    consecutive_before: int


@dataclass
class BranchState:
    previous_probe: object | None = None
    cached_residual: object | None = None
    consecutive_reuses: int = 0
    step_index: int = -1
    last_timestep: float | None = None
    layout_signature: tuple[object, ...] | None = None
    decisions: list[CacheDecision] = field(default_factory=list)


def decide(
    *,
    state: BranchState,
    total_steps: int,
    threshold: float,
    max_consecutive_reuses: int,
    whole_change: float | None,
    audio_change: float | None,
    video_change: float | None,
    have_matching_residual: bool,
) -> CacheDecision:
    """Choose full/reuse after block 0; every generated stream vetoes reuse."""

    before = state.consecutive_reuses
    if state.step_index == 0:
        decision = CacheDecision(False, "noHistory", whole_change, audio_change,
                                 video_change, before)
    elif state.step_index >= total_steps - 1:
        decision = CacheDecision(False, "cooldown", whole_change, audio_change,
                                 video_change, before)
    elif not have_matching_residual:
        decision = CacheDecision(False, "noCachedResidual", whole_change, audio_change,
                                 video_change, before)
    elif not all(value is not None and isfinite(value)
                 for value in (whole_change, audio_change, video_change)):
        decision = CacheDecision(False, "nonFinite", whole_change, audio_change,
                                 video_change, before)
    elif whole_change >= threshold:
        decision = CacheDecision(False, "wholeSequenceAboveThreshold", whole_change,
                                 audio_change, video_change, before)
    elif audio_change >= threshold:
        decision = CacheDecision(False, "audioAboveThreshold", whole_change,
                                 audio_change, video_change, before)
    elif video_change >= threshold:
        decision = CacheDecision(False, "videoAboveThreshold", whole_change,
                                 audio_change, video_change, before)
    elif before >= max_consecutive_reuses:
        decision = CacheDecision(False, "consecutiveCap", whole_change, audio_change,
                                 video_change, before)
    else:
        decision = CacheDecision(True, "belowThreshold", whole_change, audio_change,
                                 video_change, before)

    if decision.reuse:
        state.consecutive_reuses += 1
    else:
        state.consecutive_reuses = 0
    state.decisions.append(decision)
    return decision
