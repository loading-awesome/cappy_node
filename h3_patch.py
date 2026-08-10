"""ComfyUI integration for Cappy's audio-aware MiniMax H3 residual cache.

This module mirrors the current ComfyUI H3 forward only to expose a safe block
loop boundary. The cache itself is deliberately conservative: block 0 always
runs, and blocks 1..N are reused only after whole-sequence, video, and audio
probes pass their threshold.
"""

from __future__ import annotations

import logging
import json
import os
import types
from collections import Counter
from collections.abc import Callable
from typing import Any

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax import model as minimax_model

from .cache_policy import BranchState, decide
from .fast_path import ExactFastPath, fused_attention_kernel_active


LOG = logging.getLogger("cappy_h3_cache")


def _relative_l1(current: torch.Tensor, previous: torch.Tensor | None,
                 token_range: tuple[int, int] | None = None) -> float | None:
    if previous is None or current.shape != previous.shape:
        return None
    if token_range is not None:
        start, end = token_range
        if start < 0 or end <= start or end > current.shape[0]:
            return None
        current, previous = current[start:end], previous[start:end]
    current_float = current.detach().float()
    previous_float = previous.detach().float()
    return ((current_float - previous_float).abs().mean()
            / (previous_float.abs().mean() + 1e-6)).item()


def _first_range(segments: tuple[tuple[int, int, str], ...], kind: str) -> tuple[int, int] | None:
    for start, end, segment_kind in segments:
        if segment_kind == kind:
            return start, end
    return None


class AudioAwareFirstBlockCache:
    """Cache a complete stack residual, using block 0 as a fresh probe."""

    def __init__(self, threshold: float, max_consecutive_reuses: int,
                 cache_device: str, trace: str, diagnostic_group_size: int | None = None,
                 diagnostic_path: str | None = None) -> None:
        self.threshold = threshold
        self.max_consecutive_reuses = max_consecutive_reuses
        self.cache_device = cache_device
        self.trace = trace
        self.total_steps = 1
        self.branches: dict[str, BranchState] = {}
        self.diagnostic_group_size = diagnostic_group_size
        self.diagnostic_path = diagnostic_path
        self._previous_group_residuals: dict[tuple[str, int], torch.Tensor] = {}
        self._group_records: list[dict[str, Any]] = []

    def begin(self, total_steps: int) -> None:
        self.total_steps = max(1, total_steps)
        self.branches = {}
        self._previous_group_residuals = {}
        self._group_records = []

    def finish(self) -> None:
        decisions = [d for state in self.branches.values() for d in state.decisions]
        if decisions:
            reused = sum(d.reuse for d in decisions)
            reasons = Counter(d.reason for d in decisions if not d.reuse)
            LOG.info(
                "[Cappy H3 Cache] reused %d/%d block stacks (%.2fx theoretical stack speedup); refresh reasons: %s",
                reused, len(decisions), len(decisions) / max(1, len(decisions) - reused),
                ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())),
            )
        self.branches = {}
        if self.diagnostic_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.diagnostic_path)), exist_ok=True)
            with open(self.diagnostic_path, "w") as handle:
                json.dump(self._group_records, handle, indent=2)

    def _record_group(self, branch_key: str, step: int, start: int, end: int,
                      residual: torch.Tensor, segments: tuple[tuple[int, int, str], ...]) -> None:
        """Record teacher-forced block-group stability while retaining dense output."""
        key = (branch_key, start)
        previous = self._previous_group_residuals.get(key)
        audio_range = _first_range(segments, "audio")
        video_range = _first_range(segments, "video")
        self._group_records.append({
            "branch": branch_key, "step": step, "start": start, "end": end,
            "whole_change": _relative_l1(residual, previous),
            "audio_change": _relative_l1(residual, previous, audio_range),
            "video_change": _relative_l1(residual, previous, video_range),
        })
        self._previous_group_residuals[key] = residual.detach().clone()

    @staticmethod
    def _branch_key(transformer_options: dict[str, Any]) -> str:
        branch = transformer_options.get("cond_or_uncond", "default")
        if isinstance(branch, (tuple, list)):
            return ",".join(str(value) for value in branch)
        return str(branch)

    @staticmethod
    def _timestep_value(timestep: Any) -> float | None:
        if isinstance(timestep, torch.Tensor) and timestep.numel():
            return float(timestep.detach().flatten()[0].item())
        if isinstance(timestep, (float, int)):
            return float(timestep)
        return None

    def _store_residual(self, state: BranchState, residual: torch.Tensor) -> None:
        if self.cache_device == "cuda" and residual.device.type != "cuda":
            raise ValueError("Cappy H3 Cache is set to cuda, but the model is not running on CUDA.")
        try:
            state.cached_residual = (residual.detach().to("cpu", copy=True)
                                     if self.cache_device == "cpu" else residual.detach().clone())
        except torch.OutOfMemoryError:
            if self.cache_device == "cuda":
                raise
            state.cached_residual = residual.detach().to("cpu", copy=True)

    @staticmethod
    def _apply_residual(hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        return hidden_states + residual

    def __call__(self, args: dict[str, Any], _extra_options: dict[str, Any]) -> dict[str, Any]:
        model: minimax_model.MiniMaxH3Model = args["model"]
        hidden_states: torch.Tensor = args["img"]
        transformer_options: dict[str, Any] = args["transformer_options"]
        branch_key = self._branch_key(transformer_options)
        state = self.branches.setdefault(branch_key, BranchState())
        segments = tuple(args["segments"])
        layout_signature = (tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device,
                            args["block_count"], segments)
        if state.layout_signature not in (None, layout_signature):
            state = BranchState()
            self.branches[branch_key] = state
        state.layout_signature = layout_signature

        timestep = self._timestep_value(args["timestep"])
        if timestep is None:
            raise ValueError("Cappy H3 Cache could not read the sampler timestep.")
        if state.last_timestep != timestep:
            state.last_timestep = timestep
            state.step_index += 1

        # H3 blocks use in-place residual adds. Preserve an untouched input
        # snapshot; otherwise `after_first - hidden_states` is identically zero
        # because both names refer to the same mutated tensor.
        input_snapshot = hidden_states.detach().clone()
        # The first block is never skipped: its residual is the decision probe.
        after_first = run_h3_blocks(model, hidden_states.clone(), args["t_emb"], args["mod_segments"],
                                    args["rope_freqs"], transformer_options, start=0, end=1)
        probe = after_first - input_snapshot
        if self.diagnostic_group_size:
            self._record_group(branch_key, state.step_index + 1, 0, 1, probe, segments)
            output = after_first
            for start in range(1, args["block_count"], self.diagnostic_group_size):
                end = min(start + self.diagnostic_group_size, args["block_count"])
                group_input = output.detach().clone()
                output = run_h3_blocks(model, output, args["t_emb"], args["mod_segments"],
                                        args["rope_freqs"], transformer_options, start=start, end=end)
                self._record_group(branch_key, state.step_index + 1, start, end,
                                   output - group_input, segments)
            return {"img": output}
        audio_range = _first_range(segments, "audio")
        video_range = _first_range(segments, "video")
        whole_change = _relative_l1(probe, state.previous_probe)
        audio_change = _relative_l1(probe, state.previous_probe, audio_range)
        video_change = _relative_l1(probe, state.previous_probe, video_range)
        state.previous_probe = probe.detach().clone()
        residual = state.cached_residual
        matching_residual = isinstance(residual, torch.Tensor) and residual.shape == hidden_states.shape
        decision = decide(
            state=state, total_steps=self.total_steps, threshold=self.threshold,
            max_consecutive_reuses=self.max_consecutive_reuses,
            whole_change=whole_change, audio_change=audio_change, video_change=video_change,
            have_matching_residual=matching_residual,
        )
        if self.trace == "per_step":
            LOG.info("[Cappy H3 Cache] branch=%s step=%d %s whole=%s audio=%s video=%s cap=%d",
                     branch_key, state.step_index + 1, "REUSE" if decision.reuse else decision.reason,
                     _format_change(whole_change), _format_change(audio_change),
                     _format_change(video_change), decision.consecutive_before)
        if decision.reuse:
            return {"img": self._apply_residual(input_snapshot, residual)}  # type: ignore[arg-type]

        output = run_h3_blocks(model, after_first, args["t_emb"], args["mod_segments"],
                               args["rope_freqs"], transformer_options, start=1)
        self._store_residual(state, output - input_snapshot)
        return {"img": output}


def _format_change(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


class SamplingScope:
    def __init__(self, cache: AudioAwareFirstBlockCache) -> None:
        self.cache = cache

    def __call__(self, sample_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
            raise ValueError("Cappy H3 Cache could not read the sampler sigma schedule.")
        self.cache.begin(len(sigmas) - 1)
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


def run_h3_blocks(model: minimax_model.MiniMaxH3Model, hidden_states: torch.Tensor,
                  timestep_embedding: torch.Tensor, mod_segments: list[tuple[int, int, int]],
                  rope_freqs: torch.Tensor, transformer_options: dict[str, Any],
                  start: int = 0, end: int | None = None) -> torch.Tensor:
    """Run a bounded block range while preserving Comfy's block replacements."""
    replacements = transformer_options.get("patches_replace", {}).get("dit", {})
    end = len(model.blocks) if end is None else end
    blocks = list(model.blocks[start:end])
    queue = comfy.model_prefetch.make_prefetch_queue(blocks, hidden_states.device, transformer_options)
    for index in range(start, end):
        block = model.blocks[index]
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden_states.device, block)
        replacement = replacements.get(("double_block", index))
        if replacement is not None:
            def original(block_args: dict[str, Any]) -> dict[str, torch.Tensor]:
                return {"img": block(block_args["img"], block_args["t_emb"], block_args["mod_segments"],
                                     block_args["rope_freqs"], transformer_options=block_args["transformer_options"])}
            hidden_states = replacement({"img": hidden_states, "t_emb": timestep_embedding,
                                         "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                                         "transformer_options": transformer_options}, {"original_block": original})["img"]
        else:
            hidden_states = block(hidden_states, timestep_embedding, mod_segments, rope_freqs,
                                  transformer_options=transformer_options)
    if queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden_states.device, None)
    return hidden_states


def cappy_h3_forward(self: minimax_model.MiniMaxH3Model, x: list[torch.Tensor], timestep: torch.Tensor,
                     context: torch.Tensor, transformer_options: dict[str, Any] = {},
                     minimax_payload: dict[str, Any] | None = None, **kwargs: Any) -> list[torch.Tensor]:
    """ComfyUI's H3 forward with Cappy's model-patchable block-loop seam."""
    del kwargs
    fast: ExactFastPath | None = getattr(self, "_cappy_fast_path", None)
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2:5]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device, dtype = video_x.device, context.dtype
    latent_t, lat_h, lat_w, audio_t, text_len = *video_x.shape[2:5], audio_x.shape[-1], context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = minimax_model.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                            keyframes=payload.get("keyframes"), refs=payload.get("refs"),
                                            frame_count=payload.get("frame_count"))
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))
    vis_aug = float(payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP))
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
    seg_t = {"text": t_v, "video": t_v, "audio": t_a, "cond": max(t_v, vis_aug),
             "ref_img": max(t_v, vis_aug), "ref_audio": max(t_a, aud_aug)}
    unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set()) |
                      ({seg_t["ref_audio"]} if has_aud_cond else set()))
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}
    text_tags = payload.get("text_token_tags")
    mod_segments: list[tuple[int, int, int]] = []
    for start, end, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags, run_start = text_tags.view(-1).tolist(), 0
            for index in range(1, end - start + 1):
                if index == end - start or tags[index] != tags[run_start]:
                    mod_segments.append((start + run_start, start + index, row_base + int(tags[run_start])))
                    run_start = index
        else:
            mod_segments.append((start, end, row_base + seg_tag[kind]))
    img_update, audio_update = layout.img_update.to(device), layout.audio_update.to(device)
    video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
    cond_video_rows, cond_audio_rows = self._cond_video_rows(payload, device), self._cond_audio_rows(payload, device)
    all_video_rows, all_audio_rows = video_rows, audio_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update], all_video_rows[img_update] = cond_video_rows, video_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update], all_audio_rows[audio_update] = cond_audio_rows, audio_rows
    video_embed, audio_embed = self.video_patch_proj(all_video_rows).to(dtype), self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        # Depends on the prompt, not the timestep, but the stock forward rebuilds
        # it on every model evaluation.
        def refine_text() -> torch.Tensor:
            return self.token_refiner(self.condition_proj(text_states),
                                      transformer_options=transformer_options)
        text_states = fast.text_states(context, refine_text) if fast else refine_text()
    hidden_states = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    video_offset = audio_offset = 0
    for start, end, kind in layout.segments:
        length = end - start
        if kind == "text": hidden_states[start:end] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden_states[start:end] = video_embed[video_offset:video_offset + length]; video_offset += length
        else:
            hidden_states[start:end] = audio_embed[audio_offset:audio_offset + length]; audio_offset += length
    t_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        position = t_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        timestep_embedding = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    else:
        timestep_embedding = self.time_embedder(t_values).to(dtype)
    def build_rope() -> torch.Tensor:
        return minimax_model.rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

    if fast is not None:
        # Fixed by the packed layout, so constant for the whole render.
        rope_freqs = fast.rope_table((layout.signature, dtype, str(device)), build_rope)
    else:
        rope_freqs = build_rope()
    replacements = transformer_options.get("patches_replace", {}).get("dit", {})
    if ("block_loop", 0) in replacements:
        segments = tuple(layout.segments)
        def original(block_args: dict[str, Any]) -> dict[str, torch.Tensor]:
            return {"img": run_h3_blocks(self, block_args["img"], block_args["t_emb"], block_args["mod_segments"],
                                           block_args["rope_freqs"], block_args["transformer_options"],
                                           block_args.get("start", 0), block_args.get("end"))}
        hidden_states = replacements[("block_loop", 0)](
            {"model": self, "img": hidden_states, "timestep": timestep, "t_emb": timestep_embedding,
             "mod_segments": mod_segments, "rope_freqs": rope_freqs, "transformer_options": transformer_options,
             "segments": segments, "block_count": len(self.blocks)}, {"original_block": original})["img"]
    else:
        hidden_states = run_h3_blocks(self, hidden_states, timestep_embedding, mod_segments, rope_freqs, transformer_options)
    video_seg = next((start, end, t_row[seg_t["video"]]) for start, end, kind in layout.segments if kind == "video")
    audio_seg = next((start, end, t_row[seg_t["audio"]]) for start, end, kind in layout.segments if kind == "audio")
    video_result, audio_result = self.final_layer(hidden_states, timestep_embedding, video_seg, audio_seg)
    video_out = minimax_model.unpatchify_video(video_result, latent_t, lat_h // 2, lat_w // 2,
                                                self.latents_dim, self.patch_size)[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = minimax_model.unpack_audio(audio_result)
    # H3's audio-output conversion moved across the public _forward boundary.
    # v0.30.1 expects its local slope; newer Core handles it in forward().
    # Supporting both avoids either an AttributeError or a silent audio-scale
    # mismatch when a user updates ComfyUI.
    time_shift_slope = getattr(minimax_model, "time_shift_slope", None)
    if time_shift_slope is None:
        audio_velocity = -audio_out.to(audio_x.dtype)
    else:
        slope_a = time_shift_slope(sigma_v, shift_v, shift_a).to(audio_out.dtype)
        audio_velocity = (-slope_a) * audio_out.to(audio_x.dtype)
    return [-video_out.to(video_x.dtype), audio_velocity]


class FastPathScope:
    """Hand the sigma schedule to the fast path and clear it between renders."""

    def __init__(self, fast: ExactFastPath) -> None:
        self.fast = fast

    def __call__(self, sample_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        self.fast.reset()
        if isinstance(sigmas, torch.Tensor) and sigmas.ndim == 1 and len(sigmas) >= 2:
            self.fast.sigmas = sigmas.detach().to("cpu", torch.float64)
        try:
            return sample_fn(*args, **kwargs)
        finally:
            LOG.info("[Cappy H3 Fast Path] %s", self.fast.summary())
            self.fast.reset()


def patch_fast_path(model: Any, cache_invariants: bool = True) -> Any:
    """Apply only the exact optimizations. Output must be bitwise unchanged."""
    patched_model = model.clone()
    diffusion_model = patched_model.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("Cappy H3 Fast Path requires a MiniMax H3 diffusion model; received "
                         f"{diffusion_model.__class__.__name__}.")
    active, detail = fused_attention_kernel_active()
    if active:
        LOG.info("[Cappy H3 Fast Path] fused QK-RMSNorm+RoPE kernel active (%s).", detail)
    else:
        LOG.warning("[Cappy H3 Fast Path] fused QK-RMSNorm+RoPE kernel is NOT active: %s. "
                    "This costs far more than anything this node can win back.", detail)

    fast = ExactFastPath(cache_invariants=cache_invariants)
    patched_model.add_object_patch("diffusion_model._forward",
                                   types.MethodType(cappy_h3_forward, diffusion_model))
    patched_model.add_object_patch("diffusion_model._cappy_fast_path", fast)
    patched_model.add_wrapper(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                              FastPathScope(fast))
    return patched_model


def patch_model(model: Any, threshold: float, max_consecutive_reuses: int,
                cache_device: str, trace: str, diagnostic_group_size: int | None = None,
                diagnostic_path: str | None = None) -> Any:
    """Clone and patch a MiniMax H3 MODEL; other model types fail before sampling."""
    if threshold <= 0:
        raise ValueError("Cappy H3 Cache requires a threshold greater than zero.")
    patched_model = model.clone()
    diffusion_model = patched_model.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("Cappy H3 Cache requires a MiniMax H3 diffusion model; received "
                         f"{diffusion_model.__class__.__name__}.")
    cache = AudioAwareFirstBlockCache(threshold, max_consecutive_reuses, cache_device, trace,
                                      diagnostic_group_size, diagnostic_path)
    patched_model.add_object_patch("diffusion_model._forward", types.MethodType(cappy_h3_forward, diffusion_model))
    patched_model.set_model_patch_replace(cache, "dit", "block_loop", 0)
    patched_model.add_wrapper(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, SamplingScope(cache))
    return patched_model
