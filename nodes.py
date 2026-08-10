"""ComfyUI node definition."""

from __future__ import annotations

from .h3_patch import patch_model


class CappyMiniMaxH3AudioAwareCache:
    """Apply the measured, audio-aware H3 first-block residual cache."""

    CATEGORY = "advanced/model/patches"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Approximate MiniMax H3 acceleration. Runs block 0 on every step, "
        "reuses the remaining-stack residual only when whole-sequence, generated-video, "
        "and generated-audio probes are stable, and always refreshes the last step."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "relative_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Maximum relative block-0 change for reuse. This is calibrated "
                        "for Cappy's block-0 residual probe, not TeaCache or other nodes."
                    ),
                }),
                "max_consecutive_reuses": ("INT", {
                    "default": 5, "min": 1, "max": 10, "step": 1,
                    "tooltip": (
                        "Maximum age of a reused 49-block residual. Five is the measured "
                        "shipping setting; larger values require dialogue and motion testing."
                    ),
                }),
                "cache_device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Where the cached residual lives. CPU saves VRAM but adds transfer cost.",
                }),
                "trace": (["summary", "per_step"], {
                    "default": "summary",
                    "tooltip": "Logs reuse decisions, including audio vetoes, to the Comfy console.",
                }),
            }
        }

    def patch(self, model, relative_threshold, max_consecutive_reuses,
              cache_device, trace, diagnostic_group_size=None, diagnostic_path=None):
        return (patch_model(
            model=model,
            threshold=relative_threshold,
            max_consecutive_reuses=max_consecutive_reuses,
            cache_device=cache_device,
            trace=trace,
            diagnostic_group_size=diagnostic_group_size,
            diagnostic_path=diagnostic_path,
        ),)
