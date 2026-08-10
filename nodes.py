"""ComfyUI node definitions."""

from __future__ import annotations

from .h3_patch import patch_fast_path, patch_model


class CappyMiniMaxH3FastPath:
    """Exact MiniMax H3 acceleration: identical output, less work."""

    CATEGORY = "advanced/model/patches"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "The safe option, and not the fast one: MiniMax H3 acceleration that "
        "refuses to change the output, measured at 1.00x. Use "
        "CappyMiniMaxH3AudioAwareCache for actual speed. Reuses the "
        "RoPE table and refined text embedding across steps, and can run every "
        "sampler step's AdaLN projection as one GEMM per block. Each optimization "
        "is checked against the stock result at runtime and reverted unless it is "
        "bitwise equal, so enabling it is safe by construction.\n\n"
        "Measured at 864x480x49/20 steps on an RTX PRO 6000 Blackwell: 1.00x. "
        "Dense GEMM (69% of a step) and flash attention (20.5%) are already at "
        "this hardware's ceiling, so a model patch has almost nothing left to "
        "take. The real wins are the runtime and the checkpoint, not this node: "
        "PyTorch built against CUDA 13 is 1.117x and more accurate, and the W4A4 "
        "checkpoint is 1.58x. This node reports when the CUDA 13 fused kernel is "
        "missing, which is worth 10.5% and fails silently."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "batch_adaln": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Project every scheduled timestep's AdaLN in one GEMM per block, "
                        "instead of one GEMM per step. OFF because it was measured not to "
                        "work: in isolation the batched GEMM is 18x cheaper and bitwise "
                        "equal, but inside the model it is neither. It disagrees with the "
                        "stock projection by one bf16 last bit (mean_rel 1.887e-06), so the "
                        "runtime gate reverts it, and forcing it through changed the "
                        "trajectory while still measuring 1.00x. Left here, off, so the "
                        "next person does not repeat the experiment."
                    ),
                }),
                "cache_invariants": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Reuse the RoPE rotation table and the refined text embedding across "
                        "steps. Both depend on the layout and prompt, not the timestep, and "
                        "reuse is exact. Measured at 1.00x with one model evaluation per step "
                        "- under a millisecond against a one-second step. Should matter more "
                        "under CFG, which evaluates each timestep twice. Untested there."
                    ),
                }),
                "verify_exact": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Recompute each AdaLN projection the stock way and assert the batched "
                        "result is bitwise equal. Costs more than it saves; use it once after "
                        "a ComfyUI or driver update, not in production."
                    ),
                }),
            }
        }

    def patch(self, model, batch_adaln, cache_invariants, verify_exact):
        return (patch_fast_path(model=model, batch_adaln=batch_adaln,
                                cache_invariants=cache_invariants, verify=verify_exact),)


class CappyMiniMaxH3AudioAwareCache:
    """Apply the measured, audio-aware H3 first-block residual cache."""

    CATEGORY = "advanced/model/patches"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "The accelerator for MiniMax H3, and an approximation: 1.84x, paid for in "
        "fine detail on some content. Runs "
        "block 0 every step and reuses the remaining-stack residual when the "
        "whole-sequence, video and audio probes are all stable.\n\n"
        "Measured at 864x480x49/20 steps on CUDA 13: a reused step costs 0.025s "
        "against 1.00s, 9-10 of 20 stacks reuse, and DiT sampling drops from 19.2s "
        "to 10.4s - 1.84x, ahead of the 1.58x from the W4A4 checkpoint.\n\n"
        "The cost is high-frequency detail, and it depends on the content. On a "
        "detail-dense tracking shot the books visibly disappear off the library "
        "shelves, leaving flat surfaces; step-20 denoised video diverges 45.9% "
        "(cos 0.894). On a low-motion talking head the same settings preserve teeth, "
        "lip texture and skin. Judge it on your own content at full resolution, on "
        "crops of the smallest textured thing in frame - whole-frame sharpness "
        "metrics do NOT detect this loss. Lower relative_threshold to 0.08 or 0.06 "
        "if detail is going; change the threshold before the cap. Use "
        "CappyMiniMaxH3FastPath instead where output must not change at all."
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
