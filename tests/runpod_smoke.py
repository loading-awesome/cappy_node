"""Manual CUDA smoke test for Cappy on a real ComfyUI MiniMax H3 install.

Run from a ComfyUI checkout with the H3 bf16 DiT installed:

    COMFY_DIR=/runpod-volume/ComfyUI ./venv/bin/python \
      custom_nodes/cappy_node/tests/runpod_smoke.py

It deliberately uses a permissive threshold so a short four-step sampler must
take the reuse branch. This is an execution test, not a quality setting.
"""

from __future__ import annotations

import logging
import os
import sys

COMFY = os.environ.get("COMFY_DIR", ".")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import torch

import comfy.sd
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_minimax_h3 import _empty_av_latent
from cappy_node.nodes import CappyMiniMaxH3AudioAwareCache


def finite(name: str, value: torch.Tensor) -> None:
    assert torch.isfinite(value).all(), f"{name} contains NaN or infinity"
    print(f"{name}: shape={tuple(value.shape)} std={float(value.float().std()):.5f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path = os.path.join(COMFY, "models", "diffusion_models", "minimax_h3_fl2va_bf16.safetensors")
    model = comfy.sd.load_diffusion_model(path)
    # H3 accepts pre-projected text rows. Random context avoids loading the
    # 51 GB text encoder while still exercising the production DiT forward.
    context = torch.randn((1, 16, 5120), dtype=torch.bfloat16)
    conditioning = [[context, {}]]
    # Exercise the public node entry point, not merely its implementation.
    cached = CappyMiniMaxH3AudioAwareCache().patch(
        model=model, relative_threshold=1.0, max_consecutive_reuses=5,
        cache_device="auto", trace="per_step",
    )[0]
    latent, _ = _empty_av_latent(width=256, height=256, length=17)
    noise = RandomNoise.execute(noise_seed=12345).result[0]
    sampler = KSamplerSelect.execute(sampler_name="res_multistep").result[0]
    sigmas = BasicScheduler.execute(model=cached, scheduler="simple", steps=4, denoise=1.0).result[0]
    guider = BasicGuider.execute(model=cached, conditioning=conditioning).result[0]
    result = SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler,
                                           sigmas=sigmas, latent_image=latent).result[0]
    video, audio = result["samples"].unbind()
    finite("video", video)
    finite("audio", audio)
    print("CAPPY_RUNPOD_SMOKE=PASS")


if __name__ == "__main__":
    main()
