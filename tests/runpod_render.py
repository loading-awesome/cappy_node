"""Render a real decoded MiniMax H3 MP4, optionally through the Cappy node.

This is intentionally a manual RunPod validation utility, not a unit test.
Use identical arguments once without --cache and once with --cache, then
compare the resulting MP4s visually and with ffmpeg.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from fractions import Fraction

COMFY = os.environ.get("COMFY_DIR", ".")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import torch

import comfy.sd
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from comfy_api.latest._util.video_types import VideoCodec, VideoComponents, VideoContainer
from comfy_extras.nodes_audio import VAEDecodeAudio
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_minimax_h3 import _empty_av_latent
from cappy_node.nodes import CappyMiniMaxH3AudioAwareCache


PROMPT = (
    "A realistic close-up of a person speaking clearly to camera in a softly lit studio. "
    "Natural lip movement, subtle head motion, stable facial features, cinematic detail. "
    "Audio: a calm clear speaking voice, no music."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--cap", type=int, default=5)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--length", type=int, default=49)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    started = time.monotonic()

    clip_path = os.path.join(COMFY, "models", "text_encoders", "qwen3vl_32b_minimax_h3_bf16.safetensors")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.MINIMAX)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    del clip
    gc.collect()

    dit_path = os.path.join(COMFY, "models", "diffusion_models", "minimax_h3_fl2va_bf16.safetensors")
    model = comfy.sd.load_diffusion_model(dit_path)
    if args.cache:
        model = CappyMiniMaxH3AudioAwareCache().patch(
            model=model, relative_threshold=args.threshold,
            max_consecutive_reuses=args.cap, cache_device="auto", trace="per_step",
        )[0]
    latent, _ = _empty_av_latent(args.width, args.height, args.length)
    noise = RandomNoise.execute(noise_seed=args.seed).result[0]
    sampler = KSamplerSelect.execute(sampler_name="res_multistep").result[0]
    sigmas = BasicScheduler.execute(model=model, scheduler="simple", steps=args.steps, denoise=1.0).result[0]
    guider = BasicGuider.execute(model=model, conditioning=conditioning).result[0]
    result = SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler,
                                           sigmas=sigmas, latent_image=latent).result[0]
    video_latent, audio_latent = result["samples"].unbind()
    assert torch.isfinite(video_latent).all() and torch.isfinite(audio_latent).all()

    # The bf16 DiT occupies ~63 GB. Free it before loading either VAE or an
    # otherwise-valid H3 decode can OOM even on a 96 GB GPU.
    del model, guider, sampler, sigmas, latent, noise, conditioning
    gc.collect()
    torch.cuda.empty_cache()

    vvae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        os.path.join(COMFY, "models", "vae", "minimax_h3_video_vae_fp16.safetensors")))
    images = nodes.VAEDecode().decode(vae=vvae, samples=result)[0]
    del vvae
    gc.collect()
    avae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        os.path.join(COMFY, "models", "vae", "minimax_h3_audio_vae_fp32.safetensors")))
    audio = VAEDecodeAudio.execute(vae=avae, samples=result).result[0]
    VideoFromComponents(VideoComponents(images=images.float().cpu(), audio=audio,
                                        frame_rate=Fraction(24))).save_to(
        args.out, format=VideoContainer.AUTO, codec=VideoCodec.AUTO)
    print(f"CAPPY_RENDER_PASS cache={args.cache} seconds={time.monotonic() - started:.1f} out={args.out}")


if __name__ == "__main__":
    main()
