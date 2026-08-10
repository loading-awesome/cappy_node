"""Render a real decoded MiniMax H3 MP4, optionally through the Cappy node.

This is intentionally a manual RunPod validation utility, not a unit test.
Use identical arguments once without --cache and once with --cache, then
compare the resulting MP4s visually and with ffmpeg.
"""

from __future__ import annotations

import argparse
import gc
import json
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
import comfy.model_management
import comfy.nested_tensor
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
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
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
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--weight-set", choices=("bf16", "pruned-int8"), default="bf16",
                        help="Matched Comfy-Org H3 pair to load. Keep both arms on the same set.")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--cap", type=int, default=5)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--length", type=int, default=49)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--latent-out", help="Save CPU latents and exit without VAE decode.")
    parser.add_argument("--trajectory-dir",
                        help="Write CPU step-state and denoised latent taps for dense/cache comparison.")
    parser.add_argument("--video-decode-device", choices=("cuda", "cpu"), default="cuda",
                        help="Use CPU RAM for the video VAE when GPU decode would exceed VRAM.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    started = time.monotonic()

    weights = {
        "bf16": (
            "qwen3vl_32b_minimax_h3_bf16.safetensors",
            "minimax_h3_fl2va_bf16.safetensors",
        ),
        "pruned-int8": (
            "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        ),
    }
    clip_name, dit_name = weights[args.weight_set]
    clip_path = os.path.join(COMFY, "models", "text_encoders", clip_name)
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.MINIMAX)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
    del clip
    gc.collect()
    # Encoding temporarily brings the 51 GB BF16 Qwen model onto CUDA. Move
    # it back to its configured CPU offload location before admitting the 66 GB
    # BF16 DiT; otherwise both models contend for the same 96 GB card.
    comfy.model_management.unload_all_models()
    torch.cuda.empty_cache()

    dit_path = os.path.join(COMFY, "models", "diffusion_models", dit_name)
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
    trajectory_restore = None
    if args.trajectory_dir:
        os.makedirs(args.trajectory_dir, exist_ok=True)
        original_prepare_callback = custom_sampler_nodes.latent_preview.prepare_callback

        def to_cpu_parts(value):
            if getattr(value, "is_nested", False):
                return tuple(part.detach().cpu() for part in value.unbind())
            return value.detach().cpu()

        def tapped_prepare_callback(*callback_args, **callback_kwargs):
            original_callback = original_prepare_callback(*callback_args, **callback_kwargs)

            def callback(step, x0, x, total_steps):
                original_callback(step, x0, x, total_steps)
                torch.save(
                    {"step": step + 1, "total_steps": total_steps,
                     "state": to_cpu_parts(x), "denoised": to_cpu_parts(x0)},
                    os.path.join(args.trajectory_dir, f"step_{step + 1:03d}.pt"),
                )
            return callback

        custom_sampler_nodes.latent_preview.prepare_callback = tapped_prepare_callback
        trajectory_restore = original_prepare_callback
        with open(os.path.join(args.trajectory_dir, "run.json"), "w") as handle:
            json.dump({"cache": args.cache, "weight_set": args.weight_set,
                       "width": args.width, "height": args.height, "length": args.length,
                       "steps": args.steps, "seed": args.seed, "prompt": args.prompt}, handle, indent=2)
    try:
        result = SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler,
                                               sigmas=sigmas, latent_image=latent).result[0]
    finally:
        if trajectory_restore is not None:
            custom_sampler_nodes.latent_preview.prepare_callback = trajectory_restore
    video_latent, audio_latent = result["samples"].unbind()
    assert torch.isfinite(video_latent).all() and torch.isfinite(audio_latent).all()

    # Keep only CPU copies of the sampled latents. The managed DiT and the VAE
    # cannot coexist with decode activations inside 96 GB at this render size.
    result = {"samples": comfy.nested_tensor.NestedTensor((
        video_latent.detach().cpu(), audio_latent.detach().cpu(),
    ))}
    del video_latent, audio_latent
    if args.latent_out:
        torch.save(result, args.latent_out)
        print(f"CAPPY_SAMPLE_PASS weights={args.weight_set} cache={args.cache} seconds={time.monotonic() - started:.1f} latent={args.latent_out}")
        return

    # The bf16 DiT occupies ~63 GB. Free it before loading either VAE or an
    # otherwise-valid H3 decode can OOM even on a 96 GB GPU.
    model.unpatch_model(device_to=torch.device("cpu"))
    del model, guider, sampler, sigmas, latent, noise, conditioning
    gc.collect()
    comfy.model_management.unload_all_models()
    torch.cuda.empty_cache()

    video_vae_options = {}
    if args.video_decode_device == "cpu":
        # This preserves the VAE's weights and math while moving its large
        # activation footprint to host RAM. It is intentionally slower.
        video_vae_options = {"device": torch.device("cpu"), "dtype": torch.float32}
    vvae = comfy.sd.VAE(
        sd=comfy.utils.load_torch_file(
            os.path.join(COMFY, "models", "vae", "minimax_h3_video_vae_fp16.safetensors")),
        **video_vae_options,
    )
    # ComfyUI's execution engine is inference-only. This standalone utility
    # must make that explicit or PyTorch retains every VAE activation.
    with torch.inference_mode():
        images = nodes.VAEDecode().decode(vae=vvae, samples=result)[0]
    del vvae
    gc.collect()
    avae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        os.path.join(COMFY, "models", "vae", "minimax_h3_audio_vae_fp32.safetensors")))
    with torch.inference_mode():
        audio = VAEDecodeAudio.execute(vae=avae, samples=result).result[0]
    # This direct harness is outside ComfyUI's normal no-grad execution path.
    # The muxer converts the waveform to NumPy, which requires a detached tensor.
    audio = {**audio, "waveform": audio["waveform"].detach()}
    VideoFromComponents(VideoComponents(images=images.float().cpu(), audio=audio,
                                        frame_rate=Fraction(24))).save_to(
        args.out, format=VideoContainer.AUTO, codec=VideoCodec.AUTO)
    print(f"CAPPY_RENDER_PASS weights={args.weight_set} cache={args.cache} seconds={time.monotonic() - started:.1f} out={args.out}")


if __name__ == "__main__":
    main()
