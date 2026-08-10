"""Fresh-process MiniMax H3 VAE decode for a latent saved by runpod_render.py."""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

COMFY = os.environ.get("COMFY_DIR", ".")
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch

import comfy.sd
import comfy.utils
import nodes
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from comfy_api.latest._util.video_types import VideoCodec, VideoComponents, VideoContainer
from comfy_extras.nodes_audio import VAEDecodeAudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--video-decode-device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = torch.load(args.latent, map_location="cpu", weights_only=False)
    video_vae_options = {}
    if args.video_decode_device == "cpu":
        video_vae_options = {"device": torch.device("cpu"), "dtype": torch.float32}
    vvae = comfy.sd.VAE(
        sd=comfy.utils.load_torch_file(
            os.path.join(COMFY, "models", "vae", "minimax_h3_video_vae_fp16.safetensors")),
        **video_vae_options,
    )
    images = nodes.VAEDecode().decode(vae=vvae, samples=result)[0]
    avae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        os.path.join(COMFY, "models", "vae", "minimax_h3_audio_vae_fp32.safetensors")))
    audio = VAEDecodeAudio.execute(vae=avae, samples=result).result[0]
    # Comfy's video mux converts the waveform to NumPy; detach VAE output
    # because this standalone harness does not run inside the UI's no-grad path.
    audio = {**audio, "waveform": audio["waveform"].detach()}
    VideoFromComponents(VideoComponents(images=images.float().cpu(), audio=audio,
                                        frame_rate=Fraction(24))).save_to(
        args.out, format=VideoContainer.AUTO, codec=VideoCodec.AUTO)
    print(f"CAPPY_DECODE_PASS out={args.out}")


if __name__ == "__main__":
    main()
