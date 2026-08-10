"""Capture named DiT taps so a performance candidate can be judged, not guessed.

Adapted from the H3_Swift parity process (`docs/SWIFT_CANDIDATE_CONTRACT.md`).
The premise carries over unchanged: a change is accepted because a tap says so,
in pipeline order, at the first boundary that moves — not because the render
"looks fine" or the process wall-clock dropped.

Two differences from the port work, because this is optimization not porting:

* Both arms run the same PyTorch on the same GPU, so an algebraically exact
  candidate should be **bitwise** identical, not merely inside a tolerance
  class. `compare_taps.py` reports exact-match first and treats any drift on a
  supposedly-exact change as a failure to explain.
* Taps are captured per sampler step as well as per block, because a
  performance change that is exact at step 1 can still diverge once its own
  output is fed back in.

Start with `--tiny`. It runs the whole graph — patchify, refiner, all 50
blocks, both AdaLN paths, RoPE, final layer, the Euler update — in seconds. A
production render is the worst first test.

    /runpod-volume/ComfyUI/venv-cu130/bin/python \
      custom_nodes/cappy_node/tests/runpod_taps.py \
      --out /runpod-volume/taps/cu130_tiny.pt --tiny

Production shape, matching the documented validation geometry:

    ... --out /runpod-volume/taps/cu130_prod.pt --steps 20 --seed 8421
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any

COMFY = os.environ.get("COMFY_DIR", ".")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import logging

import torch

import comfy.model_management
import comfy.sd
import folder_paths
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

LIBRARY_PROMPT = (
    "A fast cinematic tracking shot inside a vast historic library. A bright red "
    "sports car races at high speed between towering bookcases, books and dust whirl "
    "in its wake, dramatic warm window light, sharp car, natural motion blur in the "
    "shelves, coherent architecture, detailed realistic film."
)


class TapRecorder:
    """Collect named tensors in pipeline order for one sampler run."""

    def __init__(self, blocks: tuple[int, ...], steps: tuple[int, ...] | None) -> None:
        self.blocks = blocks
        self.steps = steps
        self.eval_index = 0
        self.taps: dict[str, torch.Tensor] = {}
        self.order: list[str] = []

    def wanted(self) -> bool:
        return self.steps is None or self.eval_index in self.steps

    def record(self, name: str, tensor: torch.Tensor) -> None:
        key = f"step{self.eval_index:03d}.{name}"
        if key in self.taps:
            return
        # Blocks mutate the residual stream in place and return the same object,
        # so every tap must be cloned at capture time or they all alias the
        # final state. Stored in native dtype: a cast would hide bit differences.
        self.taps[key] = tensor.detach().to("cpu", copy=True)
        self.order.append(key)


def install_taps(dit: Any, recorder: TapRecorder) -> None:
    def pre_forward(_module, _args, _kwargs):
        recorder.eval_index += 1

    dit.register_forward_pre_hook(pre_forward, with_kwargs=True)

    for index in recorder.blocks:
        if index >= len(dit.blocks):
            continue

        def block_hook(_module, _inputs, output, _index=index):
            if recorder.wanted():
                recorder.record(f"block_{_index:02d}", output)

        dit.blocks[index].register_forward_hook(block_hook)

    def final_hook(_module, _inputs, output):
        if recorder.wanted():
            recorder.record("final_layer.video", output[0])
            recorder.record("final_layer.audio", output[1])

    dit.final_layer.register_forward_hook(final_hook)


def load_conditioning(prompt: str, cache_path: str | None) -> Any:
    if cache_path and os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("prompt") == prompt:
            print(f"[taps] reusing cached conditioning from {cache_path}")
            return payload["conditioning"]
    clip_path = os.path.join(COMFY, "models", "text_encoders",
                             "qwen3vl_32b_minimax_h3_bf16.safetensors")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.MINIMAX)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    del clip
    gc.collect()
    comfy.model_management.unload_all_models()
    torch.cuda.empty_cache()
    if cache_path:
        torch.save({"prompt": prompt, "conditioning": conditioning}, cache_path)
    return conditioning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=LIBRARY_PROMPT)
    parser.add_argument("--cond-cache")
    parser.add_argument("--tiny", action="store_true",
                        help="Whole-graph smoke shape: 64x64, 5 frames, 2 steps.")
    parser.add_argument("--weight-set", choices=("bf16", "pruned-int8"), default="bf16")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--length", type=int, default=49)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--blocks", default="0,1,24,49",
                        help="Block taps. Shallow and mid blocks localise; block 49 is advisory.")
    parser.add_argument("--tap-steps", default="1,2,10,20",
                        help="Model evaluations to tap, 1-based. 'all' taps every step.")
    parser.add_argument("--fast-path", action="store_true",
                        help="Apply CappyMiniMaxH3FastPath. Taps must stay bitwise identical.")
    parser.add_argument("--cache", action="store_true",
                        help="Apply the approximate residual cache instead of the exact fast path.")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--cap", type=int, default=5)
    parser.add_argument("--batch-adaln", action="store_true",
                        help="Enable the AdaLN batching experiment (measured 1.00x; off by default).")
    parser.add_argument("--verify-exact", action="store_true",
                        help="Make the fast path assert its own bitwise equality as it runs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.tiny:
        args.width, args.height, args.length, args.steps = 64, 64, 5, 2
        args.tap_steps = "all"

    blocks = tuple(int(value) for value in args.blocks.split(",") if value != "")
    steps = None if args.tap_steps == "all" else tuple(
        int(value) for value in args.tap_steps.split(",") if value != "")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    conditioning = load_conditioning(args.prompt, args.cond_cache)
    weights = {"bf16": "minimax_h3_fl2va_bf16.safetensors",
               "pruned-int8": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"}
    model = comfy.sd.load_diffusion_model(
        os.path.join(COMFY, "models", "diffusion_models", weights[args.weight_set]))
    if args.cache:
        from cappy_node.nodes import CappyMiniMaxH3AudioAwareCache
        model = CappyMiniMaxH3AudioAwareCache().patch(
            model=model, relative_threshold=args.threshold, max_consecutive_reuses=args.cap,
            cache_device="auto", trace="per_step")[0]
        print(f"[taps] residual cache applied (threshold={args.threshold}, cap={args.cap})")
    if args.fast_path:
        from cappy_node.nodes import CappyMiniMaxH3FastPath
        model = CappyMiniMaxH3FastPath().patch(
            model=model, batch_adaln=args.batch_adaln, cache_invariants=True,
            verify_exact=args.verify_exact)[0]
        print("[taps] CappyMiniMaxH3FastPath applied")

    recorder = TapRecorder(blocks, steps)
    install_taps(model.model.diffusion_model, recorder)

    latent, _ = _empty_av_latent(args.width, args.height, args.length)
    noise = RandomNoise.execute(noise_seed=args.seed).result[0]
    sampler = KSamplerSelect.execute(sampler_name="res_multistep").result[0]
    sigmas = BasicScheduler.execute(model=model, scheduler="simple", steps=args.steps,
                                    denoise=1.0).result[0]
    guider = BasicGuider.execute(model=model, conditioning=conditioning).result[0]

    original_prepare_callback = custom_sampler_nodes.latent_preview.prepare_callback
    step_seconds: list[float] = []

    def to_parts(value: Any) -> Any:
        if getattr(value, "is_nested", False):
            return tuple(part.detach().cpu() for part in value.unbind())
        return value.detach().cpu()

    def tapped_prepare_callback(*callback_args: Any, **callback_kwargs: Any):
        original_callback = original_prepare_callback(*callback_args, **callback_kwargs)
        marker = {"last": None}

        def callback(step: int, x0: Any, x: Any, total_steps: int) -> None:
            original_callback(step, x0, x, total_steps)
            torch.cuda.synchronize()
            now = time.perf_counter()
            if marker["last"] is not None:
                step_seconds.append(now - marker["last"])
            marker["last"] = now
            if steps is None or (step + 1) in steps:
                state, denoised = to_parts(x), to_parts(x0)
                for label, parts in (("state", state), ("denoised", denoised)):
                    if isinstance(parts, tuple):
                        recorder.taps[f"step{step + 1:03d}.{label}.video"] = parts[0]
                        recorder.taps[f"step{step + 1:03d}.{label}.audio"] = parts[1]
                        recorder.order += [f"step{step + 1:03d}.{label}.video",
                                           f"step{step + 1:03d}.{label}.audio"]
                    else:
                        recorder.taps[f"step{step + 1:03d}.{label}"] = parts
                        recorder.order.append(f"step{step + 1:03d}.{label}")
        return callback

    custom_sampler_nodes.latent_preview.prepare_callback = tapped_prepare_callback
    started = time.perf_counter()
    try:
        SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler,
                                      sigmas=sigmas, latent_image=latent)
        torch.cuda.synchronize()
    finally:
        custom_sampler_nodes.latent_preview.prepare_callback = original_prepare_callback
    total = time.perf_counter() - started

    clean = sorted(step_seconds)[len(step_seconds) // 4: max(1, 3 * len(step_seconds) // 4)] or step_seconds
    environment = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "fast_path": args.fast_path, "cache": args.cache,
        "threshold": args.threshold, "cap": args.cap,
        "weight_set": args.weight_set, "width": args.width, "height": args.height,
        "length": args.length, "steps": args.steps, "seed": args.seed,
        "prompt": args.prompt, "blocks": list(blocks),
        "tap_steps": "all" if steps is None else list(steps),
        "sampler_seconds": total,
        "step_seconds": step_seconds,
        "median_step_seconds": sorted(step_seconds)[len(step_seconds) // 2] if step_seconds else None,
        "interquartile_mean_step_seconds": sum(clean) / len(clean) if clean else None,
    }
    torch.save({"environment": environment, "order": recorder.order, "taps": recorder.taps}, args.out)
    print(json.dumps(environment, indent=2))
    print(f"[taps] wrote {len(recorder.taps)} taps to {args.out}")


if __name__ == "__main__":
    main()
