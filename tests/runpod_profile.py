"""Profile the dense BF16 MiniMax H3 DiT at the production render shape.

This is a manual RunPod investigation utility, not a unit test. It answers one
question: where does dense DiT step time actually go on this GPU?

It runs a real sampler at the production shape and reports three views:

1. Clean per-step wall time from a window where the profiler is OFF, so the
   headline seconds/step number is not distorted by profiling overhead.
2. A role-level breakdown (qkv/attention/out/mlp/adaln/norm/modulation) that
   aggregates all 50 DiT blocks, so a percentage means "share of a step".
3. The raw CUDA kernel and aten operator tables, so a claim about GEMM shape or
   dispatch can be checked against the kernel that actually ran.

Instrumentation is installed by wrapping module-level callables and registering
leaf-module hooks. It never edits the shipped model or `h3_patch.py`, and the
scopes are inert until the profiler window opens.

Run the standard dense profile:

    cd /runpod-volume/ComfyUI
    source venv-cu130/bin/activate
    COMFY_DIR=/runpod-volume/ComfyUI python \
      custom_nodes/cappy_node/tests/runpod_profile.py \
      --report-dir /runpod-volume/profile_bf16_dense \
      --cond-cache /runpod-volume/cond_library_bf16.pt
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import statistics
import sys
import time
from typing import Any

COMFY = os.environ.get("COMFY_DIR", ".")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

import comfy.model_management
import comfy.ops
import comfy.sd
import folder_paths
from comfy.ldm.minimax import model as minimax_model
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

try:
    import comfy.quant_ops
except ImportError:  # pragma: no cover - depends on the Comfy build
    comfy.quant_ops = None  # type: ignore[attr-defined]


LIBRARY_PROMPT = (
    "A fast cinematic tracking shot inside a vast historic library. A bright red "
    "sports car races at high speed between towering bookcases, books and dust whirl "
    "in its wake, dramatic warm window light, sharp car, natural motion blur in the "
    "shelves, coherent architecture, detailed realistic film."
)

# Scopes cost a real op each, so they stay off during the clean timing window
# and are only armed while the profiler is actually recording.
PROFILING = False
SHAPES: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)


def _scope(name: str, fn):
    def wrapped(*args: Any, **kwargs: Any):
        if not PROFILING:
            return fn(*args, **kwargs)
        with record_function(name):
            return fn(*args, **kwargs)
    return wrapped


def install_function_scopes() -> None:
    """Label the block-internal callables that are functions, not modules."""
    minimax_model.optimized_attention = _scope("h3.attn.sdpa", minimax_model.optimized_attention)
    minimax_model._mod_scale_shift = _scope("h3.mod_scale_shift", minimax_model._mod_scale_shift)
    minimax_model._mod_gate = _scope("h3.mod_gate", minimax_model._mod_gate)
    comfy.ops.linear_input_act = _scope("h3.mlp.fc2_swiglu", comfy.ops.linear_input_act)
    if comfy.quant_ops is not None and hasattr(comfy.quant_ops, "ck"):
        kernels = comfy.quant_ops.ck
        for attr in ("rms_rope_split_half_", "rms_rope_split_half"):
            if hasattr(kernels, attr):
                setattr(kernels, attr, _scope("h3.attn.qk_rmsnorm_rope", getattr(kernels, attr)))


def install_block_hooks(dit: minimax_model.MiniMaxH3Model) -> int:
    """Scope every leaf submodule of the 50 DiT blocks, aggregated by role.

    The block index is stripped from the label so `h3.block.attn.qkv_proj`
    accumulates across all blocks and reads as a share of one step.
    """
    stack: list[Any] = []
    hooked = 0
    for block in dit.blocks:
        for name, module in block.named_modules():
            if not name or any(True for _ in module.children()):
                continue  # containers would nest scopes; only leaves are labelled
            label = f"h3.block.{name}"

            def pre_hook(_module, inputs, _label=label):
                if not PROFILING:
                    return
                if inputs and isinstance(inputs[0], torch.Tensor):
                    in_features = getattr(_module, "in_features", None)
                    out_features = getattr(_module, "out_features", None)
                    SHAPES[_label][(tuple(inputs[0].shape), str(inputs[0].dtype),
                                    in_features, out_features)] += 1
                context = record_function(_label)
                context.__enter__()
                stack.append(context)

            def post_hook(_module, _inputs, _output):
                if not PROFILING or not stack:
                    return
                stack.pop().__exit__(None, None, None)

            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
            hooked += 1
    return hooked


def load_conditioning(prompt: str, cache_path: str | None) -> Any:
    """Encode the prompt, reusing a cached encode so reruns skip the 51 GB CLIP."""
    if cache_path and os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("prompt") == prompt:
            print(f"[profile] reusing cached conditioning from {cache_path}")
            return payload["conditioning"]
        print("[profile] cached conditioning prompt differs; re-encoding")
    clip_path = os.path.join(COMFY, "models", "text_encoders",
                             "qwen3vl_32b_minimax_h3_bf16.safetensors")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.MINIMAX)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    del clip
    gc.collect()
    # The BF16 Qwen encoder is ~51 GB on CUDA. It must leave the card before the
    # ~63 GB BF16 DiT is admitted, or both contend for the same 96 GB.
    comfy.model_management.unload_all_models()
    torch.cuda.empty_cache()
    if cache_path:
        torch.save({"prompt": prompt, "conditioning": conditioning}, cache_path)
        print(f"[profile] cached conditioning to {cache_path}")
    return conditioning


def summarize(prof: profile, active_steps: int, report_dir: str) -> dict[str, Any]:
    """Reduce the profile to per-step device time by role, kernel, and operator."""
    averages = prof.key_averages()

    def per_step(microseconds: float) -> float:
        return microseconds / 1000.0 / max(1, active_steps)

    scopes, kernels, operators = [], [], []
    for entry in averages:
        name = entry.key
        row = {
            "name": name,
            "count_per_step": entry.count / max(1, active_steps),
            "device_ms_per_step": per_step(entry.device_time_total),
            "self_device_ms_per_step": per_step(entry.self_device_time_total),
            "self_cpu_ms_per_step": per_step(entry.self_cpu_time_total),
        }
        if name.startswith("h3."):
            scopes.append(row)
        elif name.startswith("aten::") or name.startswith("cudnn") or "::" in name:
            operators.append(row)
        else:
            kernels.append(row)

    # Device time is wall time on the GPU: leaf self-time is the honest total.
    total_self_device = sum(row["self_device_ms_per_step"] for row in operators + kernels)
    for row in scopes:
        row["percent_of_device"] = (100.0 * row["device_ms_per_step"] / total_self_device
                                    if total_self_device else 0.0)
    for row in kernels + operators:
        row["percent_of_device"] = (100.0 * row["self_device_ms_per_step"] / total_self_device
                                    if total_self_device else 0.0)

    scopes.sort(key=lambda row: -row["device_ms_per_step"])
    kernels.sort(key=lambda row: -row["self_device_ms_per_step"])
    operators.sort(key=lambda row: -row["self_device_ms_per_step"])

    shapes = {label: [{"shape": list(key[0]), "dtype": key[1], "in_features": key[2],
                       "out_features": key[3], "calls_per_step": count / max(1, active_steps)}
                      for key, count in counter.most_common()]
              for label, counter in SHAPES.items()}

    report = {"total_self_device_ms_per_step": total_self_device, "active_steps": active_steps,
              "scopes": scopes, "kernels": kernels, "operators": operators, "shapes": shapes}
    with open(os.path.join(report_dir, "profile_report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    return report


def print_table(title: str, rows: list[dict[str, Any]], column: str, limit: int) -> None:
    print(f"\n=== {title} ===")
    print(f"{'ms/step':>10}  {'%':>6}  {'calls/step':>10}  name")
    for row in rows[:limit]:
        print(f"{row[column]:10.2f}  {row['percent_of_device']:6.2f}  "
              f"{row['count_per_step']:10.1f}  {row['name'][:110]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--prompt", default=LIBRARY_PROMPT)
    parser.add_argument("--cond-cache", help="Reuse an encoded prompt across profiling runs.")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--length", type=int, default=49)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--timing-steps", type=int, default=8,
                        help="Profiler-off steps used for the clean seconds/step median.")
    parser.add_argument("--active-steps", type=int, default=2, help="Steps recorded by the profiler.")
    parser.add_argument("--trace", action="store_true", help="Also export a chrome trace (large).")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    conditioning = load_conditioning(args.prompt, args.cond_cache)

    install_function_scopes()
    dit_path = os.path.join(COMFY, "models", "diffusion_models", "minimax_h3_fl2va_bf16.safetensors")
    model = comfy.sd.load_diffusion_model(dit_path)
    hooked = install_block_hooks(model.model.diffusion_model)
    dit = model.model.diffusion_model
    print(f"[profile] scoped {hooked} leaf modules across {len(dit.blocks)} blocks "
          f"(hidden={dit.hidden_size if hasattr(dit, 'hidden_size') else '?'})")

    latent, _ = _empty_av_latent(args.width, args.height, args.length)
    noise = RandomNoise.execute(noise_seed=args.seed).result[0]
    sampler = KSamplerSelect.execute(sampler_name="res_multistep").result[0]
    sigmas = BasicScheduler.execute(model=model, scheduler="simple", steps=args.steps,
                                    denoise=1.0).result[0]
    guider = BasicGuider.execute(model=model, conditioning=conditioning).result[0]

    step_seconds: list[float] = []
    profiler_schedule = schedule(wait=args.timing_steps, warmup=1, active=args.active_steps, repeat=1)
    original_prepare_callback = custom_sampler_nodes.latent_preview.prepare_callback

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=profiler_schedule, record_shapes=True) as prof:
        state = {"last": None}

        def tapped_prepare_callback(*callback_args: Any, **callback_kwargs: Any):
            original_callback = original_prepare_callback(*callback_args, **callback_kwargs)

            def callback(step: int, x0: Any, x: Any, total_steps: int) -> None:
                global PROFILING
                original_callback(step, x0, x, total_steps)
                torch.cuda.synchronize()
                now = time.perf_counter()
                if state["last"] is not None:
                    step_seconds.append(now - state["last"])
                state["last"] = now
                prof.step()
                # step is 0-based and fires after the step completes, so the next
                # step is the one the schedule is about to record.
                PROFILING = (step + 1) >= args.timing_steps
            return callback

        custom_sampler_nodes.latent_preview.prepare_callback = tapped_prepare_callback
        try:
            started = time.perf_counter()
            SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler,
                                          sigmas=sigmas, latent_image=latent)
            torch.cuda.synchronize()
            total = time.perf_counter() - started
        finally:
            custom_sampler_nodes.latent_preview.prepare_callback = original_prepare_callback
            PROFILING = False

    # Step 1 carries allocator and autotune warmup; the clean window is what
    # follows it, up to the point the profiler starts recording.
    clean = step_seconds[1:args.timing_steps - 1]
    report = summarize(prof, args.active_steps, args.report_dir)
    if clean:
        report["clean_step_seconds"] = {
            "samples": clean, "median": statistics.median(clean),
            "min": min(clean), "max": max(clean),
            "stdev": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        }
    report["sampler_total_seconds"] = total
    with open(os.path.join(args.report_dir, "profile_report.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    if args.trace:
        prof.export_chrome_trace(os.path.join(args.report_dir, "trace.json"))

    print(f"\n=== clean step time (profiler off, n={len(clean)}) ===")
    if clean:
        print(f"median {statistics.median(clean):.4f} s/step   min {min(clean):.4f}   "
              f"max {max(clean):.4f}   stdev {statistics.pstdev(clean):.4f}")
    print(f"sampler total {total:.1f} s for {args.steps} steps (includes profiled steps)")
    print(f"\nprofiled device time {report['total_self_device_ms_per_step']:.1f} ms/step "
          f"over {args.active_steps} recorded steps")

    print_table("role breakdown (inclusive device time, all 50 blocks aggregated)",
                report["scopes"], "device_ms_per_step", 25)
    print_table("top CUDA kernels (self device time)", report["kernels"], "self_device_ms_per_step", 25)
    print_table("top operators (self device time)", report["operators"], "self_device_ms_per_step", 25)

    print("\n=== observed linear shapes ===")
    for label, entries in sorted(report["shapes"].items()):
        for entry in entries[:2]:
            print(f"{label:34s} in={entry['in_features']} out={entry['out_features']} "
                  f"x{entry['shape']} {entry['dtype']} calls/step={entry['calls_per_step']:.0f}")
    print(f"\nreport written to {os.path.join(args.report_dir, 'profile_report.json')}")


if __name__ == "__main__":
    main()
