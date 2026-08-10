"""Find the achievable ceiling for the two kernels that own 90% of a DiT step.

Diagnostic only. Per the project rule, a speedup is never accepted on a
microbenchmark — anything promising here must then be validated end to end with
`runpod_taps.py` / `compare_taps.py`. What a microbenchmark *can* legitimately
answer is "is the current dispatch leaving anything on the table", which is a
question about the ceiling, not about a candidate.

Shapes are the measured production shape: 7126 packed tokens, hidden 5376,
inner 7168 (112 heads x 64), FFN 14336, 50 blocks.
"""

from __future__ import annotations

import argparse
import os
import sys

COMFY = os.environ.get("COMFY_DIR", ".")
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

S, HIDDEN, INNER, FFN, BLOCKS = 7126, 5376, 7168, 14336, 50
LINEARS = (("qkv", HIDDEN, 3 * INNER), ("out", INNER, HIDDEN),
           ("fc1", HIDDEN, 2 * FFN), ("fc2", FFN, HIDDEN))


def timed(fn, iterations: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def bench_gemm() -> None:
    print("=== dense linear GEMM ===")
    print(f"{'layer':>6} {'M':>6} {'K':>6} {'N':>6} {'ms':>8} {'TFLOP/s':>9}  {'x50 ms/step':>12}")
    total = 0.0
    for name, k, n in LINEARS:
        x = torch.randn(S, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ms = timed(lambda: F.linear(x, w))
        tflops = 2 * S * k * n / 1e12 / (ms / 1000)
        total += ms * BLOCKS
        print(f"{name:>6} {S:6d} {k:6d} {n:6d} {ms:8.3f} {tflops:9.1f}  {ms * BLOCKS:12.1f}")
    print(f"{'TOTAL':>6} {'':>28} {total:12.1f} ms/step\n")

    print("--- reduced-precision reduction toggle (changes numerics) ---")
    x = torch.randn(S, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2 * FFN, HIDDEN, device="cuda", dtype=torch.bfloat16)
    for allow in (True, False):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow
        ms = timed(lambda: F.linear(x, w))
        print(f"allow_bf16_reduced_precision_reduction={str(allow):5s}  fc1 {ms:7.3f} ms  "
              f"{2 * S * HIDDEN * 2 * FFN / 1e12 / (ms / 1000):7.1f} TFLOP/s")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    print("\n--- FP8 ceiling on the same shape (what precision would buy) ---")
    try:
        xf = x.to(torch.float8_e4m3fn)
        wf = w.to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, device="cuda")
        ms = timed(lambda: torch._scaled_mm(xf, wf.t(), scale_a=scale, scale_b=scale,
                                            out_dtype=torch.bfloat16))
        print(f"fp8 e4m3 _scaled_mm fc1 {ms:7.3f} ms  "
              f"{2 * S * HIDDEN * 2 * FFN / 1e12 / (ms / 1000):7.1f} TFLOP/s")
    except Exception as exc:
        print(f"fp8 _scaled_mm unavailable: {exc}")


def bench_attention() -> None:
    print("\n=== attention (1 x 112 heads x 7126 x 64) ===")
    heads, dim = INNER // 64, 64
    q, k, v = (torch.randn(1, heads, S, dim, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    flops = 4 * S * S * INNER
    outputs: dict[str, torch.Tensor] = {}
    backends = [("default", None), ("flash", SDPBackend.FLASH_ATTENTION),
                ("cudnn", SDPBackend.CUDNN_ATTENTION), ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION)]
    print(f"{'backend':>15} {'ms':>8} {'TFLOP/s':>9} {'x50 ms/step':>12}")
    for label, backend in backends:
        try:
            if backend is None:
                run = lambda: F.scaled_dot_product_attention(q, k, v)
            else:
                def run(_backend=backend):
                    with sdpa_kernel(_backend):
                        return F.scaled_dot_product_attention(q, k, v)
            ms = timed(run, iterations=10, warmup=3)
            outputs[label] = run()
            print(f"{label:>15} {ms:8.3f} {flops / 1e12 / (ms / 1000):9.1f} {ms * BLOCKS:12.1f}")
        except Exception as exc:
            print(f"{label:>15}  unavailable: {str(exc)[:70]}")

    if "flash" in outputs:
        print("\n--- agreement with the flash kernel currently in use ---")
        base = outputs["flash"].double()
        for label, out in outputs.items():
            if label == "flash":
                continue
            diff = (out.double() - base)
            rel = (diff.pow(2).mean().sqrt() / base.pow(2).mean().sqrt()).item()
            print(f"{label:>15}  bitwise={torch.equal(out, outputs['flash'])!s:>5}  rel_rms={rel:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gemm", action="store_true")
    args = parser.parse_args()
    print(f"torch {torch.__version__} cuda {torch.version.cuda} "
          f"{torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability()))}\n")
    if not args.skip_gemm:
        bench_gemm()
    bench_attention()


if __name__ == "__main__":
    main()
