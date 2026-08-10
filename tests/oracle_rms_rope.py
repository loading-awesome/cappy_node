"""Operator oracle: is comfy-kitchen's fused QK-RMSNorm+RoPE more accurate?

`Attention.forward` calls `comfy.quant_ops.ck.rms_rope_split_half_` on the bf16
path. When comfy-kitchen's CUDA backend is disabled — which is what
`ck.registry.disable("cuda")` does on any torch built against CUDA < 13 — the
same call runs an unfused fallback instead. The two are not bitwise equal, so
"which is right" cannot be settled by comparing them to each other.

This settles it against a float64 reference computed from the documented math:
RMSNorm over head_dim with the module's eps, then split-half RoPE pairing
`(i, i + rot/2)` — not interleaved.

The input tensors are generated once and written to disk, then reused by every
environment. A seed is not a shared input across runtimes; the tensor is.

    ./venv-cu130/bin/python tests/oracle_rms_rope.py --fixture /runpod-volume/taps/rope_fixture.pt
    ./venv/bin/python      tests/oracle_rms_rope.py --fixture /runpod-volume/taps/rope_fixture.pt
"""

from __future__ import annotations

import argparse
import os
import sys

COMFY = os.environ.get("COMFY_DIR", ".")
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch

import comfy.quant_ops
from comfy.ldm.minimax.model import rope_rotation_table


def make_fixture(path: str, seq: int, heads: int, head_dim: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260809)
    angles = torch.rand(seq, head_dim, generator=generator) * 6.28318530717958
    angles[:, head_dim // 2:] = angles[:, :head_dim // 2]  # rope_rotation_table expects duplicated halves
    fixture = {
        "q": torch.randn(1, seq, heads, head_dim, generator=generator).to(torch.bfloat16),
        "k": torch.randn(1, seq, heads, head_dim, generator=generator).to(torch.bfloat16),
        "angles": angles,
        "q_weight": torch.randn(head_dim, generator=generator).to(torch.bfloat16),
        "k_weight": torch.randn(head_dim, generator=generator).to(torch.bfloat16),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(fixture, path)
    return fixture


def reference(x: torch.Tensor, table: torch.Tensor, weight: torch.Tensor,
              eps: float, rot_dim: int) -> torch.Tensor:
    """float64 ground truth for RMSNorm-over-head_dim followed by split-half RoPE."""
    value = x.double()
    normed = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps) * weight.double()
    half = rot_dim // 2
    # rope_rotation_table packs [[cos, -sin], [sin, cos]] into the trailing 2x2.
    cos = table[..., 0, 0].double()
    sin = table[..., 1, 0].double()
    lower, upper = normed[..., :half], normed[..., half:rot_dim]
    out = normed.clone()
    out[..., :half] = cos * lower - sin * upper
    out[..., half:rot_dim] = sin * lower + cos * upper
    return out


def error(candidate: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    x, y = candidate.double().flatten(), truth.flatten()
    diff = x - y
    return {
        "rel_rms": (diff.pow(2).mean().sqrt() / y.pow(2).mean().sqrt()).item(),
        "max_abs": diff.abs().max().item(),
        "cos": (torch.dot(x, y) / (x.norm() * y.norm())).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--seq", type=int, default=7126)
    parser.add_argument("--heads", type=int, default=112)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--save-output", help="Write this environment's kernel output for cross-env diff.")
    args = parser.parse_args()

    if os.path.exists(args.fixture):
        fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
        print(f"[oracle] loaded fixture {args.fixture}")
    else:
        fixture = make_fixture(args.fixture, args.seq, args.heads, args.head_dim)
        print(f"[oracle] created fixture {args.fixture}")

    backend = "enabled"
    try:
        cuda_version = tuple(map(int, str(torch.version.cuda).split(".")))
        backend = "enabled" if cuda_version >= (13,) else "DISABLED (cu<13)"
    except Exception:
        backend = "unknown"
    print(f"[oracle] torch={torch.__version__} cuda={torch.version.cuda} "
          f"comfy-kitchen cuda backend: {backend}")

    device = torch.device("cuda")
    q = fixture["q"].to(device).clone()
    k = fixture["k"].to(device).clone()
    q_weight = fixture["q_weight"].to(device)
    k_weight = fixture["k_weight"].to(device)
    table = rope_rotation_table(fixture["angles"].to(device), torch.bfloat16)
    rot_dim = table.shape[-3] * 2

    truth_q = reference(fixture["q"].to(device), table, q_weight, args.eps, rot_dim)
    truth_k = reference(fixture["k"].to(device), table, k_weight, args.eps, rot_dim)

    comfy.quant_ops.ck.rms_rope_split_half_(q, k, table, q_weight, k_weight,
                                            epsilon=args.eps, rot_dim=rot_dim)
    torch.cuda.synchronize()

    print(f"\nrot_dim={rot_dim} table={tuple(table.shape)} q={tuple(q.shape)}")
    print(f"{'tensor':>8}  {'rel_rms vs fp64':>16}  {'max_abs':>11}  {'cos':>15}")
    for name, candidate, truth in (("q", q, truth_q), ("k", k, truth_k)):
        stats = error(candidate, truth)
        print(f"{name:>8}  {stats['rel_rms']:16.6e}  {stats['max_abs']:11.4e}  {stats['cos']:15.12f}")

    # A bf16 result cannot beat the spacing of its own format. Quantising the
    # fp64 truth to bf16 gives the best any correct implementation could do.
    for name, truth in (("q", truth_q), ("k", truth_k)):
        floor = error(truth.to(torch.bfloat16), truth)
        print(f"{name:>8}  bf16 representation floor: rel_rms {floor['rel_rms']:.6e}")

    if args.save_output:
        torch.save({"q": q.cpu(), "k": k.cpu(), "torch": torch.__version__,
                    "cuda": torch.version.cuda}, args.save_output)
        print(f"[oracle] wrote kernel output to {args.save_output}")


if __name__ == "__main__":
    main()
