"""Judge a performance candidate against a dense reference, tap by tap.

Reports the **first divergence in pipeline order**. Everything downstream of a
real divergence is uninterpretable, so fix that boundary and re-run rather than
reading the whole table.

Two verdict modes, because optimization admits a stronger claim than porting:

* `--exact` (default): both arms are the same PyTorch on the same GPU, so an
  algebraically exact candidate must be **bitwise** identical. Any drift at all
  is a finding that needs an explanation, even if it is numerically tiny.
* `--class`: judge against the measured tolerance classes instead, for a
  candidate that legitimately changes arithmetic (a different kernel, a
  different precision). Shallow and mid blocks localise; block 49 and the
  sampled latents are advisory, because the honest bf16 spread there is large.

    python tests/compare_taps.py --reference cu128_prod.pt --candidate cu130_prod.pt
"""

from __future__ import annotations

import argparse
import re

import torch

# Tolerances follow the H3_Swift parity classes: tight where a bug is
# localisable, advisory where the legitimate bf16 spread is already wide.
CLASSES: tuple[tuple[str, float, float, bool], ...] = (
    (r"block_(00|01)$", 0.9999, 5e-3, False),
    (r"block_24$", 0.9999, 5e-3, False),
    (r"block_49$", 0.997, 1e-1, True),
    (r"final_layer\.", 0.997, 1e-1, True),
    (r"(state|denoised)\.", 0.997, 1e-1, True),
)
DEFAULT_CLASS = (0.999, 5e-3, False)


def classify(name: str) -> tuple[float, float, bool]:
    for pattern, cos, rel, advisory in CLASSES:
        if re.search(pattern, name):
            return cos, rel, advisory
    return DEFAULT_CLASS


def step_of(name: str) -> int:
    match = re.match(r"step(\d+)\.", name)
    return int(match.group(1)) if match else 0


def rank(name: str) -> tuple[int, int]:
    """Pipeline order: by step, then by depth through the graph."""
    depth = 99
    block = re.search(r"block_(\d+)", name)
    if block:
        depth = int(block.group(1))
    elif "final_layer" in name:
        depth = 90
    elif "denoised" in name:
        depth = 95
    elif "state" in name:
        depth = 96
    return step_of(name), depth


def measure(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    if a.shape != b.shape:
        return {"shape_mismatch": f"{tuple(a.shape)} vs {tuple(b.shape)}"}
    exact = bool(torch.equal(a, b))
    # These tensors run to tens of millions of elements with magnitudes in the
    # thousands. Reducing in fp32 produced cosines above 1.0, i.e. the metric
    # was noisier than the difference it was measuring. Reduce in float64.
    x, y = a.double().flatten(), b.double().flatten()
    diff = x - y
    rel_rms = (diff.pow(2).mean().sqrt() / (y.pow(2).mean().sqrt() + 1e-30)).item()
    cos = (torch.dot(x, y) / (x.norm() * y.norm() + 1e-30)).item()
    return {"exact": exact, "rel_rms": rel_rms, "cos": cos,
            "max_abs": diff.abs().max().item(),
            "differing_frac": (diff != 0).double().mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=("exact", "class"), default="exact")
    parser.add_argument("--all", action="store_true", help="Print every tap, not just the head.")
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    for label, payload in (("reference", reference), ("candidate", candidate)):
        environment = payload["environment"]
        print(f"{label:10s} torch={environment['torch']} cuda={environment['cuda']} "
              f"weights={environment['weight_set']} "
              f"median_step={environment.get('median_step_seconds')}")
    speedup = None
    ref_step = reference["environment"].get("median_step_seconds")
    cand_step = candidate["environment"].get("median_step_seconds")
    if ref_step and cand_step:
        speedup = ref_step / cand_step
        print(f"\nstep time {ref_step:.4f} -> {cand_step:.4f} s  ({speedup:.3f}x, "
              f"{100 * (cand_step - ref_step) / ref_step:+.1f}%)")

    shared = sorted(set(reference["taps"]) & set(candidate["taps"]), key=rank)
    missing = sorted(set(reference["taps"]) ^ set(candidate["taps"]))
    if missing:
        print(f"\nWARNING: {len(missing)} taps present in only one arm: {missing[:6]}")
    if not shared:
        raise SystemExit("no shared taps: the two runs do not describe the same graph")

    print(f"\n{'tap':34s} {'exact':>6} {'rel_rms':>11} {'cos':>13} {'max_abs':>10} {'verdict':>9}")
    failures = []
    for name in shared:
        result = measure(reference["taps"][name], candidate["taps"][name])
        if "shape_mismatch" in result:
            print(f"{name:34s}   SHAPE MISMATCH {result['shape_mismatch']}")
            failures.append((name, result))
            continue
        min_cos, max_rel, advisory = classify(name)
        if args.mode == "exact":
            passed = result["exact"]
        else:
            passed = result["cos"] >= min_cos and result["rel_rms"] <= max_rel
        verdict = "pass" if passed else ("ADVISORY" if advisory else "FAIL")
        if not passed and not advisory:
            failures.append((name, result))
        if args.all or not passed or name == shared[0]:
            print(f"{name:34s} {str(result['exact']):>6} {result['rel_rms']:11.3e} "
                  f"{result['cos']:13.10f} {result['max_abs']:10.3e} {verdict:>9}")

    print()
    exact_count = sum(1 for name in shared
                      if measure(reference["taps"][name], candidate["taps"][name]).get("exact"))
    print(f"{exact_count}/{len(shared)} taps bitwise identical")
    if failures:
        first = min(failures, key=lambda item: rank(item[0]))
        print(f"\nFIRST DIVERGENCE (pipeline order): {first[0]}")
        print(f"  {first[1]}")
        print(f"  {len(failures)} tap(s) outside the {args.mode} gate. Fix this boundary "
              f"first; downstream taps are not interpretable until it is explained.")
        raise SystemExit(1)
    print(f"\nPASS: all {len(shared)} shared taps satisfy the '{args.mode}' gate.")
    if speedup:
        print(f"Speed: {speedup:.3f}x at the production shape.")


if __name__ == "__main__":
    main()
