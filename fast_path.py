"""Exact H3 acceleration: same arithmetic, less of it.

Everything here is gated on producing **bitwise identical** output. That is a
deliberately harsher bar than the residual cache in `h3_patch.py`, which is an
approximation and is measured as one. The reason for the split is what the
profile says about where a dense BF16 step actually goes on an RTX PRO 6000
Blackwell at 864x480x49, 20 steps (1000 ms/step on CUDA 13):

| component                     | ms/step | share | headroom                  |
| ----------------------------- | ------- | ----- | ------------------------- |
| dense linear GEMM             |   690   | 69.0% | none, ~80% of BF16 peak   |
| flash attention               |   205   | 20.5% | none, best backend here   |
| modulation + RMSNorm          |    53   |  5.3% | needs a fused kernel      |
| AdaLN projection              |    20   |  2.0% | tried, see below          |
| swiglu, fused RoPE, remainder |    32   |  3.2% | little                    |

Nine tenths of a step is already at the hardware ceiling, so there is no large
exact win to be had. What is left is genuine but small, and the honest way to
ship it is to prove it changes nothing.

So this module does two modest things:

1. **Render-invariant caching.** The RoPE rotation table, the token-refiner
   output and the conditioning rows depend on the layout and the prompt, not on
   the timestep, yet the stock forward rebuilds them on every model evaluation.
   Reusing them is exact. Measured at 1.00x with one evaluation per step: the
   work is real but it is under a millisecond against a one-second step. It
   should matter more under CFG, which evaluates each timestep twice. Untested
   there.
2. **Reporting whether the fused CUDA 13 kernel is active**, which is worth
   10.5% and fails silently. This is the useful part.

**AdaLN batching was tried and removed** (see git history at a0ba236). The
projection is a `[M, 2688] @ [2688, 96768]` GEMM — 40% of the model's parameters
driven by a handful of rows, so pure weight-read bandwidth — and every timestep
is known from the sigma schedule up front, so all of them could in principle run
as one GEMM per block. In isolation that was 6.96 ms -> 0.389 ms for 20 steps of
one block, and bitwise equal. In the model it was neither: batching changes the
GEMM's M and so the kernel cuBLAS selects, giving `max_abs` 1.953e-03 /
`mean_rel` 1.887e-06 against the stock projection — one bf16 last bit — and
forcing it past the gate anyway still measured 1.0005 s/step against 1.0013 s
stock. It should have been worth ~1.9% and was worth nothing; the reason was
never established. Do not re-attempt it without explaining that first.

The honest summary is that a model patch cannot make this materially faster
without giving something up. The large wins on this hardware are the CUDA 13
runtime (1.117x, and *more* accurate), precision (W4A4, 1.58x), and the
approximate residual cache in `h3_patch.py` (1.84x, at a cost in fine detail).
None of those is an exact model patch.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

LOG = logging.getLogger("cappy_h3_fast_path")


def fused_attention_kernel_active() -> tuple[bool, str]:
    """Report whether comfy-kitchen's fused QK-RMSNorm+RoPE CUDA kernel will run.

    `comfy/quant_ops.py` calls `ck.registry.disable("cuda")` whenever torch is
    built against CUDA < 13. In current ComfyUI that also switches off the fused
    kernel used by `Attention.forward` on the **bf16** path, not merely the
    quantized layouts. Measured cost of losing it at the production shape:
    memory-bound glue rises from 97 to 193 ms/step, a 10.5% slower step.

    This is worth checking loudly. The regression is invisible — same output
    shapes, same node graph, no warning that mentions speed — and it silently
    cost this project an entire round of CUDA 13 conclusions.
    """
    version = getattr(torch.version, "cuda", None)
    if version is None:
        return False, "torch is not built against CUDA"
    try:
        parsed = tuple(int(part) for part in str(version).split("."))
    except ValueError:
        return False, f"unparseable CUDA version {version!r}"
    if parsed < (13,):
        return False, (f"torch is built against CUDA {version}; comfy-kitchen's CUDA "
                       "backend is disabled below 13.0, which also disables the fused "
                       "QK-RMSNorm+RoPE kernel on the bf16 path (measured: 10.5% slower)")
    return True, f"CUDA {version}"


class ExactFastPath:
    """Bitwise-exact reuse of work the stock forward repeats."""

    def __init__(self, cache_invariants: bool = True) -> None:
        self.cache_invariants = cache_invariants
        self.reset()

    def reset(self) -> None:
        self._rope: dict[Any, torch.Tensor] = {}
        self._text: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        self.invariant_hits = 0

    def summary(self) -> str:
        return f"{self.invariant_hits} render-invariant reuses"

    # -- render-invariant tensors -------------------------------------------

    def rope_table(self, key: Any, build) -> torch.Tensor:
        if not self.cache_invariants:
            return build()
        cached = self._rope.get(key)
        if cached is None:
            self._rope[key] = cached = build()
        else:
            self.invariant_hits += 1
        return cached

    def text_states(self, context: torch.Tensor, build):
        """Cache the refined text embedding for one conditioning tensor.

        Keyed on identity, not content: hashing a conditioning tensor every step
        would cost more than the refiner it saves. The tensor itself is held in
        the key so its storage cannot be freed and the address cannot be
        recycled underneath us, which is the failure this would otherwise have.
        """
        if not self.cache_invariants:
            return build()
        key = (context.data_ptr(), tuple(context.shape), context.dtype)
        entry = self._text.get(key)
        if entry is None:
            self._text[key] = entry = (context, build())
        else:
            self.invariant_hits += 1
        return entry[1]
