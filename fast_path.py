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
| AdaLN projection              |    20   |  2.0% | **this module**           |
| swiglu, fused RoPE, remainder |    32   |  3.2% | little                    |

Nine tenths of a step is already at the hardware ceiling, so there is no large
exact win to be had. What is left is genuine but small, and the honest way to
ship it is to prove it changes nothing.

Two things are done here:

1. **AdaLN batched across sampler steps.** `adaln_proj` is a
   `[M, 2688] @ [2688, 96768]` GEMM — 40% of the model's parameters driven by a
   handful of rows, so it is pure weight-read bandwidth with no arithmetic
   intensity. Its input depends only on the timestep, and every timestep is
   known from the sigma schedule before sampling starts, so all of them can run
   as one GEMM per block that reads those weights once per render instead of
   once per step. In isolation that is 6.96 ms -> 0.389 ms for 20 steps of one
   block, and bitwise equal.

   **In the model it is neither cheaper nor bitwise equal, and is off by
   default.** Batching changes the GEMM's M, which changes the kernel cuBLAS
   selects: the measured disagreement at the production shape is `max_abs`
   1.953e-03, `mean_rel` 1.887e-06 — one bf16 last bit, about a thousandth of
   the bf16 representation floor (1.659e-03, from `tests/oracle_rms_rope.py`).
   `accept()` therefore checks each block against the stock projection on its
   first call and reverts unless it matches exactly.

   The stranger result is that forcing it through the gate — every block
   batched, confirmed active — still measured 1.0005 s/step against 1.0013 s
   stock. The AdaLN kernel is 19 ms/step in the profile and the batched form
   should have removed nearly all of it on steps 2..20, so this should have been
   worth ~1.9%. It was not, and the reason is not established. The code is kept,
   disabled, so the experiment is not repeated from scratch.
2. **Render-invariant caching.** The RoPE rotation table, the token-refiner
   output and the conditioning rows depend on the layout and the prompt, not on
   the timestep, yet the stock forward rebuilds them on every model evaluation.
   Reusing them is exact. Measured at 1.00x with one evaluation per step: the
   work is real but it is under a millisecond against a one-second step. It
   should matter more under CFG, which evaluates each timestep twice. Untested
   there.

So the honest summary is that a model patch cannot make this materially faster
without giving something up. The large wins on this hardware are the CUDA 13
runtime (1.117x, and *more* accurate) and precision (W4A4, 1.58x), neither of
which is a patch — which is why this module's other job is to shout when the
CUDA 13 fused kernel is missing.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

import comfy.model_management
from comfy.ldm.minimax import model as minimax_model

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


def unique_timesteps(sigma_value: float, shift_video: float, shift_audio: float,
                     visual_aug: float, audio_aug: float,
                     has_visual_cond: bool, has_audio_cond: bool) -> list[float]:
    """Reproduce the forward's timestep rows for one sigma.

    Kept as one function so the pre-computed schedule and the live step cannot
    drift apart: if this is wrong, the cache misses and falls back rather than
    returning a subtly different embedding.
    """
    sigma = torch.as_tensor(sigma_value, dtype=torch.float32).clamp(min=1e-6)
    t_video = float(1.0 - sigma)
    t_audio = float(1.0 - minimax_model.time_shift_sigma(sigma, shift_video, shift_audio))
    values = {t_video, t_audio}
    if has_visual_cond:
        values.add(max(t_video, visual_aug))
    if has_audio_cond:
        values.add(max(t_audio, audio_aug))
    return sorted(values)


class ExactFastPath:
    """Bitwise-exact reuse of work the stock forward repeats."""

    # Tolerance for `allow_last_bit`, which no node exposes any more: accepting
    # the batched AdaLN was measured to change the trajectory while delivering
    # 1.00x, so it is strictly worse than leaving it off. Retained only so the
    # gate has a threshold to compare against when someone re-runs the test.
    LAST_BIT_TOLERANCE = 1e-5

    def __init__(self, batch_adaln: bool = True, cache_invariants: bool = True,
                 verify: bool = False, allow_last_bit: bool = False) -> None:
        self.batch_adaln = batch_adaln
        self.cache_invariants = cache_invariants
        self.verify = verify
        self.allow_last_bit = allow_last_bit
        self.sigmas: torch.Tensor | None = None
        self.reset()

    def reset(self) -> None:
        self._verified: set[int] = set()
        self.batching_disabled = False
        self._adaln_tables: dict[int, torch.Tensor] = {}
        self._row_offsets: dict[tuple[float, ...], int] = {}
        self._schedule_embedding: torch.Tensor | None = None
        self._rope: dict[Any, torch.Tensor] = {}
        self._text: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        self.adaln_hits = 0
        self.adaln_misses = 0
        self.invariant_hits = 0
        self.last_bit_relative = 0.0

    def summary(self) -> str:
        total = self.adaln_hits + self.adaln_misses
        note = ""
        if self.last_bit_relative:
            note = f", accepted last-bit adaln (max mean_rel {self.last_bit_relative:.3e})"
        if self.batching_disabled:
            note = ", adaln batching reverted (not exact at this shape)"
        return (f"adaln batched {self.adaln_hits}/{total} projections, "
                f"{self.invariant_hits} invariant reuses{note}")

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

    # -- AdaLN across the whole schedule -------------------------------------

    def prepare_schedule(self, model: Any, dtype: torch.dtype, device: torch.device,
                         observed_sigma: float, shift_video: float, shift_audio: float,
                         visual_aug: float, audio_aug: float,
                         has_visual_cond: bool, has_audio_cond: bool) -> None:
        """Embed every timestep in the sigma schedule once, in schedule order.

        The forward reads its sigma from the `timestep` tensor, which the model
        base scales out of the scheduler's sigma. Rather than hard-code that
        scale — it is a property of the model wiring, not of this node — it is
        recovered from the first observed step. If the recovered mapping is ever
        wrong the step's key simply will not be found and every projection falls
        back to the stock path, so a bad guess costs speed, never correctness.
        """
        if self._schedule_embedding is not None or self.sigmas is None:
            return
        schedule = self.sigmas[:-1].tolist()
        if not schedule or schedule[0] == 0:
            return
        scale = observed_sigma / schedule[0]
        offsets: dict[tuple[float, ...], int] = {}
        embeddings: list[torch.Tensor] = []
        total = 0
        for sigma in schedule:
            values = unique_timesteps(sigma * scale, shift_video, shift_audio,
                                      visual_aug, audio_aug, has_visual_cond, has_audio_cond)
            key = tuple(values)
            if key in offsets:
                continue
            offsets[key] = total
            total += len(values)
            # Embedded one step at a time, at the row count the forward itself
            # uses. Embedding all steps in one call was measured to shift the
            # result by ~1.5e-05 (about two bf16 ULP at this magnitude), because
            # `time_embedder` is an MLP and changing its M changes the GEMM the
            # library picks. Only the AdaLN projection below is safe to batch.
            embeddings.append(self._embed(model, values, dtype, device))
        if not embeddings:
            return
        self._row_offsets = offsets
        self._schedule_embedding = torch.cat(embeddings, dim=0)

    @staticmethod
    def _embed(model: Any, values: list[float], dtype: torch.dtype,
               device: torch.device) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        if model.use_adaln_curves:
            table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
            position = tensor.clamp(0.0, 1.0) * (table.shape[0] - 1)
            lower = position.floor().long().clamp(max=table.shape[0] - 2)
            return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
        return model.time_embedder(tensor).to(dtype)

    def adaln(self, block_index: int, projection: Any, t_emb: torch.Tensor,
              step_values: tuple[float, ...]):
        """Return this step's AdaLN chunks from one whole-schedule GEMM."""
        offset = self._row_offsets.get(step_values)
        if (not self.batch_adaln or self.batching_disabled
                or self._schedule_embedding is None or offset is None):
            self.adaln_misses += 1
            return None
        table = self._adaln_tables.get(block_index)
        if table is None:
            # One GEMM covering every step. This only changes the GEMM's M, but
            # whether that is bitwise depends on which kernel the library then
            # picks, so the result is checked against the stock projection before
            # it is trusted -- see accept().
            table = projection.linear(
                torch.nn.functional.silu(self._schedule_embedding)
                if projection.apply_silu else self._schedule_embedding)
            self._adaln_tables[block_index] = table
        modalities, expand = projection.modalities, projection.expand
        width = expand * projection.hidden
        rows = len(step_values)
        chunk = table[offset:offset + rows].view(rows * modalities, width)
        self.adaln_hits += 1
        self.last_embedding = self._schedule_embedding[offset:offset + rows]
        self.last_input = t_emb
        return chunk.chunk(expand, dim=-1)

    def diagnose(self) -> str:
        """Say whether a verify failure came from the embedding or the GEMM."""
        mine, theirs = self.last_embedding, self.last_input
        if mine.shape != theirs.shape:
            return f"embedding shape {tuple(mine.shape)} vs forward {tuple(theirs.shape)}"
        if torch.equal(mine, theirs):
            return "embeddings are bitwise equal, so the batched GEMM is what differs"
        delta = (mine.double() - theirs.double()).abs()
        return (f"embeddings differ: max_abs={delta.max().item():.3e} "
                f"dtype {mine.dtype}/{theirs.dtype}")

    def needs_check(self, block_index: int) -> bool:
        """Check each block once per render, or every call in paranoid mode."""
        return self.verify or block_index not in self._verified

    def accept(self, block_index: int, produced, reference) -> bool:
        """Keep the batched result only if it is bitwise equal to the stock one.

        Batching changes the GEMM's M, and whether that is bitwise depends on the
        shape: at the production shape (6 rows per step, 20 steps) it is exactly
        equal, but at the tiny preset (2 rows per step) cuBLAS selects a
        different kernel and the results differ in the last bits. So the promise
        is enforced at runtime on the shape actually being rendered rather than
        asserted in advance, and the node silently returns to the stock path
        wherever it cannot be kept.
        """
        for a, b in zip(produced, reference):
            if torch.equal(a, b):
                continue
            delta = (a.double() - b.double()).abs()
            relative = (delta.mean() / b.double().abs().mean().clamp(min=1e-30)).item()
            if self.allow_last_bit and relative <= self.LAST_BIT_TOLERANCE:
                self.last_bit_relative = max(self.last_bit_relative, relative)
                continue
            self.batching_disabled = True
            self._adaln_tables.clear()
            LOG.warning(
                "[Cappy H3 Fast Path] AdaLN batching is not bitwise exact at this shape "
                "(block %d, rows=%d, max_abs=%.3e, mean_rel=%.3e); reverting to the stock "
                "projection for this render. Enable allow_last_bit_adaln to accept a "
                "difference this small in exchange for about 1.9%%.",
                block_index, b.shape[0], delta.max().item(), relative)
            return False
        self._verified.add(block_index)
        return True
