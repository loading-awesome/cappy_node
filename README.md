# Cappy H3

Two ComfyUI custom nodes for **MiniMax H3**:

- **Cappy MiniMax H3 Audio-Aware Cache** — the accelerator. Reuses the residual
  from transformer blocks 1..N when the model is stable. Measured **1.84x** on
  DiT sampling. It is an approximation and it costs fine detail on some content;
  the exact numbers are below so you can decide with them rather than around
  them.
- **Cappy MiniMax H3 Fast Path (exact)** — bitwise-identical optimizations only.
  Measured **1.00x**, i.e. it does not currently make anything faster. It is
  worth attaching anyway because it checks your runtime and warns about a 10.5%
  regression that otherwise fails silently.

Everything quoted here was measured at 864x480, 49 frames, 20 steps,
`res_multistep`/`simple`, seed 8421, BF16, on an RTX PRO 6000 Blackwell (sm_120).

## Read this first: your CUDA version is worth more than the cache settings

ComfyUI's `comfy/quant_ops.py` disables comfy-kitchen's CUDA backend whenever
PyTorch is built against CUDA < 13. That also disables the **fused
QK-RMSNorm+RoPE kernel used on the plain bf16 path**, which is not obvious from
the warning it prints. Measured cost: memory-bound work in a step rises from 97
to 193 ms and the step goes from 1.0013 s to 1.1190 s.

| Runtime | s/step |
| --- | ---: |
| PyTorch + CUDA 12.8 | 1.1190 |
| PyTorch + CUDA 13.0 | **1.0013** |

So CUDA 13 is a free **1.117x**. It is also *more accurate*: against a float64
reference the fused kernel lands at rel-RMS 2.340e-03 versus the fallback's
2.873e-03, where bf16's own representation floor is 1.659e-03.

The Fast Path node prints a warning when the fused kernel is missing. That check
is the single most valuable thing in this repository — the regression is
invisible, and it silently invalidated an entire round of this project's own
benchmarking.

## What the cache costs, measured

A reused step costs **0.025 s against 1.00 s** — the skipped stack really is
about 40x cheaper. At `relative_threshold=0.10`, `max_consecutive_reuses=5`,
9-10 of 20 stacks reuse, and DiT sampling drops from 19.2 s to 10.4 s.

The cost is high-frequency detail, and **it depends heavily on your content**:

| Content | Result |
| --- | --- |
| Fast tracking shot through a detailed library | **Visibly degraded.** At 3x zoom the books disappear off the shelves, leaving flat surfaces with only the shelf edges. Step-20 denoised video diverges 45.9% (cos 0.894). |
| Low-motion talking head | **Holds up.** Teeth, lip texture and skin all comparable to dense. |

The late sampler steps are what write fine texture, and the reused residuals are
what would have carried it. Detail-dense, fast-moving content is where that
shows; a close-up face is not.

Two warnings from validating this:

1. **Whole-frame sharpness metrics do not detect it.** The degraded library
   render scored 78.5% Laplacian variance against dense, while merely changing
   the seed scores 77.2% — the damage is inside composition noise. It was
   obvious by eye at 3x zoom and invisible to the metric. Judge crops at full
   resolution.
2. **Latent-trajectory divergence alone does not condemn a change.** The W4A4
   checkpoint diverges comparably in kind and looks clean decoded. Divergence is
   a reason to go and look, not a verdict.

## Faster still: the checkpoint

The `pruned_int8_convrot` H3 DiT is actually **W4A4**. On CUDA 13 it samples at
**0.6336 s/step against BF16's 1.0013 — 1.58x** — and loads at ~20 GB instead of
~63 GB. It passed every decoded check run against it on both a library tracking
shot and a talking head: no black frames, no flashes, no colour blobs, temporal
jerk 98.4%, audio waveform correlation +0.98.

It requires CUDA 13. On CUDA 12.8 its layout is one of the disabled ones and it
falls back to a dequantizing path that runs at 2.46 s/step — nearly 4x slower
than it should be, which is how it was originally mis-recorded as a speed loss.

Whether W4A4 and the cache compound has not been tested.

## Install

1. Quit ComfyUI.
2. Copy or clone this entire folder into ComfyUI's `custom_nodes` directory.
3. Start ComfyUI and check its console: it should report that the Cappy nodes
   were imported, with no `Failed to import` message.
4. In the node menu, search for **Cappy MiniMax H3**.

The installed layout must look like this:

```text
ComfyUI/custom_nodes/cappy_node/
  __init__.py
  nodes.py
  h3_patch.py
  fast_path.py
  cache_policy.py
```

It needs a current ComfyUI build with the native MiniMax H3 model
implementation. Both nodes intentionally fail before sampling if the connected
model is not MiniMax H3.

## First workflow

1. Open a working MiniMax H3 video workflow. Do not add this to an unrelated
   Wan, Flux, SDXL, or image workflow.
2. Locate the H3 model loader. Its `MODEL` output ordinarily connects directly
   to the H3 sampler/workflow input.
3. Add **Cappy MiniMax H3 Audio-Aware Cache** from `advanced/model/patches`.
4. Connect the loader's `MODEL` output to Cappy's `model` input.
5. Connect Cappy's `model` output to the exact socket that previously received
   the loader's `MODEL` output. Leave every other connection alone.
6. Queue the workflow normally. The node affects only the model passed through
   it; it does not change your prompt, seed, sampler, frames, or audio input.

```text
H3 Model Loader ── MODEL ──> Cappy H3 Audio-Aware Cache ── MODEL ──> H3 sampler
```

Start with:

| Setting | Start value | Meaning |
| --- | ---: | --- |
| `relative_threshold` | `0.10` | Reuse only when whole/video/audio block-0 changes are all below this value. Lower is safer and slower. This is the setting that was measured. |
| `max_consecutive_reuses` | `5` | Maximum consecutive reused stack residuals. Do not raise it until you have inspected output. |
| `cache_device` | `auto` | Keeps the residual on the model device; use `cpu` only when VRAM is the constraint. |
| `trace` | `summary` | Prints aggregate decisions; `per_step` shows why every step refreshed or reused. |

### How to decide whether it is acceptable for your work

Render the **same seed** twice, once with the node bypassed and once with it
active, on content representative of your actual job. Then compare crops at
full resolution — not thumbnails, and not a sharpness score. Look at the
smallest textured thing in frame: book spines, fabric weave, foliage, hair.
Check the final fifth of the clip, where drift has accumulated most.

If detail is going: lower `relative_threshold` to `0.08`, then `0.06`, which
reuses less often. Change the threshold before you change the cap. Neither of
those lower settings has been measured here, so treat them as your experiment.

If your content is a close-up face or another low-motion subject, the default is
likely fine and worth the 1.84x.

## What is not worth trying

Measured and closed, so nobody spends the time again:

| Avenue | Why not |
| --- | --- |
| Faster BF16 GEMM | 69% of a step, already at 402 TFLOP/s. An isolated best-case `F.linear` at the same shapes reaches only 390. |
| A better attention backend | 20.5% of a step. cuDNN has no execution plan for head_dim 64; memory-efficient attention is 2.2x slower than the flash kernel already in use. |
| `torch.compile` / CUDA graphs | The GPU is **99.8% busy**. There is no host-side gap to recover. |
| On-the-fly FP8 | 1.89x on the GEMM alone, but 1.55x once per-call activation quantization is counted, at rel-RMS 3.8e-02 — worse than W4A4 on both speed and accuracy. |
| Batching AdaLN across steps | Isolated: 18x cheaper and bitwise equal. In the model: neither. Left in `fast_path.py`, disabled, with its measurements. |

Still open: a fused modulated-RMSNorm kernel. 53 ms/step sits in
`mul_`/`addcmul_`/RMSNorm and roughly 60% of it looks recoverable, worth ~3%,
but it needs a custom Triton or CUDA kernel that clears a bitwise gate.

## Reading the console trace

`per_step` trace lines include `whole`, `audio`, and `video` changes. Useful
refresh reasons are:

- `audioAboveThreshold`: audio changed enough to require fresh transformer work.
- `videoAboveThreshold`: video changed enough to require fresh transformer work.
- `wholeSequenceAboveThreshold`: broader model state changed enough to refresh.
- `consecutiveCap`: a deliberately scheduled refresh after the maximum reuse age.
- `cooldown`: the final step is always full, to avoid finishing on stale features.

If the cache almost always reports `consecutiveCap`, the threshold is not what
limits speed; the cap is. Increasing it may be faster but is a quality
experiment, not a safe tuning tweak.

## Validating a change to this node

`tests/` carries the instruments used for every number above. They are manual
RunPod utilities, not unit tests.

| Tool | Purpose |
| --- | --- |
| `runpod_profile.py` | Where a step's time actually goes, by role and by CUDA kernel. Measures s/step in a profiler-off window so the headline number is undistorted. |
| `runpod_taps.py` | Captures named per-block and per-step taps. Start with `--tiny`: it runs the whole graph in seconds, and a production render is the worst first test. |
| `compare_taps.py` | Reports the first divergence in pipeline order. Defaults to a bitwise gate; `--mode class` for changes that legitimately alter arithmetic. |
| `oracle_rms_rope.py` | Settles which of two kernel implementations is correct, against a float64 reference, instead of comparing them to each other. |
| `bench_dispatch.py` | GEMM and attention ceilings at the production shapes. Diagnostic only — never accept a speedup on a microbenchmark. |

The comparison is trustworthy because it has a control: two separate CUDA 12.8
processes at the production shape produce **40/40 taps bitwise identical**, so a
nonzero difference is attributable to the change under test.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| The node is not in the menu | Confirm this folder is directly inside `ComfyUI/custom_nodes`, not nested one level deeper. Restart ComfyUI and inspect the startup console for an import error. |
| A warning that the fused QK-RMSNorm+RoPE kernel is inactive | Your PyTorch is built against CUDA < 13. Rebuild or reinstall against CUDA 13; it is worth 10.5% and better numerics. |
| “requires a MiniMax H3 diffusion model” | Connect Cappy only to the native MiniMax H3 model loader's `MODEL` output. |
| Import error mentioning `comfy.ldm.minimax` | Update ComfyUI to a version with native MiniMax H3 support. This node intentionally does not ship a separate H3 implementation. |
| Out of memory | Leave `cache_device` at `auto`; if you need VRAM headroom, try `cpu`. It may be slower because the residual has to transfer back to the accelerator. |
| Fine detail or texture looks washed out | Expected on detail-dense, fast-moving content. Lower `relative_threshold` to `0.08` or `0.06`. Do not increase the reuse cap. |
| Lip-sync or speech is worse | Lower `relative_threshold` with the same seed and compare. Do not increase the reuse cap. |
| No meaningful speed improvement | Turn on `per_step` trace. If nearly every step refreshes, that prompt is not cache-friendly at the current settings. |

## Compatibility and limitations

The integration mirrors ComfyUI's current H3 forward to expose its block-loop
extension point. Upstream H3 internal API changes can break these nodes. Keep
the ComfyUI revision used for validation pinned, and rerun the dense-vs-cached
comparison after any ComfyUI, model, sampler, or precision update.

The cache keeps separate state per Comfy conditioning branch. It does not
combine multiple independent renders, and it resets for each sampling run.

Every measurement here is from one GPU (RTX PRO 6000 Blackwell, sm_120) at one
shape. The ratios should travel; the absolute numbers will not.

## License and attribution

This package is distributed under **AGPL-3.0-or-later**. Its ComfyUI H3
block-loop integration is adapted from
[ComfyUI-UtilsCollection](https://github.com/silveroxides/ComfyUI-UtilsCollection)
by silveroxides (AGPL-3.0), whose H3 cache notes an earlier GPL-3.0 cache
heuristic by lihaoyun6. See [NOTICE](NOTICE) and [LICENSE](LICENSE).

The Cappy policy differs materially: it runs block 0 every step, measures that
residual rather than a pre-stack feature signature, uses whole-sequence,
video-specific, and audio-specific gates, and forces a final full refresh.
