# Cappy H3 Cache

An experimental ComfyUI custom node for **MiniMax H3** that speeds sampling by
reusing the residual from transformer blocks 1 through N when the model is
stable. It is designed around video with spoken audio: audio has an independent
veto, so a visually quiet step cannot reuse stale audio features.

This is an approximation. It can materially reduce render time, but it is not
a quality-neutral kernel optimization. Validate it against dense renders for
your model, prompt, resolution, and audio before using it in production.

## What makes it different

Most H3 cache nodes decide before entering the transformer stack. Cappy always
runs **block 0**, then measures the relative change of its residual against the
previous step. It reuses the full-stack residual only when both conditions hold:

1. The complete token sequence is below the threshold.
2. The generated-audio token range is also below the threshold.

The first and last sampler steps are always full evaluations. A consecutive
reuse cap forces periodic refreshes, limiting residual age and reducing the
warping seen with larger block-evaluation intervals.

The settings are based on an Apple Silicon H3 investigation, where a cross-step
cache was the only large, practical DiT speedup. The policy was tested at
`relative_threshold=0.10` and `max_consecutive_reuses=5`. These values are a
starting point, **not a promise of equivalent speed or output quality on CUDA**.

## Install

1. Quit ComfyUI.
2. Copy or clone this entire folder into ComfyUI's `custom_nodes` directory.
3. Start ComfyUI and check its console: it should report that the Cappy node
   was imported, with no `Failed to import` message.
4. In the node menu, search for **Cappy MiniMax H3 Audio-Aware Cache**.

The installed layout must look like this:

```text
ComfyUI/custom_nodes/cappy_node/
  __init__.py
  nodes.py
  h3_patch.py
  cache_policy.py
```

It needs a current ComfyUI build that includes the native MiniMax H3 model
implementation. The node intentionally fails before sampling if the connected
model is not MiniMax H3.

## First workflow: step by step

1. Open a working MiniMax H3 video workflow. Do not add this to an unrelated
   Wan, Flux, SDXL, or image workflow.
2. Locate the H3 model loader. Its `MODEL` output ordinarily connects directly
   to the H3 sampler/workflow input.
3. Add **Cappy MiniMax H3 Audio-Aware Cache** from
   `advanced/model/patches`.
4. Connect the loader's `MODEL` output to Cappy's `model` input.
5. Connect Cappy's `model` output to the exact socket that previously received
   the loader's `MODEL` output. Leave every other connection alone.
6. Queue the workflow normally. The node affects only the model passed through
   it; it does not change your prompt, seed, sampler, frames, or audio input.

```text
MiniMax H3 Model Loader ── MODEL ──> Cappy H3 Audio-Aware Cache ── MODEL ──> H3 sampler
```

For the first run, use the recommended settings below. Set `trace` to
`per_step` once if you want to see exactly what it is deciding, then return it
to `summary` for ordinary rendering.

Start with:

| Setting | Start value | Meaning |
| --- | ---: | --- |
| `relative_threshold` | `0.10` | Reuse only below this relative block-0 change. Lower is safer and slower. |
| `max_consecutive_reuses` | `5` | Maximum consecutive reused stack residuals. Do not raise it until you have inspected output. |
| `cache_device` | `auto` | Keeps residual on the model device; use `cpu` only when VRAM is the constraint. |
| `trace` | `summary` | Prints aggregate decisions; `per_step` shows why every step refreshed or reused. |

For a first validation, render a short spoken dialogue or singing clip using the
same seed both dense and cached. Inspect lips, consonants, fast motion, cuts,
and the final 15–20% of the render. If any look worse, lower the threshold
(try `0.08`, then `0.06`) before changing the consecutive cap.

### Safe tuning order

Keep `max_consecutive_reuses=5` while finding a threshold that preserves your
output. If you need more quality, lower `relative_threshold`. If dense and
cached output remain coherent over several representative clips and the trace
shows mostly `consecutiveCap`, you may test a cap of `6`—one value at a time.
Do not treat a single pretty clip as validation.

## Reading the console trace

`per_step` trace lines include `whole`, `audio`, and `video` changes. Useful
refresh reasons are:

- `audioAboveThreshold`: audio changed enough to require fresh transformer work.
- `wholeSequenceAboveThreshold`: broader model state changed enough to refresh.
- `consecutiveCap`: a deliberately scheduled refresh after the maximum reuse age.
- `cooldown`: final step is always full to avoid finishing on stale features.

If the cache almost always reports `consecutiveCap`, the threshold is not what
limits speed; the cap is. Increasing it may be faster but is a quality
experiment, not a safe tuning tweak.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| The node is not in the menu | Confirm this folder is directly inside `ComfyUI/custom_nodes`, not nested one level deeper. Restart ComfyUI and inspect the startup console for an import error. |
| “requires a MiniMax H3 diffusion model” | Connect Cappy only to the native MiniMax H3 model loader's `MODEL` output. |
| Import error mentioning `comfy.ldm.minimax` | Update ComfyUI to a version with native MiniMax H3 support. This node intentionally does not ship a separate H3 implementation. |
| Out of memory | Leave `cache_device` at `auto`; if you need VRAM headroom, try `cpu`. It may be slower because the residual has to transfer back to the accelerator. |
| Lip-sync, speech, or temporal coherence is worse | Disable Cappy for the job, then test `relative_threshold=0.08` or `0.06` with the same seed. Do not increase the reuse cap. |
| No meaningful speed improvement | Turn on `per_step` trace. If nearly every step refreshes, that prompt/model is not cache-friendly at the current settings. |

## Compatibility and limitations

The integration mirrors ComfyUI's current H3 forward to expose its block-loop
extension point. Upstream H3 internal API changes can break this node. Keep the
ComfyUI revision used for validation pinned, and rerun the dense-vs-cached gate
after any ComfyUI, model, sampler, or precision update.

The node keeps a separate state per Comfy conditioning branch. It does not
combine multiple independent renders, and it resets for each sampling run.

## License and attribution

This package is distributed under **AGPL-3.0-or-later**. Its ComfyUI H3
block-loop integration is adapted from
[ComfyUI-UtilsCollection](https://github.com/silveroxides/ComfyUI-UtilsCollection)
by silveroxides (AGPL-3.0), whose H3 cache notes an earlier GPL-3.0 cache
heuristic by lihaoyun6. See [NOTICE](NOTICE) and [LICENSE](LICENSE).

The Cappy policy differs materially: it runs block 0 every step, measures that
residual rather than a pre-stack feature signature, uses whole-sequence plus
audio-specific gates, and forces a final full refresh.
