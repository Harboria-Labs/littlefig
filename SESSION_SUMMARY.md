# Little Fig Research Session Summary

_Last updated: 2026-08-14 (Africa/Lagos)_

This is the durable cross-session handoff. Read it with `RESEARCH_PROGRESS.md` when
resuming work, and append a dated entry after each substantive session.

## Current objective

Finish P2: verify FigQuant on all 156 TinyLlama matrices and save the complete JSON.
Do not call the claim reproduced until the final JSON has 156 records.

## State at handoff

- Branch: `research/p1-figmezo-verify`.
- P1 FigMeZO is complete; the claimed -18.6% held-out improvement was refuted.
- P2 GPT-2 is reproduced: 50/50 wins and 5.280921% lower mean reconstruction MSE
  versus NF4.
- The user's TinyLlama run visibly completed 154 layers, with FigQuant better than
  NF4 on every completed layer. It then printed `[155/156] START lm_head.weight`
  followed by `^C`; layers 155 (`lm_head.weight`) and 156
  (`model.embed_tokens.weight`) have no result.
- That run had no checkpoint, so the 154 metrics cannot be reconstructed from the
  partial console output. This is partial evidence only.

## 2026-08-14 recovery work

The old benchmark allocated a `[65,536,000, 16]` FP32 distance tensor for the final
embedding matrix, about 3.9 GiB before other tensors. NF4 and uniform INT4 had the
same problem. A local `/content` checkpoint would also disappear after a Colab reset.

Changes made:

- `src/little_fig/engine/figquant.py`: bounded-chunk k-means assignment/update.
- `benchmark/experiment_figquant_v2.py`: atomic per-layer checkpoints, `--resume`,
  peak RSS records, bounded-memory baselines, and `--results-path`.
- `benchmark/P2_FigQuant_Colab.ipynb`: Google Drive mount, Drive checkpoint/results
  directory, TinyLlama-only execution, and resume-by-default workflow.
- `RESEARCH_PROGRESS.md`: session pointer added.

## 2026-08-14 memory-evidence extension

The P2 run now evaluates the 8 GiB objective explicitly rather than treating memory
as incidental logging. It records process peak RSS, whole-system total/available
memory, the minimum available-memory reading, per-layer workspace growth, the layer
responsible for peak RSS, budget utilization, and headroom against the configurable
`--memory-budget-gib` target (8 GiB by default).

A separate `figquant_v2_memory_<model>.json` heartbeat is atomically updated in the
Drive checkpoint directory every two seconds. If Colab kills the process inside the
final layer, this file preserves the phase, last RSS, observed peak RSS, and system
available-memory floor even though that layer never reaches its normal checkpoint.

Interpretation boundary: P2 measures the memory needed to benchmark quantization
quality while holding FP32 model weights, not the complete Fig Engine fine-tuning
pipeline. Passing 8 GiB here supports the quantizer's bounded-memory behavior; it
does not by itself prove the broader "fine-tune on 8 GB RAM" claim. That requires a
separate end-to-end training memory benchmark.

Verification: a synthetic run measured 0.397 GiB peak process RSS, correctly named
the largest matrix as the peak layer, measured 0.120 GiB peak layer workspace growth,
persisted the memory heartbeat, and passed a deliberately tight 0.5 GiB budget with
0.103 GiB headroom. The temporary verification files were removed afterward.

Notebook cleanup: the obsolete `benchmark/P1_FigMeZO_Colab.ipynb` testing notebook
was deleted. Keep `Little_Fig_Colab.ipynb` as the public/main usage notebook and
`benchmark/P2_FigQuant_Colab.ipynb` as the sole research-testing notebook.

## Completed P2 result (2026-08-14)

The new Drive-backed run completed all **156/156** TinyLlama matrices:

- Mean reconstruction MSE: FigQuant `5.6419278e-6`; NF4 `5.9652037e-6`.
- FigQuant reduction vs NF4: **5.419361%**; vs absmax INT4: **36.877870%**.
- Per-layer reduction: min **2.498559%**, median **5.743715%**, max **21.691531%**.
- SNR gain: **0.252763 dB**; losers: **0**; runtime: **1214.2 s**.
- Verdict: **REPRODUCED** for the FigQuant quality claim.

Memory result is separate: the **8.668636 GiB** collection peak came from a P2
harness bug. `collect_weights()` held the full **4.098 GiB** FP32 model while also
building **4.098 GiB** of FP32 clones. This number is not evidence for or against
Fig Engine's production training-memory claim. Quantization-only peak was
**4.932980 GiB** RSS. The harness now releases each source parameter immediately
after cloning it; the corrected collection peak still needs a new run. The actual
`FigModel.from_pretrained()` plus training path remains independently unmeasured.

## Exact next steps

1. Treat P2 FigQuant quality verification as complete; the result JSON and exact
   156-layer metrics are committed under `benchmark/`.
2. Run `benchmark/P3_8GB_Colab.ipynb` with `MEMORY_MODE = 'lowram'`, the mode
   relevant to the minimum-memory Tier-1 claim.
3. Compare absolute peak RSS with 8 GiB and incremental peak RSS with the paper's
   ~400 MB estimate. Preserve the load/quantize, dataset, and training phase split.
4. If necessary, repeat with `figcache` and `fast` to identify cache tradeoffs.
5. After P3, proceed to P4 Memory Fabric unless priorities change.

## Remaining verification

- Run P3 on TinyLlama. Local smoke could not pass model loading because this local
  environment does not have `transformers` installed.
- Re-measure the corrected P2 collector only if a new harness-specific number is
  useful; it is not required to decide the production training claim.
