# Little Fig Research Session Summary

_Last updated: 2026-08-15 (Africa/Lagos)_

This is the durable cross-session handoff. Read it with `RESEARCH_PROGRESS.md` when
resuming work, and append a dated entry after each substantive session.

## Current objective

P2 and P3 verification are complete. Run the cheap P3a isolated lowram allocator
diagnostic before proceeding to P4.

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
`FigModel.from_pretrained()` plus training path was measured separately in P3 below.

## Completed P3 result (2026-08-15)

The real TinyLlama Tier-1 CPU path completed 20 lowram steps at batch 2 and sequence
length 256. Peak RSS was **7.340488 GiB**, giving **0.659512 GiB** headroom against
8 GiB. Incremental RSS above startup was **7.126644 GiB**, so the paper's ~400 MB
estimate was not reproduced. Load/quantize peaked at 7.054710 GiB; training was the
worst phase; post-training RSS remained 6.811050 GiB.

Source audit rules out accidental full-model AdamW state. `model.py` freezes every
non-LoRA parameter, and `trainer.py` passes only `requires_grad` parameters to
AdamW. The 12,615,680 LoRA parameters need 48.13 MiB; parameters, gradients, and
two FP32 moments total about 192.5 MiB. Embeddings and lm_head are frozen FP32
storage, not optimizer state. The leading suspect for the unexplained training RSS
is lowram dequantization: each FigLinear expands packed indices to int64 and creates
a full FP32 weight in both forward and backward. Persistent 6.811 GiB RSS suggests
CPU allocator retention, but saved-activation versus dequant-workspace attribution
still needs finer per-step instrumentation.

## Exact next steps

1. Treat P2 FigQuant quality verification as complete; the result JSON and exact
   156-layer metrics are committed under `benchmark/`.
2. P3 completed in `lowram`: 7.340488 GiB absolute peak, 7.126644 GiB incremental.
   The 8 GiB budget passed; the ~400 MB estimate failed.
3. Proceed to P4 Memory Fabric unless priorities change.
4. Immediate diagnostic: run `benchmark/experiment_lowram_allocator_v1.py` in
   Colab. It tests exact TinyLlama q_proj and MLP shapes without loading the model,
   and distinguishes live growth from allocator-retained RSS with `malloc_trim(0)`.

## Remaining verification

- Optional: repeat P3 with `figcache` and `fast` to quantify cache tradeoffs; this is
  not required to decide the lowram claims.
- Re-measure the corrected P2 collector only if a new harness-specific number is
  useful; it is not required to decide the production training claim.

## 2026-08-18 session handoff

P3a completed in Colab using exact TinyLlama q_proj and MLP shapes (20 iterations,
batch 2, sequence 256, PyTorch 2.11 CPU). The isolated lowram path showed bounded
RSS retention after GC, and Linux `malloc_trim(0)` released nearly all of it:

- q_proj: 161.6 MiB retained after GC; 155.4 MiB released by trim (96%).
- mlp_proj: 94.9 MiB retained after GC; 102.7 MiB released by trim.
- No unbounded live-tensor/autograd leak was observed.

Verdict: **P3a REPRODUCED allocator retention** as a real contributor to the full
P3 RSS, but did not establish that it explains the entire 7.34 GiB peak.

P3b was implemented and pushed in commit `a1c1cbc` on
`research/p1-figmezo-verify`. `FigTrainingConfig.allocator_trim` is opt-in,
`FigTrainer` calls glibc `malloc_trim(0)` after each optimizer step, and
`benchmark/experiment_8gb_v1.py` accepts `--allocator-trim` while recording
`experiment_variant` as `baseline` or `trim_each_step`.

Run identical full TinyLlama Colab tests for baseline and trim intervention, using
separate Drive result files under `littlefig-p3b`. The Colab notebook is:
`https://colab.research.google.com/github/Harboria-Labs/littlefig/blob/research/p1-figmezo-verify/benchmark/P3_8GB_Colab.ipynb`.

Future allocator work may involve a custom C++ allocator, but first formalize the
mathematics of size classes, alignment, workspace lifetimes, reuse, fragmentation,
live bytes versus resident bytes, and synchronization. Use P3b as the comparison
baseline before designing the allocator.
