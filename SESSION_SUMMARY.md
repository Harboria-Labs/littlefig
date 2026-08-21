# Little Fig Research Session Summary

_Last updated: 2026-08-18 (Africa/Lagos)_

This is the durable cross-session handoff. Read it with `RESEARCH_PROGRESS.md` when
resuming work, and append a dated entry after each substantive session.

## Current objective

Execute the P4 FigSweep/storage-backed memory investigation in strict order,
starting with P4a only. Record, commit, and surface each result before proceeding.

## 2026-08-18 P4 plan: FigSweep wiring + storage-backed memory investigation

This sequence is interruption-safe and must not be reordered:

1. **P4a OS swap:** test whether Colab permits swap; if so, rerun the unmodified
   P3 baseline. This cheaply tests whether ordinary paging helps before custom code.
2. **P4b FigSweep wiring:** connect `figsweep_advance()` to real layer forwards,
   use window 4, and compare phase RSS/speed with P3b. This addresses the confirmed
   dormant rolling-window path.
3. **P4c LISA:** pass probe inputs and run paired sensitivity-guided vs random LISA
   on GPT-2, 8 seeds, held-out loss. Current trainer wiring never activates the
   sensitivity probe.
4. **P4d auto tier:** omit the tier under normal Colab RAM, then test constrained
   RAM downgrade if sandbox controls permit. All previous tests forced a tier.
5. **P4e memory modes:** compare lowram, figcache, and fast using the same P3
   workload, recording memory and speed. Cached modes also activate fused
   Linear+LoRA.
6. **P4f disk-stream design and proof:** only after P4a-P4e, document mmap-backed
   windowing with `saved_tensors_hooks` ownership and build an isolated 2-3-layer
   proof before touching full training. Backward lifetime must be preserved.

After every item, append concrete numbers/restrictions and a verdict to both files,
commit it separately as `research(P4x): ...`, and report it before starting the
next item.

## 2026-08-18 P4a result

**Verdict: NOT RUN / ENVIRONMENT BLOCKED.** This workspace is Windows PowerShell,
not the Colab Linux runtime used by P3. There is no Colab session or Linux swap
control available to this agent, so it cannot determine whether Colab permits a
swap file or rerun P3 with swap enabled. No substitute test was run and no swap
claim is made. P4b remains paused pending the user's Colab swap-permission result.

## 2026-08-19 implementation queue: item 1 complete

FigQuant quantization progress is now live in `src/little_fig/engine/model.py`.
Each completed target layer prints its index/name, layer time, running average,
elapsed time, ETA, cumulative compression, and current process RSS. Existing
notebook stdout capture makes this telemetry durable in the benchmark log.
Verification: `python -m py_compile src/little_fig/engine/model.py` and
`git diff --check` passed. No Colab benchmark rerun was performed; FigSweep
wiring and the remaining incomplete features are still pending.

## 2026-08-19 implementation queue: item 2 complete

`benchmark/P3_8GB_Colab.ipynb` now has a telemetry visualization cell. It parses
the durable result JSON and log, plots phase peak RSS and the 8 GiB budget, plots
training loss/speed/learning rate when available, prints quantization RSS samples,
and saves `figengine_8gb_<mode>_telemetry.png` to Drive. Notebook JSON validation
passed. This is notebook-only; no Colab rerun was performed.

## 2026-08-19 implementation queue: FigSweep wiring

`FigModel.enable_figsweep()` now installs forward pre-hooks on each registered
`FigLinear`; each hook calls `figsweep_advance()` immediately before forward.
Repeated enable removes prior hooks, and layer order follows `named_modules()`
traversal rather than lexical sorting. `model.py` compiles cleanly. This is
source-level wiring only; the P4b Colab RSS/speed/correctness comparison remains
required before declaring a memory improvement validated.

## 2026-08-19 implementation queue: LISA probe wiring complete

`FigTrainer._train_lisa()` now obtains one real batch from a fresh dataloader
iterator and passes its `input_ids` and `labels` into `LISAScheduler`. This enables
the scheduler's sensitivity probe without consuming the training iterator. The
trainer compiles cleanly. The paired GPT-2, 8-seed held-out quality test remains
required; no claim about the paper's improvement is made yet.

## 2026-08-18 source activation audit (A1-A4)

- **A1 FigSweep — PARTIALLY wired, inactive in tested lowram path.**
  `FigTrainingConfig.figsweep_window` calls `model.enable_figsweep()` only when
  explicitly greater than zero (`src/little_fig/engine/trainer.py:139-143`).
  The rolling helper `FigModel.figsweep_advance()` is defined
  (`src/little_fig/engine/model.py:658-687`) but has no call site in trainer/model
  execution. Lowram therefore uses the independent `DequantMatmul` branch in
  `src/little_fig/engine/linear.py:197-211`, with no active rolling window.
- **A2 LISA — NO for STREAMING_LORA; separate tier.** `TrainingTier.LISA` is a
  distinct enum value (`src/little_fig/engine/tier.py:25-30`) and dispatches to
  `_train_lisa()` only in its own branch (`src/little_fig/engine/trainer.py:454-462`).
  The trainer constructs `LISAScheduler` without probe inputs
  (`src/little_fig/engine/trainer.py:544-550`), so the scheduler's sensitivity
  probe condition is false (`src/little_fig/engine/lisa.py:90-96`) and selection
  falls back to uniform random (`src/little_fig/engine/lisa.py:190-210`).
- **A3 Auto tier selection — YES, but only when tier is omitted.** The trainer
  calls `select_tier(total_params)` when `config.training_tier` is `None`
  (`src/little_fig/engine/trainer.py:126-132`). `select_tier()` reads
  `psutil.virtual_memory().available` (`src/little_fig/engine/tier.py:165-167,
  200-205`), keeps 30% headroom via a 70% budget (`206-207`), and tests LISA,
  LOMO, STREAMING_LORA, then MeZO (`209-226`). It does not read storage.
- **A4 FigKernel — PARTIALLY active.** `FigModel.from_pretrained()` calls
  `_swap_rmsnorm()` when `fuse_kernels=True` (`src/little_fig/engine/model.py:229-231`),
  which is the logged RMSNorm replacement (`55-83`). Fused Linear+LoRA is only
  attempted by `FigLinear.forward()` when `_cached_W` exists, i.e. fast/cache mode
  (`src/little_fig/engine/linear.py:185-195`); lowram takes the dequant autograd
  path (`197-211`). `FigSwiGLU` and `FigCrossEntropy` are defined/exported in
  `src/little_fig/engine/figkernel.py:74-104,169-204`, but no model/trainer
  replacement or Tier-1 call site was found, so they are standalone utilities.

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

## Completed P3 result (2026-08-18)

The real TinyLlama Tier-1 CPU path completed 20 lowram steps at batch 2 and sequence
length 256. Peak RSS was **7.154205 GiB**, giving **0.845795 GiB** headroom against
8 GiB. Incremental RSS above startup was **6.940331 GiB**, so the paper's ~400 MB
estimate was not reproduced. Load/quantize was the worst phase at 7.154205 GiB;
training peaked at 6.574093 GiB; post-training RSS was 6.030830 GiB. Runtime was
3,686.7 s, with 1,403.3 s spent loading/quantizing and 2,269.9 s training.

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
2. P3 completed in `lowram`: 7.154205 GiB absolute peak, 6.940331 GiB incremental.
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

## 2026-08-19 P5 plan: lowram compute-path (dequant dtype + structure)

New angle beyond P4 windowing/wiring: the lowram COMPUTE path (`DequantMatmul`),
a suspected source of training-phase memory separate from P3a allocator retention.
Two independent variables — dequant intermediate DTYPE (FP32 vs BF16) and STRUCTURE
(full materialization vs tiled/fused). Run in order, record/commit each, surface
after P5a and P5b, do not start P5c until P5b is reviewed:

1. **P5a** — source read only, no code changes: how `DequantMatmul` dequantizes,
   the intermediate dtype, whether backward re-dequantizes, and how the fused kernel
   differs.
2. **P5b** — isolated three-variant test `benchmark/experiment_dequant_variants_v1.py`
   (P3a-style, same two shapes, 20 iters): V1 FP32 baseline, V2 BF16 dtype-only,
   V3 tiled/fused BF16. Measure RSS growth/layer, trim-reclaimed RSS, correctness vs
   V1 (values + grads), wall-clock/iter.
3. **P5c** — wire the structural fix into the real lowram path only if V3 clearly
   wins and passes correctness; re-run full P3 and require a peak clearly below the
   6.8-7.4 GiB three-run noise floor.

## 2026-08-19 P5a result: structure is the lever; dtype is minor

Source read complete (no code changed). Citations to committed code:

- **Structure (Q1): FULL materialization, NOT tiled.** `DequantMatmul.forward` builds
  the whole weight then multiplies (`linear.py:45-46`); `figquant_dequantize`
  (`figquant.py:188-210`) unpacks all nibbles to a full-size int64 tensor (`:197`),
  one `torch.gather` over the whole matrix (`:205`), full FP32 multiply (`:208`). The
  int64 index unpack (forced — `torch.gather` needs an int64 index) is the dominant
  transient, exceeding the FP32 gather result.
- **Dtype (Q2): HARDCODED FP32 intermediate; only the output is cast.** Codebook is
  FP32 (`figquant.py:111`), scales FP32 (`:93,:104`); the full FP32 matrix is built
  regardless of `x`, then `.to(dtype=x.dtype)` (`linear.py:45`) adds another copy.
  BF16 at the cast site does not lower peak; it must be pushed into the dequant, and
  even then the int64 unpack dominates — BF16-only is a modest (<~15%) win.
- **Backward (Q3): RE-DEQUANTIZES FROM SCRATCH.** `save_for_backward` stores only
  `x, indices, codebook, scales` (`linear.py:47`), not W; `backward` rebuilds and
  re-dequantizes (`linear.py:58-63`), paying the full-matrix transient twice.
  `saved_tensors_hooks` cannot catch W (never saved) — the structural dequant is the
  only lever for backward memory.
- **Fused kernel (Q4): does NOT tile — REFUTES the "adapt the fused kernel"
  hypothesis.** `_fig_fused_linear_lora_impl` (`figkernel.py:211-230`) fuses compute
  over a full pre-materialized `cached_W`; no tiling to borrow. Variant 3 must
  implement tiling essentially from scratch. Lowram also applies LoRA as a separate
  add (`linear.py:213-216`), not fused.

Verdict: Variant 3 (tiled BF16, narrowed index handling) is the strongest candidate;
Variant 2 (BF16-only) is the modest dtype control. P5b harness/variants to be built
on this basis and surfaced before P5c.

## 2026-08-19 P5b implementation and Colab handoff

Implemented and committed the isolated P5b benchmark:

- `benchmark/experiment_dequant_variants_v1.py`
- `benchmark/P5b_Dequant_Variants_Colab.ipynb`

The harness compares three clean variants: V1 full FP32 dequantization, V2 full
BF16 dequantization, and V3 BF16 dequantization tiled over output rows. It supports
the P3a TinyLlama shapes (`q_proj` 2048x2048 and `mlp_proj` 5632x2048), configurable
iterations/batch/sequence/tile size, `--results-path`, synthetic `--smoke`, sampled
RSS peaks, and isolated dequant RSS deltas. A smoke run completed successfully.

Commits on `research/p1-figmezo-verify`:

- `89704e7` initial script and notebook
- `1c4eb0a` robust results-file verification in notebook
- `0c00c98` default repository clone and branch configuration

The branch was pushed to the moved repository `Harboria-Labs/littlefig`.
Direct Colab URL:
`https://colab.research.google.com/github/Harboria-Labs/littlefig/blob/research/p1-figmezo-verify/benchmark/P5b_Dequant_Variants_Colab.ipynb`

Important Colab issue and fix: the first notebook version left `REPO_URL` blank,
so Colab stayed in `/content` and failed with
`python3: can't open file '/content/benchmark/experiment_dequant_variants_v1.py'`;
the summary cell then failed because `/content/p5b_dequant_variants_results.json`
did not exist. The current notebook clones the repo automatically, checks out
`research/p1-figmezo-verify`, runs the script via `subprocess` from the active repo,
and asserts the results file exists before reading it. Start a fresh Colab runtime,
open the direct URL, and run cells in order (or Runtime -> Run all). The `jedi`
missing-package warning is unrelated and may be ignored.

P5b is implemented but not yet run on the full TinyLlama-shaped Colab workload.
Next step is to execute the updated notebook, inspect V1/V2/V3 RSS and correctness,
then review results before beginning P5c.

## 2026-08-22 P5c corrected gate result

The harness now uses production `n_iters=8`, realistic transformer weight scale
(`std=0.02`), and direct dequantized-weight RMSE/MSE as the `correctness_pass`
gate. Matmul output RMSE remains separately labeled informational. The smoke test
passed all 9 cases. The full 27-case run (3 iterations across q_proj, k_proj, and
mlp_proj) passed **27/27** cases for V1, V2, and V3. Weight RMSE stayed around
`0.001845-0.001847` (MSE about `3.4e-6`). V3 MLP peak RSS was `577.4 MiB`, versus
`706.0 MiB` V1 and `707.7 MiB` V2, approximately 18% below V1. V3 was slower on
CPU (~2.6-2.8 s versus ~0.33-0.36 s for V1 on MLP). Conclusion: **V3 is
validated for correctness and isolated memory reduction, with a significant
throughput tradeoff; full-model training impact remains unverified.**

## 2026-08-22 P5c V3 slowdown profile

The MLP profile (5632x2048, 44 tiles) measured one tile at ~0.322 ms, estimated
44-tile tensor work at ~14.2 ms, complete `tiled_weight()` loop at 284.2 ms, and
the following BF16 CPU `F.linear` at 3253.5 ms. Thus Python/list/concat overhead is
not the main cause of V3's ~2.7 s case time; BF16 CPU matmul/backend compute is
dominant. `torch.compile` was not pursued because it targets the non-dominant
loop overhead. V3 is an isolated memory optimization with a genuine CPU speed
tradeoff; it is not yet wired into the real lowram path.

## 2026-08-22 P5c correctness diagnostic

The completed P5c smoke/full run marked every variant incorrect because its gate
compared matmul outputs, not weights: `ref = F.linear(x, original)` followed by
`err = (y.float() - ref.float()).abs()` in `benchmark/experiment_dequant_variants_p5c.py`.
P2's validated result instead measures direct dequantized-weight versus FP32-weight
reconstruction MSE. P5c also used `group_size=128, n_iters=1`; P2 and the shipped
FigQuant default use `group_size=128, n_iters=8`. A one-matrix isolation measured
weight/output RMSE `0.092678/4.191747` at n_iters=1 and `0.092280/4.180117` at
n_iters=8. Thus the ~4.2 P5c output RMSE is not comparable to P2's ~5.6e-6 weight
MSE. Classification: **(B) comparison-target mismatch**, secondary **(A) config
mismatch**, not evidence of **(C)**. Do not declare a variant winner or rerun the
correctness gate until the harness uses n_iters=8 and direct weight-level comparison.
