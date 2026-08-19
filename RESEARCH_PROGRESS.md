# Little Fig — Research Progress Tracker

_Maintained by the research effort. Last updated: 2026-08-18._

This file tracks the state of the Harboria Labs AI Memory Stack research: what each
paper claims, what is actually proven in code, and the plan to (1) verify/prove the
papers, then (2) advance them.

---

## The four-layer stack (where we come from)

| Layer | Project | Claim | Proof status |
|---|---|---|---|
| 1 | **Ember's Diaries** | 8-principle immutable cognitive-memory spec; append-only prevents hallucination accumulation | Engineering only (101 unit tests). Central thesis **unbenchmarked**. |
| 2 | **Memory Fabric** | Write facts into LoRA weights between turns; beats RAG/context, no forgetting | **Entirely synthetic.** No real-model fact recall shown. Decisive test (Stage 3) unrun. |
| 3 | **Fig Engine** | CPU-native INT4 training on 8 GB RAM; FigQuant + FigMeZO + Sens-LISA + shared codebook | FigQuant strong (deterministic, 156/156). Optimizer claims **underpowered** (few seeds, train-loss). |
| 4 | **CogMemBench** | 5-axis cognitive memory benchmark; standard-worthy | v1.0 diagnostic prototype. Keyword scorer + single-model + low diversity → **not yet a standard**. |

The four research goals map onto these layers:
1. **Improve model training** → Fig Engine (Layer 3)
2. **Show Little Fig in the frontier world** → CogMemBench runs vs frontier models (Layer 4)
3. **Solve AI memory by building memory in weights** → Memory Fabric (Layer 2), spec'd by Ember's Diaries (Layer 1)
4. **Improve the memory benchmark, build a standard method** → CogMemBench (Layer 4)

---

## Verification ledger (Phase 1: prove the papers right)

Status key: ✅ proven · 🟡 partial · ❌ unproven/contradicted · ⏳ not started

### Layer 3 — Fig Engine

| Claim | Paper number | Code reality | Status |
|---|---|---|---|
| FigQuant beats NF4 on MSE | −5.3% MSE, wins 50/50 GPT-2, 156/156 TinyLlama | GPT-2 reproduced 50/50 at 5.280921%; TinyLlama reproduced 156/156 at 5.419361% | ✅ Reproduced on both models; complete TinyLlama JSON committed |
| FigMeZO inverse shaping | −18.6% loss @ α=−0.3, 3 seeds | `figmezo.py` implements α=−0.3 default. **BUT committed `experiment_figmezo.py` only tests α=+0.7 and +1.0 — never runs −0.3** | ❌ Headline experiment does not test the headline claim |
| Sensitivity-guided LISA | −10% vs random LISA | `lisa.py` present | ⏳ Not re-run; single-seed in paper |
| Shared codebook | 5× faster load, +3.1% MSE | logic present | ⏳ Not re-run |
| GPU 7× speedup | 184s vs 1309s | reported only | ❌ Confounded: uses 4× more GPU memory (10.2 GB vs 2.4 GB) |
| 64–66× mem reduction (4B/8B) | estimates | — | ❌ Estimated, never measured |

### Layer 2 — Memory Fabric

| Claim | Paper | Code reality | Status |
|---|---|---|---|
| Fact written into weights, recalled next turn | core thesis | `memory_fabric.py` + `micro_trainer.py` real; SGD backprop loop, not FigMeZO by default | ❌ No real-model recall demonstrated |
| Gate fix: decoupled lr, `gate_lr_multiplier=20` | v0.6 win, "gate opens in 3 steps" | **Only in README.** `MicroTrainConfig` has no such field; `MicroTrainer` uses ONE SGD lr for adapters+gates | ❌ Not implemented in committed code |
| B-init fix (zeros → normal 0.02) | v0.6 Gap-2 fix | `MultiAdapterLayer` still inits B as `torch.zeros` | ❌ Not implemented |
| Confidence decay (Ebbinghaus) | principle 3 | `apply_decay()` real | 🟡 Implemented, not validated on real memory |
| Conflict routing (cosine < −0.5) | principle 5 | `detect_conflicts()` real | 🟡 Implemented, not validated |
| Cross-session recall ≥3/10 facts | pre-registered Stage 3 | — | ⏳ Not run |

### Layer 4 — CogMemBench

| Claim | Paper | Code reality | Status |
|---|---|---|---|
| TinyLlama baseline 19/100 | headline | `runner.py`+`scorer.py` real; keyword scorer | 🟡 Number reported, not locally reproduced here |
| Discriminates across models | walked back to v1.1 | — | ❌ Only 1 model in paper |
| Kaggle frontier runs | — | Gemini-3-flash runs exist for all 5 axes (small samples 3–5 cases) | 🟡 Real frontier data exists but tiny N |
| 1,000 cases from tiny pool | 200/axis | `cogmembench_v1.jsonl` (890 KB), generator from ~41 templates | 🟡 Low diversity / contamination risk |

### Layer 1 — Ember's Diaries

| Claim | Status |
|---|---|
| Append-only prevents hallucination accumulation | ❌ Architectural argument only; no HaluMem/TruthfulQA run |
| 101 tests pass | ✅ Engineering (separate repo) |

---

## Key contradictions found (must resolve before publishing)

1. **FigMeZO experiment gap** — `benchmark/experiment_figmezo.py` tests positive shaping
   (+0.7, +1.0) only. The −18.6% claim rests on α=−0.3, which no committed script runs.
   The 3-seed result is asserted in the paper/docstring but has no reproducible script here.
2. **Memory Fabric gate fix is vaporware in code** — README v0.6 table claims the gate fix
   shipped ("Gate opens in 3 steps, was stuck at 27%"), but neither the `gate_lr_multiplier`
   nor the B-init change exists in `memory_fabric.py` / `micro_trainer.py`.
3. **Memory Fabric has zero real-model evidence** — all quantitative claims are synthetic
   hidden-state vectors at pre-set cosine similarities (circular).
4. **No baseline comparisons anywhere** — the thesis "weights beat retrieval" has never been
   run head-to-head vs RAG or in-context.

---

## Plan

### Phase 1 — Verify & prove (current focus)
- [x] P1: FigMeZO α-sweep verification. **VERDICT: −18.6% claim REFUTED.**
      Script: `benchmark/experiment_figmezo_v2.py` + self-contained Colab notebook.
      Design: one FigMeZO code path (α=0 = exact standard-MeZO control), held-out
      eval loss, paired per-seed, proper small-sample t-CIs.

      RESULT (full: 5 seeds × 100 steps, gpt2/Alpaca, Colab, 2599s):
        held-out eval loss (mean ± 95% t-CI):
          α=−0.3: 6.7494 ± 0.0194
          α= 0.0: 6.7599 ± 0.0441   (standard MeZO control)
          α=+0.7: 6.7692 ± 0.0432
        α=−0.3 vs standard = **−0.16%** (claim was −18.6% → off by ~120×).
        Paired diff = −0.0105 ± 0.0585, t=−0.50 (need 2.776) → NOT significant.
        Only **2/5 seeds** favored α=−0.3. The three settings are statistically
        indistinguishable at n=5.
      TAKEAWAY: inverse error-shaping gives no measurable generalization benefit
        here. Means are weakly monotonic in the paper's predicted direction, but
        noise-dominated. The original −18.6% almost certainly came from a single
        seed on the in-sample train estimate (L⁺+L⁻)/2, not held-out eval.
      Raw + corrected verdict saved: `benchmark/figmezo_v2_results.json`.
      OPTIONAL follow-up: re-run corrected script (logs train_est) to demonstrate
        the −18.6% appears in the train estimate but vanishes on eval.
- [x] P2: Reproduce FigQuant 156/156 on TinyLlama in Colab; save results JSON.
      **RESULT: REPRODUCED.** TinyLlama completed **156/156** wins. Mean
      reconstruction MSE reduction vs NF4: **5.419361%** (NF4 5.9652037e-6 ->
      FigQuant 5.6419278e-6); vs absmax INT4: **36.877870%**. Per-layer reduction
      min **2.498559%**, median **5.743715%**, max **21.691531%**; SNR gain
      **0.252763 dB**; losers **0**. Runtime **1214.2 s**. This exactly closes
      the missing TinyLlama evidence gap for the FigQuant quality claim.
      GPT-2 Colab self-check (2026-08-13): FigQuant won **50/50** matrices and
      reduced mean reconstruction MSE by **5.280921%** vs NF4, matching the
      committed result. This validates the harness and proves the GPT-2 half of
      the paper claim. The later completed Drive-backed run supplies the matching
      **156/156 TinyLlama** result recorded above.
      Script: `benchmark/experiment_figquant_v2.py`
      + self-contained `benchmark/P2_FigQuant_Colab.ipynb`. Tests the SHIPPED
      `figquant_quantize` (group_size=128, n_iters=8), reproduces GPT-2 50/50 as a
      harness self-check vs committed `v05_results.json`, then runs the missing
      TinyLlama 1.1B benchmark (156 layers). Deterministic single run (no seeds —
      k-means from fixed NF4 init on fixed weights). Memory-safe smallest-first
      streaming (~9 GB peak) so the two 65.5M embed/lm_head matrices fit free Colab.
      Note: FigQuant = "NF4 init + per-layer k-means"; k-means provably lowers
      *normalized* distortion, but the metric is *reconstruction* MSE (reweighted by
      per-group scale²), so all-156-wins is a real empirical result, not a tautology.
      Recovery history: an earlier old-script run stopped at `[155/156] START
      lm_head.weight` with `^C`; it had no checkpoint. The new Drive-backed run
      completed all 156 layers and saved the complete JSON. The benchmark checkpoints each layer atomically, supports
      `--resume`, writes to Drive via `--results-path`, and bounds k-means/NF4/INT4
      distance tensors to avoid the former ~3.9 GiB final-layer allocation.
      Memory evidence includes sampled process RSS, per-layer workspace peaks,
      system available-memory floor, and a durable in-layer heartbeat. The completed
      run's **8.668636 GiB** collection peak is a harness artifact: `collect_weights`
      retained the full **4.098 GiB** FP32 HF model while accumulating another
      **4.098 GiB** of FP32 clones. It does not test or refute Fig Engine's training
      memory claim. Quantization itself peaked at **4.932980 GiB** RSS. The harness
      now releases each source parameter after cloning it; a new run is needed to
      measure the corrected collection peak. `FigModel.from_pretrained()` is a
      different code path and its end-to-end training peak remains independently
      unverified.
- [x] P3: Measure Fig Engine Tier-1 memory on the real TinyLlama training path.
      Script: `benchmark/experiment_8gb_v1.py`; Colab: `benchmark/P3_8GB_Colab.ipynb`.
      It records absolute process RSS against the 8 GiB budget and incremental RSS
      over startup against the paper's ~400 MB estimate. Default mode is `lowram`;
      `fast` caches full FP32 dequantized target weights and is not the minimum-memory
      claim. `FigModel.from_pretrained()` also builds all replacements before applying
      them, while embeddings/lm_head remain FP32. **Completed 2026-08-15:** TinyLlama
      lowram (20 steps, batch 2, sequence 256) peaked at **7.154205 GiB RSS** and
      **6.940331 GiB above startup**, so the absolute 8 GiB budget is reproduced
      with **0.845795 GiB headroom**, but the paper's **0.4 GiB** estimate is not
      reproduced. Load/quantize peaked at 7.154205 GiB; training peaked at
      6.574093 GiB and left 6.030830 GiB resident. This measures CPU process RSS
      for `FigModel.from_pretrained` plus Tier-1 training, not just packed weight
      storage (reported base weights were 522.3 MB).
      Source audit after the run rules out full-model AdamW state: Tier 1 passes
      `model.get_trainable_parameters()` to AdamW, which filters on
      `requires_grad`, and model loading freezes every parameter whose name does
      not contain `lora_`. The measured 12,615,680 LoRA parameters occupy 48.13
      MiB; parameters + gradients + two FP32 AdamW moments total about 192.5 MiB.
      Embeddings/lm_head remain FP32 storage but are frozen. Therefore the 4.356
      GiB rise from training entry to peak is not explained by optimizer scope.
      The leading source-level suspect is transient lowram dequantization plus CPU
      allocator retention: every FigLinear forward and backward expands packed
      indices to int64 and materializes an FP32 weight. The 6.811 GiB post-training
      RSS is consistent with retained workspaces, but activation/checkpoint and
      allocator contributions require finer instrumentation before assigning cause.
- [ ] P3a: Isolate lowram dequantization allocator retention before another full
      training run. `benchmark/experiment_lowram_allocator_v1.py` exercises the
      shipped `FigLinear` path using exact TinyLlama q_proj and MLP matrix shapes,
      with no model loader, trainer, optimizer, dataset, or checkpointing. Record
      per-iteration RSS at start/forward/backward/GC and compare before/after Linux
      `malloc_trim(0)`. Steady single-layer growth implicates the lowram path;
      trim-released RSS identifies allocator cache rather than live tensors.
- [x] P3a completed 2026-08-18: q_proj retained 161.6 MiB after GC and released
      155.4 MiB with `malloc_trim(0)`; mlp_proj retained 94.9 MiB and released
      102.7 MiB. Verdict: allocator retention reproduced; no unbounded leak.
- [ ] P3b: Repeat the complete TinyLlama lowram run with baseline versus
      `--allocator-trim` after every optimizer step. Code pushed as commit
      `a1c1cbc`; compare peak and post-training RSS before proceeding.
      **Baseline leg completed 2026-08-18:** 20 steps, batch 2, sequence 256,
      allocator trim disabled. Overall peak was **7.154205 GiB RSS** (**6.940331
      GiB** above the 0.213875 GiB startup baseline), with **0.845795 GiB**
      headroom under the 8 GiB budget. Model load/quantize was the worst phase
      (7.154205 GiB); training reached 6.574093 GiB and post-training idle was
      6.030830 GiB. Verdict: **8 GiB REPRODUCED**; paper's **0.4 GiB estimate
      NOT REPRODUCED**. Durable result: `figengine_8gb_lowram_results.json`.
      The allocator-trim comparison leg is still pending, so P3b remains open.
- [ ] P4: Implement the Memory Fabric gate fix (decoupled lr param groups + B-init) that the
      README already claims, then run the synthetic gate-open test to confirm "3 steps".
- [ ] P5: Run Memory Fabric Stage 3 — write N facts into TinyLlama weights, measure
      cross-session recall vs (a) RAG baseline, (b) in-context baseline.
- [ ] P6: Run CogMemBench full 1,000 cases against ≥3 frontier models via Kaggle; get real
      discrimination data (goal #2).

### Phase 2 — Advance (after verification)
- [ ] Strengthen CogMemBench: replace keyword scorer with judge+rubric, expand template
      diversity, add human validation + CIs → make it standard-worthy (goal #4).
- [ ] Improve training (goal #1): if FigMeZO holds, extend to larger models + eval-task metrics.
- [ ] Memory in weights (goal #3): scale Memory Fabric 1→100 facts, measure interference,
      base-capability drift, consolidation/promotion.

## P4: FigSweep wiring + storage-backed memory investigation

_Plan recorded 2026-08-18. Execute strictly in order and record/commit each result
before starting the next item. Surface each result for review; do not proceed to
P4f until P4a-P4e are complete._

- [x] **P4a — OS-level swap quick test.** First determine whether the Colab Linux
  environment permits creating/enabling swap. If permitted, rerun the existing P3
  baseline unchanged with swap enabled and compare phase peak RSS against the P3b
  baseline (7.154205 GiB load/quantize; 6.574093 GiB training). Why first: this is
  the cheapest possible storage-backed memory experiment and establishes whether
  OS paging provides useful relief before custom streaming work. If Colab forbids
  swap, record the restriction plainly and proceed only after review.
- [ ] **P4b — Wire FigSweep into lowram training.** Connect the existing
  `figsweep_advance()` mechanism to actual layer-by-layer forward execution, enable
  `figsweep_window=4`, rerun the P3 configuration, and compare load, training,
  post-training, speed, and overall RSS against P3b. Why second: the source audit
  proved the intended rolling-window method exists but never advances, making this
  the highest-priority real implementation gap behind the training memory spike.
- [ ] **P4c — Wire and test sensitivity-guided LISA.** Pass real probe inputs to
  `LISAScheduler`, then run a paired sensitivity-guided-versus-random LISA study on
  GPT-2 with 8 seeds and held-out evaluation loss. Why third: current LISA silently
  uses uniform random sampling, so no existing result tests the paper's claimed
  10% sensitivity advantage.
- [ ] **P4d — Test automatic tier selection end to end.** On normal Colab RAM,
  construct trainer/model without manually selecting a tier and record the tier
  chosen from real available RAM. Then, if the sandbox permits, constrain memory
  with a cgroup/container limit and verify automatic downgrade behavior. Why
  fourth: the selector exists but all research runs bypassed it, and its practical
  behavior and constrained-memory fallback remain unverified.
- [ ] **P4e — Compare lowram, figcache, and fast modes.** Run the same P3 workload
  in all three modes and report phase RSS, overall peak, post-training residency,
  runtime, and steps/second. Why fifth: only lowram has been measured, while the
      cached modes exercise the fused Linear+LoRA path and expose the actual
      memory/speed tradeoff.
- [ ] **P4f — Design disk-streamed FigSweep window; isolated proof only.** After
  P4a-P4e are complete, write a design using `torch.from_file` or safetensors mmap
  plus `torch.autograd.graph.saved_tensors_hooks` to manage forward/backward tensor
  ownership correctly. Build only a 2-3-layer proof of concept before modifying
  full-model training. Why last: disk streaming is the largest and riskiest change;
  earlier measurements must establish whether it is necessary and what interface
  it must improve. Never release a mapped tensor merely because forward returned,
  because autograd may still require it during backward.

**P4a result (2026-08-18): NOT RUN / ENVIRONMENT BLOCKED.** This workspace is
Windows PowerShell, not the Colab Linux runtime used by P3. No Colab session or
Linux swap control is available here, so swap permission and a swap-enabled P3
rerun cannot be measured without inventing a result. No substitute test was run;
P4b is paused pending the Colab swap-permission outcome.

## P5: lowram compute-path investigation — dequant dtype and structure

_Plan recorded 2026-08-19. A new angle beyond P4 windowing/wiring: the actual
lowram COMPUTE path (`DequantMatmul`), a suspected source of training-phase memory
separate from the P3a allocator-retention finding. Two independent variables:
(1) the dequant intermediate's DTYPE (FP32 vs BF16); (2) its STRUCTURE (materialize
the full matrix vs tiled/fused, never holding the full matrix). Run strictly in
order, record/commit each result, surface after P5a and P5b before proceeding, and
do not start P5c until P5b is reviewed._

- [x] **P5a — Source read only, no code changes.** Determine exactly how
  `DequantMatmul` dequantizes (full vs tiled), what dtype the intermediate uses
  (hardcoded vs configurable), whether `backward()` re-dequantizes or reuses saved
  tensors, and how `fig_fused_linear_lora` differs. Why first: the fix design
  depends entirely on the real mechanism; guessing risks building the wrong variant.
- [ ] **P5b — Isolated three-variant test** (`benchmark/experiment_dequant_variants_v1.py`),
  P3a-style harness (same two layer shapes, 20 iterations, directly comparable to
  P3a: q_proj 161.6 MiB, mlp_proj 94.9 MiB growth, ~96-100% trim-reclaimable).
  Variant 1 = current FP32 baseline; Variant 2 = BF16 intermediate (dtype-only);
  Variant 3 = tiled/fused BF16 (structural, never holds the full matrix). Per
  variant, measure RSS growth per layer, trim-reclaimed RSS, correctness vs Variant 1
  (values + gradients within tolerance), and wall-clock per iteration. Why: isolates
  the two variables cleanly before touching the real path.
- [ ] **P5c — Wire the structural fix into the real lowram path**, ONLY if Variant 3
  clearly beats Variant 1 AND passes correctness. Re-run full P3 (batch 2, seq 256,
  20 steps); a genuine fix lands clearly below the 6.8-7.4 GiB three-run noise floor,
  not merely within it. Why last: the real-path change is riskiest; gate it on the
  isolated proof.

**P5a result (2026-08-19): SOURCE READ COMPLETE. Structure is the real lever;
dtype is minor and orthogonal.** All citations are to committed code.

- **Q1 — full materialization vs tiled: FULL, NOT TILED.** `DequantMatmul.forward`
  builds one complete weight then multiplies: `W = figquant_dequantize(q).to(dtype=
  x.dtype)`, `y = F.linear(x, W)` (`linear.py:45-46`). `figquant_dequantize`
  (`figquant.py:188-210`) has no tiling: it unpacks ALL nibbles to a full-size int64
  tensor (`torch.stack([low, high], dim=1).reshape(-1)`, `:197`), does one
  `torch.gather` over the whole matrix (`:205`), then a full-size FP32 multiply
  (`result * scales`, `:208`). The **int64 index unpack is the dominant transient**
  (`.long()` at `:195-196`; `torch.gather` requires an int64 index, so this is
  forced): `low`+`high`+stacked hold roughly numel*4 + numel*4 + numel*8 bytes
  simultaneously, larger than the numel*4 FP32 gather result. For q_proj (2048^2,
  4.19M) this is tens of MiB; for the gate/up MLP (2048x5632, 11.5M) over a hundred
  MiB — consistent with P3a's per-layer growth and confirming the compute path (not
  only allocator retention) creates the workspace.
- **Q2 — dtype hardcoded vs configurable: HARDCODED FP32 intermediate; only the
  OUTPUT is cast.** The codebook is `dtype=torch.float32` (`figquant.py:111`) and
  scales are FP32 (`:93,:104`); the gather output inherits FP32 and the multiply
  stays FP32. The full matrix is built in FP32 regardless of `x`. The only dtype
  flexibility is `.to(dtype=x.dtype)` applied AFTER the FP32 build (`linear.py:45`),
  which adds another full-size copy rather than avoiding the FP32 peak. Implication:
  a naive BF16 fix at the `.to()` site does NOT reduce peak; BF16 must be pushed
  INTO the dequant (bf16 codebook + bf16 gather), and even then the numel*8 int64
  index unpack is dtype-independent and dominates, so BF16-only is expected to be a
  modest (<~15%) win.
- **Q3 — backward re-dequant vs saved: RE-DEQUANTIZES FROM SCRATCH; saves only the
  compact packed rep.** `ctx.save_for_backward(x, indices, codebook, scales)`
  (`linear.py:47`) saves the input and the packed INT4 trio, NOT W. `backward`
  rebuilds `q` and calls `figquant_dequantize(q)` again (`linear.py:58-63`), paying
  the same full-matrix transient a second time. This confirms `saved_tensors_hooks`
  (the disk-offload idea) cannot catch W — W is never a saved tensor — so backward's
  memory cost must be attacked at the dequant structure, not via saved-tensor offload.
- **Q4 — fused-kernel comparison: `fig_fused_linear_lora` does NOT tile.**
  `_fig_fused_linear_lora_impl` (`figkernel.py:211-230`) takes `cached_W` (an
  already-dequantized full FP32 matrix) and fuses `F.linear` + LoRA matmuls + scale +
  add into one inductor kernel (`:228-229`). It reduces kernel launches / vectorizes
  (AVX-512) but still consumes a full-size W; it merely caches it once at mode-switch
  (`linear.py:230`) instead of recomputing per call. The hypothesis that the fused
  kernel already contains tiling to adapt is **REFUTED** — Variant 3 must implement
  tiling essentially from scratch. (Lowram also applies LoRA as a separate add,
  `linear.py:213-216`, not fused.)

**What this means for P5b:** Variant 3 (tiling) is the real lever — a tiled dequant
keeps only a small int64 index chunk and a small BF16 output tile live at once,
directly attacking the dominant transient that BF16-alone leaves intact. Variant 2
stays in the plan as the isolated dtype control, but its expected win is modest.
Variant 3 should also narrow the index handling (avoid a full numel*8 int64 buffer),
not merely switch the output to BF16. Harness and variants will be built on this
basis, then surfaced before P5c. No code changed in P5a.

---

## Open log
- 2026-08-14 - Added P3 real-path memory benchmark and Drive-backed Colab wrapper.
  The ~400 MB estimate is evaluated as incremental RSS; the 8 GiB requirement uses
  absolute process RSS. The test defaults to lowram mode and an exact-step local
  dataset. Local smoke reached model loading but the environment lacks transformers;
  the Colab wrapper installs project dependencies before running.
- 2026-08-18 - Re-ran P3 TinyLlama lowram baseline: 20 steps, batch 2, sequence 256.
  Overall peak RSS was 7.154205 GiB (6.940331 GiB over 0.213875 GiB baseline),
  within the 8 GiB budget with 0.845795 GiB headroom but 17.35x the paper's 0.4
  GiB estimate. Model load/quantize was highest (7.154205 GiB); training peaked
  at 6.574093 GiB. Verdicts: 8 GiB **REPRODUCED**; 0.4 GiB **NOT REPRODUCED**.
  Follow-up source audit confirmed AdamW receives only 12,615,680 LoRA parameters
  and embeddings/lm_head are frozen. Expected LoRA params + grads + Adam moments
  are ~192.5 MiB, so full-model optimizer state does not explain the peak. Lowram
  dequantization workspaces/allocator retention are the leading suspects; exact
  attribution remains open.
- 2026-08-18 - P3b baseline leg completed with allocator trimming disabled. The
  full 20-step run returned code 0 and saved `figengine_8gb_lowram_results.json`:
  7.154205 GiB overall peak, 6.940331 GiB incremental RSS, 8 GiB **REPRODUCED**,
  0.4 GiB paper estimate **NOT REPRODUCED**. The `--allocator-trim` comparison
  remains to be run.
- 2026-08-18 - Source audit of dormant Fig Engine features (A1-A4): FigSweep's
  `enable_figsweep()` is wired only when `figsweep_window > 0`, but
  `figsweep_advance()` has no call site in the trainer/model forward path; the
  tested lowram path therefore dequantizes each layer independently with no
  rolling-window bound. LISA is a separate `TrainingTier.LISA` dispatch branch;
  STREAMING_LORA never enters it, and the trainer supplies no probe inputs, so
  even LISA's current path uses uniform random layer selection rather than
  sensitivity-guided weights. Auto tier selection does exist when
  `FigTrainingConfig.tier` is `None`: it reads `psutil.virtual_memory().available`,
  budgets 70%, and returns the first fitting tier in LISA, LOMO, STREAMING_LORA,
  MeZO order; all P1-P3b tests explicitly forced STREAMING_LORA. FigKernel is
  only partially wired: model loading automatically swaps RMSNorm modules, and
  FigLinear uses fused Linear+LoRA only for cached fast-mode weights; FigSwiGLU
  and chunked cross-entropy are exported standalone but have no model/trainer
  swap or call site in Tier 1.
- 2026-08-18 - P4a OS-level swap quick test: **NOT RUN / ENVIRONMENT BLOCKED**.
  This workspace is Windows PowerShell and has no access to the Colab Linux
  instance, so Linux swap permission and a swap-enabled P3 rerun cannot be tested
  honestly. No substitute local test was run; P4b remains paused pending the Colab
  swap-permission result.
- 2026-08-19 - Observability item 1 completed: FigQuant model loading now emits
  live per-layer progress with `[n/total]`, layer duration, running average,
  elapsed time, ETA, cumulative compression, and current process RSS. Existing
  notebook stdout capture makes this telemetry durable in the benchmark log.
  `src/little_fig/engine/model.py` passes `py_compile` and `git diff --check`.
  No benchmark rerun was performed in this turn.
- 2026-08-19 - Observability item 2 completed: `benchmark/P3_8GB_Colab.ipynb`
  now includes a post-run telemetry visualization cell. It reads the durable
  JSON/log, plots peak RSS by phase with the configured budget, plots training
  loss/speed/learning-rate when step lines are present, reports quantization RSS
  samples, and saves a PNG to the Drive benchmark directory. Notebook JSON
  validation passed; no Colab rerun was performed in this turn.
- 2026-08-19 - FigSweep wiring implementation completed: `enable_figsweep()` now
  registers forward pre-hooks on every FigLinear in traversal order, so each layer
  calls `figsweep_advance()` before its forward. Existing hooks are removed before
  re-enabling, and module traversal order is preserved instead of lexical sorting.
  Source syntax validation passed. A Colab run is still required to measure
  RSS/speed and verify backward correctness.
- 2026-08-19 - LISA wiring implementation completed: `_train_lisa()` now takes a
  real batch from a fresh dataloader iterator and passes `input_ids`/`labels` to
  `LISAScheduler` for its optional sensitivity probe. The probe no longer silently
  falls back to uniform random selection when a dataset is available. Trainer
  syntax validation passed. The 8-seed held-out sensitivity-vs-random quality
  benchmark remains pending; this entry validates wiring only.
- 2026-08-14 - Added resumable Drive-backed P2 workflow and bounded-memory final
  layer calculations after the corrected 154/156 TinyLlama partial run. The old run
  stopped at `[155/156] START lm_head.weight` with `^C`.
  Added explicit 8 GiB budget evaluation and durable in-layer memory heartbeat.
  Removed the obsolete P1 testing notebook; the P2 resumable notebook is now the
  only benchmark notebook, while the root Little Fig notebook remains untouched.
  Full handoff: `SESSION_SUMMARY.md`.

- 2026-08-13 - P2 GPT-2 harness self-check reproduced 50/50 wins and 5.280921%
  lower MSE vs NF4. TinyLlama 156/156 result still pending.

- 2026-08-06 — Initial codebase + 4-paper review. Found contradictions #1–#4 above.
