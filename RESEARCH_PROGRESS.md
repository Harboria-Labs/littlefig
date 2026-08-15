# Little Fig — Research Progress Tracker

_Maintained by the research effort. Last updated: 2026-08-15._

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
      lowram (20 steps, batch 2, sequence 256) peaked at **7.340488 GiB RSS** and
      **7.126644 GiB above startup**, so the absolute 8 GiB budget is reproduced
      with **0.659512 GiB headroom**, but the paper's **0.4 GiB** estimate is not
      reproduced. Load/quantize peaked at 7.054710 GiB; training peaked higher at
      7.340488 GiB and left 6.811050 GiB resident. This measures CPU process RSS
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

---

## Open log
- 2026-08-14 - Added P3 real-path memory benchmark and Drive-backed Colab wrapper.
  The ~400 MB estimate is evaluated as incremental RSS; the 8 GiB requirement uses
  absolute process RSS. The test defaults to lowram mode and an exact-step local
  dataset. Local smoke reached model loading but the environment lacks transformers;
  the Colab wrapper installs project dependencies before running.
- 2026-08-15 - Completed P3 TinyLlama lowram run: 20 steps, batch 2, sequence 256.
  Overall peak RSS was 7.340488 GiB (7.126644 GiB over 0.213844 GiB baseline),
  within the 8 GiB budget but 17.8x the paper's 0.4 GiB estimate. The highest
  phase was training (7.340488 GiB), narrowly above model load/quantize (7.054710
  GiB). Verdicts: 8 GiB **REPRODUCED**; 0.4 GiB **NOT REPRODUCED**.
  Follow-up source audit confirmed AdamW receives only 12,615,680 LoRA parameters
  and embeddings/lm_head are frozen. Expected LoRA params + grads + Adam moments
  are ~192.5 MiB, so full-model optimizer state does not explain the peak. Lowram
  dequantization workspaces/allocator retention are the leading suspects; exact
  attribution remains open.
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
