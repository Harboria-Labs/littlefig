# Little Fig — Research Progress Tracker

_Maintained by the research effort. Last updated: 2026-08-06._

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
| FigQuant beats NF4 on MSE | −5.3% MSE, wins 50/50 GPT-2, 156/156 TinyLlama | `figquant.py` real; `v05_results.json` shows −5.28% MSE, 50/50 wins on GPT-2 | 🟡 GPT-2 reproduced locally; TinyLlama 156/156 not in local results |
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
- [~] P1: FigMeZO α-sweep verification. Confirm or refute −18.6%.
      Script: `benchmark/experiment_figmezo_v2.py` + self-contained Colab notebook.
      Design: one FigMeZO code path (α=0 = exact standard-MeZO control), held-out
      eval loss, paired per-seed, proper small-sample t-CIs, logs BOTH eval loss and
      the paper's original train-estimate metric.

      PROVISIONAL (smoke: 2 seeds × 8 steps, gpt2/Alpaca, Colab):
        α=−0.3 eval 5.3069 | α=0 eval 5.3148 | α=+0.7 eval 5.3117
        → α=−0.3 is −0.15% vs standard on HELD-OUT EVAL LOSS.
        → Direction matches paper (−0.3 best), but magnitude is ~180× smaller
          than the claimed −18.6%.
      HYPOTHESIS: the −18.6% lives in the noisy in-sample train-estimate
        (L⁺+L⁻)/2 (single seed), not in generalization. Full 5-seed/100-step run
        in progress to confirm. Likely outcome: "direction-only, magnitude refuted."
- [ ] P2: Reproduce FigQuant 156/156 on TinyLlama locally; save results JSON.
- [ ] P3: Implement the Memory Fabric gate fix (decoupled lr param groups + B-init) that the
      README already claims, then run the synthetic gate-open test to confirm "3 steps".
- [ ] P4: Run Memory Fabric Stage 3 — write N facts into TinyLlama weights, measure
      cross-session recall vs (a) RAG baseline, (b) in-context baseline.
- [ ] P5: Run CogMemBench full 1,000 cases against ≥3 frontier models via Kaggle; get real
      discrimination data (goal #2).

### Phase 2 — Advance (after verification)
- [ ] Strengthen CogMemBench: replace keyword scorer with judge+rubric, expand template
      diversity, add human validation + CIs → make it standard-worthy (goal #4).
- [ ] Improve training (goal #1): if FigMeZO holds, extend to larger models + eval-task metrics.
- [ ] Memory in weights (goal #3): scale Memory Fabric 1→100 facts, measure interference,
      base-capability drift, consolidation/promotion.

---

## Open log
- 2026-08-06 — Initial codebase + 4-paper review. Found contradictions #1–#4 above.
