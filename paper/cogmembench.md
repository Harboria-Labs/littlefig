# CogMemBench: An Axis-Decomposed Benchmark for Cognitive Memory Reasoning in Large Language Models

**Authors:** 0xticketguy (Harboria Labs)
**Repository:** https://github.com/Harboria-Labs/littlefig/tree/main/cogmembench
**Version:** 1.0 — Dataset: 1,000 evaluation cases across 5 cognitive axes
**License:** AGPL-3.0

> **Harboria Labs Research Stack**
> This paper is Layer 4 of a four-part research program.
> Layer 1 — Ember's Diaries: cognitive memory specification
> Layer 2 — Memory Fabric: neural weight-space implementation
> Layer 3 — Fig Engine: training infrastructure
> Layer 4 — **CogMem Benchmark** *(this paper)*: evaluation

---

## Abstract

AI memory systems have proliferated rapidly — OpenAI ChatGPT Memory, Google Gemini Memory, Anthropic Claude Projects, and neural architectures including MemoryLLM, M+, and Memory Fabric — but no standardized, reproducible evaluation framework exists for comparing them. Existing benchmarks either evaluate static knowledge (MMLU, HumanEval), measure hallucination at the operation level (HaluMem), or evaluate system-level retention over simulated long conversations (MemoryBench). None isolates the five core cognitive capabilities that distinguish a genuine memory system from in-context reading.

We present CogMemBench, an axis-decomposed benchmark that evaluates five fundamental cognitive memory capabilities grounded in established cognitive psychology: (1) fact acquisition, (2) goal-directed recall (Conway's Self-Memory System, 2005), (3) graceful decay of outdated knowledge (Ebbinghaus, 1885), (4) conflict detection between stored facts (motivated by HaluMem findings), and (5) consolidation through repeated exposure (Atkinson-Shiffrin, 1968). The dataset contains 1,000 programmatically generated evaluation cases (200 per axis) with deterministic seeding.

We evaluate TinyLlama 1.1B as a baseline and find a CogMem Score of 19.0/100, with strong acquisition (75%) but near-zero performance on goal-directed recall (10%), temporal decay (0%), conflict detection (0%), and consolidation (10%). This establishes the floor that memory-augmented models must surpass.

We also describe a planned cross-session evaluation track, which extends CogMemBench to evaluate persistent weight-space and database memory systems — architectures where memories are not injected into the prompt but must be retrieved from the system's internal state. This track is the correct methodology for evaluating Memory Fabric and similar persistent memory architectures.

---

## 1. Introduction

Every major AI organization shipped memory features in 2025-2026: OpenAI ChatGPT Memory, Google Gemini Memory, Anthropic Claude Projects. Research architectures including MemoryLLM, M+, Memory³, MEGa, MemOS, and Harboria Labs Memory Fabric represent different technical approaches to the same problem. Yet there is no independent, reproducible way to compare these implementations. Nobody knows which one actually works.

This gap exists for a structural reason: existing benchmarks do not measure the capabilities that define a genuine memory system.

**MMLU, HumanEval, and similar benchmarks** evaluate static knowledge from pretraining. A model that cannot update its knowledge cannot improve on these benchmarks through memory alone.

**HaluMem** [2024] evaluates hallucinations in memory system operations (extraction, updating, question answering) across very long conversations (1,500–2,600 turns, 1M+ token context). It is the appropriate benchmark for evaluating hallucination behavior in memory operations, and CogMemBench positions itself as complementary rather than competing.

**MemoryBench** [2025] evaluates LLM systems on memorizing and learning from user feedback, grounded in the same cognitive psychology literature as CogMemBench (Atkinson-Shiffrin, Ebbinghaus). It tests the system-level capability to retain and use information across interactions.

CogMemBench occupies a different but related niche: **axis-decomposed evaluation of cognitive reasoning over provided memory contexts.** Rather than testing end-to-end memory system behavior (HaluMem, MemoryBench), CogMemBench decomposes the cognitive layer into five constituent capabilities, each grounded in a specific cognitive science framework, and evaluates them independently. This decomposition enables targeted diagnosis — a model that fails goal-directed recall but passes acquisition has a different architecture problem than one that fails conflict detection.

CogMemBench is designed for two distinct evaluation targets:

**Cognitive reasoning track (in-context).** Memories are provided in the prompt. The model must demonstrate goal-directed retrieval, temporal discrimination, conflict detection, and consolidation reasoning over supplied context. This measures the model's *ability to reason about memory*, independently of whether it has a persistent memory architecture.

**Cross-session track (persistent, forthcoming).** No memories are injected into the prompt. The model must retrieve from its own internal state (weight-space adapters, external database, or cache). This measures actual persistent memory effectiveness, and is the correct track for evaluating Memory Fabric, MemoryLLM, and similar architectures.

### 1.1 Position in the Harboria Labs Research Stack

CogMemBench is Layer 4 of the Harboria Labs research stack. It evaluates whether Layers 1-3 (Ember's Diaries specification, Memory Fabric neural implementation, Fig Engine training infrastructure) actually produce a better memory system. Critically, CogMemBench evaluates any memory architecture — it is not specific to Memory Fabric. Any system that can be queried with memory scenarios can be scored on CogMemBench.

---

## 2. Related Work

### 2.1 Memory System Benchmarks

**HaluMem** [arXiv:2511.03506, 2024] is the first operation-level hallucination evaluation benchmark for memory systems. It defines three evaluation tasks — memory extraction, memory updating, and memory question answering — across conversations averaging 1,500–2,600 turns with context lengths exceeding 1 million tokens. HaluMem's focus is on hallucination accumulation during memory operations. CogMemBench's focus is on cognitive reasoning capabilities (goal-directed recall, temporal awareness, conflict detection) that determine *why* hallucinations occur.

**MemoryBench** [2025] is grounded in the same cognitive psychology literature as CogMemBench (Atkinson & Shiffrin 1968, Ebbinghaus 1885) and evaluates LLM systems on their ability to memorize and learn from user feedback. It uses a simulation platform to test cross-session retention. CogMemBench differs by providing standardized in-context evaluation cases that any model can be scored on without requiring a full multi-session deployment.

**LongMemEval** evaluates memory capabilities in long-context settings through single-session multi-turn conversations. It focuses on factual consistency over extended conversations rather than the cognitive capabilities CogMemBench targets.

**ZeroScrolls, HELMET** evaluate long-context understanding. They test whether a model can find information in a long context, not whether it can reason cognitively about that information.

### 2.2 What CogMemBench Does Differently

The key structural distinction is axis decomposition. Rather than a single aggregate memory score, CogMemBench measures five cognitive capabilities independently, each grounded in a specific theoretical framework. A model can fail acquisition (basic retention) while passing recall (goal-direction), which reveals something specific about its architecture. This diagnostic granularity is what CogMemBench contributes to the evaluation landscape.

The second distinction is scope: CogMemBench tests *whether a model can reason about memory cognitively*, not just whether it can retrieve facts. Goal-directed recall (Conway, 2005) requires understanding which memory is relevant to a goal, not just which memory is topically similar. This is the capability that distinguishes a useful AI memory system from a keyword search engine.

---

## 3. Evaluation Framework

### 3.1 Cognitive Axes

CogMemBench evaluates five axes, each grounded in cognitive psychology:

| Axis | Measures | Grounded In | Weight |
|---|---|---|---|
| **Acquisition** | Can the model encode and retain a new fact? | Basic memory encoding | 20% |
| **Goal-directed Recall** | Does it retrieve by task-relevance or topic-similarity? | Conway's Self-Memory System (2005) | 25% |
| **Graceful Decay** | Does outdated knowledge become less certain? | Ebbinghaus Forgetting Curve (1885) | 20% |
| **Conflict Detection** | Can it identify contradictions between stored facts? | HaluMem (2024) findings | 20% |
| **Consolidation** | Does repeated exposure strengthen knowledge? | Atkinson-Shiffrin Model (1968) | 15% |

Goal-directed recall receives the highest weight because it is the most discriminating capability. Keyword search retrieves by topic similarity. A genuine cognitive memory system retrieves by goal relevance. This is the capability that distinguishes the two.

### 3.2 Why Goal-Directed Recall Matters

The distinction between topic-similarity retrieval and goal-directed retrieval is the core theoretical contribution of Conway's Self-Memory System (2005): human memory retrieval is organized around *current goals*, not *association strength*.

CogMemBench makes this concrete with distractor design. Each recall test case presents a set of memories and a current goal. One memory is goal-relevant (e.g., "Sarah's birthday is June 12th" for the goal "plan a birthday surprise"). One distractor is topically similar but goal-irrelevant (e.g., "Sarah loves Italian food"). A model that retrieves by topic similarity will select either with equal probability. A model with genuine goal-directed recall will consistently select the goal-relevant fact.

RAG-based retrieval systems typically fail this test because vector similarity is a proxy for topic similarity, not goal relevance. This is the key empirical motivation for Memory Fabric's learned gating approach.

### 3.3 The Decay Axis — Conceptual Note

The decay axis evaluates whether a model expresses differential confidence in facts from different time periods. This requires careful interpretation:

The Ebbinghaus forgetting curve describes biological memory decay — the observer forgets stable facts over time. CogMemBench tests a different phenomenon: **verification recency**. A fact that was verified yesterday is more reliably current than a fact that was verified a year ago, even if the underlying fact is stable.

Test cases are designed around facts that can legitimately change (office locations, API endpoints, project statuses) rather than stable facts (birthdays, names). The correct model behavior is to express higher certainty about recently verified information and appropriate hedging about information that may have changed ("might have moved," "should verify").

A model that treats all facts as equally reliable regardless of when they were verified fails this axis. A model that appropriately hedges on unverified older facts passes.

---

## 4. Dataset

### 4.1 Format

Each test case is a JSON object:

```json
{
    "id": "abc123",
    "axis": "recall",
    "prompt": "Current goal: Plan a dinner for my wife's birthday...",
    "context": {
        "goal": "Plan a dinner for my wife's birthday",
        "memories": [
            "Wife's birthday is June 12th",
            "Wife loves Italian food",
            "Restaurant nearby is open Saturdays",
            "Budget for dinner is $150"
        ]
    },
    "correct_answer": "Wife's birthday is June 12th",
    "distractor": "Wife loves Italian food",
    "difficulty": "medium",
    "metadata": {
        "reasoning": "Birthday date is needed for timing; food preference is secondary"
    }
}
```

### 4.2 Statistics

| Property | Value |
|---|---|
| Total cases | 1,000 |
| Cases per axis | 200 |
| Difficulty distribution | 33% easy / 34% medium / 33% hard |
| Average prompt length | ~150 tokens |
| Deterministic (seed=42) | Yes |
| Format | JSONL |
| Approximate file size | ~1.2 MB |

### 4.3 Data Generation

All cases are programmatically generated from a curated pool:

- 15 personal facts (with question/answer pairs)
- 10 goals (with task contexts)
- 8 goal-conditioned recall scenarios
- 8 conflict scenarios (with type and resolution)

The generator is deterministic — the same seed produces identical dataset. This ensures reproducibility across evaluations and enables fair comparison between models.

---

## 5. Scoring

### 5.1 Per-Axis Evaluation

Each axis uses task-specific evaluation:

**Acquisition.** Fuzzy keyword match: ≥70% of answer keywords must be present in the response to count as correct. Designed to tolerate reasonable paraphrase while rejecting hallucinations.

**Recall.** Binary: the correct (goal-relevant) memory must be identified AND the distractor must not be selected as the primary answer. This tests goal-direction, not merely topic proximity.

**Decay.** LLM-as-judge: the evaluator assesses whether the response expresses appropriate differential confidence (higher certainty for recently verified facts, appropriate hedging for older or potentially outdated information).

**Conflict.** Keyword detection + citation verification: response must contain explicit conflict language ("contradict," "inconsistent," "conflict," "mismatch") AND reference both conflicting facts specifically.

**Consolidation.** LLM-as-judge: the evaluator assesses whether the model expresses higher confidence in the repeatedly-confirmed fact compared to the single-mention fact, with reasoning about why.

### 5.2 CogMem Score

Weighted average of per-axis accuracy:

```
CogMem Score = 20% × Acquisition
             + 25% × Recall
             + 20% × Decay
             + 20% × Conflict
             + 15% × Consolidation
```

### 5.3 Scoring Limitations

**Keyword matching is gameable.** A model could include all correct keywords without genuine reasoning. Future versions use LLM-as-judge for all axes. For the current version, recall and conflict use keywords as a first-pass filter with human spot-checking recommended for published results.

**False negatives from shared tokens.** In the current implementation, common words shared between the correct answer and the distractor can cause correct responses to score as failures. Specifically: a response that mentions "Sarah's birthday" may also trigger the distractor keyword "Sarah" from "Sarah loves Italian food." Stopword filtering and name-aware matching will be added in v1.1.

**In-context evaluation scope.** The in-context track measures cognitive *reasoning ability* over provided memories, not persistent memory *retrieval effectiveness*. These are different capabilities. A model can score 80% on the in-context track while having no persistent memory at all.

---

## 6. Baseline Results

### 6.1 TinyLlama 1.1B (Chat, FP16, no memory training)

| Axis | Accuracy | Score Contribution |
|---|---|---|
| Acquisition | 75.0% | 15.0 |
| Recall | 10.0% | 2.5 |
| Decay | 0.0% | 0.0 |
| Conflict | 0.0% | 0.0 |
| Consolidation | 10.0% | 1.5 |
| **CogMem Score** | | **19.0 / 100** |

Runtime: 262.7 seconds. Hardware: CPU (baseline configuration).

### 6.2 Interpretation

**Acquisition (75%):** The model reads and repeats facts from its prompt — basic reading comprehension working correctly. Not a memory capability in the cognitive sense; it establishes that the evaluation format is functioning.

**Recall (10%):** Near-random performance. The model selects topic-similar memories rather than goal-relevant ones. This is the expected behavior of any model without goal-directed retrieval — it is essentially selecting by keyword overlap.

**Decay (0%):** The model treats all memories as equally reliable regardless of verification recency. No temporal awareness. This is the expected behavior of a stateless LLM.

**Conflict (0%):** Cannot detect contradictions between stored facts. Would generate text by implicitly averaging conflicting facts, a primary source of hallucinations identified by HaluMem.

**Consolidation (10%):** Near-random. Does not understand that repeated verification increases trustworthiness versus single-mention facts.

### 6.3 Expected Score Ranges

| System Type | Expected CogMem Score |
|---|---|
| Standard LLM (no memory, no training) | 10 – 25 |
| LLM with RAG (vector retrieval) | 25 – 45 |
| LLM with cognitive memory training | 50 – 80 |
| Perfect cognitive memory system | 100 |

RAG-based systems are expected to score higher on acquisition (facts are reliably injected) but remain low on recall (vector similarity ≠ goal relevance), decay (no temporal model), and conflict (retrieval doesn't detect contradictions). The expected 25-45 range for RAG represents the ceiling of retrieval-augmented approaches on cognitive capabilities.

### 6.4 Validation Status

The baseline result establishes the floor (score of 19 for a standard LLM with no memory training). To fully validate that CogMemBench discriminates between systems with different memory capabilities, scores across a capability gradient are needed. The following evaluations are planned:

- GPT-3.5 / GPT-4 (frontier LLMs, no memory architecture)
- TinyLlama + RAG (same model with vector retrieval)
- Memory Fabric (Stage 3 validation, when complete)

The "validates discrimination" claim in this paper is held to the existing baseline only. Full discrimination validation will be reported in v1.1.

---

## 7. The Cross-Session Track (Forthcoming)

The in-context track measures cognitive reasoning ability over supplied memories. It cannot evaluate persistent memory systems — a model with a perfect weight-space memory and a model with no memory architecture receive identical inputs.

The cross-session track fixes this.

### 7.1 Protocol

**Storage phase.** The memory system is given a fact to store using its native storage mechanism:
- For database systems (Ember's Diaries): `protocol.remember("My API key expires September 1st")`
- For weight-space systems (Memory Fabric): `trainer.write_memory(model, namespace="personal", input_ids=...)`
- For in-context systems (RAG): fact indexed into the vector store

**Gap.** Context is cleared. No memory is accessible in the prompt.

**Retrieval phase.** The model receives only the query — no memories injected:
```
"When does my API key expire?"
```

The system must answer from its own internal state.

### 7.2 Axis Mapping for Cross-Session

| CogMemBench Axis | Cross-Session Analog |
|---|---|
| Acquisition | Fact stored in Session A retrieved correctly in Session B |
| Recall | Goal-relevant fact retrieved over topically similar stored facts |
| Decay | Facts stored longer ago retrieved with lower confidence |
| Conflict | Two stored conflicting facts produce hedged or surfaced-conflict output |
| Consolidation | Fact stored 3× retrieved more reliably than fact stored 1× |

### 7.3 Why This Track is Necessary

A system that scores 80% on the in-context track but 5% on the cross-session track is a good reasoner with broken persistent memory. A system that scores 30% on the in-context track but 70% on the cross-session track has strong persistent memory but weak cognitive reasoning. Both failure modes are invisible without both tracks.

CogMemBench v1.1 will include the cross-session track as the primary evaluation methodology for Memory Fabric, and will make it available for any persistent memory architecture.

---

## 8. Usage

### 8.1 Installation

```bash
pip install git+https://github.com/ticketguy/littlefig.git
```

### 8.2 Run the Benchmark

```python
from cogmembench import CogMemRunner

runner = CogMemRunner(per_axis=200)  # Full 1,000 cases
results = runner.run(
    model_fn=lambda prompt: your_model.generate(prompt),
)
print(f"CogMem Score: {results['cogmem_score']}/100")
print(f"Per-axis: {results['axis_scores']}")
```

### 8.3 Generate Dataset Only

```python
from cogmembench import CogMemGenerator

gen = CogMemGenerator(seed=42)
cases = gen.generate_all(per_axis=200)
gen.save_jsonl(cases, "cogmembench_v1.jsonl")
```

---

## 9. Leaderboard Submission Format

```json
{
    "model_name": "your-model-name",
    "model_size": "1.1B",
    "memory_architecture": "none | rag | weight-space | database | hybrid",
    "track": "in-context | cross-session",
    "cogmem_score": 19.0,
    "axis_scores": {
        "acquisition":    0.75,
        "recall":         0.10,
        "decay":          0.00,
        "conflict":       0.00,
        "consolidation":  0.10
    },
    "runtime_seconds": 262.7,
    "hardware": "CPU / GPU model",
    "notes": "Baseline, no memory training"
}
```

The `memory_architecture` field is required to prevent comparison between in-context and cross-session scores, which measure fundamentally different capabilities.

---

## 10. Limitations

**In-context track scope.** The current track provides memories in the prompt. It measures cognitive reasoning ability, not persistent memory retrieval. Systems are evaluated equally regardless of their memory architecture. This is by design for the in-context track, but means the benchmark cannot yet distinguish a stateless GPT-4 from a fully deployed Memory Fabric system. The cross-session track (v1.1) addresses this.

**Programmatic generation.** Test cases are generated from a curated pool of scenarios. Real-world memory interactions are more complex and contextually varied. CogMemBench tests fundamental cognitive capabilities, not production-level memory management.

**English only.** All test cases are in English. Multilingual evaluation is future work.

**Single baseline model.** Current results include only TinyLlama 1.1B. The score range estimates (Table 6.3) are projections based on architectural reasoning, not measurements. A multi-model leaderboard is the priority for v1.1.

**Keyword scorer noise.** False negatives from shared tokens between correct answers and distractors are a known issue. Scores on the recall axis may understate true model performance for models that correctly identify goal-relevant memories but happen to also mention distractor keywords in their reasoning.

---

## 11. Future Work

**v1.1 priorities:**

1. Cross-session track implementation — the most important addition. Enables evaluation of Memory Fabric, MemoryLLM, Ember's Diaries database, and any other persistent memory architecture.

2. Multi-model leaderboard — at least 5 models across a capability gradient (TinyLlama 1.1B, Llama 3 8B, GPT-3.5, GPT-4, Memory Fabric) to validate that CogMemBench discriminates between capability levels.

3. LLM-as-judge scoring for all axes — replace keyword matching with a judge model for open-ended evaluation. Eliminates false negatives from token overlap and enables more nuanced scoring.

4. Stopword filtering and name-aware matching in the keyword scorer — fixes the false-negative issue in the current implementation.

5. HaluMem alignment — evaluate models on CogMemBench and HaluMem simultaneously to characterize the relationship between cognitive reasoning capability and hallucination rate. Our hypothesis: models that fail conflict detection on CogMemBench will show higher hallucination rates on HaluMem's updating task.

---

## 12. Conclusion

CogMemBench provides axis-decomposed evaluation of cognitive memory reasoning, grounded in established cognitive psychology frameworks. It measures five distinct capabilities — acquisition, goal-directed recall, graceful decay, conflict detection, and consolidation — that collectively define the cognitive layer of a memory system.

The baseline evaluation of TinyLlama 1.1B (19.0/100) establishes the floor: standard LLMs without memory training or architecture can acquire facts from context (75%) but fail entirely on the cognitive capabilities that distinguish memory systems from keyword search (0% on decay, 0% on conflict, 10% on recall and consolidation).

The planned cross-session track will extend CogMemBench to evaluate persistent memory architectures directly, making it the appropriate evaluation methodology for Memory Fabric, Ember's Diaries, and any other system where memories must be retrieved from internal state rather than injected into context.

The industry needs what this benchmark provides: an independent, reproducible way to compare AI memory systems on the cognitive capabilities that actually matter.

---

## Citation

```bibtex
@misc{cogmembench2026,
    title     = {CogMemBench: An Axis-Decomposed Benchmark for Cognitive Memory
                 Reasoning in Large Language Models},
    author    = {0xticketguy},
    year      = {2026},
    publisher = {Harboria Labs},
    url       = {https://github.com/Harboria-Labs/littlefig/tree/main/cogmembench}
}
```

---

## References

1. Conway, M.A. "Memory and the Self." Journal of Memory and Language, 2005.
2. Ebbinghaus, H. "Über das Gedächtnis." 1885.
3. Atkinson, R.C. & Shiffrin, R.M. "Human Memory: A Proposed System." 1968.
4. HaluMem. "Evaluating Hallucinations in Memory Systems of Agents." arXiv:2511.03506, 2024.
5. Wang, Y., et al. "MEMORYLLM: Towards Self-Updatable Large Language Models." arXiv:2402.04624, 2024.
6. Wang, Y., et al. "MemoryBench: Towards Comprehensive Evaluation of Memory Mechanisms." 2025.
7. Li, Z., et al. "MemOS: A Memory OS for AI System." arXiv:2507.03724, 2025.
8. 0xticketguy (Harboria Labs). "Ember's Diaries: An Immutable Cognitive Database Engine for Grounded AI Memory." 2026.
9. 0xticketguy (Harboria Labs). "Memory Fabric: Neural Weight-Space Implementation of Ember's Diaries." 2026.
10. 0xticketguy (Harboria Labs). "Fig Engine: CPU-Native Training Infrastructure for Large Language Models." 2026.

---

*Code: https://github.com/Harboria-Labs/littlefig/tree/main/cogmembench*
*License: AGPL-3.0*
*Built by 0xticketguy / Harboria Labs*
