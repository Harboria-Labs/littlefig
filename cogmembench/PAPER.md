# CogMemBench: A Benchmark for Continuous Cognitive Memory in Large Language Models

**Authors:** 0xticketguy (Harboria Labs)
**Version:** 1.0
**Dataset:** 1,000 evaluation cases across 5 cognitive axes
**License:** CC-BY-4.0

---

## Abstract

We present CogMemBench, the first benchmark designed to evaluate whether large language models can function as cognitive memory systems — not merely recalling stored text, but demonstrating goal-directed retrieval, temporal awareness, conflict detection, and knowledge consolidation. Current LLM benchmarks (MMLU, HumanEval, etc.) evaluate static knowledge. CogMemBench evaluates dynamic knowledge management — the cognitive layer that every AI agent needs but nobody can currently measure.

We evaluate TinyLlama 1.1B as a baseline and find a CogMem Score of 19.0/100, demonstrating that standard LLMs perform well on basic acquisition (75%) but fail completely on goal-directed recall (10%), temporal decay awareness (0%), conflict detection (0%), and consolidation reasoning (10%). These results validate that CogMemBench discriminates between models with and without cognitive memory capabilities.

---

## 1. Motivation

Every major AI company shipped "memory" features in 2025-2026:
- OpenAI: ChatGPT Memory
- Google: Gemini Memory
- Anthropic: Claude Projects

Yet there is no independent, reproducible way to compare these implementations. Nobody knows which one actually works. The industry lacks a standard benchmark for AI memory quality.

CogMemBench fills this gap by evaluating five fundamental cognitive memory capabilities grounded in established psychology:

| Axis | Measures | Grounded In |
|------|----------|-------------|
| Acquisition | Can the model learn and retain a new fact? | Basic memory encoding |
| Goal-directed Recall | Does it retrieve by task-relevance or topic-similarity? | Conway's Self-Memory System (2005) |
| Graceful Decay | Does unused knowledge become less certain? | Ebbinghaus Forgetting Curve (1885) |
| Conflict Detection | Can it identify contradictions between stored facts? | HaluMem (2024) findings |
| Consolidation | Does repeated exposure strengthen knowledge? | Atkinson-Shiffrin Model (1968) |

---

## 2. Dataset Description

### Format

Each test case is a JSON object:

```json
{
    "id": "abc123",
    "axis": "recall",
    "prompt": "Current goal: Plan a dinner for my wife's birthday...",
    "context": {"goal": "...", "memories": [...]},
    "correct_answer": "Wife's birthday is June 12th",
    "distractor": "Loves Italian food",
    "difficulty": "medium",
    "metadata": {"reasoning": "Birthday date needed for timing"}
}
```

### Statistics

| Property | Value |
|----------|-------|
| Total cases | 1,000 |
| Cases per axis | 200 |
| Difficulty distribution | 33% easy, 34% medium, 33% hard |
| Average prompt length | ~150 tokens |
| Deterministic (seed=42) | Yes |
| File format | JSONL |
| File size | ~1.2 MB |

### Data Generation

All cases are programmatically generated from a curated pool of:
- 15 personal facts (with question/answer pairs)
- 10 goals (with task contexts)
- 8 goal-conditioned recall scenarios
- 8 conflict scenarios (with type and resolution)

The generator is deterministic — same seed produces identical dataset.

---

## 3. Scoring

### Per-Axis Scoring

Each axis uses task-specific evaluation:

- **Acquisition:** Fuzzy keyword match (≥70% of answer keywords present = correct)
- **Recall:** Correct memory mentioned AND distractor not selected
- **Decay:** Model expresses differential confidence (recent > old)
- **Conflict:** Conflicting pair identified + conflict language used
- **Consolidation:** Model trusts repeated fact more than single-mention fact

### CogMem Score (0-100)

Weighted average of per-axis accuracy:

```
CogMem Score = 20% × Acquisition + 25% × Recall + 20% × Decay + 20% × Conflict + 15% × Consolidation
```

Recall gets the highest weight because goal-directed retrieval is the most discriminating capability and the most important for real-world AI agents.

---

## 4. Baseline Results

### TinyLlama 1.1B (Chat, FP16, no memory training)

| Axis | Accuracy | Score Contribution |
|------|:--------:|:------------------:|
| Acquisition | 75.0% | 15.0 |
| Recall | 10.0% | 2.5 |
| Decay | 0.0% | 0.0 |
| Conflict | 0.0% | 0.0 |
| Consolidation | 10.0% | 1.5 |
| **CogMem Score** | | **19.0/100** |

### Interpretation

- **Acquisition (75%):** The model can read and repeat facts from its prompt — basic reading comprehension. Not a memory capability.
- **Recall (10%):** Random performance. The model picks topic-similar memories, not goal-relevant ones. No cognitive retrieval.
- **Decay (0%):** Complete failure. The model treats all memories as equally reliable regardless of age. No temporal awareness.
- **Conflict (0%):** Cannot detect contradictions. Would hallucinate by averaging conflicting facts.
- **Consolidation (10%):** Nearly random. Doesn't understand that repeated verification increases trustworthiness.

### What These Results Mean

A score of 19/100 means TinyLlama has **no cognitive memory capabilities** beyond basic reading comprehension. It can parrot facts but cannot reason about them cognitively. This establishes the baseline that memory-enhanced models must beat.

Expected ranges:
- Standard LLM (no memory): 10-25/100
- LLM with RAG: 25-45/100 (better recall, still no decay/conflict)
- LLM with cognitive memory training: 50-80/100 (target)
- Perfect cognitive memory system: 100/100

---

## 5. Usage

### Installation

```bash
pip install git+https://github.com/ticketguy/littlefig.git
```

### Run Benchmark

```python
from cogmembench import CogMemRunner

runner = CogMemRunner(per_axis=200)  # Full 1000 cases
results = runner.run(
    model_fn=lambda prompt: your_model.generate(prompt),
)
print(f"CogMem Score: {results['cogmem_score']}/100")
```

### Generate Dataset Only

```python
from cogmembench import CogMemGenerator

gen = CogMemGenerator(seed=42)
cases = gen.generate_all(per_axis=200)
gen.save_jsonl(cases, "cogmembench_v1.jsonl")
```

---

## 6. Leaderboard Submission Format

Models are evaluated by running the benchmark and reporting:

```json
{
    "model_name": "your-model-name",
    "model_size": "1.1B",
    "cogmem_score": 19.0,
    "axis_scores": {
        "acquisition": 0.75,
        "recall": 0.10,
        "decay": 0.00,
        "conflict": 0.00,
        "consolidation": 0.10
    },
    "runtime_seconds": 262.7,
    "notes": "Baseline, no memory training"
}
```

---

## 7. Limitations

1. **Evaluation is text-match based** — a model could game the scoring by including keywords without genuine reasoning. Future versions will use LLM-as-judge for open-ended evaluation.

2. **Test cases are programmatically generated** — real-world memory scenarios are more complex. The benchmark tests fundamental capabilities, not production-level memory management.

3. **English only** — all test cases are in English. Multilingual cognitive memory evaluation is future work.

4. **Small model baseline only** — we've only tested TinyLlama 1.1B. Larger models (7B+, GPT-4, Claude) will likely score higher on acquisition and possibly recall, but may still fail on decay/conflict/consolidation.

---

## 8. Citation

```bibtex
@misc{cogmembench2026,
    title={CogMemBench: A Benchmark for Continuous Cognitive Memory in Large Language Models},
    author={0xticketguy},
    year={2026},
    publisher={Harboria Labs},
    url={https://github.com/ticketguy/littlefig/tree/main/cogmembench}
}
```

---

## References

1. Conway, M.A. (2005). "Memory and the Self." Journal of Memory and Language.
2. Ebbinghaus, H. (1885). "Über das Gedächtnis."
3. Atkinson, R.C. & Shiffrin, R.M. (1968). "Human Memory: A Proposed System."
4. HaluMem (2024). "Evaluating Hallucinations in Memory Systems of Agents." arXiv:2511.03506.
5. Wang, Y., et al. (2024). "MEMORYLLM: Towards Self-Updatable LLMs." arXiv:2402.04624.

---

*Built by 0xticketguy / Harboria Labs*
*Code: https://github.com/ticketguy/littlefig/tree/main/cogmembench*
