---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - cognitive-memory
  - benchmark
  - llm-evaluation
  - memory-systems
size_categories:
  - 1K<n<10K
---

# CogMemBench v1.0

5-axis benchmark for evaluating cognitive memory in LLMs.

## Axes
1. **Acquisition** (200 cases): Learn a fact, retain it
2. **Goal-directed Recall** (200 cases): Retrieve by task-relevance
3. **Graceful Decay** (200 cases): Old = less certain
4. **Conflict Detection** (200 cases): Spot contradictions
5. **Consolidation** (200 cases): Repeated = stronger

## Scoring
CogMem Score (0-100): weighted average across axes.

## Baseline
TinyLlama 1.1B: 19.0/100 (no memory training)

## Usage
```python
from cogmembench import CogMemRunner
results = CogMemRunner().run(model_fn=your_model_fn)
```
