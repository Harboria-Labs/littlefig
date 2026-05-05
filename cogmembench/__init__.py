"""
CogMemBench — Continuous Cognitive Memory Benchmark for LLMs

5-axis evaluation of whether LLMs can function as cognitive memory systems:
  1. Acquisition: Learn a fact, retain it across distraction
  2. Goal-directed recall: Retrieve by task-relevance, not topic-similarity
  3. Graceful decay: Older unused memories should be less certain
  4. Conflict detection: Surface contradictions, don't hallucinate
  5. Consolidation: Repeated exposure should strengthen knowledge

Original work by 0xticketguy / Harboria Labs.
"""

__version__ = "0.1.0"

from .generator import CogMemGenerator
from .scorer import CogMemScorer
from .runner import CogMemRunner
