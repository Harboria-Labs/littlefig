"""
CogMemBench — Runner

Runs the benchmark against any model that can generate text.
Supports: Little Fig models, HuggingFace models, API models (via callable).
"""

import json
import time
from typing import List, Dict, Callable, Optional
from .generator import CogMemGenerator, TestCase
from .scorer import CogMemScorer


class CogMemRunner:
    """
    Runs CogMemBench end-to-end.
    
    Usage:
        runner = CogMemRunner()
        
        # With a local model:
        results = runner.run(model_fn=lambda prompt: model.generate(prompt))
        
        # Or generate dataset only:
        runner.generate_dataset("cogmembench_v1.jsonl", per_axis=200)
    """

    def __init__(self, seed: int = 42, per_axis: int = 200):
        self.generator = CogMemGenerator(seed=seed)
        self.scorer = CogMemScorer()
        self.per_axis = per_axis

    def generate_dataset(self, output_path: str, per_axis: Optional[int] = None):
        """Generate and save the benchmark dataset."""
        n = per_axis or self.per_axis
        cases = self.generator.generate_all(per_axis=n)
        self.generator.save_jsonl(cases, output_path)
        return cases

    def run(
        self,
        model_fn: Callable[[str], str],
        per_axis: Optional[int] = None,
        max_cases: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full benchmark.
        
        Args:
            model_fn: function that takes a prompt string and returns generated text
            per_axis: cases per axis (default: self.per_axis)
            max_cases: cap total cases (for testing)
            verbose: print progress
            
        Returns:
            Full results dict with CogMem Score and per-axis breakdown.
        """
        n = per_axis or self.per_axis
        cases = self.generator.generate_all(per_axis=n)
        
        if max_cases:
            cases = cases[:max_cases]
        
        if verbose:
            print(f"🧠 CogMemBench: Running {len(cases)} cases across 5 axes...")
        
        responses = []
        t0 = time.time()
        
        for i, case in enumerate(cases):
            try:
                response = model_fn(case.prompt)
            except Exception as e:
                response = f"ERROR: {e}"
            responses.append(response)
            
            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"   {i+1}/{len(cases)} ({rate:.1f} cases/sec)")
        
        total_time = time.time() - t0
        
        # Score
        results = self.scorer.score_batch(cases, responses)
        results["runtime_seconds"] = round(total_time, 1)
        results["cases_per_second"] = round(len(cases) / total_time, 2)
        
        if verbose:
            self._print_results(results)
        
        return results

    def run_on_cases(
        self,
        cases: List[TestCase],
        model_fn: Callable[[str], str],
        verbose: bool = True,
    ) -> Dict:
        """Run on pre-loaded cases (e.g., from JSONL file)."""
        responses = []
        for case in cases:
            try:
                responses.append(model_fn(case.prompt))
            except Exception as e:
                responses.append(f"ERROR: {e}")
        
        results = self.scorer.score_batch(cases, responses)
        if verbose:
            self._print_results(results)
        return results

    def _print_results(self, results: Dict):
        """Pretty-print results."""
        print(f"\n{'='*50}")
        print(f"  🧠 CogMemBench Results")
        print(f"{'='*50}")
        print(f"\n  CogMem Score: {results['cogmem_score']}/100")
        print(f"\n  Per-axis accuracy:")
        for ax, acc in results["axis_accuracy"].items():
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"    {ax:>15}: {bar} {acc*100:.1f}%")
        print(f"\n  Total: {results['total_correct']}/{results['total_cases']} correct")
        print(f"  Runtime: {results['runtime_seconds']}s")
        print(f"{'='*50}")


def load_cases_from_jsonl(path: str) -> List[TestCase]:
    """Load test cases from a JSONL file."""
    cases = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            cases.append(TestCase(**data))
    return cases
