"""
CogMemBench — Scoring System

Evaluates model responses against ground truth.
Produces per-axis accuracy + weighted CogMem Score (0-100).
"""

import re
from typing import List, Dict, Tuple
from .generator import TestCase


class CogMemScorer:
    """
    Scores model responses on CogMemBench test cases.
    
    Scoring per axis:
    - Acquisition: exact or fuzzy match of recalled fact
    - Recall: did model pick the correct memory?
    - Decay: did model express more confidence in recent vs old?
    - Conflict: did model identify the conflicting pair?
    - Consolidation: did model trust the repeated fact more?
    
    Final CogMem Score: weighted average (0-100).
    """

    AXIS_WEIGHTS = {
        "acquisition": 0.20,
        "recall": 0.25,
        "decay": 0.20,
        "conflict": 0.20,
        "consolidation": 0.15,
    }

    def score_response(self, case: TestCase, model_response: str) -> Tuple[bool, float, str]:
        """
        Score a single response.
        Returns: (correct: bool, score: float 0-1, explanation: str)
        """
        response_lower = model_response.lower().strip()
        
        if case.axis == "acquisition":
            return self._score_acquisition(case, response_lower)
        elif case.axis == "recall":
            return self._score_recall(case, response_lower)
        elif case.axis == "decay":
            return self._score_decay(case, response_lower)
        elif case.axis == "conflict":
            return self._score_conflict(case, response_lower)
        elif case.axis == "consolidation":
            return self._score_consolidation(case, response_lower)
        else:
            return (False, 0.0, f"Unknown axis: {case.axis}")

    def score_batch(self, cases: List[TestCase], responses: List[str]) -> Dict:
        """
        Score a full batch of responses.
        Returns per-axis accuracy + overall CogMem Score.
        """
        axis_correct = {ax: 0 for ax in self.AXIS_WEIGHTS}
        axis_total = {ax: 0 for ax in self.AXIS_WEIGHTS}
        details = []

        for case, response in zip(cases, responses):
            correct, score, explanation = self.score_response(case, response)
            axis_correct[case.axis] += int(correct)
            axis_total[case.axis] += 1
            details.append({
                "id": case.id,
                "axis": case.axis,
                "correct": correct,
                "score": score,
                "explanation": explanation,
            })

        # Per-axis accuracy
        axis_accuracy = {}
        for ax in self.AXIS_WEIGHTS:
            if axis_total[ax] > 0:
                axis_accuracy[ax] = axis_correct[ax] / axis_total[ax]
            else:
                axis_accuracy[ax] = 0.0

        # Weighted CogMem Score (0-100)
        cogmem_score = sum(
            axis_accuracy[ax] * weight * 100
            for ax, weight in self.AXIS_WEIGHTS.items()
        )

        return {
            "cogmem_score": round(cogmem_score, 1),
            "axis_accuracy": {ax: round(acc, 4) for ax, acc in axis_accuracy.items()},
            "axis_counts": axis_total,
            "total_correct": sum(axis_correct.values()),
            "total_cases": sum(axis_total.values()),
            "details": details,
        }

    # ── Per-axis scoring logic ────────────────────────────────────────────────

    def _score_acquisition(self, case: TestCase, response: str) -> Tuple[bool, float, str]:
        """Check if the model recalled the fact correctly."""
        answer = case.correct_answer.lower()
        # Exact substring match
        if answer in response:
            return (True, 1.0, "Exact recall")
        # Fuzzy: check key words
        key_words = [w for w in answer.split() if len(w) > 3]
        matches = sum(1 for w in key_words if w in response)
        if key_words and matches / len(key_words) >= 0.7:
            return (True, 0.8, f"Fuzzy match ({matches}/{len(key_words)} keywords)")
        # Check if distractor was chosen instead
        if case.distractor.lower() in response:
            return (False, 0.0, "Chose distractor / said doesn't remember")
        return (False, 0.2, "Partial or no recall")

    def _score_recall(self, case: TestCase, response: str) -> Tuple[bool, float, str]:
        """Check if model picked the goal-relevant memory."""
        correct = case.correct_answer.lower()
        distractor = case.distractor.lower()
        
        # Key phrases from correct answer
        correct_phrases = [p.strip() for p in correct.split(",") if len(p.strip()) > 5]
        distractor_phrases = [p.strip() for p in distractor.split(",") if len(p.strip()) > 5]
        
        correct_mentioned = any(p in response for p in correct_phrases) or correct[:30] in response
        distractor_mentioned = any(p in response for p in distractor_phrases) or distractor[:30] in response
        
        if correct_mentioned and not distractor_mentioned:
            return (True, 1.0, "Selected goal-relevant memory")
        elif correct_mentioned and distractor_mentioned:
            # Both mentioned — check which is presented as the answer
            correct_pos = response.find(correct_phrases[0]) if correct_phrases else 999
            distractor_pos = response.find(distractor_phrases[0]) if distractor_phrases else 999
            if correct_pos < distractor_pos:
                return (True, 0.7, "Correct memory mentioned first")
            return (False, 0.3, "Both mentioned but distractor emphasized")
        elif distractor_mentioned:
            return (False, 0.0, "Selected topic-similar distractor")
        return (False, 0.1, "Neither memory clearly selected")

    def _score_decay(self, case: TestCase, response: str) -> Tuple[bool, float, str]:
        """Check if model expressed appropriate uncertainty about old memory."""
        # Look for signals of differential confidence
        recent_fact = case.context["recent"]["fact"].lower()[:20]
        old_fact = case.context["old"]["fact"].lower()[:20]
        
        # Positive signals: "more confident about recent", "less sure about old", "may be outdated"
        uncertainty_words = ["uncertain", "less confident", "may be outdated", "not sure", 
                           "less certain", "might have changed", "could be wrong"]
        confidence_words = ["more confident", "more certain", "reliable", "trust", "sure about"]
        
        has_uncertainty_about_old = any(w in response for w in uncertainty_words)
        has_confidence_about_recent = any(w in response for w in confidence_words)
        
        # Check for "equally reliable" type distractor responses
        equal_signals = ["equally", "both reliable", "same confidence", "no difference"]
        chose_equal = any(w in response for w in equal_signals)
        
        if chose_equal:
            return (False, 0.0, "Claimed equal confidence (wrong)")
        if has_uncertainty_about_old and has_confidence_about_recent:
            return (True, 1.0, "Correctly expressed differential confidence")
        if has_uncertainty_about_old or has_confidence_about_recent:
            return (True, 0.7, "Partial differential confidence")
        return (False, 0.2, "No clear confidence differentiation")

    def _score_conflict(self, case: TestCase, response: str) -> Tuple[bool, float, str]:
        """Check if model detected the contradiction."""
        pair = case.context["conflicting_pair"]
        mem_a = pair[0].lower()[:25]
        mem_b = pair[1].lower()[:25]
        
        # Must mention both conflicting memories
        mentions_a = mem_a in response or any(w in response for w in mem_a.split()[:3] if len(w) > 4)
        mentions_b = mem_b in response or any(w in response for w in mem_b.split()[:3] if len(w) > 4)
        
        # Conflict indicators
        conflict_words = ["contradict", "conflict", "inconsisten", "changed", "differs",
                         "can't both", "updated", "supersede", "no longer", "was moved"]
        detected_conflict = any(w in response for w in conflict_words)
        
        # "No conflicts" is the distractor
        no_conflict_signals = ["no conflict", "all consistent", "no contradiction"]
        missed = any(w in response for w in no_conflict_signals)
        
        if missed:
            return (False, 0.0, "Said no conflicts (missed)")
        if detected_conflict and (mentions_a or mentions_b):
            return (True, 1.0, "Detected conflict and identified memories")
        if detected_conflict:
            return (True, 0.7, "Detected conflict but didn't specify pair")
        return (False, 0.2, "No clear conflict detection")

    def _score_consolidation(self, case: TestCase, response: str) -> Tuple[bool, float, str]:
        """Check if model trusts the repeated fact more."""
        repeated = case.context["repeated"]["fact"].lower()[:25]
        single = case.context["single"]["fact"].lower()[:25]
        
        # Signals that repeated is trusted more
        trust_repeated = ["confirmed", "well-established", "more confident", "trust more",
                         "multiple times", "repeatedly", "stronger"]
        trust_equal = ["equally", "both", "same level", "no difference"]
        
        trusts_repeated = any(w in response for w in trust_repeated)
        trusts_equal = any(w in response for w in trust_equal)
        
        if trusts_equal:
            return (False, 0.0, "Claimed equal trust (wrong)")
        if trusts_repeated:
            return (True, 1.0, "Correctly trusts repeated fact more")
        return (False, 0.3, "No clear preference expressed")
