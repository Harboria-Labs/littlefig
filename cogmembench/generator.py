"""
CogMemBench — Test Case Generator

Generates evaluation cases for all 5 axes.
Deterministic given a seed. Each case is self-contained JSON.
"""

import json
import random
import hashlib
from typing import List, Dict
from dataclasses import dataclass, asdict


@dataclass
class TestCase:
    id: str
    axis: str  # acquisition | recall | decay | conflict | consolidation
    prompt: str  # what to send to the model
    context: dict  # memories, goals, metadata the model receives
    correct_answer: str  # ground truth
    distractor: str  # plausible wrong answer
    difficulty: str  # easy | medium | hard
    metadata: dict  # additional info for scoring


class CogMemGenerator:
    """Generates benchmark test cases for all 5 cognitive memory axes."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._facts_pool = self._build_facts()
        self._goals_pool = self._build_goals()

    def generate_all(self, per_axis: int = 200) -> List[TestCase]:
        """Generate full benchmark dataset."""
        cases = []
        cases.extend(self.gen_acquisition(per_axis))
        cases.extend(self.gen_recall(per_axis))
        cases.extend(self.gen_decay(per_axis))
        cases.extend(self.gen_conflict(per_axis))
        cases.extend(self.gen_consolidation(per_axis))
        return cases

    def save_jsonl(self, cases: List[TestCase], path: str):
        """Save cases as JSONL."""
        with open(path, "w") as f:
            for case in cases:
                f.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
        print(f"📊 CogMemBench: {len(cases)} cases saved to {path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Axis 1: ACQUISITION — learn a fact, retain it
    # ═══════════════════════════════════════════════════════════════════════════

    def gen_acquisition(self, n: int = 200) -> List[TestCase]:
        """
        Format: model is told a fact, then given N distractor turns,
        then asked about the fact. Must recall it exactly.
        """
        cases = []
        difficulties = [
            ("easy", 2),    # 2 distractor turns
            ("medium", 5),  # 5 distractor turns
            ("hard", 10),   # 10 distractor turns
        ]

        for i in range(n):
            fact = self.rng.choice(self._facts_pool)
            diff_label, n_distractors = self.rng.choice(difficulties)

            distractors = [self.rng.choice([
                "What's the weather like today?",
                "Tell me a joke.",
                "Explain quantum computing briefly.",
                "What's 15 times 23?",
                "Write a haiku about rain.",
                "What are the primary colors?",
                "How do airplanes fly?",
                "Name three countries in South America.",
                "What's the difference between RAM and ROM?",
                "Summarize the plot of Romeo and Juliet.",
            ]) for _ in range(n_distractors)]

            case_id = hashlib.md5(f"acq_{i}_{fact['statement']}".encode()).hexdigest()[:12]

            cases.append(TestCase(
                id=case_id,
                axis="acquisition",
                prompt=f"Earlier in our conversation, I told you: \"{fact['statement']}\"\n\nAfter several other topics, I'm now asking: {fact['question']}",
                context={
                    "stored_fact": fact["statement"],
                    "distractor_turns": distractors,
                    "turns_since_storage": n_distractors,
                },
                correct_answer=fact["answer"],
                distractor=fact.get("distractor", "I don't remember"),
                difficulty=diff_label,
                metadata={"category": fact["category"]},
            ))
        return cases

    # ═══════════════════════════════════════════════════════════════════════════
    # Axis 2: GOAL-DIRECTED RECALL — retrieve by usefulness, not similarity
    # ═══════════════════════════════════════════════════════════════════════════

    def gen_recall(self, n: int = 200) -> List[TestCase]:
        """
        Format: model has a set of memories + a current goal + a question.
        The correct memory is TASK-RELEVANT but not TOPIC-SIMILAR.
        The distractor is TOPIC-SIMILAR but not TASK-RELEVANT.
        """
        cases = []
        scenarios = self._build_recall_scenarios()

        for i in range(n):
            s = self.rng.choice(scenarios)
            # Add 3-5 random irrelevant memories as noise
            noise = self.rng.sample(
                [f["statement"] for f in self._facts_pool if f["statement"] != s["correct_memory"]],
                k=min(4, len(self._facts_pool) - 1)
            )
            all_memories = [s["correct_memory"], s["distractor_memory"]] + noise
            self.rng.shuffle(all_memories)

            case_id = hashlib.md5(f"recall_{i}_{s['goal']}".encode()).hexdigest()[:12]

            cases.append(TestCase(
                id=case_id,
                axis="recall",
                prompt=f"Current goal: {s['goal']}\n\nQuestion: {s['question']}\n\nAvailable memories:\n" +
                       "\n".join(f"- {m}" for m in all_memories) +
                       "\n\nWhich memory is most useful for achieving the goal?",
                context={
                    "goal": s["goal"],
                    "all_memories": all_memories,
                },
                correct_answer=s["correct_memory"],
                distractor=s["distractor_memory"],
                difficulty=s.get("difficulty", "medium"),
                metadata={"reasoning": s["reasoning"]},
            ))
        return cases

    # ═══════════════════════════════════════════════════════════════════════════
    # Axis 3: GRACEFUL DECAY — old memories less certain, recent ones strong
    # ═══════════════════════════════════════════════════════════════════════════

    def gen_decay(self, n: int = 200) -> List[TestCase]:
        """
        Format: model has memories with different ages (turns since stored).
        When asked about old vs recent memories, it should express more
        uncertainty about old ones and more confidence about recent ones.
        """
        cases = []
        for i in range(n):
            recent_fact = self.rng.choice(self._facts_pool)
            old_fact = self.rng.choice([f for f in self._facts_pool if f != recent_fact])

            recent_age = self.rng.randint(1, 5)
            old_age = self.rng.randint(50, 200)

            case_id = hashlib.md5(f"decay_{i}_{old_fact['statement']}".encode()).hexdigest()[:12]

            cases.append(TestCase(
                id=case_id,
                axis="decay",
                prompt=(
                    f"You have two memories:\n"
                    f"- (stored {recent_age} turns ago): \"{recent_fact['statement']}\"\n"
                    f"- (stored {old_age} turns ago): \"{old_fact['statement']}\"\n\n"
                    f"How confident are you about each? Which is more likely to still be accurate?"
                ),
                context={
                    "recent": {"fact": recent_fact["statement"], "age_turns": recent_age},
                    "old": {"fact": old_fact["statement"], "age_turns": old_age},
                },
                correct_answer=f"More confident about: \"{recent_fact['statement']}\" (recent). Less certain about: \"{old_fact['statement']}\" (old, may be outdated).",
                distractor="Both are equally reliable.",
                difficulty="easy" if old_age > 100 else "medium",
                metadata={"recent_age": recent_age, "old_age": old_age},
            ))
        return cases

    # ═══════════════════════════════════════════════════════════════════════════
    # Axis 4: CONFLICT DETECTION — spot contradictions
    # ═══════════════════════════════════════════════════════════════════════════

    def gen_conflict(self, n: int = 200) -> List[TestCase]:
        """
        Format: model has two memories that contradict each other.
        Must identify the contradiction and surface it.
        """
        cases = []
        conflicts = self._build_conflicts()

        for i in range(n):
            c = self.rng.choice(conflicts)
            # Add non-conflicting memories as noise
            noise = self.rng.sample(
                [f["statement"] for f in self._facts_pool],
                k=min(3, len(self._facts_pool))
            )

            case_id = hashlib.md5(f"conflict_{i}_{c['mem_a']}".encode()).hexdigest()[:12]

            cases.append(TestCase(
                id=case_id,
                axis="conflict",
                prompt=(
                    f"Your stored memories include:\n"
                    f"- \"{c['mem_a']}\"\n"
                    f"- \"{c['mem_b']}\"\n" +
                    "".join(f"- \"{m}\"\n" for m in noise) +
                    f"\nDo any of these memories contradict each other? If so, which ones and how?"
                ),
                context={
                    "conflicting_pair": [c["mem_a"], c["mem_b"]],
                    "noise_memories": noise,
                },
                correct_answer=f"Conflict: \"{c['mem_a']}\" contradicts \"{c['mem_b']}\" — {c['explanation']}",
                distractor="No conflicts detected. All memories are consistent.",
                difficulty=c.get("difficulty", "medium"),
                metadata={"conflict_type": c["type"]},
            ))
        return cases

    # ═══════════════════════════════════════════════════════════════════════════
    # Axis 5: CONSOLIDATION — repeated exposure strengthens knowledge
    # ═══════════════════════════════════════════════════════════════════════════

    def gen_consolidation(self, n: int = 200) -> List[TestCase]:
        """
        Format: model has seen a fact either 1 time or 5+ times.
        Should be more confident/accurate about the repeated one.
        """
        cases = []
        for i in range(n):
            repeated_fact = self.rng.choice(self._facts_pool)
            single_fact = self.rng.choice([f for f in self._facts_pool if f != repeated_fact])

            n_exposures = self.rng.randint(4, 8)

            case_id = hashlib.md5(f"consol_{i}_{repeated_fact['statement']}".encode()).hexdigest()[:12]

            cases.append(TestCase(
                id=case_id,
                axis="consolidation",
                prompt=(
                    f"You have two pieces of knowledge:\n"
                    f"- \"{repeated_fact['statement']}\" (confirmed {n_exposures} times across different conversations)\n"
                    f"- \"{single_fact['statement']}\" (mentioned once)\n\n"
                    f"Which do you trust more? Why?"
                ),
                context={
                    "repeated": {"fact": repeated_fact["statement"], "exposures": n_exposures},
                    "single": {"fact": single_fact["statement"], "exposures": 1},
                },
                correct_answer=f"Higher confidence in: \"{repeated_fact['statement']}\" — confirmed {n_exposures} times, making it well-established knowledge.",
                distractor="Both are equally trustworthy since they're both stored.",
                difficulty="easy" if n_exposures >= 6 else "medium",
                metadata={"n_exposures": n_exposures},
            ))
        return cases

    # ═══════════════════════════════════════════════════════════════════════════
    # Data pools
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_facts(self) -> List[Dict]:
        return [
            {"statement": "My daughter's birthday is June 12th", "question": "When is my daughter's birthday?", "answer": "June 12th", "category": "personal", "distractor": "I'm not sure about the date"},
            {"statement": "I'm allergic to peanuts and shellfish", "question": "What are my food allergies?", "answer": "Peanuts and shellfish", "category": "health", "distractor": "I don't recall any allergies"},
            {"statement": "The project deadline is March 15 for the API migration", "question": "When is the API migration due?", "answer": "March 15", "category": "work", "distractor": "Sometime in March"},
            {"statement": "My wife's name is Sarah and our anniversary is October 3rd", "question": "When is my wedding anniversary?", "answer": "October 3rd", "category": "personal", "distractor": "I think it's in the fall"},
            {"statement": "I prefer Python over JavaScript for backend work", "question": "What language do I prefer for backend?", "answer": "Python", "category": "preferences", "distractor": "JavaScript"},
            {"statement": "Team standup is every day at 9:15am", "question": "What time is the daily standup?", "answer": "9:15am", "category": "schedule", "distractor": "9:00am"},
            {"statement": "My budget for personal expenses is $500 per month", "question": "What's my monthly personal budget?", "answer": "$500 per month", "category": "finance", "distractor": "Around $400-600"},
            {"statement": "I started learning Rust in January 2026", "question": "When did I start learning Rust?", "answer": "January 2026", "category": "learning", "distractor": "Sometime last year"},
            {"statement": "My home address is 42 Oak Street, Portland", "question": "What's my home address?", "answer": "42 Oak Street, Portland", "category": "personal", "distractor": "Somewhere in Portland"},
            {"statement": "I have a Tesla Model 3, blue, 2024", "question": "What car do I drive?", "answer": "Tesla Model 3, blue, 2024", "category": "personal", "distractor": "A Tesla"},
            {"statement": "My favorite restaurant is Osteria Marco on 16th Street", "question": "What's my favorite restaurant?", "answer": "Osteria Marco on 16th Street", "category": "preferences", "distractor": "An Italian place downtown"},
            {"statement": "My son Jake plays soccer on Saturdays at 10am", "question": "When does Jake play soccer?", "answer": "Saturdays at 10am", "category": "schedule", "distractor": "On weekends"},
            {"statement": "I'm training for a half marathon in November", "question": "What am I training for?", "answer": "A half marathon in November", "category": "health", "distractor": "Some kind of race"},
            {"statement": "My work laptop password was last changed on April 1st", "question": "When did I last change my work password?", "answer": "April 1st", "category": "work", "distractor": "Recently"},
            {"statement": "I take 10mg of lisinopril daily for blood pressure", "question": "What medication do I take?", "answer": "10mg of lisinopril daily", "category": "health", "distractor": "Blood pressure medication"},
        ]

    def _build_goals(self) -> List[str]:
        return [
            "Plan a surprise birthday party",
            "Book a restaurant for anniversary dinner",
            "Prepare for an overseas work trip",
            "Schedule a team lunch on Tuesday",
            "Choose a gift for wife under budget",
            "Decide whether to take on extra project",
            "Plan family vacation dates",
            "Prepare healthy meal for guest with dietary restrictions",
            "Optimize weekly schedule for more free time",
            "Find a running partner for training",
        ]

    def _build_recall_scenarios(self) -> List[Dict]:
        return [
            {"goal": "Plan a surprise birthday party for my daughter", "question": "I need to plan an event next month", "correct_memory": "My daughter's birthday is June 12th", "distractor_memory": "Team standup is every day at 9:15am", "reasoning": "The birthday date determines party timing", "difficulty": "medium"},
            {"goal": "Order catering for office party safely", "question": "What should I keep in mind for the food order?", "correct_memory": "I'm allergic to peanuts and shellfish", "distractor_memory": "My favorite restaurant is Osteria Marco on 16th Street", "reasoning": "Allergy is a safety issue that must constrain food choices", "difficulty": "medium"},
            {"goal": "Book a quiet anniversary dinner", "question": "Help me find a good restaurant", "correct_memory": "My wife's name is Sarah and our anniversary is October 3rd", "distractor_memory": "My favorite restaurant is Osteria Marco on 16th Street", "reasoning": "Need the date to make a reservation + wife's preferences matter", "difficulty": "hard"},
            {"goal": "Decide if I can take on a freelance project this month", "question": "Do I have capacity for more work?", "correct_memory": "The project deadline is March 15 for the API migration", "distractor_memory": "I started learning Rust in January 2026", "reasoning": "Existing deadline constrains available capacity", "difficulty": "medium"},
            {"goal": "Plan family vacation that works for everyone", "question": "When should we go?", "correct_memory": "My son Jake plays soccer on Saturdays at 10am", "distractor_memory": "I have a Tesla Model 3, blue, 2024", "reasoning": "Soccer schedule constrains travel dates", "difficulty": "hard"},
            {"goal": "Prepare a meal for a guest with unknown dietary needs", "question": "What should I cook?", "correct_memory": "I'm allergic to peanuts and shellfish", "distractor_memory": "My favorite restaurant is Osteria Marco on 16th Street", "reasoning": "Must avoid allergens when cooking for others — safety first", "difficulty": "easy"},
            {"goal": "Choose a birthday gift within budget", "question": "What can I afford?", "correct_memory": "My budget for personal expenses is $500 per month", "distractor_memory": "My wife's name is Sarah and our anniversary is October 3rd", "reasoning": "Budget determines affordable gift range", "difficulty": "easy"},
            {"goal": "Plan a running route for Saturday morning", "question": "What time should I go?", "correct_memory": "My son Jake plays soccer on Saturdays at 10am", "distractor_memory": "I'm training for a half marathon in November", "reasoning": "Must finish run before soccer at 10am", "difficulty": "hard"},
        ]

    def _build_conflicts(self) -> List[Dict]:
        return [
            {"mem_a": "Team meeting is every Tuesday at 2pm", "mem_b": "Team meeting was moved to Wednesday at 3pm", "explanation": "Meeting time changed — both can't be current", "type": "temporal_update", "difficulty": "easy"},
            {"mem_a": "I work at Google as a senior engineer", "mem_b": "Just started a new role at Meta last week", "explanation": "Employer changed — can't work at both simultaneously", "type": "value_change", "difficulty": "easy"},
            {"mem_a": "Budget is $500/month for personal expenses", "mem_b": "Increased personal budget to $750/month starting this quarter", "explanation": "Budget amount changed — newer supersedes older", "type": "value_change", "difficulty": "medium"},
            {"mem_a": "Daughter's school pickup is at 3:15pm", "mem_b": "School switched to early dismissal at 1:30pm on Fridays", "explanation": "Partial conflict — Friday schedule differs from other days", "type": "partial_conflict", "difficulty": "hard"},
            {"mem_a": "User prefers Python for all backend work", "mem_b": "User said they've switched to Go for new microservices", "explanation": "Preference evolved — Python was general, Go is for specific new work", "type": "evolution", "difficulty": "hard"},
            {"mem_a": "Favorite restaurant is Osteria Marco", "mem_b": "Osteria Marco closed permanently last month", "explanation": "Factual change — the restaurant no longer exists", "type": "external_change", "difficulty": "medium"},
            {"mem_a": "Takes 10mg lisinopril daily", "mem_b": "Doctor increased dosage to 20mg last visit", "explanation": "Medical update — dosage changed", "type": "value_change", "difficulty": "easy"},
            {"mem_a": "Training for half marathon in November", "mem_b": "Had to stop training due to knee injury in September", "explanation": "Plan invalidated by new event", "type": "plan_invalidation", "difficulty": "medium"},
        ]
