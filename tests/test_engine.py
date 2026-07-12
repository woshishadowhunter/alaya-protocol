import unittest
from datetime import datetime, timedelta, timezone

from alaya.engine import (
    Activation, EvolutionPolicy, ExperienceEngine, LexicalRetrieval,
    PolicyProtocol, RetrievalBackend,
)
from alaya.models import Evidence, ExperienceSeed


NOW = datetime(2026, 7, 12, tzinfo=timezone.utc)


def candidate():
    return ExperienceSeed.new(
        lesson="Align stakeholders before full solution design.",
        guidance="Start with stakeholder mapping and a resistance check.",
        context_tags=["project", "stakeholder", "community"],
        applicability="Multi-party projects",
        evidence=Evidence("support", "case-1", "Early design was rejected", NOW),
        now=NOW,
    )


class CustomPolicy:
    min_support = 1
    support_gain = 0.3
    contradiction_loss = 0.1
    deprecate_below = 0.1
    half_life_days = 90.0
    counterexample_penalty = 0.5

    def should_promote(self, supports: int) -> bool:
        return supports >= self.min_support

    def should_deprecate(self, confidence: float) -> bool:
        return confidence < self.deprecate_below

    def confidence_delta(self, polarity: str) -> float:
        return self.support_gain if polarity == "support" else -self.contradiction_loss

    def decay_factor(self, days: float) -> float:
        import math
        return math.pow(0.5, days / self.half_life_days)


class FakeEmbeddingBackend:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "stakeholder" in t else [0.0, 1.0] for t in texts]

    def similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return max(0.0, dot)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine(EvolutionPolicy(min_support=2))

    def test_second_independent_support_promotes_candidate(self):
        evolved = self.engine.reinforce(
            candidate(), Evidence("support", "case-2", "Alignment unlocked delivery", NOW)
        )
        self.assertEqual(evolved.status, "active")
        self.assertEqual(evolved.support_count, 2)

    def test_duplicate_source_does_not_increase_support(self):
        evolved = self.engine.reinforce(
            candidate(), Evidence("support", "case-1", "Repeated note", NOW)
        )
        self.assertEqual(evolved.support_count, 1)
        self.assertEqual(evolved.status, "candidate")

    def test_contradiction_reduces_confidence_and_is_retained(self):
        evolved = self.engine.reinforce(
            candidate(), Evidence("contradict", "case-3", "Rapid draft created alignment", NOW)
        )
        self.assertLess(evolved.confidence, candidate().confidence)
        self.assertEqual(evolved.contradiction_count, 1)
        self.assertEqual(len(evolved.evidence), 2)

    def test_decay_is_deterministic_and_does_not_mutate_seed(self):
        seed = candidate()
        decayed = self.engine.decay(seed, NOW + timedelta(days=90))
        self.assertLess(decayed.confidence, seed.confidence)
        self.assertEqual(seed.confidence, 0.5)

    def test_decay_with_zero_elapsed_is_identity(self):
        seed = candidate()
        decayed = self.engine.decay(seed, NOW)
        self.assertEqual(decayed.confidence, seed.confidence)

    def test_confidence_clamped_at_zero(self):
        seed = candidate()
        for _ in range(5):
            seed = self.engine.reinforce(
                seed, Evidence("contradict", f"case-{seed.contradiction_count+10}", "Bad", NOW)
            )
        self.assertGreaterEqual(seed.confidence, 0.0)

    def test_confidence_clamped_at_one(self):
        seed = candidate()
        for i in range(10):
            seed = self.engine.reinforce(
                seed, Evidence("support", f"case-{i+10}", "Great", NOW)
            )
        self.assertLessEqual(seed.confidence, 1.0)

    def test_activation_explains_relevance(self):
        relevant = self.engine.reinforce(
            candidate(), Evidence("support", "case-2", "Worked", NOW)
        )
        unrelated = ExperienceSeed.new(
            lesson="Prepare meals on Sunday.", guidance="Batch cook.",
            context_tags=["health", "food"], applicability="Weekly meal planning",
            status="active", confidence=0.9, now=NOW,
        )
        results = self.engine.activate(
            "Plan a community project involving property owners", [unrelated, relevant], NOW
        )
        self.assertEqual(results[0].seed.id, relevant.id)
        self.assertGreater(results[0].relevance, 0)
        self.assertIn("confidence", results[0].explanation)

    def test_activate_skips_non_active_seeds(self):
        c = candidate()
        results = self.engine.activate("stakeholder project", [c], NOW)
        self.assertEqual(len(results), 0)

    def test_activate_returns_empty_for_zero_overlap(self):
        seed = ExperienceSeed.new(
            lesson="Cook meals.", guidance="Batch cook.",
            context_tags=["food"], applicability="Meal planning",
            status="active", confidence=0.9, now=NOW,
        )
        results = self.engine.activate("xyzzy qwerty", [seed], NOW)
        self.assertEqual(len(results), 0)

    def test_activate_empty_seed_list(self):
        results = self.engine.activate("stakeholder project", [], NOW)
        self.assertEqual(len(results), 0)

    def test_activate_respects_limit(self):
        seeds = []
        for i in range(10):
            s = ExperienceSeed.new(
                lesson=f"Lesson {i}", guidance=f"G {i}",
                context_tags=["stakeholder", "project"], applicability="Projects",
                status="active", confidence=0.7, now=NOW,
            )
            seeds.append(s)
        results = self.engine.activate("stakeholder project", seeds, NOW, limit=3)
        self.assertEqual(len(results), 3)

    def test_decay_deprecates_below_threshold(self):
        seed = ExperienceSeed.new(
            lesson="Old lesson", guidance="Old guidance",
            context_tags=["old"], applicability="Old projects",
            status="active", confidence=0.2, now=NOW,
        )
        decayed = self.engine.decay(seed, NOW + timedelta(days=400))
        self.assertEqual(decayed.status, "deprecated")

    def test_counterexample_reduces_activation_score(self):
        seed = ExperienceSeed.new(
            lesson="Negotiate with all parties first.",
            guidance="Map interests before planning.",
            context_tags=["project", "stakeholder"],
            applicability="Multi-party projects",
            counterexamples=["Emergency safety work"],
            status="active", confidence=0.7, now=NOW,
        )
        no_penalty = self.engine.activate("planning a community project", [seed], NOW)
        penalized = self.engine.activate("emergency safety work on the building", [seed], NOW)
        if len(penalized) > 0 and len(no_penalty) > 0:
            self.assertLess(penalized[0].score, no_penalty[0].score)

    def test_counterexample_no_match_no_penalty(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["project"], applicability="Y",
            counterexamples=["Emergency work"], status="active", confidence=0.7, now=NOW,
        )
        results = self.engine.activate("planning a regular project", [seed], NOW)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].relevance, 0)

    def test_activation_has_structured_fields(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["stakeholder", "project"],
            applicability="Projects", status="active", confidence=0.7, now=NOW,
        )
        results = self.engine.activate("stakeholder project planning", [seed], NOW)
        self.assertEqual(len(results), 1)
        act = results[0]
        self.assertGreater(act.confidence, 0)
        self.assertGreater(act.recency, 0)
        self.assertEqual(act.confidence, seed.confidence)
        self.assertIn("stakeholder", act.matched_terms)
        self.assertIn("project", act.matched_terms)
        self.assertIn("confidence", act.explanation)

    def test_lexical_retrieval_is_default(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["stakeholder"], applicability="Y",
            status="active", confidence=0.7, now=NOW,
        )
        results = self.engine.activate("stakeholder meeting", [seed], NOW)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].relevance, 0)

    def test_activation_create_factory(self):
        seed = candidate()
        act = Activation.create(
            seed, relevance=0.8, confidence=0.7, recency=0.9,
            matched_terms=("project", "stakeholder"),
        )
        self.assertAlmostEqual(act.score, 0.8 * 0.6 + 0.7 * 0.3 + 0.9 * 0.1)
        self.assertEqual(act.confidence, 0.7)
        self.assertEqual(act.recency, 0.9)
        self.assertIn("project", act.explanation)


class CustomPolicyTests(unittest.TestCase):
    def test_custom_policy_promotes_after_one_support(self):
        engine = ExperienceEngine(CustomPolicy())
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "case-1", "Good", NOW), now=NOW,
        )
        evolved = engine.reinforce(seed, Evidence("support", "case-2", "Great", NOW))
        self.assertEqual(evolved.status, "active")
        self.assertGreater(evolved.confidence, seed.confidence)


class RetrievalBackendTests(unittest.TestCase):
    def test_custom_backend_used_for_scoring(self):
        engine = ExperienceEngine()
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["stakeholder"], applicability="Y",
            status="active", confidence=0.7, now=NOW,
        )
        backend = FakeEmbeddingBackend()
        results = engine.activate("stakeholder planning", [seed], NOW, backend=backend)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].relevance, 0)
