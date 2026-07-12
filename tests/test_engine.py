import unittest
from datetime import datetime, timedelta, timezone

from alaya.engine import EvolutionPolicy, ExperienceEngine
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
        c = candidate()  # status = "candidate"
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
