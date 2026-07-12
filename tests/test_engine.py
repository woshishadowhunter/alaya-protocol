import unittest
from datetime import datetime, timedelta, timezone

from alaya.engine import (
    Activation, EvolutionPolicy, ExperienceEngine, LexicalRetrieval,
    PolicyProtocol, RetrievalBackend,
)
from alaya.models import Evidence, ExperienceSeed, Nature


NOW = datetime(2026, 7, 12, tzinfo=timezone.utc)


def candidate():
    return ExperienceSeed.new(
        lesson="Align stakeholders before full solution design.",
        guidance="Start with stakeholder mapping and a resistance check.",
        context_tags=["project", "stakeholder", "community"],
        applicability="Multi-party projects",
        evidence=Evidence("support", "case-1", "Early design was rejected", NOW, channel="direct"),
        now=NOW,
    )


class CustomPolicy:
    min_support = 1
    support_gain = 0.3
    contradiction_loss = 0.1
    deprecate_below = 0.1
    half_life_days = 90.0
    counterexample_penalty = 0.5
    speculative_promotion = 1
    conditional_promotion = 1
    principle_auto_promotion_supports = 3
    principle_min_contradictions = 2
    speculative_contradiction_loss = 0.15
    principle_contradiction_loss = 0.1
    speculative_half_life = 60.0
    principle_half_life = 200.0
    direct_gain = 0.3
    reflective_gain = 0.15
    interactive_gain = 0.2
    direct_support_weight = 1.0
    reflective_support_weight = 0.5
    interactive_support_weight = 0.7
    min_channels_for_principle = 2

    def should_promote(self, supports):
        return supports >= self.min_support
    def should_deprecate(self, confidence):
        return confidence < self.deprecate_below
    def confidence_delta(self, polarity):
        return self.support_gain if polarity == "support" else -self.contradiction_loss
    def decay_factor(self, days):
        import math; return math.pow(0.5, days / self.half_life_days)
    def half_life_for(self, nature):
        if nature == "speculative": return self.speculative_half_life
        if nature == "principle": return self.principle_half_life
        return self.half_life_days
    def contradiction_loss_for(self, nature):
        if nature == "speculative": return self.speculative_contradiction_loss
        if nature == "principle": return self.principle_contradiction_loss
        return self.contradiction_loss
    def counterexample_penalty_for(self, nature):
        if nature == "speculative": return 0.2
        if nature == "principle": return 0.5
        return self.counterexample_penalty
    def confidence_weight_for(self, nature):
        if nature == "speculative": return 0.2
        if nature == "principle": return 0.4
        return 0.3
    def gain_for(self, channel):
        if channel == "direct": return self.direct_gain
        if channel == "interactive": return self.interactive_gain
        return self.reflective_gain
    def support_weight_for(self, channel):
        if channel == "direct": return self.direct_support_weight
        if channel == "interactive": return self.interactive_support_weight
        return self.reflective_support_weight


class FakeEmbeddingBackend:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "stakeholder" in t else [0.0, 1.0] for t in texts]

    def similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return max(0.0, dot)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine(EvolutionPolicy(min_support=2))

    def test_speculative_promotes_after_three_direct_supports(self):
        seed = candidate()
        seed = self.engine.reinforce(seed, Evidence("support", "case-2", "Good", NOW, channel="direct"))
        seed = self.engine.reinforce(seed, Evidence("support", "case-3", "Great", NOW, channel="direct"))
        self.assertEqual(seed.status, "active")
        self.assertEqual(seed.support_count, 3)

    def test_second_independent_support_still_candidate_for_speculative(self):
        seed = candidate()
        seed = self.engine.reinforce(seed, Evidence("support", "case-2", "Good", NOW, channel="direct"))
        self.assertEqual(seed.status, "candidate")
        self.assertEqual(seed.support_count, 2)

    def test_duplicate_source_does_not_increase_support(self):
        seed = candidate()
        seed = self.engine.reinforce(seed, Evidence("support", "case-1", "Repeated note", NOW, channel="direct"))
        self.assertEqual(seed.support_count, 1)
        self.assertEqual(seed.status, "candidate")

    def test_contradiction_reduces_confidence_and_is_retained(self):
        seed = candidate()
        seed = self.engine.reinforce(seed, Evidence("contradict", "case-3", "Rapid draft created alignment", NOW, channel="direct"))
        self.assertLess(seed.confidence, candidate().confidence)
        self.assertEqual(seed.contradiction_count, 1)
        self.assertEqual(len(seed.evidence), 2)

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
                seed, Evidence("contradict", f"case-{seed.contradiction_count+10}", "Bad", NOW, channel="direct")
            )
        self.assertGreaterEqual(seed.confidence, 0.0)

    def test_confidence_clamped_at_one(self):
        seed = candidate()
        for i in range(10):
            seed = self.engine.reinforce(
                seed, Evidence("support", f"case-{i+10}", "Great", NOW, channel="direct")
            )
        self.assertLessEqual(seed.confidence, 1.0)

    def test_activation_explains_relevance(self):
        seed = candidate()
        seed = self.engine.reinforce(seed, Evidence("support", "case-2", "Worked", NOW, channel="direct"))
        seed = self.engine.reinforce(seed, Evidence("support", "case-3", "Worked", NOW, channel="direct"))
        unrelated = ExperienceSeed.new(
            lesson="Prepare meals on Sunday.", guidance="Batch cook.",
            context_tags=["health", "food"], applicability="Weekly meal planning",
            status="active", confidence=0.9, now=NOW,
        )
        results = self.engine.activate(
            "Plan a community project involving property owners", [unrelated, seed], NOW
        )
        self.assertEqual(results[0].seed.id, seed.id)
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

    def test_principle_not_deprecated_by_decay(self):
        seed = ExperienceSeed.new(
            lesson="Universal truth", guidance="Always apply",
            context_tags=["math", "logic"], applicability="All reasoning",
            status="active", nature="principle", confidence=0.2, now=NOW,
        )
        decayed = self.engine.decay(seed, NOW + timedelta(days=1000))
        self.assertEqual(decayed.status, "active")

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


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine()

    def test_direct_evidence_counts_fully(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "src-1", "Tool output confirmed", NOW, channel="direct"), now=NOW,
        )
        self.assertEqual(seed.confidence, 0.5)
        seed = self.engine.reinforce(seed, Evidence("support", "src-2", "Also confirmed", NOW, channel="direct"))
        self.assertGreater(seed.confidence, 0.5 + 0.15 - 0.01)

    def test_reflective_evidence_half_weight_gain(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "src-1", "LLM suggested", NOW, channel="reflective"), now=NOW,
        )
        self.assertEqual(seed.confidence, 0.5)
        seed = self.engine.reinforce(seed, Evidence("support", "src-2", "LLM again", NOW, channel="reflective"))
        self.assertAlmostEqual(seed.confidence, 0.575)  # 0.5 + 0.075

    def test_mixed_channels_enable_promotion(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "src-1", "Tool output", NOW, channel="direct"), now=NOW,
        )
        seed = self.engine.reinforce(seed, Evidence("support", "src-2", "LLM reflection", NOW, channel="reflective"))
        seed = self.engine.reinforce(seed, Evidence("support", "src-3", "User feedback", NOW, channel="interactive"))
        # weighted: 1.0 + 0.5 + 0.7 = 2.2 < 3, still candidate
        self.assertEqual(seed.status, "candidate")
        self.assertEqual(seed.distinct_channels, {"direct", "reflective", "interactive"})

    def test_single_channel_cannot_reach_principle(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "src-1", "Good", NOW, channel="direct"), now=NOW,
        )
        for i in range(8):
            seed = self.engine.reinforce(seed, Evidence("support", f"src-{i+2}", "Great", NOW, channel="direct"))
        # With many supports it reaches conditional, but single-channel blocks principle
        self.assertEqual(seed.nature, "conditional")


class NatureTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine()

    def test_speculative_starts_as_default(self):
        seed = ExperienceSeed.new(lesson="L", guidance="G", context_tags=["x"], applicability="Y")
        self.assertEqual(seed.nature, "speculative")

    def test_nature_survives_round_trip(self):
        for n in ("speculative", "conditional", "principle"):
            seed = ExperienceSeed.new(lesson="L", guidance="G", context_tags=["x"], applicability="Y", nature=n)
            restored = ExperienceSeed.from_dict(seed.to_dict())
            self.assertEqual(restored.nature, n)

    def test_old_dict_without_nature_defaults_to_speculative(self):
        data = {
            "schema_version": "1.0", "id": "test-id", "lesson": "L", "guidance": "G",
            "context_tags": ["x"], "applicability": "Y", "counterexamples": [],
            "confidence": 0.5, "status": "candidate",
            "evidence": [],
            "created_at": "2026-07-12T00:00:00+00:00",
            "updated_at": "2026-07-12T00:00:00+00:00",
            "last_activated_at": None,
        }
        seed = ExperienceSeed.from_dict(data)
        self.assertEqual(seed.nature, "speculative")

    def test_contradiction_reduces_confidence_more_for_speculative(self):
        spec = ExperienceSeed.new(lesson="L", guidance="G", context_tags=["x"], applicability="Y", confidence=0.5, nature="speculative")
        cond = ExperienceSeed.new(lesson="L2", guidance="G2", context_tags=["x"], applicability="Y", confidence=0.5, nature="conditional")
        spec_ev = self.engine.reinforce(spec, Evidence("contradict", "src-1", "Bad", NOW, channel="direct"))
        cond_ev = self.engine.reinforce(cond, Evidence("contradict", "src-1", "Bad", NOW, channel="direct"))
        self.assertLess(spec_ev.confidence, cond_ev.confidence)

    def test_nature_promotes_to_conditional_after_three_direct_supports(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "src-1", "Good", NOW, channel="direct"), now=NOW,
        )
        seed = self.engine.reinforce(seed, Evidence("support", "src-2", "Good", NOW, channel="direct"))
        seed = self.engine.reinforce(seed, Evidence("support", "src-3", "Good", NOW, channel="direct"))
        self.assertEqual(seed.nature, "conditional")
        self.assertEqual(seed.status, "active")

    def test_principle_has_higher_activation_score(self):
        engine = ExperienceEngine()
        spec = ExperienceSeed.new(lesson="Spec lesson", guidance="Spec G", context_tags=["stakeholder", "project"], applicability="Y", status="active", nature="speculative", confidence=0.6, now=NOW)
        prin = ExperienceSeed.new(lesson="Prin lesson", guidance="Prin G", context_tags=["stakeholder", "project"], applicability="Y", status="active", nature="principle", confidence=0.6, now=NOW)
        results = engine.activate("stakeholder project", [spec, prin], NOW)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].seed.id, prin.id)

    def test_invalid_nature_rejected(self):
        with self.assertRaises(ValueError):
            ExperienceSeed.new(lesson="L", guidance="G", context_tags=["x"], applicability="Y", nature="invalid")

    def test_invalid_channel_rejected(self):
        with self.assertRaises(ValueError):
            Evidence("support", "src", "summary", NOW, channel="invalid")


class CustomPolicyTests(unittest.TestCase):
    def test_custom_policy_promotes_after_one_support(self):
        engine = ExperienceEngine(CustomPolicy())
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "case-1", "Good", NOW, channel="direct"), now=NOW,
        )
        evolved = engine.reinforce(seed, Evidence("support", "case-2", "Great", NOW, channel="direct"))
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
