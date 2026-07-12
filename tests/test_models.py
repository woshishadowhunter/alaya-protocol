import unittest
from datetime import datetime, timezone

from alaya.models import Evidence, ExperienceSeed


class ModelTests(unittest.TestCase):
    def test_seed_round_trip_preserves_provenance(self):
        now = datetime(2026, 7, 12, tzinfo=timezone.utc)
        seed = ExperienceSeed.new(
            lesson="Map stakeholders before drafting a complete plan.",
            guidance="Run a stakeholder and resistance check first.",
            context_tags=["community", "project-planning"],
            applicability="Multi-party projects with unresolved interests",
            evidence=Evidence("support", "pilot-1", "Plan stalled before alignment", now),
            now=now,
        )

        restored = ExperienceSeed.from_dict(seed.to_dict())

        self.assertEqual(restored, seed)

    def test_rejects_out_of_range_confidence(self):
        with self.assertRaises(ValueError):
            ExperienceSeed.new(
                lesson="x",
                guidance="y",
                context_tags=["z"],
                applicability="a",
                confidence=1.1,
            )

    def test_rejects_invalid_polarity(self):
        with self.assertRaises(ValueError):
            Evidence("neutral", "src", "summary")

    def test_rejects_empty_source_id(self):
        with self.assertRaises(ValueError):
            Evidence("support", "  ", "summary")

    def test_rejects_naive_datetime(self):
        naive = datetime(2026, 7, 12)
        with self.assertRaises(ValueError):
            Evidence("support", "src", "summary", naive)

    def test_counterexamples_survive_round_trip(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            counterexamples=["Emergency work", "Statutory deadlines"],
        )
        restored = ExperienceSeed.from_dict(seed.to_dict())
        self.assertEqual(restored.counterexamples, seed.counterexamples)

    def test_confidence_zero_is_valid(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            confidence=0.0,
        )
        self.assertEqual(seed.confidence, 0.0)

    def test_confidence_one_is_valid(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            confidence=1.0,
        )
        self.assertEqual(seed.confidence, 1.0)
