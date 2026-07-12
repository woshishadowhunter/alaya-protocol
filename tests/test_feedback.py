import gc
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from alaya.engine import EvolutionPolicy, ExperienceEngine
from alaya.models import Decision, Evidence, ExperienceSeed, Observation
from alaya.store import DecisionStore, SQLiteSeedStore

NOW = datetime(2026, 7, 12, tzinfo=timezone.utc)


class DecisionModelTests(unittest.TestCase):
    def test_decision_round_trip(self):
        d = Decision.new("project planning", ["id1", "id2"], ["id3"], "Ran workshop", NOW)
        data = d.to_dict()
        self.assertEqual(data["context"], "project planning")
        self.assertEqual(data["chosen_seeds"], ["id1", "id2"])
        self.assertEqual(data["excluded_seeds"], ["id3"])

    def test_decision_rejects_empty_action(self):
        with self.assertRaises(ValueError):
            Decision.new("ctx", ["id1"], [], "  ", NOW)


class ObservationModelTests(unittest.TestCase):
    def test_observation_to_evidence_success(self):
        obs = Observation("dec-1", "Workshop solved 4 conflicts", "success", "src-1", NOW)
        evidence = obs.to_evidence()
        self.assertEqual(evidence.polarity, "support")
        self.assertEqual(evidence.source_id, "src-1")

    def test_observation_to_evidence_failure(self):
        obs = Observation("dec-1", "Workshop failed", "failure", "src-2", NOW)
        evidence = obs.to_evidence()
        self.assertEqual(evidence.polarity, "contradict")

    def test_observation_rejects_invalid_polarity(self):
        with self.assertRaises(ValueError):
            Observation("d", "o", "neutral", "s", NOW)


class FeedbackLoopTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine(EvolutionPolicy())

    def _active_seed(self, lesson="L", tags=None):
        seed = ExperienceSeed.new(
            lesson=lesson, guidance="G",
            context_tags=tags or ["project"], applicability="Y",
            confidence=0.6, now=NOW,
        )
        # Mark as active by reinforcing with 3 supports (speculative→conditional→active)
        return seed.with_changes(status="active")

    def test_apply_returns_decision_and_activations(self):
        seed = self._active_seed()
        decision, activations = self.engine.apply(
            "planning a project", [seed], [seed.id], "Ran workshop", NOW,
        )
        self.assertEqual(decision.action, "Ran workshop")
        self.assertEqual(decision.chosen_seeds, (seed.id,))
        self.assertEqual(len(activations), 1)

    def test_observe_chosen_success_reinforces_as_support(self):
        seed = self._active_seed()
        decision = Decision.new("ctx", [seed.id], [], "Action", NOW)
        obs = Observation(decision.id, "It worked", "success", "src-1", NOW)
        evolved = self.engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 1)
        self.assertEqual(evolved[0].support_count, 1)
        self.assertGreater(evolved[0].confidence, seed.confidence)

    def test_observe_chosen_failure_reinforces_as_contradict(self):
        seed = self._active_seed()
        decision = Decision.new("ctx", [seed.id], [], "Action", NOW)
        obs = Observation(decision.id, "It failed", "failure", "src-1", NOW)
        evolved = self.engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 1)
        self.assertEqual(evolved[0].contradiction_count, 1)
        self.assertLess(evolved[0].confidence, seed.confidence)

    def test_observe_chosen_mixed_reinforces_as_support(self):
        seed = self._active_seed()
        decision = Decision.new("ctx", [seed.id], [], "Action", NOW)
        obs = Observation(decision.id, "Partial success", "mixed", "src-1", NOW)
        evolved = self.engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 1)
        self.assertEqual(evolved[0].support_count, 1)

    def test_observe_excluded_success_reinforces_as_contradict(self):
        seed = self._active_seed()
        decision = Decision.new("ctx", [], [seed.id], "Action", NOW)
        obs = Observation(decision.id, "Worked without seed", "success", "src-1", NOW)
        evolved = self.engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 1)
        self.assertEqual(evolved[0].contradiction_count, 1)

    def test_observe_excluded_failure_reinforces_as_support(self):
        seed = self._active_seed()
        decision = Decision.new("ctx", [], [seed.id], "Action", NOW)
        obs = Observation(decision.id, "Failed without seed", "failure", "src-1", NOW)
        evolved = self.engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 1)
        self.assertEqual(evolved[0].support_count, 1)

    def test_observe_excluded_mixed_no_reinforcement(self):
        seed = self._active_seed()
        decision = Decision.new("ctx", [], [seed.id], "Action", NOW)
        obs = Observation(decision.id, "Partial", "mixed", "src-1", NOW)
        evolved = self.engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 0)

    def test_observe_missing_seed_skipped(self):
        seed = self._active_seed()
        decision = Decision.new("ctx", ["missing-id"], [], "Action", NOW)
        obs = Observation(decision.id, "Result", "success", "src-1", NOW)
        evolved = self.engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 0)

    def test_full_feedback_loop(self):
        engine = ExperienceEngine(EvolutionPolicy())
        seed = ExperienceSeed.new(
            lesson="Stakeholder mapping prevents deadlocks.",
            guidance="Map interests first.",
            context_tags=["stakeholder", "project"],
            applicability="Multi-party projects",
            evidence=Evidence("support", "src-1", "Mapping resolved conflict", NOW, channel="direct"),
            now=NOW,
        )
        seed = engine.reinforce(seed, Evidence("support", "src-2", "Workshop succeeded", NOW, channel="direct"))
        seed = engine.reinforce(seed, Evidence("support", "src-3", "Third confirmation", NOW, channel="direct"))
        self.assertEqual(seed.nature, "conditional")

        decision, activations = engine.apply(
            "community renewal project", [seed], [seed.id], "Ran stakeholder mapping workshop", NOW,
        )
        self.assertEqual(len(activations), 1)

        obs = Observation(decision.id, "Workshop resolved 4 of 5 conflicts", "success", "src-4", NOW)
        evolved = engine.observe(decision, obs, [seed], NOW)
        self.assertEqual(len(evolved), 1)
        self.assertEqual(evolved[0].support_count, 4)
        self.assertGreater(evolved[0].confidence, seed.confidence)


class DecisionStoreTests(unittest.TestCase):
    def test_save_and_get(self):
        tmp = tempfile.mkdtemp()
        try:
            store = DecisionStore(Path(tmp) / "test.db")
            d = Decision.new("ctx", ["a", "b"], ["c"], "action", NOW)
            store.save(d)
            restored = store.get(d.id)
            self.assertEqual(restored.id, d.id)
            self.assertEqual(restored.action, d.action)
            self.assertEqual(restored.chosen_seeds, ("a", "b"))
            store.close()
            gc.collect()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
