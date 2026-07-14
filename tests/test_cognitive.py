import gc
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from alaya.engine import EvolutionPolicy, ExperienceEngine
from alaya.models import ActivationLog, Evidence, ExperienceSeed, Nature
from alaya.store import SQLiteSeedStore

NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)


class RuleSeedTests(unittest.TestCase):
    def test_new_rule_has_rule_type_and_triggers(self):
        rule = ExperienceSeed.new_rule(
            lesson="If uncertain about user intent, ask before acting.",
            guidance="Detect ambiguity → pause → ask clarifying question.",
            rule_type="heuristic",
            trigger_tokens=["unclear", "ambiguous", "maybe", "possibly", "不确定"],
            applicability="All agent interactions where user intent could be multiply interpreted",
        )
        self.assertEqual(rule.rule_type, "heuristic")
        self.assertEqual(rule.trigger_tokens, ("unclear", "ambiguous", "maybe", "possibly", "不确定"))
        self.assertEqual(rule.status, "active")
        self.assertEqual(rule.nature, "conditional")

    def test_new_rule_round_trip(self):
        rule = ExperienceSeed.new_rule(
            lesson="Avoid over-fitting to a single user preference.",
            guidance="One correction does not make a permanent preference.",
            rule_type="boundary",
            trigger_tokens=["always", "never", "every time"],
            applicability="User feedback interpretation",
            evidence=Evidence("support", "correction-001", "User said 'don't always assume'", NOW, channel="interactive"),
            now=NOW,
        )
        restored = ExperienceSeed.from_dict(rule.to_dict())
        self.assertEqual(restored.rule_type, "boundary")
        self.assertEqual(restored.trigger_tokens, ("always", "never", "every time"))

    def test_rule_preserved_after_reinforce(self):
        engine = ExperienceEngine()
        rule = ExperienceSeed.new_rule(
            lesson="Test before deploying.", guidance="Always run the test suite.",
            rule_type="correction", trigger_tokens=["deploy", "push"],
            applicability="Code deployment workflows",
        )
        evolved = engine.reinforce(rule, Evidence("support", "src-2", "Confirmed", NOW, channel="direct"), NOW)
        self.assertEqual(evolved.rule_type, "correction")
        self.assertEqual(evolved.trigger_tokens, ("deploy", "push"))


class RuleCheckTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine()
        self.rules = [
            ExperienceSeed.new_rule(
                lesson="Don't guess user intent when unclear.",
                guidance="Pause and ask a clarifying question.",
                rule_type="heuristic",
                trigger_tokens=["unclear", "ambiguous", "not sure", "maybe"],
                applicability="Agent decision-making",
                evidence=Evidence("support", "src-1", "Shadowhunter corrected", NOW, channel="interactive"),
                now=NOW,
            ),
            ExperienceSeed.new_rule(
                lesson="Test before deployment always.",
                guidance="Run full test suite before any push.",
                rule_type="correction",
                trigger_tokens=["deploy", "push", "ship"],
                applicability="Code deployment",
                evidence=Evidence("support", "src-2", "Deployment failed without tests", NOW, channel="direct"),
                now=NOW,
            ),
        ]

    def test_rule_triggers_on_matching_context(self):
        results = self.engine.check_rules("I'm not sure what the user wants, maybe I should guess", self.rules)
        self.assertGreater(len(results), 0)
        triggered_ids = {r[0].id for r in results}
        self.assertIn(self.rules[0].id, triggered_ids)

    def test_non_matching_context_returns_empty(self):
        results = self.engine.check_rules("What is the capital of France", self.rules)
        self.assertEqual(len(results), 0)

    def test_rule_without_triggers_is_skipped(self):
        seed = ExperienceSeed.new(lesson="L", guidance="G", context_tags=["x"], applicability="Y", status="active", now=NOW)
        results = self.engine.check_rules("unclear intent", [seed])
        self.assertEqual(len(results), 0)

    def test_non_active_rule_is_skipped(self):
        rule = ExperienceSeed.new_rule(
            lesson="L", guidance="G", rule_type="heuristic", trigger_tokens=["test"],
            applicability="Y",
        ).with_changes(status="candidate")
        results = self.engine.check_rules("test context", [rule])
        self.assertEqual(len(results), 0)


class AuditHealthTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine()

    def test_empty_store_returns_empty(self):
        result = self.engine.audit_health([])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["status"], "empty")

    def test_bootstrapping_with_few_seeds(self):
        seeds = [
            ExperienceSeed.new(lesson="L", guidance="G", context_tags=["x"], applicability="Y", now=NOW),
        ]
        result = self.engine.audit_health(seeds)
        self.assertEqual(result["health"], "bootstrapping")
        self.assertEqual(result["natures"]["speculative"], 1)

    def test_mixed_natures_and_statuses(self):
        seeds = [
            ExperienceSeed.new(lesson="A", guidance="GA", context_tags=["a"], applicability="Y", nature="speculative", status="active", now=NOW),
            ExperienceSeed.new(lesson="B", guidance="GB", context_tags=["b"], applicability="Y", nature="conditional", status="active", now=NOW),
            ExperienceSeed.new(lesson="C", guidance="GC", context_tags=["c"], applicability="Y", nature="principle", status="active", now=NOW),
            ExperienceSeed.new(lesson="D", guidance="GD", context_tags=["d"], applicability="Y", nature="speculative", status="deprecated", now=NOW),
        ]
        result = self.engine.audit_health(seeds)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["natures"]["principle"], 1)
        self.assertEqual(result["statuses"]["active"], 3)
        self.assertGreater(result["avg_confidence"], 0)
        self.assertIn("timestamp", result)

    def test_channel_diversity_tracked(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["x"], applicability="Y",
            evidence=Evidence("support", "src-1", "Direct obs", NOW, channel="direct"), now=NOW,
        )
        result = self.engine.audit_health([seed])
        self.assertEqual(result["single_channel"], 1)


class AuditBlindspotTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExperienceEngine()

    def test_low_confidence_single_source_detected(self):
        seed = ExperienceSeed.new(
            lesson="L", guidance="G", context_tags=["untested-domain"], applicability="Y",
            confidence=0.3, status="candidate", now=NOW,
        )
        result = self.engine.audit_blindspots([seed])
        self.assertIn("untested-domain", result["low_confidence_tags"])

    def test_domain_coverage_checked(self):
        seeds = [
            ExperienceSeed.new(lesson="A", guidance="GA", context_tags=["python", "api"], applicability="Y", status="active", now=NOW),
            ExperienceSeed.new(lesson="B", guidance="GB", context_tags=["rust", "cli"], applicability="Y", status="active", now=NOW),
        ]
        result = self.engine.audit_blindspots(seeds, ["python backend", "frontend react", "rust cli"])
        self.assertTrue(result["coverage"]["python backend"])
        self.assertTrue(result["coverage"]["rust cli"])
        self.assertFalse(result["coverage"]["frontend react"])


class ActivationLogTests(unittest.TestCase):
    def test_log_round_trip(self):
        log = ActivationLog.new("main-agent", "planning a project", ["id1", "id2"], ["id1"], ["id2"], NOW)
        data = log.to_dict()
        self.assertEqual(data["agent_id"], "main-agent")
        self.assertEqual(data["chosen_seed_ids"], ["id1"])
        self.assertEqual(data["excluded_seed_ids"], ["id2"])
