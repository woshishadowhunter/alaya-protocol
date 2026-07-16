"""Conformance checks for auditable agent experience.

The evaluation is intentionally deterministic: no model, network, or vector
database is required. It checks the minimum behavior a framework adapter should
preserve before calling itself Alaya-compatible.
"""

from datetime import datetime, timezone

from alaya import Decision, Evidence, ExperienceEngine, ExperienceSeed, Observation


def active_seed(now: datetime) -> ExperienceSeed:
    seed = ExperienceSeed.new(
        lesson="Align stakeholders before full solution design.",
        guidance="Map interests, veto points, and minimum acceptable outcomes first.",
        context_tags=["project", "stakeholder", "community"],
        applicability="Multi-party community projects",
        confidence=0.7,
        status="active",
        nature="conditional",
        evidence=Evidence("support", "case-001", "A complete plan stalled before alignment.", now, channel="direct"),
        now=now,
    )
    return seed


def run() -> dict[str, object]:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    engine = ExperienceEngine()
    seed = active_seed(now)

    activation = engine.activate("community project with residents and property owners", [seed], now)
    activation_ok = bool(activation and activation[0].seed.id == seed.id and activation[0].matched_terms)

    decision = Decision.new(
        "community project with residents and property owners",
        [seed.id],
        [],
        "Run stakeholder mapping before writing the full plan.",
        now,
    )
    evolved = engine.observe(
        decision,
        Observation(decision.id, "Stakeholder workshop unlocked delivery.", "success", "case-002", now),
        [seed],
        now,
    )
    feedback_ok = bool(evolved and evolved[0].confidence > seed.confidence and evolved[0].support_count == 2)

    contradicted = engine.reinforce(
        seed,
        Evidence("contradict", "case-003", "Emergency repair succeeded without stakeholder mapping.", now),
        now,
    )
    contradiction_ok = contradicted.confidence < seed.confidence and contradicted.contradiction_count == 1

    passed = activation_ok and feedback_ok and contradiction_ok
    return {
        "passed": passed,
        "cases": {
            "activation_is_explainable": activation_ok,
            "decision_feedback_updates_seed": feedback_ok,
            "contradiction_lowers_confidence": contradiction_ok,
        },
    }


if __name__ == "__main__":
    result = run()
    print(result)
    raise SystemExit(0 if result["passed"] else 1)

