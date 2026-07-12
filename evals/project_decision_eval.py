"""Deterministic smoke evaluation; no model or network required."""

from datetime import datetime, timezone

from alaya import Evidence, ExperienceEngine, ExperienceSeed


def run() -> dict[str, object]:
    now = datetime.now(timezone.utc)
    engine = ExperienceEngine()
    relevant = ExperienceSeed.new(
        lesson="Align stakeholders before full project design.", guidance="Map interests first.",
        context_tags=["project", "stakeholder", "community"], applicability="Multi-party projects",
        status="active", confidence=0.8,
    )
    irrelevant = ExperienceSeed.new(
        lesson="Batch meal preparation saves time.", guidance="Prepare meals weekly.",
        context_tags=["food", "health"], applicability="Meal planning", status="active", confidence=0.95,
    )
    matches = engine.activate("community stakeholder project", [irrelevant, relevant], now)
    passed = len(matches) == 1 and matches[0].seed.id == relevant.id
    return {"passed": passed, "activated": len(matches), "top_lesson": matches[0].seed.lesson if matches else None}


if __name__ == "__main__":
    result = run()
    print(result)
    raise SystemExit(0 if result["passed"] else 1)

