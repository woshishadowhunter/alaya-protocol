"""Minimal adapter pattern for agent frameworks.

Real framework adapters can wrap LangGraph, OpenAI Agents SDK, AutoGen, CrewAI,
or a custom runner. The important boundary is stable: activate experience before
a decision, record which seeds influenced the decision, then observe the outcome.
"""

from datetime import datetime, timezone

from alaya import Decision, Evidence, ExperienceEngine, ExperienceSeed, Observation


class AgentExperienceAdapter:
    def __init__(self, seeds: list[ExperienceSeed] | None = None) -> None:
        self.engine = ExperienceEngine()
        self.seeds = seeds or []

    def before_decision(self, context: str) -> list[dict[str, object]]:
        matches = self.engine.activate(context, self.seeds)
        return [
            {
                "seed_id": item.seed.id,
                "guidance": item.seed.guidance,
                "score": item.score,
                "explanation": item.explanation,
            }
            for item in matches
        ]

    def after_decision(self, context: str, chosen_seed_ids: list[str], action: str) -> Decision:
        decision, _ = self.engine.apply(context, self.seeds, chosen_seed_ids, action)
        return decision

    def after_outcome(self, decision: Decision, outcome: str, polarity: str, source_id: str) -> None:
        observation = Observation(decision.id, outcome, polarity, source_id)
        evolved = self.engine.observe(decision, observation, self.seeds)
        by_id = {seed.id: seed for seed in self.seeds}
        for seed in evolved:
            by_id[seed.id] = seed
        self.seeds = list(by_id.values())


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    seed = ExperienceSeed.new(
        lesson="Do not design the full plan before stakeholder alignment.",
        guidance="Map interests and veto points first.",
        context_tags=["project", "stakeholder", "community"],
        applicability="Multi-party community work",
        status="active",
        confidence=0.75,
        evidence=Evidence("support", "case-1", "Alignment removed a delivery blocker.", now),
        now=now,
    )

    adapter = AgentExperienceAdapter([seed])
    context = "Plan a community activity with several property owners."
    suggestions = adapter.before_decision(context)
    decision = adapter.after_decision(context, [suggestions[0]["seed_id"]], "Run stakeholder mapping.")
    adapter.after_outcome(decision, "The mapping found the real blocker.", "success", "demo-outcome-1")
    print(adapter.before_decision(context)[0])

