from datetime import datetime, timezone

from alaya import Evidence, ExperienceEngine, ExperienceSeed


now = datetime.now(timezone.utc)
engine = ExperienceEngine()
seed = ExperienceSeed.new(
    lesson="Multi-party renewal projects stall when complete plans precede interest alignment.",
    guidance="Map stakeholders, interests, veto points, and minimum acceptable outcomes first.",
    context_tags=["community", "renewal", "stakeholder", "project"],
    applicability="Projects involving residents, property managers, developers, and regulators",
    counterexamples=["Emergency safety work requiring immediate statutory action"],
    evidence=Evidence("support", "renewal-review-1", "A complete plan triggered resistance", now),
    now=now,
)
seed = engine.reinforce(
    seed,
    Evidence("support", "renewal-review-2", "Interest mapping produced an executable minimum agreement", now),
)
seed = engine.reinforce(
    seed,
    Evidence("support", "renewal-review-3", "Alignment workshop resolved a 6-month deadlock", now),
)

for match in engine.activate("A community renewal project with residents and property managers", [seed], now):
    print(match.seed.guidance)
    print(match.explanation)

