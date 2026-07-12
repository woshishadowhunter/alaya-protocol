from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

from .models import Evidence, ExperienceSeed, utc_now


@dataclass(frozen=True)
class EvolutionPolicy:
    min_support: int = 2
    support_gain: float = 0.15
    contradiction_loss: float = 0.2
    deprecate_below: float = 0.15
    half_life_days: float = 180.0


@dataclass(frozen=True)
class Activation:
    seed: ExperienceSeed
    score: float
    relevance: float
    explanation: str


class ExperienceEngine:
    def __init__(self, policy: EvolutionPolicy | None = None) -> None:
        self.policy = policy or EvolutionPolicy()

    def reinforce(self, seed: ExperienceSeed, evidence: Evidence, now: datetime | None = None) -> ExperienceSeed:
        if any(item.source_id == evidence.source_id for item in seed.evidence):
            return seed
        timestamp = now or evidence.observed_at
        change = self.policy.support_gain if evidence.polarity == "support" else -self.policy.contradiction_loss
        confidence = min(1.0, max(0.0, seed.confidence + change))
        items = seed.evidence + (evidence,)
        supports = len({e.source_id for e in items if e.polarity == "support"})
        status = seed.status
        if confidence < self.policy.deprecate_below:
            status = "deprecated"
        elif supports >= self.policy.min_support and status == "candidate":
            status = "active"
        return seed.with_changes(evidence=items, confidence=confidence, status=status, updated_at=timestamp)

    def decay(self, seed: ExperienceSeed, now: datetime | None = None) -> ExperienceSeed:
        timestamp = now or utc_now()
        days = max(0.0, (timestamp - seed.updated_at).total_seconds() / 86400)
        factor = math.pow(0.5, days / self.policy.half_life_days)
        confidence = round(seed.confidence * factor, 6)
        status = "deprecated" if confidence < self.policy.deprecate_below else seed.status
        return seed.with_changes(confidence=confidence, status=status)

    def activate(
        self, context: str, seeds: list[ExperienceSeed], now: datetime | None = None,
        limit: int = 5,
    ) -> list[Activation]:
        timestamp = now or utc_now()
        context_tokens = _tokens(context)
        results: list[Activation] = []
        for seed in seeds:
            if seed.status != "active":
                continue
            seed_tokens = _tokens(" ".join(seed.context_tags) + " " + seed.applicability + " " + seed.lesson)
            overlap = len(context_tokens & seed_tokens)
            relevance = overlap / max(1, len(context_tokens))
            if relevance == 0:
                continue
            age_days = max(0.0, (timestamp - seed.updated_at).total_seconds() / 86400)
            recency = math.pow(0.5, age_days / self.policy.half_life_days)
            score = relevance * 0.6 + seed.confidence * 0.3 + recency * 0.1
            explanation = (
                f"relevance={relevance:.3f}; confidence={seed.confidence:.3f}; "
                f"recency={recency:.3f}; matched_terms={','.join(sorted(context_tokens & seed_tokens))}"
            )
            results.append(Activation(seed, round(score, 6), round(relevance, 6), explanation))
        return sorted(results, key=lambda item: (-item.score, item.seed.id))[:limit]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()) if len(token) > 1}

