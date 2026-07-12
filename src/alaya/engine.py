from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .models import Evidence, ExperienceSeed, Nature, utc_now


class PolicyProtocol(Protocol):
    min_support: int
    support_gain: float
    contradiction_loss: float
    deprecate_below: float
    half_life_days: float
    counterexample_penalty: float
    speculative_promotion: int
    conditional_promotion: int
    principle_auto_promotion_supports: int
    principle_min_contradictions: int
    speculative_contradiction_loss: float
    principle_contradiction_loss: float
    speculative_half_life: float
    principle_half_life: float

    def should_promote(self, supports: int) -> bool:
        return supports >= self.min_support

    def should_deprecate(self, confidence: float) -> bool:
        return confidence < self.deprecate_below

    def confidence_delta(self, polarity: str) -> float:
        return self.support_gain if polarity == "support" else -self.contradiction_loss

    def decay_factor(self, days: float) -> float:
        return math.pow(0.5, days / self.half_life_days)

    def half_life_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return self.speculative_half_life
        if nature == "principle":
            return self.principle_half_life
        return self.half_life_days

    def contradiction_loss_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return self.speculative_contradiction_loss
        if nature == "principle":
            return self.principle_contradiction_loss
        return self.contradiction_loss

    def counterexample_penalty_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return 0.2
        if nature == "principle":
            return 0.5
        return self.counterexample_penalty

    def confidence_weight_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return 0.2
        if nature == "principle":
            return 0.4
        return 0.3


class RetrievalBackend(Protocol):
    def encode(self, texts: list[str]) -> list[Any]: ...
    def similarity(self, a: Any, b: Any) -> float: ...


@dataclass(frozen=True)
class LexicalRetrieval:
    def encode(self, texts: list[str]) -> list[set[str]]:
        return [_tokens(t) for t in texts]

    def similarity(self, a: set[str], b: set[str]) -> float:
        overlap = len(a & b)
        return overlap / max(1, len(a))


@dataclass(frozen=True)
class EvolutionPolicy:
    min_support: int = 2
    support_gain: float = 0.15
    contradiction_loss: float = 0.2
    deprecate_below: float = 0.15
    half_life_days: float = 180.0
    counterexample_penalty: float = 0.3
    speculative_promotion: int = 3
    conditional_promotion: int = 2
    principle_auto_promotion_supports: int = 5
    principle_min_contradictions: int = 3
    speculative_contradiction_loss: float = 0.25
    principle_contradiction_loss: float = 0.15
    speculative_half_life: float = 90.0
    principle_half_life: float = 365.0

    def should_promote(self, supports: int) -> bool:
        return supports >= self.min_support

    def should_deprecate(self, confidence: float) -> bool:
        return confidence < self.deprecate_below

    def confidence_delta(self, polarity: str) -> float:
        return self.support_gain if polarity == "support" else -self.contradiction_loss

    def decay_factor(self, days: float) -> float:
        return math.pow(0.5, days / self.half_life_days)

    def half_life_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return self.speculative_half_life
        if nature == "principle":
            return self.principle_half_life
        return self.half_life_days

    def contradiction_loss_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return self.speculative_contradiction_loss
        if nature == "principle":
            return self.principle_contradiction_loss
        return self.contradiction_loss

    def counterexample_penalty_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return 0.2
        if nature == "principle":
            return 0.5
        return self.counterexample_penalty

    def confidence_weight_for(self, nature: Nature) -> float:
        if nature == "speculative":
            return 0.2
        if nature == "principle":
            return 0.4
        return 0.3


@dataclass(frozen=True)
class Activation:
    seed: ExperienceSeed
    score: float
    relevance: float
    confidence: float = 0.0
    recency: float = 0.0
    matched_terms: tuple[str, ...] = ()
    explanation: str = ""

    @classmethod
    def create(
        cls, seed: ExperienceSeed, relevance: float, confidence: float,
        recency: float, matched_terms: tuple[str, ...],
        nature_weight: float = 0.3,
    ) -> "Activation":
        score = round(relevance * 0.6 + confidence * nature_weight + recency * 0.1, 6)
        explanation = (
            f"relevance={relevance:.3f}; confidence={confidence:.3f}; "
            f"recency={recency:.3f}; nature={seed.nature}; "
            f"matched_terms={','.join(sorted(matched_terms))}"
        )
        return cls(seed, score, round(relevance, 6), confidence, recency, matched_terms, explanation)


class ExperienceEngine:
    def __init__(self, policy: PolicyProtocol | None = None) -> None:
        self.policy = policy or EvolutionPolicy()

    def reinforce(self, seed: ExperienceSeed, evidence: Evidence, now: datetime | None = None) -> ExperienceSeed:
        if any(item.source_id == evidence.source_id for item in seed.evidence):
            return seed
        timestamp = now or evidence.observed_at
        if evidence.polarity == "support":
            change = self.policy.support_gain
        else:
            change = -self.policy.contradiction_loss_for(seed.nature)
        confidence = min(1.0, max(0.0, seed.confidence + change))
        items = seed.evidence + (evidence,)
        supports = len({e.source_id for e in items if e.polarity == "support"})
        contradicts = len({e.source_id for e in items if e.polarity == "contradict"})
        status = seed.status
        nature = self._evolve_nature(seed.nature, supports, contradicts, seed.created_at)
        if self.policy.should_deprecate(confidence):
            status = "deprecated"
        elif self._should_activate(nature, supports) and status == "candidate":
            status = "active"
        return seed.with_changes(
            evidence=items, confidence=confidence, status=status, nature=nature, updated_at=timestamp,
        )

    def decay(self, seed: ExperienceSeed, now: datetime | None = None) -> ExperienceSeed:
        timestamp = now or utc_now()
        days = max(0.0, (timestamp - seed.updated_at).total_seconds() / 86400)
        half_life = self.policy.half_life_for(seed.nature)
        factor = math.pow(0.5, days / half_life)
        confidence = round(seed.confidence * factor, 6)
        if seed.nature == "principle":
            return seed.with_changes(confidence=confidence)
        status = "deprecated" if self.policy.should_deprecate(confidence) else seed.status
        return seed.with_changes(confidence=confidence, status=status)

    def activate(
        self, context: str, seeds: list[ExperienceSeed], now: datetime | None = None,
        limit: int = 5, backend: RetrievalBackend | None = None,
    ) -> list[Activation]:
        timestamp = now or utc_now()
        retriever = backend or LexicalRetrieval()
        context_encoded = retriever.encode([context])[0]
        results: list[Activation] = []
        for seed in seeds:
            if seed.status != "active":
                continue
            seed_text = " ".join(seed.context_tags) + " " + seed.applicability + " " + seed.lesson
            seed_encoded = retriever.encode([seed_text])[0]
            relevance = retriever.similarity(context_encoded, seed_encoded)
            if relevance == 0:
                continue
            penalty = self.policy.counterexample_penalty_for(seed.nature)
            for ce in seed.counterexamples:
                ce_encoded = retriever.encode([ce])[0]
                if retriever.similarity(context_encoded, ce_encoded) > 0.3:
                    relevance *= penalty
                    break
            matched_terms: tuple[str, ...] = ()
            if isinstance(context_encoded, set) and isinstance(seed_encoded, set):
                matched_terms = tuple(sorted(context_encoded & seed_encoded))
            half_life = self.policy.half_life_for(seed.nature)
            age_days = max(0.0, (timestamp - seed.updated_at).total_seconds() / 86400)
            recency = math.pow(0.5, age_days / half_life)
            nature_weight = self.policy.confidence_weight_for(seed.nature)
            results.append(Activation.create(
                seed, relevance, seed.confidence, round(recency, 6), matched_terms,
                nature_weight=nature_weight,
            ))
        return sorted(results, key=lambda item: (-item.score, item.seed.id))[:limit]

    def _should_activate(self, nature: Nature, supports: int) -> bool:
        if nature == "speculative":
            return supports >= self.policy.speculative_promotion
        return supports >= self.policy.conditional_promotion

    @staticmethod
    def _evolve_nature(current: Nature, supports: int, contradicts: int, created_at: datetime) -> Nature:
        if current == "speculative":
            if supports >= 3:
                return "conditional"
        elif current == "conditional":
            if supports >= 5 and contradicts == 0 and (utc_now() - created_at).days >= 30:
                return "principle"
            if contradicts >= 1:
                return "speculative"
        elif current == "principle":
            if contradicts >= 3:
                return "conditional"
        return current


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w一-鿿]+", text.lower()) if len(token) > 1}
