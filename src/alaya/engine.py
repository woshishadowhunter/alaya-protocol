from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .models import Channel, Decision, Evidence, ExperienceSeed, Nature, Observation, RuleType, utc_now


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
    direct_gain: float = 0.15
    reflective_gain: float = 0.075
    interactive_gain: float = 0.10
    direct_support_weight: float = 1.0
    reflective_support_weight: float = 0.5
    interactive_support_weight: float = 0.7
    min_channels_for_principle: int = 2

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

    def gain_for(self, channel: Channel) -> float:
        if channel == "direct":
            return self.direct_gain
        if channel == "interactive":
            return self.interactive_gain
        return self.reflective_gain

    def support_weight_for(self, channel: Channel) -> float:
        if channel == "direct":
            return self.direct_support_weight
        if channel == "interactive":
            return self.interactive_support_weight
        return self.reflective_support_weight


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
            change = self.policy.gain_for(evidence.channel)
        else:
            change = -self.policy.contradiction_loss_for(seed.nature)
        confidence = min(1.0, max(0.0, seed.confidence + change))
        items = seed.evidence + (evidence,)
        weighted_supports = sum(
            self.policy.support_weight_for(e.channel)
            for e in {e.source_id: e for e in items if e.polarity == "support"}.values()
        )
        contradicts = len({e.source_id for e in items if e.polarity == "contradict"})
        channels = {e.channel for e in items}
        status = seed.status
        nature = self._evolve_nature(seed.nature, weighted_supports, contradicts, channels, seed.created_at)
        if self.policy.should_deprecate(confidence):
            status = "deprecated"
        elif self._should_activate(nature, weighted_supports) and status == "candidate":
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

    def apply(
        self, context: str, seeds: list[ExperienceSeed], chosen_ids: list[str], action: str,
        now: datetime | None = None, limit: int = 5, backend: RetrievalBackend | None = None,
    ) -> tuple[Decision, list[Activation]]:
        """Record a decision influenced by activated experience.

        Returns (Decision, activations) so the caller sees which seeds were surfaced
        and which were chosen vs excluded.
        """
        timestamp = now or utc_now()
        activations = self.activate(context, seeds, timestamp, limit, backend)
        excluded_ids = [a.seed.id for a in activations if a.seed.id not in chosen_ids]
        decision = Decision.new(context, chosen_ids, excluded_ids, action, timestamp)
        # Mark seeds as activated
        updated: list[ExperienceSeed] = []
        for a in activations:
            if a.seed.id in chosen_ids:
                updated.append(a.seed.with_changes(last_activated_at=timestamp))
        return decision, activations

    def observe(
        self, decision: Decision, observation: Observation,
        seeds: list[ExperienceSeed], now: datetime | None = None,
    ) -> list[ExperienceSeed]:
        """Observe outcome and auto-reinforce related seeds.

        Feedback rules:
        - chosen + success → support
        - chosen + failure → contradict
        - chosen + mixed → support (weakly)
        - excluded + success → contradict (seed was correctly ignored)
        - excluded + failure → support (seed should have been heeded)
        """
        timestamp = now or observation.observed_at
        seed_map = {s.id: s for s in seeds}
        evolved: list[ExperienceSeed] = []

        for seed_id in decision.chosen_seeds:
            seed = seed_map.get(seed_id)
            if seed is None:
                continue
            if observation.polarity == "success":
                ev = Evidence("support", observation.source_id, observation.outcome, timestamp, channel="interactive")
                evolved.append(self.reinforce(seed, ev, timestamp))
            elif observation.polarity == "failure":
                ev = Evidence("contradict", observation.source_id, observation.outcome, timestamp, channel="interactive")
                evolved.append(self.reinforce(seed, ev, timestamp))
            elif observation.polarity == "mixed":
                ev = Evidence("support", observation.source_id, observation.outcome, timestamp, channel="interactive")
                evolved.append(self.reinforce(seed, ev, timestamp))

        for seed_id in decision.excluded_seeds:
            seed = seed_map.get(seed_id)
            if seed is None:
                continue
            if observation.polarity == "success":
                ev = Evidence("contradict", observation.source_id, observation.outcome, timestamp, channel="interactive")
                evolved.append(self.reinforce(seed, ev, timestamp))
            elif observation.polarity == "failure":
                ev = Evidence("support", observation.source_id, observation.outcome, timestamp, channel="interactive")
                evolved.append(self.reinforce(seed, ev, timestamp))
            # mixed: no reinforcement for excluded seeds

        return evolved

    def check_rules(
        self, context: str, seeds: list[ExperienceSeed],
        backend: RetrievalBackend | None = None,
    ) -> list[tuple[ExperienceSeed, float, str]]:
        """Check context against hardened inference rules.

        Returns list of (rule_seed, relevance, breach_status).
        breach_status is 'triggered' (rule applies, no breach detected)
        or 'warning' (rule applies, potential breach).
        """
        retriever = backend or LexicalRetrieval()
        context_encoded = retriever.encode([context])[0]
        context_tokens = _tokens(context)
        results: list[tuple[ExperienceSeed, float, str]] = []
        for seed in seeds:
            if seed.rule_type is None or seed.status != "active":
                continue
            if not seed.trigger_tokens:
                continue
            trigger_hit = any(t in context_tokens for t in seed.trigger_tokens)
            if not trigger_hit:
                continue
            rule_text = seed.lesson + " " + seed.guidance
            rule_encoded = retriever.encode([rule_text])[0]
            relevance = retriever.similarity(context_encoded, rule_encoded)
            if relevance == 0:
                continue
            breach = "triggered"
            if relevance > 0.6 and seed.nature == "principle":
                breach = "warning"
            results.append((seed, round(relevance, 4), breach))
        return sorted(results, key=lambda r: (-r[1], r[0].id))

    def audit_health(self, seeds: list[ExperienceSeed]) -> dict[str, object]:
        """Cognitive health audit — 自证分 (svasamvitti-bhāga)."""
        now = utc_now()
        total = len(seeds)
        if total == 0:
            return {"total": 0, "status": "empty"}

        natures: dict[str, int] = {"speculative": 0, "conditional": 0, "principle": 0}
        statuses: dict[str, int] = {"candidate": 0, "active": 0, "deprecated": 0}
        channels: dict[str, int] = {"direct": 0, "reflective": 0, "interactive": 0}
        rule_types: dict[str, int] = {"correction": 0, "heuristic": 0, "boundary": 0, "pattern": 0}
        confidences: list[float] = []
        stale_count = 0

        for s in seeds:
            natures[s.nature] += 1
            statuses[s.status] += 1
            for c in s.distinct_channels:
                channels[c] += 1
            if s.rule_type:
                rule_types[s.rule_type] += 1
            confidences.append(s.confidence)
            days_since_update = (now - s.updated_at).days
            if days_since_update > 30 and s.status == "active":
                stale_count += 1

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        principle_ratio = natures["principle"] / total if total > 0 else 0
        active_ratio = statuses["active"] / total if total > 0 else 0

        health = "healthy"
        if principle_ratio < 0.05 and total > 5:
            health = "needs-principles"
        if active_ratio < 0.1 and total > 3:
            health = "stagnant"
        if stale_count > total * 0.5:
            health = "decaying"
        if total < 3:
            health = "bootstrapping"

        return {
            "total": total,
            "health": health,
            "natures": natures,
            "statuses": statuses,
            "channels": channels,
            "rule_types": rule_types,
            "avg_confidence": round(avg_confidence, 4),
            "principle_ratio": round(principle_ratio, 4),
            "active_ratio": round(active_ratio, 4),
            "stale_active": stale_count,
            "single_channel": sum(1 for s in seeds if len(s.distinct_channels) == 1),
            "timestamp": now.isoformat(),
        }

    def audit_blindspots(
        self, seeds: list[ExperienceSeed], known_domains: list[str] | None = None,
    ) -> dict[str, object]:
        """Blindspot detection — 证自证分 (meta-awareness of gaps)."""
        all_tags: set[str] = set()
        tag_confidence: dict[str, float] = {}
        for s in seeds:
            for t in s.context_tags:
                all_tags.add(t)
                tag_confidence[t] = max(tag_confidence.get(t, 0.0), s.confidence)

        active_tags = {s.context_tags for s in seeds if s.status == "active"}
        flat_active: set[str] = set()
        for tags in active_tags:
            flat_active.update(tags)

        low_confidence_tags = [t for t, c in tag_confidence.items() if c < 0.4]
        single_source_tags: list[str] = []
        for t in all_tags:
            seeds_with_tag = [s for s in seeds if t in s.context_tags]
            if len(seeds_with_tag) == 1 and seeds_with_tag[0].support_count < 2:
                single_source_tags.append(t)

        domain_coverage: dict[str, bool] = {}
        if known_domains:
            for domain in known_domains:
                domain_tokens = _tokens(domain)
                domain_coverage[domain] = any(
                    domain_tokens & set(s.context_tags) for s in seeds
                )

        return {
            "total_domains": len(all_tags),
            "active_domains": len(flat_active),
            "low_confidence_tags": low_confidence_tags,
            "single_source_tags": single_source_tags,
            "coverage": domain_coverage,
            "recommendation": (
                "explore: " + ", ".join(low_confidence_tags[:5])
                if low_confidence_tags
                else "sufficient coverage"
            ),
        }

    def _should_activate(self, nature: Nature, weighted_supports: float) -> bool:
        if nature == "speculative":
            return weighted_supports >= self.policy.speculative_promotion
        return weighted_supports >= self.policy.conditional_promotion

    def _evolve_nature(self, current: Nature, weighted_supports: float,
                      contradicts: int, channels: set[Channel], created_at: datetime) -> Nature:
        if current == "speculative":
            if weighted_supports >= 3:
                return "conditional"
        elif current == "conditional":
            if (weighted_supports >= 5 and contradicts == 0
                    and (utc_now() - created_at).days >= 30
                    and len(channels) >= self.policy.min_channels_for_principle):
                return "principle"
            if contradicts >= 1:
                return "speculative"
        elif current == "principle":
            if contradicts >= self.policy.principle_min_contradictions:
                return "conditional"
        return current


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w一-鿿]+", text.lower()) if len(token) > 1}
