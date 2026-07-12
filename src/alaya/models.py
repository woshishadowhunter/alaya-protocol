from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

Polarity = Literal["support", "contradict"]
Status = Literal["candidate", "active", "deprecated"]
Nature = Literal["speculative", "conditional", "principle"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed


@dataclass(frozen=True)
class Evidence:
    polarity: Polarity
    source_id: str
    summary: str
    observed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.polarity not in ("support", "contradict"):
            raise ValueError("polarity must be support or contradict")
        if not self.source_id.strip() or not self.summary.strip():
            raise ValueError("evidence requires source_id and summary")
        _dt(self.observed_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "polarity": self.polarity,
            "source_id": self.source_id,
            "summary": self.summary,
            "observed_at": self.observed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(data["polarity"], data["source_id"], data["summary"], _dt(data["observed_at"]))


@dataclass(frozen=True)
class ExperienceSeed:
    id: str
    lesson: str
    guidance: str
    context_tags: tuple[str, ...]
    applicability: str
    counterexamples: tuple[str, ...]
    confidence: float
    status: Status
    nature: Nature
    evidence: tuple[Evidence, ...]
    created_at: datetime
    updated_at: datetime
    last_activated_at: datetime | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.id or not self.lesson.strip() or not self.guidance.strip():
            raise ValueError("seed requires id, lesson, and guidance")
        if not self.applicability.strip() or not self.context_tags:
            raise ValueError("seed requires applicability and context_tags")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.status not in ("candidate", "active", "deprecated"):
            raise ValueError("invalid status")
        if self.nature not in ("speculative", "conditional", "principle"):
            raise ValueError("invalid nature")
        _dt(self.created_at); _dt(self.updated_at)
        if self.last_activated_at is not None:
            _dt(self.last_activated_at)

    @property
    def support_count(self) -> int:
        return len({e.source_id for e in self.evidence if e.polarity == "support"})

    @property
    def contradiction_count(self) -> int:
        return len({e.source_id for e in self.evidence if e.polarity == "contradict"})

    @classmethod
    def new(
        cls, *, lesson: str, guidance: str, context_tags: list[str], applicability: str,
        counterexamples: list[str] | None = None, confidence: float = 0.5,
        status: Status = "candidate", nature: Nature = "speculative",
        evidence: Evidence | None = None,
        now: datetime | None = None,
    ) -> "ExperienceSeed":
        timestamp = now or utc_now()
        return cls(
            id=str(uuid4()), lesson=lesson, guidance=guidance,
            context_tags=tuple(dict.fromkeys(t.strip().lower() for t in context_tags if t.strip())),
            applicability=applicability, counterexamples=tuple(counterexamples or ()),
            confidence=confidence, status=status, nature=nature,
            evidence=(evidence,) if evidence else (),
            created_at=timestamp, updated_at=timestamp,
        )

    def with_changes(self, **changes: Any) -> "ExperienceSeed":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "id": self.id, "lesson": self.lesson,
            "guidance": self.guidance, "context_tags": list(self.context_tags),
            "applicability": self.applicability, "counterexamples": list(self.counterexamples),
            "confidence": self.confidence, "status": self.status, "nature": self.nature,
            "evidence": [item.to_dict() for item in self.evidence],
            "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat(),
            "last_activated_at": self.last_activated_at.isoformat() if self.last_activated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceSeed":
        return cls(
            id=data["id"], lesson=data["lesson"], guidance=data["guidance"],
            context_tags=tuple(data["context_tags"]), applicability=data["applicability"],
            counterexamples=tuple(data.get("counterexamples", ())),
            confidence=float(data["confidence"]),
            status=data["status"],
            nature=data.get("nature", "speculative"),
            evidence=tuple(Evidence.from_dict(e) for e in data.get("evidence", ())),
            created_at=_dt(data["created_at"]), updated_at=_dt(data["updated_at"]),
            last_activated_at=_dt(data["last_activated_at"]) if data.get("last_activated_at") else None,
            schema_version=data.get("schema_version", "1.0"),
        )
