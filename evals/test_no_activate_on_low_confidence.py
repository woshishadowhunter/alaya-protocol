"""Deterministic eval: lesson must not activate without sufficient evidence."""

from __future__ import annotations


def should_activate(lesson: dict, context: dict) -> bool:
    """Mirror protocol gate: require min confidence + matching tags + evidence."""
    if not lesson.get("evidence"):
        return False
    conf = float(lesson.get("confidence", 0))
    min_conf = float(context.get("min_confidence", 0.7))
    if conf < min_conf:
        return False
    tags = set(lesson.get("tags") or [])
    needed = set(context.get("required_tags") or [])
    if needed and not needed.issubset(tags):
        return False
    return True


def test_does_not_activate_when_evidence_missing():
    lesson = {
        "id": "lesson-no-evidence",
        "confidence": 0.95,
        "tags": ["python", "testing"],
        "evidence": [],
    }
    context = {"min_confidence": 0.7, "required_tags": {"python"}}
    assert should_activate(lesson, context) is False


def test_does_not_activate_below_confidence_threshold():
    lesson = {
        "id": "lesson-low-conf",
        "confidence": 0.4,
        "tags": ["python"],
        "evidence": ["tests/test_foo.py::test_bar passed"],
    }
    context = {"min_confidence": 0.7, "required_tags": {"python"}}
    assert should_activate(lesson, context) is False


def test_activates_when_gates_pass():
    lesson = {
        "id": "lesson-ok",
        "confidence": 0.9,
        "tags": ["python", "testing"],
        "evidence": ["CI green on main"],
    }
    context = {"min_confidence": 0.7, "required_tags": {"python"}}
    assert should_activate(lesson, context) is True
