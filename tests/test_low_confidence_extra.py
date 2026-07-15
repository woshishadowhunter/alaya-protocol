def should_activate(lesson, min_confidence=0.7):
    if not lesson.get("evidence"):
        return False
    return float(lesson.get("confidence", 0)) >= min_confidence

def test_boundary_confidence():
    assert should_activate({"confidence": 0.7, "evidence": ["x"]}, 0.7) is True
    assert should_activate({"confidence": 0.699, "evidence": ["x"]}, 0.7) is False
