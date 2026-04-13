import pytest

from src.evaluation.metrics import (
    AnswerQualitySample,
    RoutingSampleResult,
    plan_quality_rate,
    reference_source_duplication_rate,
    reference_inclusion_rate,
    routing_accuracy,
)


def test_routing_accuracy() -> None:
    samples = [
        RoutingSampleResult(expected_route="clause_only", predicted_route="clause_only"),
        RoutingSampleResult(expected_route="full_review", predicted_route="advice_only"),
        RoutingSampleResult(expected_route="risk_only", predicted_route="risk_only"),
    ]
    assert routing_accuracy(samples) == 2 / 3


def test_reference_inclusion_rate() -> None:
    samples = [
        AnswerQualitySample(references=["a"], revision_plan=["1", "2", "3", "4"]),
        AnswerQualitySample(references=[], revision_plan=["1", "2", "3", "4"]),
    ]
    assert reference_inclusion_rate(samples) == 0.5


def test_plan_quality_rate() -> None:
    samples = [
        AnswerQualitySample(references=["a"], revision_plan=["1", "2", "3", "4"]),
        AnswerQualitySample(references=["a"], revision_plan=["1", "2"]),
    ]
    assert plan_quality_rate(samples) == 0.5


def test_reference_source_duplication_rate() -> None:
    samples = [
        AnswerQualitySample(
            references=[
                {"source": "a.md"},
                {"source": "a.md"},
                {"source": "b.md"},
            ],
            revision_plan=["1", "2", "3", "4"],
        ),
        AnswerQualitySample(
            references=[
                {"source": "c.md"},
                {"source": "d.md"},
            ],
            revision_plan=["1", "2", "3", "4"],
        ),
    ]
    # sample1 duplicate ratio = 1 - (2/3), sample2 = 0 -> mean = 1/6
    assert reference_source_duplication_rate(samples) == pytest.approx(1 / 6)
