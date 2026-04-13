"""Evaluation helpers for differentiation metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingSampleResult:
    expected_route: str
    predicted_route: str


@dataclass(frozen=True)
class AnswerQualitySample:
    references: list[object]
    revision_plan: list[str]


def routing_accuracy(samples: list[RoutingSampleResult]) -> float:
    """Return Top-1 routing accuracy in [0, 1]."""
    if not samples:
        return 0.0
    correct = sum(
        1
        for item in samples
        if item.expected_route.strip().lower() == item.predicted_route.strip().lower()
    )
    return correct / len(samples)


def reference_inclusion_rate(samples: list[AnswerQualitySample], min_references: int = 1) -> float:
    """Return ratio of answers that include at least min_references refs."""
    if not samples:
        return 0.0
    included = sum(1 for item in samples if len(item.references) >= min_references)
    return included / len(samples)


def plan_quality_rate(samples: list[AnswerQualitySample], min_plan_items: int = 4) -> float:
    """Return ratio of answers that include enough actionable revision plan items."""
    if not samples:
        return 0.0
    qualified = sum(1 for item in samples if len(item.revision_plan) >= min_plan_items)
    return qualified / len(samples)


def reference_source_duplication_rate(samples: list[AnswerQualitySample]) -> float:
    """Return macro average duplicate-source ratio across samples in [0, 1]."""
    if not samples:
        return 0.0
    per_sample: list[float] = []
    for item in samples:
        if not item.references:
            per_sample.append(0.0)
            continue
        sources: list[str] = []
        for ref in item.references:
            if isinstance(ref, dict):
                source = str(ref.get("source", "")).strip().lower()
            else:
                source = str(ref).strip().lower()
            if source:
                sources.append(source)
        if not sources:
            per_sample.append(0.0)
            continue
        unique_count = len(set(sources))
        total_count = len(sources)
        duplicate_ratio = max(0.0, min(1.0, 1.0 - (unique_count / total_count)))
        per_sample.append(duplicate_ratio)
    return sum(per_sample) / len(per_sample)
