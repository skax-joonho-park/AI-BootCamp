"""Run automatic checks for routing accuracy / reference inclusion / plan quality."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    AnswerQualitySample,
    RoutingSampleResult,
    plan_quality_rate,
    reference_source_duplication_rate,
    reference_inclusion_rate,
    routing_accuracy,
)
from src.config.settings import load_settings  # noqa: E402
from src.workflow import LegalPilotService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate differentiation metrics")
    parser.add_argument(
        "--cases",
        default="data/eval/sample_queries.json",
        help="JSON file path containing evaluation queries",
    )
    parser.add_argument("--session-prefix", default="eval-session")
    parser.add_argument("--min-routing-accuracy", type=float, default=0.5)
    parser.add_argument("--min-reference-rate", type=float, default=0.8)
    parser.add_argument("--min-plan-quality-rate", type=float, default=0.8)
    parser.add_argument(
        "--min-per-labeled-route",
        type=int,
        default=4,
        help="Minimum number of cases for each labeled route(full_review/clause_only/risk_only/advice_only).",
    )
    parser.add_argument(
        "--min-ambiguous-cases",
        type=int,
        default=4,
        help="Minimum number of ambiguous(no expected_route) cases.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path. When set, writes the full report with UTF-8 encoding.",
    )
    parser.add_argument(
        "--compare-rerank-on-off",
        action="store_true",
        help="Run the same cases with RERANK_ENABLED=true/false and report source-duplication delta.",
    )
    parser.add_argument(
        "--min-rerank-diversity-gain",
        type=float,
        default=0.0,
        help="Minimum required gain where gain=(dup_ratio_off - dup_ratio_on).",
    )
    return parser.parse_args()


def _load_cases(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation cases JSON must be a list.")
    return [item for item in raw if isinstance(item, dict)]


def _requires_plan(route: str) -> bool:
    route_key = (route or "").strip().lower()
    return route_key in {"full_review", "advice_only"}


def _normalized_route(value: str) -> str:
    return (value or "").strip().lower()


def _case_distribution(cases: list[dict]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        route = _normalized_route(str(case.get("expected_route", "")))
        counter[route if route else "ambiguous"] += 1
    return counter


def _validate_case_distribution(
    counter: Counter[str],
    min_per_labeled_route: int,
    min_ambiguous_cases: int,
) -> list[str]:
    failures: list[str] = []
    for route in ("clause_only", "risk_only", "advice_only", "full_review"):
        if counter.get(route, 0) < min_per_labeled_route:
            failures.append(
                f"[FAIL] case distribution: '{route}'={counter.get(route, 0)} < {min_per_labeled_route}"
            )
    if counter.get("ambiguous", 0) < min_ambiguous_cases:
        failures.append(
            f"[FAIL] case distribution: 'ambiguous'={counter.get('ambiguous', 0)} < {min_ambiguous_cases}"
        )
    return failures


def _confusion_matrix(samples: list[RoutingSampleResult]) -> tuple[list[str], dict[str, dict[str, int]]]:
    labels = sorted(
        {
            _normalized_route(item.expected_route)
            for item in samples
            if _normalized_route(item.expected_route)
        }
        | {
            _normalized_route(item.predicted_route)
            for item in samples
            if _normalized_route(item.predicted_route)
        }
    )
    matrix: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }
    for item in samples:
        expected = _normalized_route(item.expected_route)
        predicted = _normalized_route(item.predicted_route)
        if expected in matrix and predicted in matrix[expected]:
            matrix[expected][predicted] += 1
    return labels, matrix


def _run_once(
    *,
    cases: list[dict],
    session_prefix: str,
    rerank_enabled_override: bool | None = None,
) -> dict[str, object]:
    original_rerank_env = os.environ.get("RERANK_ENABLED")
    try:
        if rerank_enabled_override is None:
            if "RERANK_ENABLED" in os.environ:
                del os.environ["RERANK_ENABLED"]
        else:
            os.environ["RERANK_ENABLED"] = "true" if rerank_enabled_override else "false"
        load_settings.cache_clear()

        service = LegalPilotService()
        routing_samples: list[RoutingSampleResult] = []
        reference_samples: list[AnswerQualitySample] = []
        plan_quality_samples: list[AnswerQualitySample] = []
        details: list[dict[str, object]] = []
        for i, case in enumerate(cases, start=1):
            session_id = f"{session_prefix}-{i}"
            state = {
                "session_id": session_id,
                "user_query": str(case.get("query", "")),
                "document_type": str(case.get("document_type", "근로계약서")),
                "contract_text": str(case.get("contract_text", "")),
                "reference_text": str(case.get("reference_text", "")),
            }
            result = service.graph.invoke(
                state,
                config={"configurable": {"thread_id": session_id}},
            )
            final_answer = result.get("final_answer", {}) if isinstance(result, dict) else {}
            refs = final_answer.get("references", []) if isinstance(final_answer, dict) else []
            plans = final_answer.get("revision_plan", []) if isinstance(final_answer, dict) else []
            references = refs if isinstance(refs, list) else []
            revision_plan = plans if isinstance(plans, list) else []

            predicted_route = str(result.get("route", "unknown"))
            expected_route = str(case.get("expected_route", ""))
            eval_route = (expected_route or predicted_route).strip().lower()
            if expected_route:
                routing_samples.append(
                    RoutingSampleResult(
                        expected_route=_normalized_route(expected_route),
                        predicted_route=_normalized_route(predicted_route),
                    )
                )
            sample = AnswerQualitySample(
                references=references,
                revision_plan=[str(item) for item in revision_plan],
            )
            reference_samples.append(sample)
            if _requires_plan(eval_route):
                plan_quality_samples.append(sample)
            details.append(
                {
                    "query": state["user_query"],
                    "expected_route": expected_route,
                    "predicted_route": predicted_route,
                    "eval_route": eval_route,
                    "references_count": len(references),
                    "plan_items_count": len(revision_plan),
                }
            )
        routing = routing_accuracy(routing_samples) if routing_samples else 0.0
        ref_rate = reference_inclusion_rate(reference_samples)
        plan_rate = plan_quality_rate(plan_quality_samples)
        dup_rate = reference_source_duplication_rate(reference_samples)
        return {
            "routing_samples": routing_samples,
            "reference_samples": reference_samples,
            "routing": routing,
            "ref_rate": ref_rate,
            "plan_rate": plan_rate,
            "dup_rate": dup_rate,
            "details": details,
        }
    finally:
        if original_rerank_env is None:
            os.environ.pop("RERANK_ENABLED", None)
        else:
            os.environ["RERANK_ENABLED"] = original_rerank_env
        load_settings.cache_clear()


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    cases = _load_cases(cases_path)
    case_counter = _case_distribution(cases)
    baseline = _run_once(cases=cases, session_prefix=args.session_prefix)
    routing_samples = baseline["routing_samples"]
    routing = float(baseline["routing"])
    ref_rate = float(baseline["ref_rate"])
    plan_rate = float(baseline["plan_rate"])
    dup_rate = float(baseline["dup_rate"])
    details = baseline["details"]

    report_lines: list[str] = []
    report_lines.append("=== Differentiation Metrics ===")
    report_lines.append(f"Cases: {len(cases)}")
    report_lines.append(
        "Case distribution: "
        f"clause_only={case_counter.get('clause_only', 0)}, "
        f"risk_only={case_counter.get('risk_only', 0)}, "
        f"advice_only={case_counter.get('advice_only', 0)}, "
        f"full_review={case_counter.get('full_review', 0)}, "
        f"ambiguous={case_counter.get('ambiguous', 0)}"
    )
    if routing_samples:
        report_lines.append(f"Routing accuracy: {routing:.2%}")
    else:
        report_lines.append("Routing accuracy: n/a (expected_route not provided)")
    report_lines.append(f"Reference inclusion rate: {ref_rate:.2%}")
    report_lines.append(f"Plan quality rate (full_review/advice_only): {plan_rate:.2%}")
    report_lines.append(f"Reference duplicate-source ratio: {dup_rate:.2%}")
    if routing_samples:
        labels, matrix = _confusion_matrix(routing_samples)
        expected_labels = sorted({_normalized_route(item.expected_route) for item in routing_samples})
        report_lines.append("")
        report_lines.append("Routing confusion matrix (rows=expected, cols=predicted):")
        header = ["expected\\pred", *labels]
        report_lines.append(" | ".join(header))
        report_lines.append(" | ".join(["---"] * len(header)))
        for expected in labels:
            row = [expected, *[str(matrix[expected][predicted]) for predicted in labels]]
            report_lines.append(" | ".join(row))
        report_lines.append("")
        report_lines.append("Route recall:")
        recalls: list[float] = []
        for route in expected_labels:
            total = sum(matrix[route].values())
            tp = matrix[route].get(route, 0)
            recall = (tp / total) if total else 0.0
            recalls.append(recall)
            report_lines.append(f"- {route}: {recall:.2%} ({tp}/{total})")
        macro_recall = (sum(recalls) / len(recalls)) if recalls else 0.0
        report_lines.append(f"- macro_recall: {macro_recall:.2%}")
    report_lines.append("")
    report_lines.append("=== Case Details ===")
    for item in details:
        report_lines.append(json.dumps(item, ensure_ascii=False))

    rerank_gain = 0.0
    if args.compare_rerank_on_off:
        rerank_on = _run_once(
            cases=cases,
            session_prefix=f"{args.session_prefix}-rerank-on",
            rerank_enabled_override=True,
        )
        rerank_off = _run_once(
            cases=cases,
            session_prefix=f"{args.session_prefix}-rerank-off",
            rerank_enabled_override=False,
        )
        dup_on = float(rerank_on["dup_rate"])
        dup_off = float(rerank_off["dup_rate"])
        rerank_gain = dup_off - dup_on
        report_lines.append("")
        report_lines.append("=== Rerank On/Off Diversity Comparison ===")
        report_lines.append(f"duplicate-source ratio (rerank=on): {dup_on:.2%}")
        report_lines.append(f"duplicate-source ratio (rerank=off): {dup_off:.2%}")
        report_lines.append(f"diversity gain (off-on): {rerank_gain:.2%}")

    distribution_failures = _validate_case_distribution(
        case_counter,
        min_per_labeled_route=max(1, args.min_per_labeled_route),
        min_ambiguous_cases=max(1, args.min_ambiguous_cases),
    )
    if distribution_failures:
        report_lines.extend(distribution_failures)
        _emit_report(report_lines, args.output)
        raise SystemExit(distribution_failures[0])

    if routing_samples and routing < args.min_routing_accuracy:
        report_lines.append(
            f"[FAIL] routing accuracy {routing:.2%} < {args.min_routing_accuracy:.2%}"
        )
        _emit_report(report_lines, args.output)
        raise SystemExit(report_lines[-1])
    if ref_rate < args.min_reference_rate:
        report_lines.append(f"[FAIL] reference inclusion rate {ref_rate:.2%} < {args.min_reference_rate:.2%}")
        _emit_report(report_lines, args.output)
        raise SystemExit(report_lines[-1])
    if plan_rate < args.min_plan_quality_rate:
        report_lines.append(f"[FAIL] plan quality rate {plan_rate:.2%} < {args.min_plan_quality_rate:.2%}")
        _emit_report(report_lines, args.output)
        raise SystemExit(report_lines[-1])
    if args.compare_rerank_on_off and rerank_gain < args.min_rerank_diversity_gain:
        report_lines.append(
            f"[FAIL] rerank diversity gain {rerank_gain:.2%} < {args.min_rerank_diversity_gain:.2%}"
        )
        _emit_report(report_lines, args.output)
        raise SystemExit(report_lines[-1])

    report_lines.append("[PASS] All thresholds satisfied.")
    _emit_report(report_lines, args.output)


def _emit_report(lines: list[str], output_path: str) -> None:
    text = "\n".join(lines)
    print(text)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

