from src.workflow.engine import (
    _cache_record_payload_for_request,
    _sanitize_evidence_map,
    _upsert_cache_record,
    derive_node_status,
    enforce_final_answer_policy,
    heuristic_route_from_query,
    normalize_final_answer_by_route,
    route_minimums,
)
from src.workflow.contracts import enforce_chat_response_contract


def test_route_minimums_for_clause_and_risk_only() -> None:
    clause_rules = route_minimums("clause_only")
    risk_rules = route_minimums("risk_only")
    assert clause_rules["revision_plan"] == 0
    assert risk_rules["revision_plan"] == 0


def test_normalize_final_answer_by_route_blanks_irrelevant_sections() -> None:
    payload = {
        "summary": "요약",
        "clause_analysis": ["c1"],
        "risk_findings": ["r1"],
        "revision_plan": ["p1"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("advice_only", payload)
    assert normalized["clause_analysis"] == []
    assert normalized["risk_findings"] == []
    assert normalized["revision_plan"] == ["p1"]


def test_advice_only_summary_is_clipped_to_one_or_two_sentences() -> None:
    payload = {
        "summary": "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장은 잘려야 합니다.",
        "clause_analysis": [],
        "risk_findings": [],
        "revision_plan": ["p1"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("advice_only", payload)
    assert "세 번째 문장" not in normalized["summary"]
    assert normalized["summary"].count(".") <= 2


def test_advice_only_summary_applies_char_guardrail() -> None:
    long_summary = "A" * 200
    payload = {
        "summary": long_summary,
        "clause_analysis": [],
        "risk_findings": [],
        "revision_plan": ["p1"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("advice_only", payload)
    # _enforce_plan_only_summary clips at 140 chars (+3 for ellipsis)
    assert len(normalized["summary"]) <= 143  # max_chars + ellipsis allowance


def test_summary_is_not_notice_only_when_model_returns_notice_text() -> None:
    payload = {
        "summary": "법률 자문이 아닙니다.",
        "clause_analysis": ["c1", "c2"],
        "risk_findings": ["r1", "r2"],
        "revision_plan": ["p1", "p2"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("full_review", payload)
    assert "조항" in normalized["summary"] or "위험" in normalized["summary"] or "수정" in normalized["summary"]


def test_enforce_final_answer_policy_adds_refs_and_citation() -> None:
    payload = {
        "summary": "요약",
        "clause_analysis": ["핵심 조항 진단"],
        "risk_findings": ["위험 조항 탐지"],
        "revision_plan": [],
        "references": [],
    }
    enforced = enforce_final_answer_policy(
        route="full_review",
        payload=payload,
        rag_refs=[
            {
                "rank": 1,
                "source": "근로기준법_요약.md",
                "chunk_id": 1,
                "location": "n/a",
                "score": 0.8,
                "category": "statutes",
                "snippet": "근로기준법 제17조",
            }
        ],
    )
    assert enforced["references"]
    assert isinstance(enforced["references"][0], dict)
    assert all("[1]" in item for item in enforced["clause_analysis"])
    assert len(enforced["revision_plan"]) >= 4
    assert enforced.get("input_gap_notice")


def test_heuristic_route_from_query_respects_explicit_exclusion() -> None:
    route = heuristic_route_from_query("조항 분석만 해줘. 위험 제외, 수정 계획 제외")
    assert route is not None
    assert route[0] == "clause_only"


def test_heuristic_route_from_query_advice_only_keywords() -> None:
    route = heuristic_route_from_query("수정 계획만 간단히 작성해줘")
    assert route is not None
    assert route[0] == "advice_only"


def test_heuristic_route_ignores_negated_exclusion_phrase() -> None:
    route = heuristic_route_from_query("위험 제외하고 싶진 않지만 우선 조항만 먼저 보고 싶어.")
    # Negated exclusion should not force a heuristic route.
    # Let the LLM router decide with full context.
    assert route is None


def test_heuristic_route_does_not_force_advice_only_on_negated_advice_exclusion() -> None:
    route = heuristic_route_from_query("조항 분석만 보고 싶고 수정 계획 제외는 아니야.")
    assert route is not None
    assert route[0] == "clause_only"


def test_heuristic_route_does_not_force_advice_only_on_include_advice_phrase() -> None:
    route = heuristic_route_from_query("조항만 먼저 보고, 수정 계획은 제외하지 말고 같이 보자.")
    assert route is not None
    assert route[0] == "clause_only"


def test_normalize_final_answer_by_route_cleans_none_notice() -> None:
    payload = {
        "summary": "요약",
        "clause_analysis": ["c1"],
        "risk_findings": ["r1"],
        "revision_plan": ["p1"],
        "input_gap_notice": None,
        "references": [],
    }
    normalized = normalize_final_answer_by_route("full_review", payload)
    assert normalized.get("input_gap_notice") is None


def test_enforce_chat_response_contract_fills_min_schema() -> None:
    payload = enforce_chat_response_contract({"summary": "", "cached_state_hit": "yes"})
    assert payload["summary"]
    assert isinstance(payload["clause_analysis"], list)
    assert isinstance(payload["references"], list)
    assert payload["cached_state_hit"] is True


def test_enforce_chat_response_contract_normalizes_node_status() -> None:
    payload = enforce_chat_response_contract(
        {
            "summary": "ok",
            "node_status": {
                "clause": {
                    "status": "degraded",
                    "error_code": "STRUCTURED_OUTPUT_CLAUSE_FALLBACK",
                    "detail": "fallback used",
                }
            },
        }
    )
    assert isinstance(payload["node_status"], dict)
    assert payload["node_status"]["clause"]["status"] == "degraded"


def test_derive_node_status_includes_skip_reason_by_route() -> None:
    state = {
        "final_answer": {"summary": "요약"},
    }
    status = derive_node_status("advice_only", state)
    assert status["clause"]["status"] == "skipped"
    assert status["risk"]["status"] == "skipped"
    assert "route=advice_only" in str(status["clause"]["detail"])
    assert "advice-focused" in str(status["risk"]["detail"])


def test_sanitize_evidence_map_clips_out_of_range_indices() -> None:
    evidence_map = {
        "항목A": [1, 2, 7, 0, -1, 2],
        "항목B": ["3", "x", None],
        "": [1],
    }
    sanitized = _sanitize_evidence_map(evidence_map, max_ref=3)
    assert sanitized == {"항목A": [1, 2], "항목B": [3]}


def test_sanitize_evidence_map_returns_empty_when_no_refs() -> None:
    evidence_map = {"항목A": [1, 2, 3]}
    assert _sanitize_evidence_map(evidence_map, max_ref=0) == {}


def test_cache_record_supports_signature_keyed_multi_entry() -> None:
    record = _upsert_cache_record({}, "sig-a", {"summary": "A"}, max_items=3)
    record = _upsert_cache_record(record, "sig-b", {"summary": "B"}, max_items=3)
    payload_a = _cache_record_payload_for_request(record, "sig-a")
    payload_b = _cache_record_payload_for_request(record, "sig-b")
    assert payload_a == {"summary": "A"}
    assert payload_b == {"summary": "B"}


def test_cache_record_applies_lru_trim_per_session() -> None:
    record = {}
    for idx in range(1, 5):
        record = _upsert_cache_record(record, f"sig-{idx}", {"summary": str(idx)}, max_items=2)
    assert _cache_record_payload_for_request(record, "sig-4") == {"summary": "4"}
    assert _cache_record_payload_for_request(record, "sig-3") == {"summary": "3"}
    assert _cache_record_payload_for_request(record, "sig-2") is None


def test_enforce_final_answer_policy_keeps_reference_metadata_fields() -> None:
    payload = {
        "summary": "요약",
        "clause_analysis": ["핵심 조항 진단"],
        "risk_findings": [],
        "revision_plan": [],
        "references": [
            {
                "rank": 1,
                "source": "근로기준법_요약.md",
                "chunk_id": 3,
                "location": "page=1",
                "score": 0.9,
                "category": "statutes",
                "snippet": "근로기준법 제17조",
                "collected_at": "2026-03-09",
                "source_url": "https://example.com",
                "curator": "legalpilot-team",
                "license": "CC-BY-4.0",
            }
        ],
    }
    enforced = enforce_final_answer_policy(route="clause_only", payload=payload, rag_refs=[])
    ref = enforced["references"][0]
    assert ref["collected_at"] == "2026-03-09"
    assert ref["source_url"] == "https://example.com"
    assert ref["curator"] == "legalpilot-team"
    assert ref["license"] == "CC-BY-4.0"
