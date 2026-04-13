"""LangGraph multi-agent workflow for LegalPilot AI."""

from __future__ import annotations

import json
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Literal, TypeVar

from filelock import FileLock
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing import TypedDict

from src.agents.schemas import ClauseNotes, FinalAnswer, RevisionNotes, RiskNotes
from src.agents.tools import clause_keyword_match_score, contract_reference_gap_score, legal_issue_bank
from src.common import ErrorCodes, LegalPilotError
from src.config import get_chat_model, load_settings
from src.retrieval import HybridRetriever, SearchHit, rerank_hits
from src.utils.memory import SessionMemory
from src.utils.io import atomic_write_text
from src.workflow.contracts import (
    STRUCTURED_OUTPUT_REPAIR_ENABLED,
    STRUCTURED_OUTPUT_RETRY_MAX,
    ChatRequest,
    ChatResponse,
    enforce_chat_response_contract,
)
from src.workflow.prompts import (
    ADVICE_ONLY_MARKERS,
    CLAUSE_ONLY_MARKERS,
    EXCLUSION_TERMS,
    EXCLUSION_PATTERNS,
    INTENT_KEYWORDS,
    NEGATION_PATTERNS,
    PROMPT_RULE_ALL_FIELDS_KOREAN,
    PROMPT_RULE_RESULT_ONLY,
    RISK_ONLY_MARKERS,
    STRUCTURED_JSON_REPAIR_INSTRUCTION,
    TOOL_LOOP_FINALIZATION_INSTRUCTION,
    build_common_policy_block,
    build_route_options_block,
    build_specialist_system_prompt,
    build_supervisor_routing_rules_block,
)


class AgentState(TypedDict, total=False):
    session_id: str
    user_query: str
    document_type: str
    contract_text: str
    reference_text: str
    memory_messages: list[dict[str, str]]
    rag_context: str
    rag_refs: list[dict[str, Any]]
    rag_low_confidence: bool
    route: str
    routing_reason: str
    clause_notes: dict[str, Any]
    risk_notes: dict[str, Any]
    revision_notes: dict[str, Any]
    final_answer: dict[str, Any]
    node_status: dict[str, Any]


class RouteDecision(BaseModel):
    route: Literal["clause_only", "risk_only", "full_review", "advice_only"] = Field(
        description="Supervisor가 선택한 라우팅 결과"
    )
    reason: str = Field(description="해당 라우팅을 선택한 이유")


class RagPlan(BaseModel):
    rewritten_queries: list[str] = Field(default_factory=list, description="재검색용 쿼리 목록")
    source_hint: str = Field(
        default="", description="우선 확인할 문서/키워드 힌트(예: backend, data, pm)"
    )


class SpecialistDecision(BaseModel):
    need_tool_call: bool = Field(default=False, description="도구 호출 필요 여부")
    focus_areas: list[str] = Field(default_factory=list, description="이번 노드에서 강조할 포인트")
    omit_areas: list[str] = Field(default_factory=list, description="이번 노드에서 생략할 포인트")
    reason: str = Field(default="", description="의사결정 이유")


TStructured = TypeVar("TStructured", bound=BaseModel)


def heuristic_route_from_query(user_query: str) -> tuple[str, str] | None:
    """Apply deterministic exclusions/only-intent routing before LLM router."""
    text = " ".join(str(user_query or "").lower().split())
    if not text:
        return None

    def _has_any(keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _has_exclusion_intent(target_terms: tuple[str, ...]) -> bool:
        """Detect explicit exclusion intent while avoiding negated contexts.

        Example negations to avoid:
        - "면접 제외하지 말고 ..."
        - "면접 제외하고 싶진 않지만 ..."
        - "면접은 제외가 아니고 ..."
        """
        for term in target_terms:
            if not term:
                continue
            has_exclusion = any(f"{term} {pattern}" in text or f"{term}{pattern}" in text for pattern in EXCLUSION_PATTERNS)
            if not has_exclusion:
                continue
            has_negation = any(
                re.search(
                    rf"{re.escape(term)}.{{0,24}}(제외|빼).{{0,24}}{re.escape(neg)}",
                    text,
                )
                for neg in NEGATION_PATTERNS
            )
            if has_negation:
                continue
            return True
        return False

    def _has_plan_only_marker_intent() -> bool:
        if not _has_any(ADVICE_ONLY_MARKERS):
            return False
        plan_term_window = "|".join(re.escape(term) for term in EXCLUSION_TERMS["advice"] if term)
        has_negated_plan_exclusion = any(
            re.search(
                rf"({plan_term_window}).{{0,20}}(제외|빼|말고).{{0,20}}{re.escape(neg)}",
                text,
            )
            for neg in NEGATION_PATTERNS
        )
        if has_negated_plan_exclusion:
            return False
        return True

    exclude_resume = _has_exclusion_intent(EXCLUSION_TERMS["clause"])
    exclude_interview = _has_exclusion_intent(EXCLUSION_TERMS["risk"])
    exclude_plan = _has_exclusion_intent(EXCLUSION_TERMS["advice"])
    want_resume = _has_any(INTENT_KEYWORDS["clause"])
    want_interview = _has_any(INTENT_KEYWORDS["risk"])
    want_plan = _has_any(INTENT_KEYWORDS["advice"])

    if _has_plan_only_marker_intent():
        return ("advice_only", "휴리스틱 라우팅: 수정 계획 전용 요청 키워드 감지")
    if _has_any(CLAUSE_ONLY_MARKERS) and not want_interview:
        return ("clause_only", "휴리스틱 라우팅: 조항 분석 전용/위험 제외 키워드 감지")
    if _has_any(RISK_ONLY_MARKERS) and not want_resume:
        return ("risk_only", "휴리스틱 라우팅: 위험 조항 전용/조항 분석 제외 키워드 감지")

    if exclude_interview and exclude_plan and not exclude_resume:
        return ("clause_only", "휴리스틱 라우팅: 위험/수정 제외 요청")
    if exclude_resume and exclude_plan and not exclude_interview:
        return ("risk_only", "휴리스틱 라우팅: 조항/수정 제외 요청")
    if exclude_resume and exclude_interview and not exclude_plan:
        return ("advice_only", "휴리스틱 라우팅: 조항/위험 제외 요청")
    if exclude_interview and want_resume and not want_plan:
        return ("clause_only", "휴리스틱 라우팅: 위험 제외 + 조항 분석 의도")
    if exclude_resume and want_interview and not want_plan:
        return ("risk_only", "휴리스틱 라우팅: 조항 제외 + 위험 탐지 의도")

    return None


def route_minimums(route: str) -> dict[str, int]:
    route_key = (route or "full_review").lower()
    if route_key == "clause_only":
        return {
            "clause_analysis": 4,
            "risk_findings": 0,
            "revision_plan": 0,
        }
    if route_key == "risk_only":
        return {
            "clause_analysis": 0,
            "risk_findings": 4,
            "revision_plan": 0,
        }
    if route_key == "advice_only":
        return {
            "clause_analysis": 0,
            "risk_findings": 0,
            "revision_plan": 4,
        }
    return {
        "clause_analysis": 4,
        "risk_findings": 4,
        "revision_plan": 4,
    }


def _split_summary_sentences(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return []
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", normalized) if chunk.strip()]
    if len(sentences) <= 1:
        sentences = [
            chunk.strip()
            for chunk in re.split(r"(?<=다\.)\s+|(?<=요\.)\s+|(?<=함\.)\s+|(?<=임\.)\s+", normalized)
            if chunk.strip()
        ]
    return sentences or [normalized]


def _enforce_plan_only_summary(summary: str, max_chars: int = 140) -> str:
    base = "요청에 따라 수정 계획 중심으로 핵심만 요약해 제공합니다."
    raw_text = str(summary or "")
    normalized = " ".join(raw_text.split()).strip()
    if not normalized:
        normalized = base
    line_units = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if line_units:
        clipped = "\n".join(line_units[:2]).strip()
    else:
        clipped = ""
    sentences = _split_summary_sentences(normalized)
    if not clipped:
        clipped = " ".join(sentences[:2]).strip() if sentences else base
    else:
        # If raw lines are too coarse, still prefer sentence-level concise summary.
        sentence_candidate = " ".join(sentences[:2]).strip() if sentences else ""
        if sentence_candidate and len(sentence_candidate) <= len(clipped):
            clipped = sentence_candidate
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip() + "..."
    return clipped or base


def _scope_notice(route: str) -> str:
    route_key = (route or "full_review").lower()
    if route_key == "advice_only":
        return "본 검토 의견은 법적 자문이 아니며, 실제 계약 체결 전 반드시 전문 법률가의 검토를 받으세요."
    return (
        "안내: 본 답변은 일반적인 계약서 검토 참고 정보이며 법률 자문이 아닙니다. "
        "실제 계약 체결 전 반드시 전문 법률가의 확인이 필요합니다."
    )


def _attach_scope_notice(route: str, summary: str) -> str:
    text = " ".join(str(summary or "").split()).strip()
    if not text:
        text = "요청에 대한 핵심 요약입니다."
    if "법률 자문이 아닙니다" in text or "전문 법률가" in text:
        return text
    return f"{text} {_scope_notice(route)}".strip()


def _compose_core_summary(route: str, answer: dict[str, Any]) -> str:
    route_key = (route or "full_review").lower()
    clause_len = len(list(answer.get("clause_analysis", []) or []))
    risk_len = len(list(answer.get("risk_findings", []) or []))
    revision_len = len(list(answer.get("revision_plan", []) or []))
    if route_key == "advice_only":
        return f"수정 계획 핵심 {max(revision_len, 1)}개를 우선순위 중심으로 정리했습니다."
    if route_key == "clause_only":
        return f"조항 분석 핵심 {max(clause_len, 1)}개를 우선순위로 정리했습니다."
    if route_key == "risk_only":
        return f"위험 조항 탐지 핵심 {max(risk_len, 1)}개를 리스크/대응 방향 중심으로 정리했습니다."
    return (
        f"조항 분석 {max(clause_len, 1)}개, 위험 조항 {max(risk_len, 1)}개, "
        f"수정 계획 {max(revision_len, 1)}개를 통합 정리했습니다."
    )


def _looks_notice_only(summary: str) -> bool:
    text = " ".join(str(summary or "").split()).strip()
    if not text:
        return True
    has_notice = ("법률 자문이 아닙니다" in text or "전문 법률가" in text)
    if not has_notice:
        return False
    core_keywords = ("조항", "위험", "수정", "검토", "핵심", "정리", "액션", "우선순위")
    return not any(keyword in text for keyword in core_keywords)


def _clean_optional_notice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "nan"}:
        return None
    return text


def normalize_final_answer_by_route(route: str, answer: dict[str, Any]) -> dict[str, Any]:
    route_key = (route or "full_review").lower()
    summary = str(answer.get("summary", "")).strip()
    if not summary:
        if route_key == "advice_only":
            summary = "요청에 따라 수정 계획 중심으로 핵심만 요약해 제공합니다."
        else:
            summary = "요청에 대한 핵심 요약입니다."
    if _looks_notice_only(summary):
        summary = _compose_core_summary(route_key, answer)
    summary = _attach_scope_notice(route_key, summary)
    if _looks_notice_only(summary):
        summary = _attach_scope_notice(route_key, _compose_core_summary(route_key, answer))
    if route_key == "advice_only":
        summary = _enforce_plan_only_summary(summary)
    payload = {
        "summary": summary,
        "clause_analysis": list(answer.get("clause_analysis", []) or []),
        "risk_findings": list(answer.get("risk_findings", []) or []),
        "revision_plan": list(answer.get("revision_plan", []) or []),
        "input_gap_notice": _clean_optional_notice(answer.get("input_gap_notice")),
        "references": list(answer.get("references", []) or []),
    }
    if route_key in {"risk_only", "advice_only"}:
        payload["clause_analysis"] = []
    if route_key in {"clause_only", "advice_only"}:
        payload["risk_findings"] = []
    return payload


def _ensure_citation(item: str, default_tag: str) -> str:
    text = str(item).strip()
    if not text:
        return text
    if "[" in text and "]" in text:
        return text
    return f"{text} {default_tag}".strip()


def _has_valid_citation(item: str, max_ref: int) -> bool:
    if max_ref <= 0:
        return False
    numbers = re.findall(r"\[(\d+)\]", str(item))
    if not numbers:
        return False
    return any(1 <= int(n) <= max_ref for n in numbers)


def _normalize_reference_records(references: list[Any] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(references or [], start=1):
        if isinstance(item, dict):
            normalized.append(
                {
                    "rank": int(item.get("rank", idx)),
                    "source": str(item.get("source", "unknown")),
                    "chunk_id": item.get("chunk_id"),
                    "location": str(item.get("location", "n/a")),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "category": item.get("category"),
                    "snippet": str(item.get("snippet", "")),
                    "score_breakdown": (
                        item.get("score_breakdown")
                        if isinstance(item.get("score_breakdown"), dict)
                        else None
                    ),
                    "collected_at": _clean_optional_notice(item.get("collected_at")),
                    "source_url": _clean_optional_notice(item.get("source_url")),
                    "curator": _clean_optional_notice(item.get("curator")),
                    "license": _clean_optional_notice(item.get("license")),
                }
            )
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(
            {
                "rank": idx,
                "source": text,
                "chunk_id": None,
                "location": "n/a",
                "score": 0.0,
                "category": None,
                "snippet": text[:120],
                "score_breakdown": None,
                "collected_at": None,
                "source_url": None,
                "curator": None,
                "license": None,
            }
        )
    return normalized


def _extract_error_code_from_text(text: str) -> str | None:
    raw = str(text or "")
    if not raw:
        return None
    matched = re.search(
        r"(STRUCTURED_OUTPUT_[A-Z_]+|SERVICE_RUNTIME_ERROR|CONFIG_MISSING_ENV)",
        raw,
    )
    if matched:
        return matched.group(1)
    return None


def _collect_text_values(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for value in payload.values():
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend(str(item) for item in value if isinstance(item, str))
    return texts


def _sanitize_evidence_map(raw_map: Any, max_ref: int) -> dict[str, list[int]]:
    """Keep only evidence indices that point to existing references."""
    if not isinstance(raw_map, dict) or max_ref <= 0:
        return {}
    sanitized: dict[str, list[int]] = {}
    for key, value in raw_map.items():
        key_text = str(key).strip()
        if not key_text or not isinstance(value, list):
            continue
        normalized_indices: list[int] = []
        for item in value:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if idx < 1 or idx > max_ref or idx in normalized_indices:
                continue
            normalized_indices.append(idx)
        if normalized_indices:
            sanitized[key_text] = normalized_indices
    return sanitized


def _sanitize_notes_evidence_map(notes: dict[str, Any], max_ref: int) -> dict[str, Any]:
    if not isinstance(notes, dict):
        return {}
    normalized = dict(notes)
    normalized["evidence_map"] = _sanitize_evidence_map(normalized.get("evidence_map"), max_ref=max_ref)
    return normalized


def _merge_reference_records(
    primary: list[dict[str, Any]] | None,
    secondary: list[dict[str, Any]] | None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _normalize_reference_records(primary) + _normalize_reference_records(secondary):
        key = "::".join(
            [
                str(item.get("source", "")),
                str(item.get("chunk_id", "")),
                str(item.get("location", "")),
                str(item.get("snippet", "")),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max(1, limit):
            break
    for idx, item in enumerate(merged, start=1):
        item["rank"] = idx
    return merged


def _cache_record_payload_for_request(record: Any, signature: str) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    kind = str(record.get("cache_kind", ""))
    if kind == "final_answer_payload_v1":
        if record.get("signature") != signature:
            return None
        payload = record.get("payload")
        return payload if isinstance(payload, dict) else None
    if kind == "final_answer_payload_v2":
        items = record.get("items", [])
        if not isinstance(items, list):
            return None
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("signature") != signature:
                continue
            payload = item.get("payload")
            if isinstance(payload, dict):
                return payload
    return None


def _upsert_cache_record(
    record: Any,
    signature: str,
    payload: dict[str, Any],
    max_items: int,
) -> dict[str, Any]:
    max_entries = max(1, int(max_items))
    items: list[dict[str, Any]] = []
    if isinstance(record, dict):
        kind = str(record.get("cache_kind", ""))
        if kind == "final_answer_payload_v2" and isinstance(record.get("items"), list):
            items = [item for item in record["items"] if isinstance(item, dict)]
        elif kind == "final_answer_payload_v1":
            old_signature = record.get("signature")
            old_payload = record.get("payload")
            if isinstance(old_signature, str) and isinstance(old_payload, dict):
                items = [
                    {
                        "signature": old_signature,
                        "payload": old_payload,
                        "updated_at": float(record.get("updated_at", 0.0) or 0.0),
                    }
                ]
    items = [item for item in items if item.get("signature") != signature]
    items.insert(
        0,
        {
            "signature": signature,
            "payload": payload,
            "updated_at": time.time(),
        },
    )
    return {
        "cache_kind": "final_answer_payload_v2",
        "items": items[:max_entries],
    }


def derive_node_status(route: str, state: AgentState) -> dict[str, dict[str, Any]]:
    """Build partial-failure status for each node without failing whole response."""
    route_key = (route or "full_review").lower()
    run_clause = route_key in {"full_review", "clause_only"}
    run_risk = route_key in {"full_review", "risk_only"}
    run_advice = route_key in {"full_review", "advice_only"}
    node_status: dict[str, dict[str, Any]] = {
        "supervisor": {"status": "ok", "error_code": None, "detail": None},
        "rag": {"status": "ok", "error_code": None, "detail": None},
        "clause": {"status": "skipped", "error_code": None, "detail": None},
        "risk": {"status": "skipped", "error_code": None, "detail": None},
        "advice": {"status": "skipped", "error_code": None, "detail": None},
        "synthesis": {"status": "ok", "error_code": None, "detail": None},
    }
    if run_clause:
        node_status["clause"]["status"] = "ok"
    if run_risk:
        node_status["risk"]["status"] = "ok"
    if run_advice:
        node_status["advice"]["status"] = "ok"

    def _fill_skip_detail(node_name: str) -> None:
        if node_status.get(node_name, {}).get("status") != "skipped":
            return
        detail = f"Skipped by route policy (route={route_key})."
        if route_key == "advice_only" and node_name in {"clause", "risk"}:
            detail = f"Skipped by route policy (route={route_key}; advice-focused execution)."
        elif route_key == "clause_only" and node_name in {"risk", "advice"}:
            detail = f"Skipped by route policy (route={route_key}; clause-focused execution)."
        elif route_key == "risk_only" and node_name in {"clause", "advice"}:
            detail = f"Skipped by route policy (route={route_key}; risk-focused execution)."
        node_status[node_name]["detail"] = detail

    def _mark_if_fallback(node_name: str, payload: dict[str, Any] | None) -> None:
        if not payload:
            node_status[node_name]["status"] = "degraded"
            node_status[node_name]["error_code"] = ErrorCodes.STRUCTURED_OUTPUT_CONTRACT_ERROR
            node_status[node_name]["detail"] = "Node output missing."
            return
        for text in _collect_text_values(payload):
            code = _extract_error_code_from_text(text)
            if code:
                node_status[node_name]["status"] = "degraded"
                node_status[node_name]["error_code"] = code
                node_status[node_name]["detail"] = text[:240]
                return

    if run_clause:
        _mark_if_fallback("clause", state.get("clause_notes"))
    if run_risk:
        _mark_if_fallback("risk", state.get("risk_notes"))
    if run_advice:
        _mark_if_fallback("advice", state.get("revision_notes"))
    for skipped_node in ("clause", "risk", "advice"):
        _fill_skip_detail(skipped_node)

    final_answer = state.get("final_answer") or {}
    if isinstance(final_answer, dict):
        summary_text = str(final_answer.get("summary", ""))
        code = _extract_error_code_from_text(summary_text)
        if code:
            node_status["synthesis"] = {
                "status": "degraded",
                "error_code": code,
                "detail": summary_text[:240],
            }
    return node_status


def _make_insufficient_input_item(field_name: str, idx: int) -> str:
    if field_name == "revision_plan":
        defaults = [
            "계약서 내 문제 조항을 우선순위별로 목록화하고 수정 방향을 초안으로 작성하세요.",
            "참조 법령/표준 계약서와 대조하여 누락된 필수 조항을 추가 초안으로 작성하세요.",
            "위험 조항별 대안 문구를 법적 근거와 함께 작성하고 상대방과 협의 항목을 선별하세요.",
            "수정 완료 후 체크리스트를 작성하여 모든 조항이 법령 기준에 부합하는지 검증하세요.",
        ]
        return defaults[min(max(idx - 1, 0), len(defaults) - 1)]

    prompts = {
        "clause_analysis": (
            "근거/입력 정보가 부족해 맞춤 조항 분석 항목 생성이 제한됩니다. "
            "계약서 원문, 계약 유형, 당사자 정보를 추가로 제공해 주세요."
        ),
        "risk_findings": (
            "근거/입력 정보가 부족해 맞춤 위험 조항 탐지 항목 생성이 제한됩니다. "
            "계약서 원문, 참조 법령, 주요 우려 사항을 추가로 알려주세요."
        ),
        "revision_plan": (
            "근거/입력 정보가 부족해 수정 계획 세부화가 제한됩니다. "
            "계약서 원문, 협의 가능 범위, 우선순위 조항을 추가로 제공해 주세요."
        ),
    }
    base = prompts.get(
        field_name,
        "근거/입력 정보가 부족합니다. 계약서 원문, 계약 유형, 주요 우려 사항을 추가로 알려주세요.",
    )
    return f"{base} (추가 확인 {idx})"


def enforce_final_answer_policy(
    route: str,
    payload: dict[str, Any],
    rag_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    min_rules = route_minimums(route)
    refs = payload.get("references", [])
    references = _normalize_reference_records(refs if isinstance(refs, list) else [])
    if not references and isinstance(rag_refs, list) and rag_refs:
        references = _normalize_reference_records(rag_refs[:4])

    citation_tag = "[1]" if references else ""
    input_gap_notice = _clean_optional_notice(payload.get("input_gap_notice")) or ""
    for field_name, minimum in min_rules.items():
        items = payload.get(field_name, [])
        normalized_items = [str(item).strip() for item in items] if isinstance(items, list) else []
        normalized_items = [item for item in normalized_items if item]
        if minimum <= 0:
            payload[field_name] = []
            continue
        if len(normalized_items) < minimum:
            if field_name == "revision_plan" and not input_gap_notice:
                input_gap_notice = (
                    "입력 정보가 제한되어 기본 수정 계획을 함께 제시했습니다. "
                    "계약서 원문, 협의 가능 범위, 우선순위 조항을 추가로 입력하면 더 구체화할 수 있습니다."
                )
            for idx in range(len(normalized_items) + 1, minimum + 1):
                normalized_items.append(_make_insufficient_input_item(field_name, idx))
        if citation_tag:
            normalized_items = [_ensure_citation(item, citation_tag) for item in normalized_items]
        payload[field_name] = normalized_items

    payload["input_gap_notice"] = input_gap_notice or None
    payload["references"] = references
    return payload


def _run_tool_loop_structured(
    prompt: str,
    tools: list[BaseTool],
    output_schema: type[TStructured],
    max_steps: int = 4,
    max_tool_rounds: int = 2,
    model_temperature: float = 0.2,
    system_prompt: str | None = None,
) -> TStructured:
    structured, _ = _run_tool_loop_structured_with_trace(
        prompt=prompt,
        tools=tools,
        output_schema=output_schema,
        max_steps=max_steps,
        max_tool_rounds=max_tool_rounds,
        model_temperature=model_temperature,
        system_prompt=system_prompt,
    )
    return structured


def _run_tool_loop_structured_with_trace(
    prompt: str,
    tools: list[BaseTool],
    output_schema: type[TStructured],
    max_steps: int = 4,
    max_tool_rounds: int = 2,
    model_temperature: float = 0.2,
    system_prompt: str | None = None,
) -> tuple[TStructured, list[str]]:
    # Tool-calling is autonomous here:
    # 1) bind_tools exposes available tools to the model
    # 2) model emits tool_calls when it decides tools are needed
    # 3) loop executes tools and returns ToolMessage for iterative reasoning
    llm = get_chat_model(temperature=model_temperature).bind_tools(tools)
    messages = [HumanMessage(content=prompt)]
    if system_prompt:
        messages = [SystemMessage(content=system_prompt), *messages]
    tool_outputs: list[str] = []
    seen_calls: set[str] = set()
    tool_round_count = 0
    for _ in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_round_count += 1
        if tool_round_count > max_tool_rounds:
            messages.append(
                HumanMessage(
                    content=(
                        "도구 호출 라운드가 충분히 수행되었습니다. "
                        "이제 도구 호출을 중단하고 최종 결과만 작성하세요."
                    )
                )
            )
            break

        repeated_detected = False

        for call in response.tool_calls:
            matched_tool = next((tool for tool in tools if tool.name == call["name"]), None)
            args = call.get("args", {})
            signature = f"{call['name']}::{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
            if not matched_tool:
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {call['name']}",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                continue
            if signature in seen_calls:
                repeated_detected = True
                messages.append(
                    ToolMessage(
                        content="Repeated tool+args call detected. Execution skipped.",
                        tool_call_id=call["id"],
                        name=matched_tool.name,
                    )
                )
                continue
            seen_calls.add(signature)

            output = matched_tool.invoke(args)
            output_text = str(output)
            tool_outputs.append(output_text)
            messages.append(
                ToolMessage(
                    content=output_text,
                    tool_call_id=call["id"],
                    name=matched_tool.name,
                )
            )
        if repeated_detected:
            messages.append(
                HumanMessage(
                    content=(
                        "동일한 도구 호출이 반복 감지되었습니다. "
                        "추가 도구 호출 없이 최종 결과를 정리하세요."
                    )
                )
            )
            break

    parser = get_chat_model(temperature=model_temperature).with_structured_output(output_schema)
    structured = parser.invoke(
        [
            *messages,
            HumanMessage(
                content=TOOL_LOOP_FINALIZATION_INSTRUCTION
            ),
        ],
    )
    return structured, tool_outputs


def _needs_citation_rewrite(payload: dict[str, Any], references: list[dict[str, Any]]) -> bool:
    if not references:
        return False
    max_ref = len(references)
    for field_name in ("clause_analysis", "risk_findings", "revision_plan"):
        items = payload.get(field_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not _has_valid_citation(str(item), max_ref=max_ref):
                return True
    return False


def _load_few_shot_bank(
    *,
    agent_name: str,
    default_bank: dict[str, str],
    base_dir: Path,
) -> dict[str, str]:
    """Load few-shot examples from disk with safe fallback to built-in defaults."""
    loaded: dict[str, str] = {}
    for key, default_text in default_bank.items():
        file_path = base_dir / f"{agent_name}_{key}.md"
        try:
            text = file_path.read_text(encoding="utf-8").strip()
            loaded[key] = text or default_text.strip()
        except Exception:
            loaded[key] = default_text.strip()
    return loaded


def build_graph(retriever: HybridRetriever, memory: SessionMemory):
    settings = load_settings()
    supervisor_llm = get_chat_model(temperature=0.0)
    rag_planner_llm = get_chat_model(temperature=0.1)
    plan_llm = get_chat_model(temperature=0.15)
    synthesis_llm = get_chat_model(temperature=0.1)
    clause_system_prompt = build_specialist_system_prompt(
        "Clause Analyzer Agent", "계약서 조항 분석 및 표준 계약 대비 갭 분석에 집중하고 필요 시 도구를 자율 호출하세요."
    )
    risk_system_prompt = build_specialist_system_prompt(
        "Risk Detection Agent", "근거 기반 위험 조항 탐지 및 법적 리스크 코칭에 집중하고 필요 시 도구를 자율 호출하세요."
    )
    advice_system_prompt = build_specialist_system_prompt(
        "Revision Advisor Agent", "근거 기반 수정 계획을 작성하고 필요 시 추가 검색 도구를 자율 호출하세요."
    )
    clause_few_shot_defaults = {
        "근로계약서": """
[Few-shot 예시]
입력:
- 계약서 유형: 근로계약서
- 계약서 요약: "포괄임금제 적용, 근로시간 명시 없음"
출력 스타일:
- 핵심 조항 진단:
  1) 근로기준법 제17조 위반 가능성: 근로시간·임금 항목 미기재
  2) 포괄임금제 유효 요건 미충족(개별 합의서 부재)
  3) 연장·야간·휴일수당 산정 기준 불명확
""".strip(),
        "임대차계약서": """
[Few-shot 예시]
입력:
- 계약서 유형: 임대차계약서
- 계약서 요약: "임대인 일방 해지 조항, 원상복구 범위 불명확"
출력 스타일:
- 핵심 조항 진단:
  1) 주택임대차보호법 제6조 위반 가능성: 일방 해지 요건 미충족
  2) 원상복구 범위가 과도하게 광범위하게 설정됨
  3) 보증금 반환 기한 미명시
""".strip(),
        "nda": """
[Few-shot 예시]
입력:
- 계약서 유형: NDA
- 계약서 요약: "비밀정보 정의 포괄적, 유효기간 영구"
출력 스타일:
- 핵심 조항 진단:
  1) 비밀정보 정의가 지나치게 광범위해 정상 사업 활동 제한 가능
  2) 영구 유효기간은 계약법상 공서양속 위반 논란 가능
  3) 공개예외 조항(법령상 공개) 누락
""".strip(),
    }

    risk_few_shot_defaults = {
        "근로계약서": """
[Few-shot 예시]
입력: 근로계약서 위험 조항 검토
출력 스타일:
- 위험 조항: 포괄임금제 + 경업금지 조항
- 법적 우려: 연장근로수당 미지급, 직업선택 자유 침해
- 대응 방향: 개별 수당 명시 또는 포괄임금제 동의서 별도 작성
""".strip(),
        "임대차계약서": """
[Few-shot 예시]
입력: 임대차계약서 위험 조항 검토
출력 스타일:
- 위험 조항: 임대인 일방 해지, 과도한 원상복구 범위
- 법적 우려: 임차인의 주거 안정권 침해, 과도한 복구비용 청구
- 대응 방향: 해지 조건을 법령 기준으로 한정, 원상복구 범위 명확화
""".strip(),
        "nda": """
[Few-shot 예시]
입력: NDA 위험 조항 검토
출력 스타일:
- 위험 조항: 포괄적 비밀정보 정의 + 무제한 손해배상
- 법적 우려: 영업 자유 제한, 예측 불가능한 손해배상 리스크
- 대응 방향: 비밀정보 범위 구체화, 손해배상 상한 설정
""".strip(),
    }

    few_shot_dir = Path(__file__).resolve().parents[2] / "data" / "prompts" / "few_shots"
    clause_few_shot_bank = _load_few_shot_bank(
        agent_name="clause",
        default_bank=clause_few_shot_defaults,
        base_dir=few_shot_dir,
    )
    risk_few_shot_bank = _load_few_shot_bank(
        agent_name="risk",
        default_bank=risk_few_shot_defaults,
        base_dir=few_shot_dir,
    )

    def _few_shot_key(document_type: str) -> str:
        dtype = (document_type or "").lower()
        if "임대" in dtype or "임차" in dtype or "전세" in dtype:
            return "임대차계약서"
        if "nda" in dtype or "비밀" in dtype or "기밀" in dtype:
            return "nda"
        return "근로계약서"

    def _select_few_shots(bank: dict[str, str], document_type: str) -> str:
        max_examples = max(0, int(settings.few_shot_max_examples))
        if max_examples <= 0:
            return "없음 (운영 비용 제어를 위해 Few-shot 생략)"
        primary = _few_shot_key(document_type)
        ordered = [primary, *[key for key in bank.keys() if key != primary]]
        selected: list[str] = []
        for key in ordered[:max_examples]:
            shot = bank.get(key, "").strip()
            if shot:
                selected.append(shot)
        return "\n\n".join(selected) if selected else "없음"

    def _invoke_structured_with_repair(
        prompt: str,
        schema: type[TStructured],
        base_temperature: float,
        system_prompt: str | None = None,
    ) -> TStructured:
        attempts = max(0, int(STRUCTURED_OUTPUT_RETRY_MAX))
        last_error: Exception | None = None
        current_prompt = prompt
        for attempt in range(attempts + 1):
            try:
                parser = get_chat_model(temperature=max(base_temperature - (attempt * 0.1), 0.0)).with_structured_output(schema)
                if system_prompt:
                    return parser.invoke(
                        [SystemMessage(content=system_prompt), HumanMessage(content=current_prompt)]
                    )
                return parser.invoke(current_prompt)
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                if attempt >= attempts:
                    break
                if STRUCTURED_OUTPUT_REPAIR_ENABLED:
                    current_prompt = (
                        f"{prompt}\n\n"
                        "[JSON_REPAIR]\n"
                        f"{STRUCTURED_JSON_REPAIR_INSTRUCTION}"
                    )
        assert last_error is not None
        raise last_error

    def _format_clause_notes(notes: dict[str, Any]) -> str:
        if not notes:
            return "없음"
        lines: list[str] = []
        for item in notes.get("key_clauses", []):
            lines.append(f"- 핵심 조항 진단: {item}")
        for item in notes.get("problematic_clauses", []):
            lines.append(f"- 문제 조항: {item}")
        for item in notes.get("evidence_snippets", []):
            lines.append(f"- 근거: {item}")
        evidence_map = notes.get("evidence_map", {})
        if isinstance(evidence_map, dict) and evidence_map:
            lines.append(f"- 근거 매핑: {evidence_map}")
        return "\n".join(lines) if lines else str(notes)

    def _format_risk_notes(notes: dict[str, Any]) -> str:
        if not notes:
            return "없음"
        lines: list[str] = []
        for item in notes.get("risk_areas", []):
            lines.append(f"- 위험 조항: {item}")
        for item in notes.get("legal_concerns", []):
            lines.append(f"- 법적 우려: {item}")
        for item in notes.get("mitigation_advice", []):
            lines.append(f"- 대응 방향: {item}")
        for item in notes.get("evidence_snippets", []):
            lines.append(f"- 근거: {item}")
        evidence_map = notes.get("evidence_map", {})
        if isinstance(evidence_map, dict) and evidence_map:
            lines.append(f"- 근거 매핑: {evidence_map}")
        return "\n".join(lines) if lines else str(notes)

    def _format_revision_notes(notes: dict[str, Any]) -> str:
        if not notes:
            return "없음"
        lines: list[str] = []
        for item in notes.get("priorities", []):
            lines.append(f"- 우선순위: {item}")
        for item in notes.get("revision_steps", []):
            lines.append(f"- 수정 단계: {item}")
        for item in notes.get("validation_checks", []):
            lines.append(f"- 검증: {item}")
        for item in notes.get("evidence_snippets", []):
            lines.append(f"- 근거: {item}")
        evidence_map = notes.get("evidence_map", {})
        if isinstance(evidence_map, dict) and evidence_map:
            lines.append(f"- 근거 매핑: {evidence_map}")
        return "\n".join(lines) if lines else str(notes)

    def _extract_tool_payloads(tool_outputs: list[str]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for item in tool_outputs:
            try:
                loaded = json.loads(item)
            except Exception:
                continue
            if isinstance(loaded, dict):
                payloads.append(loaded)
        return payloads

    def _clause_tool_reflected(notes: ClauseNotes, tool_outputs: list[str]) -> bool:
        payloads = _extract_tool_payloads(tool_outputs)
        if not payloads:
            return True
        merged_text = " ".join([*notes.key_clauses, *notes.problematic_clauses]).lower()
        checked = 0
        for payload in payloads:
            if str(payload.get("tool", "")).strip().lower() == "specialist_retrieve":
                continue
            expected_tokens: list[str] = []
            matched = payload.get("matched_keywords", [])
            missing = payload.get("missing_keywords", [])
            score = payload.get("match_score", payload.get("required_match_rate"))
            missing_required = payload.get("missing_required_top", [])
            missing_preferred = payload.get("missing_preferred_top", [])
            if isinstance(matched, list):
                expected_tokens.extend(str(item).lower() for item in matched if str(item).strip())
            if isinstance(missing, list):
                expected_tokens.extend(str(item).lower() for item in missing if str(item).strip())
            if isinstance(missing_required, list):
                expected_tokens.extend(str(item).lower() for item in missing_required if str(item).strip())
            if isinstance(missing_preferred, list):
                expected_tokens.extend(str(item).lower() for item in missing_preferred if str(item).strip())
            if score is not None:
                expected_tokens.extend([str(score), f"{score}%"])
            checked += 1
            if any(token and token in merged_text for token in expected_tokens):
                return True
        return checked == 0

    def _risk_tool_reflected(notes: RiskNotes, tool_outputs: list[str]) -> bool:
        payloads = _extract_tool_payloads(tool_outputs)
        if not payloads:
            return True
        merged_issues = " ".join(notes.risk_areas).lower()
        checked = 0
        for payload in payloads:
            if str(payload.get("tool", "")).strip().lower() == "specialist_retrieve":
                continue
            issues = payload.get("issues", [])
            if not isinstance(issues, list):
                continue
            checked += 1
            for issue in issues:
                q = str(issue).strip().lower()
                if not q:
                    continue
                if q in merged_issues or q[:12] in merged_issues:
                    return True
        return checked == 0

    def _tool_outputs_preview(tool_outputs: list[str], limit: int = 2) -> str:
        if not tool_outputs:
            return "[]"
        shortened = [str(item)[:240] for item in tool_outputs[:limit]]
        return json.dumps(shortened, ensure_ascii=False)

    def _normalize_fallback_reason(reason: Any, error_code: str) -> str:
        text = " ".join(str(reason or "").split()).strip()
        if not text:
            return error_code
        text = re.sub(r"\bErrorCodes\.", "", text)
        if text.startswith(error_code):
            return text
        return f"{error_code}: {text}"

    def _as_string_list(values: list[Any] | None, *, fallback: str, max_items: int = 4) -> list[str]:
        normalized: list[str] = []
        for raw in list(values or []):
            item = " ".join(str(raw or "").split()).strip()
            if item:
                normalized.append(item)
            if len(normalized) >= max_items:
                break
        return normalized if normalized else [fallback]

    def _fallback_rag_snippets(state: AgentState, reason: str, *, max_items: int = 2) -> list[str]:
        reason_line = " ".join(str(reason or "").split()).strip()
        rag_context = " ".join(str(state.get("rag_context", "") or "").split()).strip()
        snippets = _as_string_list(
            [reason_line, rag_context[:300] if rag_context else "RAG 근거 없음"],
            fallback="RAG 근거 없음",
            max_items=max_items,
        )
        return snippets

    def _fallback_clause_notes(state: AgentState, reason: str) -> dict[str, Any]:
        normalized_reason = _normalize_fallback_reason(reason, ErrorCodes.STRUCTURED_OUTPUT_CLAUSE_FALLBACK)
        return {
            "key_clauses": _as_string_list(
                ["구조화 파싱 실패로 최소 진단 결과로 대체되었습니다.", normalized_reason],
                fallback="구조화 파싱 실패로 기본 조항 진단을 제공합니다.",
            ),
            "problematic_clauses": _as_string_list(
                [
                    "임금·근로시간 등 핵심 조항의 법령 기준 부합 여부를 우선 확인",
                    "일방적 불이익 조항(해지·원상복구·손해배상) 점검",
                    "법령상 필수 기재사항 누락 여부 확인",
                    "모호한 표현 및 해석 분쟁 가능성 조항 정리",
                ],
                fallback="법령 기준과 핵심 조항 정합성을 우선 보강",
            ),
            "evidence_snippets": _fallback_rag_snippets(state, normalized_reason, max_items=2),
            "evidence_map": {},
        }

    def _fallback_risk_notes(state: AgentState, reason: str) -> dict[str, Any]:
        normalized_reason = _normalize_fallback_reason(reason, ErrorCodes.STRUCTURED_OUTPUT_RISK_FALLBACK)
        return {
            "risk_areas": _as_string_list(
                [
                    "포괄임금제·경업금지 등 근로자 불이익 조항",
                    "일방적 계약 해지 조항",
                    "과도한 손해배상·위약벌 조항",
                    "비밀정보 범위 과도 설정",
                ],
                fallback="일방에게 불이익한 핵심 위험 조항을 우선 탐지",
            ),
            "legal_concerns": _as_string_list(
                [
                    "관련 법령(근로기준법·주택임대차보호법 등) 위반 가능성",
                    "공서양속·신의칙 위반 소지",
                    "계약 무효·취소 가능성",
                    "분쟁 발생 시 불리한 증거 될 수 있는 조항",
                ],
                fallback="법령 위반 및 계약 분쟁 가능성을 우선 검토",
            ),
            "mitigation_advice": _as_string_list(
                [
                    "문제 조항별 대안 문구를 법령 기준에 맞게 작성",
                    "상대방과 협의 가능한 항목 선별",
                    "전문 법률가 검토 필요 항목 표시",
                    "계약 체결 전 반드시 최종 확인",
                ],
                fallback="위험 조항별 대안 문구 초안을 작성하고 전문가 검토를 받을 것",
            ),
            "evidence_snippets": _fallback_rag_snippets(state, normalized_reason, max_items=2),
            "evidence_map": {},
        }

    def _fallback_revision_notes(state: AgentState, reason: str) -> dict[str, Any]:
        normalized_reason = _normalize_fallback_reason(reason, ErrorCodes.STRUCTURED_OUTPUT_ADVICE_FALLBACK)
        return {
            "priorities": _as_string_list(
                [
                    "법령 위반 가능성이 높은 조항부터 우선 수정",
                    "일방에게 불이익한 조항 대안 문구 작성",
                    "누락된 필수 기재사항 추가",
                ],
                fallback="법령 기준 위반 조항부터 우선 수정",
            ),
            "revision_steps": _as_string_list(
                [
                    "1단계: 문제 조항 목록화 및 우선순위 설정",
                    "2단계: 각 조항별 대안 문구 초안 작성",
                    "3단계: 상대방과 협의 후 최종 문구 확정",
                    "4단계: 전문 법률가 최종 검토 후 서명",
                ],
                fallback="조항별 수정 초안 → 협의 → 법률가 검토 순서로 진행",
            ),
            "validation_checks": _as_string_list(
                [
                    "수정 후 전체 조항의 법령 기준 부합 여부 재확인",
                    "상대방 서명 전 최종 체크리스트 완료 여부 점검",
                    "필수 기재사항 누락 없음 확인",
                ],
                fallback="수정 완료 후 법령 기준 부합 여부를 재점검",
            ),
            "evidence_snippets": _fallback_rag_snippets(state, normalized_reason, max_items=2),
            "evidence_map": {},
        }

    def _route_minimums(route: str) -> dict[str, int]:
        return route_minimums(route)

    def _enforce_final_answer_policy(
        route: str,
        payload: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        rag_refs = state.get("rag_refs", [])
        return enforce_final_answer_policy(
            route=route,
            payload=payload,
            rag_refs=rag_refs if isinstance(rag_refs, list) else [],
        )

    def _normalize_final_answer_by_route(route: str, answer: dict[str, Any]) -> dict[str, Any]:
        return normalize_final_answer_by_route(route, answer)

    def _fallback_final_answer(state: AgentState, reason: str) -> dict[str, Any]:
        refs = state.get("rag_refs") or []
        references = _normalize_reference_records(refs[:4] if isinstance(refs, list) else [])
        if not references:
            references = _normalize_reference_records(
                [
                    {
                        "rank": 1,
                        "source": "RAG 근거 부족: 일반 가이드 기반 결과",
                        "chunk_id": None,
                        "location": "n/a",
                        "score": 0.0,
                        "category": "fallback",
                        "snippet": "RAG 근거 부족",
                    "collected_at": None,
                    "source_url": None,
                    "curator": None,
                    "license": None,
                    }
                ]
            )
        return {
            "summary": f"일부 구조화 단계 실패로 안전 모드 응답을 제공합니다. ({reason})",
            "clause_analysis": [
                "계약서 유형별 필수 기재사항 누락 여부 확인",
                "임금·근로시간·보증금 등 핵심 조항의 법령 기준 부합 여부 점검",
                "일방적 불이익 조항(해지·원상복구·손해배상) 확인",
                "모호한 표현 및 해석 분쟁 가능성 조항 정리",
            ],
            "risk_findings": [
                "포괄임금제·경업금지 등 일방에게 불이익한 조항 탐지",
                "법령(근로기준법·주택임대차보호법 등) 위반 가능성 검토",
                "계약 해지·무효 가능성이 있는 조항 점검",
                "손해배상·위약벌 조항의 적정성 확인",
            ],
            "revision_plan": [
                "법령 위반 가능성이 높은 조항부터 우선 수정 초안 작성",
                "일방에게 불이익한 조항 대안 문구 작성 후 상대방과 협의",
                "누락된 필수 기재사항 추가 및 전문 법률가 검토 요청",
                "수정 완료 후 전체 법령 기준 부합 여부 재확인",
            ],
            "input_gap_notice": "일부 입력/근거 정보가 부족해 기본 계약 검토 가이드를 제공합니다.",
            "references": references,
        }

    def _route_by_supervisor(state: AgentState) -> AgentState:
        router = supervisor_llm.with_structured_output(RouteDecision)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("memory_messages", [])
        ) or "없음"
        contract_text = (state.get("contract_text") or "").strip()
        reference_text = (state.get("reference_text") or "").strip()
        heuristic = heuristic_route_from_query(state.get("user_query", ""))
        if heuristic:
            route, reason = heuristic
            if not contract_text and route == "clause_only":
                route = "advice_only"
                reason = (
                    f"{reason} / 계약서 원문이 없어 clause_only 대신 advice_only로 조정"
                )
            return {"route": route, "routing_reason": reason}
        try:
            decision = router.invoke(
                f"""
너는 Supervisor Planner다. 사용자 요청을 보고 실행 라우트를 고른다.

{build_route_options_block()}

[사용자 요청]
{state['user_query']}

[계약서 유형]
{state['document_type']}

[참조 법령/표준 계약서 제공 여부]
{"제공됨" if reference_text else "미제공"}

[참조 법령/표준 계약서 길이]
{len(reference_text)}

[계약서 원문 제공 여부]
{"제공됨" if contract_text else "미제공"}

[계약서 원문 길이]
{len(contract_text)}

[최근 메모리]
{history_text}

{build_supervisor_routing_rules_block()}
"""
            )
            route = decision.route
            reason = decision.reason
            if not contract_text and route == "clause_only":
                route = "advice_only"
                reason = (
                    f"{reason} / 계약서 원문이 없어 clause_only 대신 advice_only로 조정"
                )
            return {"route": route, "routing_reason": reason}
        except Exception:
            return {"route": "full_review", "routing_reason": "파싱 실패로 full_review 라우트 기본 적용"}

    def supervisor_node(state: AgentState) -> AgentState:
        base_state: AgentState = {
            "memory_messages": memory.get(state["session_id"], limit=8),
        }
        base_state.update(_route_by_supervisor(base_state | state))
        return base_state

    def _infer_role_hint(document_type: str) -> str:
        dtype = (document_type or "").lower()
        if "임대" in dtype or "임차" in dtype or "전세" in dtype:
            return "임대차, 보증금, 임대료, 주택임대차보호법"
        if "nda" in dtype or "비밀" in dtype or "기밀" in dtype:
            return "nda, 비밀정보, 기밀유지, 손해배상"
        if "용역" in dtype or "도급" in dtype or "위탁" in dtype:
            return "용역, 도급, 지적재산권, 납기, 하자보증"
        return "근로계약, 임금, 근로시간, 근로기준법"

    def _category_filter_for_route(route: str) -> set[str] | None:
        route_key = (route or "full_review").lower()
        fallback_category = {"uncategorized"} if settings.allow_uncategorized_in_filter else set()
        if route_key == "clause_only":
            return {"statutes", "standard_contracts", "contract_examples", "contract_upload"} | fallback_category
        if route_key == "risk_only":
            return {"case_guides", "statutes", "standard_contracts"} | fallback_category
        if route_key == "advice_only":
            return {"statutes", "standard_contracts", "case_guides"} | fallback_category
        return None

    def _chunk_inline_text(text: str, chunk_size: int = 500, chunk_overlap: int = 120) -> list[str]:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return []
        if len(cleaned) <= chunk_size:
            return [cleaned]
        step = max(chunk_size - chunk_overlap, 50)
        chunks: list[str] = []
        for start in range(0, len(cleaned), step):
            chunk = cleaned[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(cleaned):
                break
        return chunks

    def _lexical_overlap_score(query: str, text: str) -> float:
        query_tokens = {token for token in re.findall(r"[0-9A-Za-z가-힣]+", query.lower()) if len(token) > 1}
        if not query_tokens:
            return 0.0
        text_tokens = {token for token in re.findall(r"[0-9A-Za-z가-힣]+", text.lower()) if len(token) > 1}
        if not text_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        return min(1.0, overlap)

    def _ephemeral_input_hits(
        query: str,
        reference_text: str,
        contract_text: str,
        ref_base_score: float,
        contract_base_score: float,
        overlap_weight: float,
        top_k: int = 2,
    ) -> list[SearchHit]:
        ephemeral_hits: list[SearchHit] = []
        safe_overlap_weight = max(0.0, min(overlap_weight, 1.0))

        def _append_hits(raw_text: str, source: str, source_type: str, base_score: float, start_chunk_id: int) -> None:
            for idx, chunk in enumerate(_chunk_inline_text(raw_text)[:4], start=1):
                overlap_score = _lexical_overlap_score(query, chunk)
                if overlap_score <= 0.0:
                    continue
                score = round(base_score + (overlap_score * safe_overlap_weight), 4)
                ephemeral_hits.append(
                    SearchHit(
                        content=chunk,
                        source=source,
                        score=score,
                        metadata={
                            "category": "standard_contracts" if source_type == "reference_upload" else "contract_upload",
                            "source_type": source_type,
                            "location": f"inline_chunk={idx}",
                            "chunk_id": start_chunk_id - idx,
                        },
                    )
                )

        _append_hits(
            reference_text,
            "uploaded_reference_text",
            "reference_upload",
            base_score=ref_base_score,
            start_chunk_id=-1000,
        )
        _append_hits(
            contract_text,
            "uploaded_contract_text",
            "contract_upload",
            base_score=contract_base_score,
            start_chunk_id=-2000,
        )
        ephemeral_hits.sort(key=lambda item: item.score, reverse=True)
        return ephemeral_hits[:top_k]

    def _location_label(metadata: dict[str, Any]) -> str:
        if "location" in metadata:
            return str(metadata["location"])
        if "page_number" in metadata:
            return f"page={metadata['page_number']}"
        if "paragraph_number" in metadata:
            return f"paragraph={metadata['paragraph_number']}"
        if "sheet_name" in metadata and "row_number" in metadata:
            return f"sheet={metadata['sheet_name']}, row={metadata['row_number']}"
        if "row_number" in metadata:
            return f"row={metadata['row_number']}"
        return "n/a"

    @tool("specialist_retrieve")
    def specialist_retrieve(
        query: str,
        route_hint: str = "full",
        top_k: int = 3,
        broaden: bool = False,
    ) -> str:
        """Specialist RAG retrieval tool for node-level autonomous evidence expansion."""
        safe_k = max(1, min(int(top_k), 6))
        route_key = (route_hint or "full").strip().lower()
        retrieval_source_cap = (
            settings.retrieval_max_chunks_per_file
            if not settings.rerank_enabled
            else 0
        )
        category_filter = None if broaden else _category_filter_for_route(route_key)
        hits = retriever.search(
            query,
            top_k=safe_k,
            category_filter=category_filter,
            max_chunks_per_file=retrieval_source_cap,
        )
        if (not hits) and category_filter:
            hits = retriever.search(
                query,
                top_k=safe_k,
                category_filter=None,
                max_chunks_per_file=retrieval_source_cap,
            )
        payload = [
            {
                "rank": i,
                "source": hit.source,
                "score": hit.score,
                "category": hit.metadata.get("category"),
                "chunk_id": hit.metadata.get("chunk_id"),
                "location": _location_label(hit.metadata),
                "snippet": hit.content[:180].replace("\n", " "),
            }
            for i, hit in enumerate(hits, start=1)
        ]
        return json.dumps(
            {
                "tool": "specialist_retrieve",
                "query": query,
                "route_hint": route_key,
                "hits": payload,
            },
            ensure_ascii=False,
        )

    def _specialist_local_recovery(
        state: AgentState,
        specialist: Literal["clause", "risk", "advice"],
    ) -> tuple[str, list[dict[str, Any]], bool]:
        base_context = str(state.get("rag_context", "") or "")
        base_refs = _normalize_reference_records(
            state.get("rag_refs", []) if isinstance(state.get("rag_refs"), list) else []
        )
        if not state.get("rag_low_confidence"):
            return base_context, base_refs, False

        settings = load_settings()
        retrieval_source_cap = (
            settings.retrieval_max_chunks_per_file
            if not settings.rerank_enabled
            else 0
        )
        route = str(state.get("route", "full_review"))
        route_categories = _category_filter_for_route(route)
        role_hint = _infer_role_hint(state["document_type"])
        specialist_hint_map = {
            "clause": "계약서 조항 분석 핵심 항목",
            "risk": "위험 조항 탐지 법적 리스크",
            "advice": "수정 계획 우선순위 및 검증",
        }
        specialist_hint = specialist_hint_map.get(specialist, "계약 검토 핵심")
        reference_text = (state.get("reference_text") or "").strip()
        contract_text = (state.get("contract_text") or "").strip()
        query_candidates = [
            f"{state['document_type']} {specialist_hint} {state['user_query']}",
            f"{state['document_type']} {state['user_query']}",
        ]
        if reference_text:
            query_candidates.append(f"표준 계약 요건 {reference_text[:220]}")
        if specialist == "risk" and contract_text:
            query_candidates.append(f"계약서 위험 조항 탐지 {contract_text[:220]}")
        if specialist == "advice":
            query_candidates.append(f"수정 우선순위 단계별 검증 {state['user_query']}")

        merged_hits: dict[str, SearchHit] = {}
        for candidate in query_candidates[:3]:
            for hit in retriever.search(
                candidate,
                top_k=3,
                category_filter=None,
                max_chunks_per_file=retrieval_source_cap,
            ):
                key = f"{hit.source}::{hit.content[:120]}"
                current = merged_hits.get(key)
                if not current or hit.score > current.score:
                    merged_hits[key] = hit

        ranked_hits = rerank_hits(
            hits=list(merged_hits.values()),
            query=f"{state['document_type']} {state['user_query']}",
            role_hint=role_hint,
            route_categories=route_categories,
            top_k=3,
            max_per_source=settings.rerank_max_per_source,
            provider=settings.rerank_provider,
            enabled=settings.rerank_enabled,
        )
        if not ranked_hits:
            return base_context, base_refs, True

        top_score = ranked_hits[0].score
        low_confidence = top_score < settings.rag_evidence_score_threshold
        extra_context_lines: list[str] = []
        extra_refs: list[dict[str, Any]] = []
        for hit in ranked_hits:
            extra_context_lines.append(
                f"[{specialist}-recovery] ({hit.source}, score={hit.score})\n{hit.content}"
            )
            extra_refs.append(
                {
                    "source": hit.source,
                    "score": hit.score,
                    "category": hit.metadata.get("category"),
                    "chunk_id": hit.metadata.get("chunk_id"),
                    "location": _location_label(hit.metadata),
                    "snippet": hit.content[:120].replace("\n", " "),
                    "score_breakdown": hit.metadata.get("score_breakdown"),
                    "collected_at": _clean_optional_notice(hit.metadata.get("collected_at")),
                    "source_url": _clean_optional_notice(hit.metadata.get("source_url")),
                    "curator": _clean_optional_notice(hit.metadata.get("curator")),
                    "license": _clean_optional_notice(hit.metadata.get("license")),
                }
            )
        merged_context = base_context
        if extra_context_lines:
            merged_context = (
                f"{base_context}\n\n[specialist-local-recovery] "
                f"{specialist} 노드가 근거 보강을 위해 추가 검색을 수행했습니다.\n"
                + "\n\n".join(extra_context_lines)
            ).strip()
        merged_refs = _merge_reference_records(base_refs, extra_refs, limit=8)
        return merged_context, merged_refs, low_confidence

    def _specialist_decision(
        state: AgentState,
        specialist: Literal["clause", "risk", "advice"],
        has_contract_text: bool,
        reference_text: str,
    ) -> dict[str, Any]:
        fallback: dict[str, Any] = {
            "need_tool_call": specialist in {"clause", "risk"} and has_contract_text,
            "focus_areas": [],
            "omit_areas": [],
            "reason": "휴리스틱 기본 결정",
        }
        try:
            planner = get_chat_model(temperature=0.0).with_structured_output(SpecialistDecision)
            decision = planner.invoke(
                f"""
너는 {specialist} specialist planner다.
이번 요청에서 무엇을 강조/생략할지 결정하라.

specialist={specialist}
user_query={state['user_query']}
document_type={state['document_type']}
route={state.get('route', 'full_review')}
has_contract_text={"yes" if has_contract_text else "no"}
has_reference_text={"yes" if bool(reference_text) else "no"}
rag_low_confidence={state.get('rag_low_confidence', False)}

규칙:
- clause/risk는 계약서 원문이 없으면 need_tool_call=false 우선.
- advice는 기본적으로 need_tool_call=false를 우선하되, rag_low_confidence=true이면 true를 선택할 수 있다.
- focus_areas는 2~4개, omit_areas는 최대 3개.
"""
            )
            return decision.model_dump()
        except Exception:
            return fallback

    def rag_node(state: AgentState) -> AgentState:
        planner = rag_planner_llm.with_structured_output(RagPlan)
        settings = load_settings()
        role_hint = _infer_role_hint(state["document_type"])
        route = state.get("route", "full_review")
        retrieval_top_k = 2 if route == "advice_only" else 4
        query_limit = 2 if route == "advice_only" else 3
        retrieval_source_cap = (
            settings.retrieval_max_chunks_per_file
            if not settings.rerank_enabled
            else 0
        )
        reference_text = (state.get("reference_text") or "").strip()
        contract_text = (state.get("contract_text") or "").strip()
        category_filter = _category_filter_for_route(route)
        try:
            rag_plan = planner.invoke(
                f"""
너는 RAG Agent다. 검색 품질을 높이기 위해 쿼리를 2~3개로 재작성하라.
route={route}
user_query={state['user_query']}
document_type={state['document_type']}
reference_text_present={"yes" if reference_text else "no"}
reference_text_preview={reference_text[:400] if reference_text else "none"}
source_hint는 계약서 유형 연관 법령/키워드로 짧게 작성하라.
"""
            )
            query_candidates = [state["user_query"], *rag_plan.rewritten_queries]
            source_hint = rag_plan.source_hint or role_hint
        except Exception:
            query_candidates = [state["user_query"], f"{state['document_type']} {state['user_query']}"]
            source_hint = role_hint

        merged_hits: dict[str, SearchHit] = {}
        for candidate in query_candidates[:query_limit]:
            filtered_hits = retriever.search(
                candidate,
                top_k=retrieval_top_k,
                category_filter=category_filter,
                max_chunks_per_file=retrieval_source_cap,
            )
            # Fallback: if route-aware filter over-prunes results, retry without filter.
            candidate_hits = (
                retriever.search(
                    candidate,
                    top_k=retrieval_top_k,
                    category_filter=None,
                    max_chunks_per_file=retrieval_source_cap,
                )
                if (category_filter and not filtered_hits)
                else filtered_hits
            )
            for hit in candidate_hits:
                key = f"{hit.source}::{hit.content[:120]}"
                current = merged_hits.get(key)
                if not current or hit.score > current.score:
                    merged_hits[key] = hit

        # Blend uploaded reference/contract text as ephemeral evidence for gap-analysis grounding.
        for hit in _ephemeral_input_hits(
            query=state["user_query"],
            reference_text=reference_text,
            contract_text=contract_text,
            ref_base_score=settings.ephemeral_ref_base_score,
            contract_base_score=settings.ephemeral_contract_base_score,
            overlap_weight=settings.ephemeral_overlap_weight,
            top_k=2,
        ):
            key = f"{hit.source}::{hit.content[:120]}"
            current = merged_hits.get(key)
            if not current or hit.score > current.score:
                merged_hits[key] = hit

        ranked_hits = rerank_hits(
            hits=list(merged_hits.values()),
            query=state["user_query"],
            role_hint=source_hint,
            route_categories=category_filter,
            top_k=retrieval_top_k,
            max_per_source=settings.rerank_max_per_source,
            provider=settings.rerank_provider,
            enabled=settings.rerank_enabled,
        )
        top_score = ranked_hits[0].score if ranked_hits else 0.0
        low_confidence = (not ranked_hits) or (top_score < settings.rag_evidence_score_threshold)
        if low_confidence:
            # Agent-local recovery policy: in low-confidence mode, broaden query/filter once.
            recovery_query = f"{state['document_type']} 핵심 조항 {state['user_query']}"
            recovery_hits = retriever.search(
                recovery_query,
                top_k=retrieval_top_k + 1,
                category_filter=None,
                max_chunks_per_file=retrieval_source_cap,
            )
            for hit in recovery_hits:
                key = f"{hit.source}::{hit.content[:120]}"
                current = merged_hits.get(key)
                if not current or hit.score > current.score:
                    merged_hits[key] = hit
            ranked_hits = rerank_hits(
                hits=list(merged_hits.values()),
                query=state["user_query"],
                role_hint=source_hint,
                route_categories=category_filter,
                top_k=retrieval_top_k,
                max_per_source=settings.rerank_max_per_source,
                provider=settings.rerank_provider,
                enabled=settings.rerank_enabled,
            )
            top_score = ranked_hits[0].score if ranked_hits else 0.0
            low_confidence = (not ranked_hits) or (top_score < settings.rag_evidence_score_threshold)
        context_parts: list[str] = []
        refs: list[dict[str, Any]] = []
        for i, hit in enumerate(ranked_hits, start=1):
            context_parts.append(f"[{i}] ({hit.source}, score={hit.score})\n{hit.content}")
            refs.append(
                {
                    "rank": i,
                    "source": hit.source,
                    "score": hit.score,
                    "category": hit.metadata.get("category"),
                    "chunk_id": hit.metadata.get("chunk_id"),
                    "location": _location_label(hit.metadata),
                    "snippet": hit.content[:120].replace("\n", " "),
                    "score_breakdown": hit.metadata.get("score_breakdown"),
                }
            )

        if not context_parts:
            context_parts = ["검색 결과가 부족하여 일반적인 법률 가이드를 기반으로 응답합니다."]
        if low_confidence:
            context_parts.append(
                f"[rag-safety] 근거 점수가 임계치({settings.rag_evidence_score_threshold:.2f}) 미만이므로 "
                "보수적 표현과 조건부 제안을 우선하세요."
            )

        rag_context = "\n\n".join(context_parts)
        rag_context += (
            f"\n\n[rag-agent-note] route={route}, source_hint={source_hint}, "
            f"category_filter={sorted(category_filter) if category_filter else 'none'}, "
            f"top_score={top_score:.4f}, low_confidence={low_confidence}"
        )
        return {
            "rag_context": rag_context,
            "rag_refs": refs,
            "rag_low_confidence": low_confidence,
        }

    def advice_node(state: AgentState) -> AgentState:
        reference_text = (state.get("reference_text") or "").strip()
        has_contract_text = bool((state.get("contract_text") or "").strip())
        specialist_rag_context, specialist_rag_refs, specialist_low_confidence = _specialist_local_recovery(
            state,
            "advice",
        )
        specialist_decision = _specialist_decision(
            state=state,
            specialist="advice",
            has_contract_text=has_contract_text,
            reference_text=reference_text,
        )
        clause_notes = _format_clause_notes(state.get("clause_notes", {}))
        risk_notes = _format_risk_notes(state.get("risk_notes", {}))
        try:
            structured, _ = _run_tool_loop_structured_with_trace(
                prompt=f"""
당신은 Revision Advisor Agent입니다.
{build_common_policy_block(include_plan_citation=True)}

[사용자 요청]
{state['user_query']}

[계약서 유형]
{state['document_type']}

[Supervisor route]
{state.get('route', 'full_review')}

[참조 법령/표준 계약서]
{reference_text or '미제공'}

[Clause Analyzer 결과]
{clause_notes if state.get('clause_notes') else '이번 라우트에서는 Clause Analyzer를 생략했습니다.'}

[Risk Detection 결과]
{risk_notes if state.get('risk_notes') else '이번 라우트에서는 Risk Detection을 생략했습니다.'}

[RAG 근거]
{specialist_rag_context}

[Revision Advisor 의사결정]
{json.dumps(specialist_decision, ensure_ascii=False)}

요구사항:
- priorities, revision_steps, validation_checks를 균형 있게 작성
- evidence_map 필수: key는 수정 항목, value는 근거 번호 목록(예: [1,2])
- 근거가 부족하거나 모호하면 specialist_retrieve 도구를 호출해 추가 근거를 확보하세요.
""",
                tools=[specialist_retrieve],
                output_schema=RevisionNotes,
                model_temperature=0.15,
                system_prompt=advice_system_prompt,
            )
            payload = _sanitize_notes_evidence_map(
                structured.model_dump(),
                max_ref=len(specialist_rag_refs),
            )
            return {
                "revision_notes": payload,
                "rag_context": specialist_rag_context,
                "rag_refs": specialist_rag_refs,
                "rag_low_confidence": specialist_low_confidence,
            }
        except Exception as exc:
            return {
                "revision_notes": _fallback_revision_notes(
                    state, f"{ErrorCodes.STRUCTURED_OUTPUT_ADVICE_FALLBACK}: {exc}"
                )
            }

    def clause_node(state: AgentState) -> AgentState:
        has_contract_text = bool((state.get("contract_text") or "").strip())
        reference_text = (state.get("reference_text") or "").strip()
        specialist_decision = _specialist_decision(
            state=state,
            specialist="clause",
            has_contract_text=has_contract_text,
            reference_text=reference_text,
        )
        specialist_rag_context, specialist_rag_refs, specialist_low_confidence = _specialist_local_recovery(
            state,
            "clause",
        )
        prompt = f"""
당신은 Clause Analyzer Agent입니다.
{build_common_policy_block()}
계약서 유형: {state['document_type']}
사용자 요청: {state['user_query']}
참조 법령/표준 계약서:
{reference_text or '참조 텍스트가 제공되지 않았습니다.'}
계약서 원문:
{state['contract_text'] or '계약서 원문이 제공되지 않았습니다.'}

RAG 근거:
{specialist_rag_context}

[Clause Analyzer 의사결정]
{json.dumps(specialist_decision, ensure_ascii=False)}

지시:
1) clause_keyword_match_score 도구를 활용해 필수 조항 커버리지를 반영하세요.
1-1) 참조 텍스트가 있으면 contract_reference_gap_score 도구로 필수/우대 조항 매칭률과 누락 항목 top-N을 반영하세요.
2) 도구 호출이 필요 없다고 판단되면 즉시 최종 결과를 작성하세요.
3) 아래 Few-shot 스타일을 참고해 동일한 형식으로 작성하세요.
4) 근거 문장/출처 기반으로만 작성하고, 근거 없는 단정은 금지합니다.
5) 최종 출력은 반드시 ClauseNotes 스키마를 따르세요.
6) 로컬 정책: 계약서 원문이 없으면 개별 조항 교정보다 "계약서 유형별 필수 조항 체크리스트" 중심으로 작성하세요.
7) 참조 텍스트가 있으면 참조 요건과 계약서 간 갭을 항목별로 명시적으로 비교하세요.
8) evidence_map 필수: key는 조항/진단 문장, value는 관련 근거 번호 목록(예: [1,2])입니다.
9) 도구를 호출했다면 도구 결과(JSON)의 score/keywords/missing_required_top을 최소 1회 이상 key_clauses 또는 problematic_clauses에 반영하세요.
10) 근거가 부족하거나 모호하면 specialist_retrieve 도구로 추가 검색을 수행하세요.

[Few-shot]
{_select_few_shots(clause_few_shot_bank, state['document_type'])}
"""
        try:
            if not has_contract_text:
                structured = _invoke_structured_with_repair(
                    prompt=f"""
계약서 원문이 없는 상황입니다.
아래 조건을 반영해 ClauseNotes를 작성하세요.
- 개별 계약 단정 금지
- 계약서 유형 기준 필수 조항 체크리스트/갭 분석 우선 제시
- 한국어로만 작성
- evidence_map 필수: 각 조항 항목을 근거 번호([1],[2]...)와 매핑

[계약서 유형]
{state['document_type']}

[참조 법령/표준 계약서]
{reference_text or '없음'}

[사용자 요청]
{state['user_query']}

[RAG 근거]
{specialist_rag_context}
""",
                    schema=ClauseNotes,
                    base_temperature=0.1,
                )
            else:
                if not bool(specialist_decision.get("need_tool_call", True)):
                    structured = _invoke_structured_with_repair(
                        prompt=f"{prompt}\n\n[로컬 정책]\n이번 요청은 도구 호출 없이 직접 분석 결과를 작성하세요.",
                        schema=ClauseNotes,
                        base_temperature=0.1,
                    )
                    payload = _sanitize_notes_evidence_map(
                        structured.model_dump(),
                        max_ref=len(specialist_rag_refs),
                    )
                    return {
                        "clause_notes": payload,
                        "rag_context": specialist_rag_context,
                        "rag_refs": specialist_rag_refs,
                        "rag_low_confidence": specialist_low_confidence,
                    }
                structured, tool_outputs = _run_tool_loop_structured_with_trace(
                    prompt=prompt,
                    tools=[clause_keyword_match_score, contract_reference_gap_score, specialist_retrieve],
                    output_schema=ClauseNotes,
                    model_temperature=0.15,
                    system_prompt=clause_system_prompt,
                )
                if tool_outputs and (not _clause_tool_reflected(structured, tool_outputs)):
                    retry_prompt = (
                        f"{prompt}\n\n"
                        "[검증 피드백]\n"
                        "직전 결과에서 도구 결과 반영이 명확히 감지되지 않았습니다.\n"
                        f"도구 결과 요약: {_tool_outputs_preview(tool_outputs)}\n"
                        "반드시 도구 결과의 score/keywords를 key_clauses 또는 problematic_clauses에 최소 1회 포함하세요."
                    )
                    structured, _ = _run_tool_loop_structured_with_trace(
                        prompt=retry_prompt,
                        tools=[clause_keyword_match_score, contract_reference_gap_score, specialist_retrieve],
                        output_schema=ClauseNotes,
                        model_temperature=0.15,
                        system_prompt=clause_system_prompt,
                    )
            payload = _sanitize_notes_evidence_map(
                structured.model_dump(),
                max_ref=len(specialist_rag_refs),
            )
            return {
                "clause_notes": payload,
                "rag_context": specialist_rag_context,
                "rag_refs": specialist_rag_refs,
                "rag_low_confidence": specialist_low_confidence,
            }
        except Exception as exc:
            return {
                "clause_notes": _fallback_clause_notes(
                    state, f"{ErrorCodes.STRUCTURED_OUTPUT_CLAUSE_FALLBACK}: {exc}"
                )
            }

    def risk_node(state: AgentState) -> AgentState:
        has_contract_text = bool((state.get("contract_text") or "").strip())
        reference_text = (state.get("reference_text") or "").strip()
        specialist_decision = _specialist_decision(
            state=state,
            specialist="risk",
            has_contract_text=has_contract_text,
            reference_text=reference_text,
        )
        specialist_rag_context, specialist_rag_refs, specialist_low_confidence = _specialist_local_recovery(
            state,
            "risk",
        )
        prompt = f"""
당신은 Risk Detection Agent입니다.
{build_common_policy_block()}
계약서 유형: {state['document_type']}
사용자 요청: {state['user_query']}
참조 법령/표준 계약서:
{reference_text or '참조 텍스트가 제공되지 않았습니다.'}

RAG 근거:
{specialist_rag_context}

[Risk Detection 의사결정]
{json.dumps(specialist_decision, ensure_ascii=False)}

지시:
1) legal_issue_bank 도구를 활용해 유형별 주요 법적 이슈 세트를 참고하세요.
2) 도구 호출이 필요 없다고 판단되면 즉시 최종 결과를 작성하세요.
3) 아래 Few-shot 스타일을 참고해 동일한 형식으로 작성하세요.
4) 근거 문장/출처 기반으로만 작성하고, 근거 없는 단정은 금지합니다.
5) 최종 출력은 반드시 RiskNotes 스키마를 따르세요.
6) 로컬 정책: 계약서 원문이 없으면 개별 조항 단정 대신 계약서 유형별 공통 위험 패턴/대응 프레임워크 중심으로 작성하세요.
7) 참조 텍스트가 있으면 참조 요건을 기준으로 위험 우선순위를 조정하세요.
8) evidence_map 필수: key는 위험/우려 문장, value는 관련 근거 번호 목록(예: [1,2])입니다.
9) 도구를 호출했다면 도구 결과(JSON)의 issues를 최소 1회 이상 risk_areas에 반영하세요.
10) 근거가 부족하거나 모호하면 specialist_retrieve 도구로 추가 검색을 수행하세요.

[Few-shot]
{_select_few_shots(risk_few_shot_bank, state['document_type'])}
"""
        try:
            if not has_contract_text:
                structured = _invoke_structured_with_repair(
                    prompt=f"""
계약서 원문이 없는 상황입니다.
아래 조건을 반영해 RiskNotes를 작성하세요.
- 개별 계약 단정 금지
- 계약서 유형 공통 위험 패턴, 법적 우려, 대응 방향을 우선 제시
- 한국어로만 작성
- evidence_map 필수: 각 위험/우려 항목을 근거 번호([1],[2]...)와 매핑

[계약서 유형]
{state['document_type']}

[참조 법령/표준 계약서]
{reference_text or '없음'}

[사용자 요청]
{state['user_query']}

[RAG 근거]
{specialist_rag_context}
""",
                    schema=RiskNotes,
                    base_temperature=0.2,
                )
            else:
                if not bool(specialist_decision.get("need_tool_call", True)):
                    structured = _invoke_structured_with_repair(
                        prompt=f"{prompt}\n\n[로컬 정책]\n이번 요청은 도구 호출 없이 직접 분석 결과를 작성하세요.",
                        schema=RiskNotes,
                        base_temperature=0.2,
                    )
                    payload = _sanitize_notes_evidence_map(
                        structured.model_dump(),
                        max_ref=len(specialist_rag_refs),
                    )
                    return {
                        "risk_notes": payload,
                        "rag_context": specialist_rag_context,
                        "rag_refs": specialist_rag_refs,
                        "rag_low_confidence": specialist_low_confidence,
                    }
                structured, tool_outputs = _run_tool_loop_structured_with_trace(
                    prompt=prompt,
                    tools=[legal_issue_bank, specialist_retrieve],
                    output_schema=RiskNotes,
                    model_temperature=0.25,
                    system_prompt=risk_system_prompt,
                )
                if tool_outputs and (not _risk_tool_reflected(structured, tool_outputs)):
                    retry_prompt = (
                        f"{prompt}\n\n"
                        "[검증 피드백]\n"
                        "직전 결과에서 도구 결과 반영이 명확히 감지되지 않았습니다.\n"
                        f"도구 결과 요약: {_tool_outputs_preview(tool_outputs)}\n"
                        "반드시 도구 결과의 issues 중 최소 1개를 risk_areas에 반영하세요."
                    )
                    structured, _ = _run_tool_loop_structured_with_trace(
                        prompt=retry_prompt,
                        tools=[legal_issue_bank, specialist_retrieve],
                        output_schema=RiskNotes,
                        model_temperature=0.25,
                        system_prompt=risk_system_prompt,
                    )
            payload = _sanitize_notes_evidence_map(
                structured.model_dump(),
                max_ref=len(specialist_rag_refs),
            )
            return {
                "risk_notes": payload,
                "rag_context": specialist_rag_context,
                "rag_refs": specialist_rag_refs,
                "rag_low_confidence": specialist_low_confidence,
            }
        except Exception as exc:
            return {
                "risk_notes": _fallback_risk_notes(
                    state, f"{ErrorCodes.STRUCTURED_OUTPUT_RISK_FALLBACK}: {exc}"
                )
            }

    def synthesis_node(state: AgentState) -> AgentState:
        route = state.get("route", "full_review")
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("memory_messages", [])
        )
        clause_notes = (
            _format_clause_notes(state.get("clause_notes", {}))
            if state.get("clause_notes")
            else "이번 라우트에서는 Clause Analyzer를 생략했습니다."
        )
        risk_notes = (
            _format_risk_notes(state.get("risk_notes", {}))
            if state.get("risk_notes")
            else "이번 라우트에서는 Risk Detection을 생략했습니다."
        )
        revision_notes = (
            _format_revision_notes(state.get("revision_notes", {}))
            if state.get("revision_notes")
            else "이번 라우트에서는 Revision Advisor를 생략했습니다."
        )
        synthesis_prompt = f"""
당신은 LegalPilot AI의 Supervisor입니다.
아래 정보를 통합하여 반드시 JSON 스키마에 맞는 결과를 생성하세요.
{PROMPT_RULE_RESULT_ONLY}
{PROMPT_RULE_ALL_FIELDS_KOREAN}

[사용자 요청]
{state['user_query']}

[계약서 유형]
{state['document_type']}

[참조 법령/표준 계약서]
{state.get('reference_text', '') or '미제공'}

[최근 대화 메모리]
{history_text or '없음'}

[Supervisor 라우팅]
- route: {route}
- reason: {state.get('routing_reason', '기본 라우트')}

[Clause Analyzer 결과]
{clause_notes}

[Risk Detection 결과]
{risk_notes}

[Revision Advisor 결과]
{revision_notes}

[RAG 근거]
{state.get('rag_context', 'RAG 단계를 생략했습니다.')}

[RAG 신뢰도 모드]
{"근거 부족 모드(보수적 표현 우선)" if state.get('rag_low_confidence') else "일반 모드"}

[참고 출처]
{state.get('rag_refs', [])}

[필수 규칙]
- summary는 4문장 이내.
- advice_only에서도 summary는 1~2문장으로 반드시 작성.
- route가 full_review 또는 advice_only인 경우 Revision Advisor 결과를 우선 반영해 revision_plan 작성.
- references는 임의 생성하지 말고 비워두거나 제공된 구조를 유지하세요(후처리에서 rag_refs 기준으로 정규화).
- references가 있을 때만 액션 불릿에 유효 citation(`[1]`~`[{len(state.get("rag_refs", []) or [])}]`)을 표기하고, references가 비어 있으면 citation을 생략하세요.
- 책임 한계 고지 템플릿을 summary에 포함:
  "본 검토 의견은 법률 자문이 아니며, 실제 계약 체결 전 반드시 전문 법률가의 확인이 필요합니다."

[권장 규칙]
- input_gap_notice는 필요한 경우에만 짧게 작성하고, revision_plan에는 구체적 수정 액션만 포함하세요.
- route별 최소 개수/빈 배열 정책은 후처리에서 강제되므로 우선 의미 일관성에 집중.
"""
        try:
            answer = _invoke_structured_with_repair(
                prompt=synthesis_prompt,
                schema=FinalAnswer,
                base_temperature=0.1,
            )
            draft_payload = answer.model_dump()
            if _needs_citation_rewrite(draft_payload, state.get("rag_refs", []) or []):
                answer = _invoke_structured_with_repair(
                    prompt=f"""
기존 초안에서 일부 불릿이 references 번호와 연결되지 않았습니다.
아래 정보를 유지하면서 각 불릿에 최소 1개의 유효 citation([1]~[{len(state.get("rag_refs", []) or [])}])을 연결해 1회 재작성하세요.
생각 과정을 노출하지 말고 JSON 결과만 출력하세요.

[기존 초안]
{json.dumps(draft_payload, ensure_ascii=False)}

[참고 출처]
{state.get('rag_refs', [])}
""",
                    schema=FinalAnswer,
                    base_temperature=0.0,
                )
                draft_payload = answer.model_dump()
            normalized = _normalize_final_answer_by_route(route, draft_payload)
            normalized = _enforce_final_answer_policy(route, normalized, state)
            return {"final_answer": normalized}
        except Exception as exc:
            fallback = _fallback_final_answer(
                state, f"{ErrorCodes.STRUCTURED_OUTPUT_SYNTHESIS_FALLBACK}: {exc}"
            )
            normalized = _normalize_final_answer_by_route(route, fallback)
            normalized = _enforce_final_answer_policy(route, normalized, state)
            return {"final_answer": normalized}

    def route_after_supervisor(state: AgentState) -> str:
        return "rag"

    def route_after_rag(state: AgentState) -> str:
        route = state.get("route", "full_review")
        if route in {"full_review", "clause_only"}:
            return "clause"
        if route == "risk_only":
            return "risk"
        if route == "advice_only":
            return "advice"
        return "synthesis"

    def route_after_clause(state: AgentState) -> str:
        return "risk" if state.get("route", "full_review") == "full_review" else "synthesis"

    def route_after_risk(state: AgentState) -> str:
        return "advice" if state.get("route", "full_review") == "full_review" else "synthesis"

    def route_after_advice(state: AgentState) -> str:
        return "synthesis"

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("clause", clause_node)
    graph.add_node("risk", risk_node)
    graph.add_node("advice", advice_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"rag": "rag"},
    )
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {"clause": "clause", "risk": "risk", "advice": "advice", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "clause",
        route_after_clause,
        {"risk": "risk", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {"advice": "advice", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "advice",
        route_after_advice,
        {"synthesis": "synthesis"},
    )
    graph.add_edge("synthesis", END)
    # Runtime checkpointer only (in-process): survives node-to-node retries within the same process.
    # Cross-restart persistence is handled separately by session memory + final-answer cache file.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


class LegalPilotService:
    """High-level service that owns retriever, workflow and memory."""

    def __init__(self) -> None:
        settings = load_settings()
        self.settings = settings
        if settings.state_store_backend != "file":
            # Storage backend switch scaffold: non-file backends are reserved for future integration.
            # Current stable implementation keeps file-based stores to preserve behavior.
            print(
                "[WARN] STATE_STORE_BACKEND is set to "
                f"'{settings.state_store_backend}', but only 'file' backend is currently implemented. "
                "Falling back to file-based persistence."
            )
        self.memory = SessionMemory(
            storage_path=settings.index_dir / "session_memory.json",
            max_sessions=settings.memory_max_sessions,
            ttl_seconds=settings.memory_ttl_seconds,
            persist_enabled=settings.session_memory_persist_enabled,
            pii_mask_enabled=settings.session_memory_pii_mask_enabled,
        )
        self.final_answer_cache_path = settings.index_dir / "final_answer_cache.json"
        self.final_answer_cache_lock = FileLock(str(settings.index_dir / "final_answer_cache.json.lock"))
        # Legacy path for backward compatibility with old cache file name.
        self.legacy_final_answer_cache_path = settings.index_dir / "graph_state_cache.json"
        # NOTE: This is a final-answer payload cache (normalized ChatResponse),
        # not a full LangGraph node-by-node checkpoint store.
        self.retriever = HybridRetriever.build()
        self.graph = build_graph(retriever=self.retriever, memory=self.memory)

    def _request_signature(self, req: ChatRequest) -> str:
        raw = "||".join(
            [
                req.session_id,
                req.user_query,
                req.document_type,
                req.contract_text,
                req.reference_text,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _should_bypass_cache(self, req: ChatRequest) -> bool:
        if not self.settings.final_answer_cache_enabled:
            return True
        if not self.settings.final_answer_cache_bypass_contextual:
            return False
        query = (req.user_query or "").lower()
        contextual_triggers = (
            "이전 대화",
            "이전 맥락",
            "앞서",
            "방금",
            "다시",
            "재작성",
            "이어서",
            "기존 맥락",
        )
        return any(token in query for token in contextual_triggers)

    def _load_final_answer_cache(self) -> dict[str, Any]:
        cache_path = self.final_answer_cache_path
        if not cache_path.exists() and self.legacy_final_answer_cache_path.exists():
            cache_path = self.legacy_final_answer_cache_path
        if not cache_path.exists():
            return {}
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save_final_answer_cache(self, payload: dict[str, Any]) -> None:
        atomic_write_text(
            self.final_answer_cache_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_cached_final_answer(self, req: ChatRequest) -> dict[str, Any] | None:
        if self._should_bypass_cache(req):
            return None
        signature = self._request_signature(req)
        with self.final_answer_cache_lock:
            cache = self._load_final_answer_cache()
        record = cache.get(req.session_id, {})
        return _cache_record_payload_for_request(record, signature)

    def _upsert_final_answer_cache(self, req: ChatRequest, result: AgentState) -> None:
        if not self.settings.final_answer_cache_enabled:
            return
        route = str(result.get("route", "full"))
        node_status = derive_node_status(route, result)
        payload = enforce_chat_response_contract(
            {
            "summary": result.get("final_answer", {}).get("summary", ""),
            "clause_analysis": result.get("final_answer", {}).get("clause_analysis", []),
            "risk_findings": result.get("final_answer", {}).get("risk_findings", []),
            "revision_plan": result.get("final_answer", {}).get("revision_plan", []),
            "input_gap_notice": result.get("final_answer", {}).get("input_gap_notice"),
            "references": _normalize_reference_records(
                result.get("final_answer", {}).get("references", [])
            ),
            "route": result.get("route"),
            "routing_reason": result.get("routing_reason"),
            "rag_low_confidence": result.get("rag_low_confidence"),
            "node_status": node_status,
            "cached_state_hit": False,
            }
        )
        signature = self._request_signature(req)
        with self.final_answer_cache_lock:
            cache = self._load_final_answer_cache()
            cache[req.session_id] = _upsert_cache_record(
                record=cache.get(req.session_id, {}),
                signature=signature,
                payload=payload,
                max_items=self.settings.final_answer_cache_max_per_session,
            )
            self._save_final_answer_cache(cache)

    def run(self, req: ChatRequest) -> ChatResponse:
        try:
            self.memory.add(req.session_id, "user", req.user_query)
            cached_payload = self._get_cached_final_answer(req)
            if cached_payload:
                cached_payload["references"] = _normalize_reference_records(
                    cached_payload.get("references", [])
                )
                cached_payload["cached_state_hit"] = True
                response = ChatResponse(
                    session_id=req.session_id,
                    **enforce_chat_response_contract(cached_payload),
                )
                self.memory.add(req.session_id, "assistant", response.summary)
                return response

            state: AgentState = {
                "session_id": req.session_id,
                "user_query": req.user_query,
                "document_type": req.document_type,
                "contract_text": req.contract_text,
                "reference_text": req.reference_text,
            }
            result = self.graph.invoke(
                state,
                config={"configurable": {"thread_id": req.session_id}},
            )
            if not isinstance(result, dict) or "final_answer" not in result:
                raise LegalPilotError(
                    error_code=ErrorCodes.STRUCTURED_OUTPUT_CONTRACT_ERROR,
                    detail="Missing final_answer in graph result.",
                    status_code=500,
                )
            self._upsert_final_answer_cache(req, result)
            route = str(result.get("route", "full"))
            node_status = derive_node_status(route, result)
            payload = enforce_chat_response_contract(
                {
                **result["final_answer"],
                "references": _normalize_reference_records(
                    result.get("final_answer", {}).get("references", [])
                ),
                "route": result.get("route"),
                "routing_reason": result.get("routing_reason"),
                "rag_low_confidence": result.get("rag_low_confidence"),
                "node_status": node_status,
                "cached_state_hit": False,
                }
            )
            response = ChatResponse(session_id=req.session_id, **payload)

            self.memory.add(req.session_id, "assistant", response.summary)
            return response
        except LegalPilotError:
            raise
        except Exception as exc:
            raise LegalPilotError(
                error_code=ErrorCodes.SERVICE_RUNTIME_ERROR,
                detail=f"LegalPilot service failed: {exc}",
                status_code=500,
            ) from exc
