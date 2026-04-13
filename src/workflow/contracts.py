"""Request/response contracts used across API/UI/CLI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


STRUCTURED_OUTPUT_RETRY_MAX = 1
STRUCTURED_OUTPUT_REPAIR_ENABLED = True
STRUCTURED_OUTPUT_FALLBACK_STRATEGY = "minimal_schema_enforcement"


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    user_query: str
    document_type: str = Field(default="근로계약서")
    contract_text: str = Field(default="")
    reference_text: str = Field(default="")


class ReferenceItem(BaseModel):
    rank: int
    source: str
    chunk_id: int | str | None = None
    location: str = "n/a"
    score: float = 0.0
    category: str | None = None
    snippet: str = ""
    score_breakdown: dict[str, float] | None = None
    collected_at: str | None = None
    source_url: str | None = None
    curator: str | None = None
    license: str | None = None


class NodeExecutionStatus(BaseModel):
    status: str = "ok"  # ok | degraded | skipped
    error_code: str | None = None
    detail: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    summary: str
    clause_analysis: list[str] = Field(default_factory=list)
    risk_findings: list[str] = Field(default_factory=list)
    revision_plan: list[str] = Field(default_factory=list)
    input_gap_notice: str | None = None
    references: list[ReferenceItem] = Field(default_factory=list)
    route: str | None = None
    routing_reason: str | None = None
    rag_low_confidence: bool | None = None
    node_status: dict[str, NodeExecutionStatus] | None = None
    cached_state_hit: bool = False


def enforce_chat_response_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Force minimal ChatResponse schema keys to keep UI/API stable on parse drift."""
    raw_node_status = payload.get("node_status")
    normalized_node_status: dict[str, Any] | None = None
    if isinstance(raw_node_status, dict):
        normalized_node_status = {}
        for node_name, node_payload in raw_node_status.items():
            if not isinstance(node_name, str) or not isinstance(node_payload, dict):
                continue
            normalized_node_status[node_name] = {
                "status": str(node_payload.get("status", "ok") or "ok").strip() or "ok",
                "error_code": (
                    None
                    if node_payload.get("error_code") in (None, "", "None", "null", "nan")
                    else str(node_payload.get("error_code")).strip()
                ),
                "detail": (
                    None
                    if node_payload.get("detail") in (None, "", "None", "null", "nan")
                    else str(node_payload.get("detail")).strip()
                ),
            }

    return {
        "summary": str(payload.get("summary", "") or "").strip() or "요청 결과를 요약합니다.",
        "clause_analysis": list(payload.get("clause_analysis", []) or []),
        "risk_findings": list(payload.get("risk_findings", []) or []),
        "revision_plan": list(payload.get("revision_plan", []) or []),
        "input_gap_notice": (
            None
            if payload.get("input_gap_notice") in (None, "", "None", "null", "nan")
            else str(payload.get("input_gap_notice")).strip()
        ),
        "references": list(payload.get("references", []) or []),
        "route": payload.get("route"),
        "routing_reason": payload.get("routing_reason"),
        "rag_low_confidence": payload.get("rag_low_confidence"),
        "node_status": normalized_node_status,
        "cached_state_hit": bool(payload.get("cached_state_hit", False)),
    }
