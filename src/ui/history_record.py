"""History record builder for UI persistence with configurable data minimization."""

from __future__ import annotations

import hashlib
from typing import Any

HISTORY_RECORD_VERSION = 1


def _sha256_short(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _clip(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _summary_response(payload: dict[str, Any]) -> dict[str, Any]:
    references = payload.get("references", [])
    compact_refs: list[dict[str, Any]] = []
    if isinstance(references, list):
        for item in references[:3]:
            if isinstance(item, dict):
                compact_refs.append(
                    {
                        "rank": item.get("rank"),
                        "source": item.get("source"),
                        "score": item.get("score"),
                    }
                )
    return {
        "summary": str(payload.get("summary", "") or ""),
        "run_id": str(payload.get("run_id", "") or ""),
        "executed_at": str(payload.get("executed_at", "") or ""),
        "result_source": str(payload.get("result_source", "") or ""),
        "route": payload.get("route"),
        "routing_reason": payload.get("routing_reason"),
        "rag_low_confidence": payload.get("rag_low_confidence"),
        "cached_state_hit": payload.get("cached_state_hit", False),
        "clause_analysis": list(payload.get("clause_analysis", []) or [])[:5],
        "risk_findings": list(payload.get("risk_findings", []) or [])[:5],
        "revision_plan": list(payload.get("revision_plan", []) or [])[:5],
        "input_gap_notice": payload.get("input_gap_notice"),
        "references": compact_refs,
    }


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def migrate_history_record(item: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a history record to the latest schema version."""
    normalized = dict(item)
    version = _to_int(normalized.get("record_version", 0), default=0)
    if version <= 0:
        # v0 -> v1: add explicit schema version and normalize minimal required fields.
        normalized["record_version"] = 1
    else:
        normalized["record_version"] = version

    normalized["storage_mode"] = (
        str(normalized.get("storage_mode", "full"))
        if str(normalized.get("storage_mode", "full")) in {"summary", "full"}
        else "full"
    )
    normalized["query"] = str(normalized.get("query", "") or "")
    normalized["document_type"] = str(normalized.get("document_type", "근로계약서") or "근로계약서")
    normalized["session_id"] = str(normalized.get("session_id", "") or "")
    normalized["run_id"] = str(normalized.get("run_id", "") or "")
    normalized["contract_len"] = _to_int(normalized.get("contract_len", 0), default=0)
    normalized["ref_len"] = _to_int(normalized.get("ref_len", 0), default=0)
    normalized["contract_text"] = str(normalized.get("contract_text", "") or "")
    normalized["reference_text"] = str(normalized.get("reference_text", "") or "")
    if "response" in normalized and not isinstance(normalized.get("response"), dict):
        normalized["response"] = {}
    return normalized


def build_history_record(
    *,
    session_id: str,
    query: str,
    document_type: str,
    contract_text: str,
    reference_text: str,
    response_payload: dict[str, Any],
    run_id: str = "",
    storage_mode: str = "summary",
) -> dict[str, Any]:
    mode = storage_mode if storage_mode in {"summary", "full"} else "summary"
    base = {
        "record_version": HISTORY_RECORD_VERSION,
        "session_id": session_id,
        "run_id": str(run_id or ""),
        "query": query.strip(),
        "document_type": document_type,
        "contract_len": len(contract_text or ""),
        "ref_len": len(reference_text or ""),
        "contract_hash": _sha256_short(contract_text or ""),
        "ref_hash": _sha256_short(reference_text or ""),
        "storage_mode": mode,
    }
    if mode == "full":
        return {
            **base,
            "contract_text": contract_text,
            "reference_text": reference_text,
            "response": response_payload,
        }
    return {
        **base,
        "contract_text": "",
        "reference_text": "",
        "contract_preview": _clip(contract_text, 500),
        "ref_preview": _clip(reference_text, 500),
        "response": _summary_response(response_payload),
    }
