from src.ui.history_record import build_history_record, migrate_history_record


def _sample_response() -> dict:
    return {
        "summary": "요약",
        "clause_analysis": ["c1", "c2"],
        "risk_findings": ["r1"],
        "revision_plan": ["p1", "p2"],
        "references": [{"rank": 1, "source": "doc.md", "score": 0.9}],
        "route": "full_review",
        "routing_reason": "test",
        "rag_low_confidence": False,
        "cached_state_hit": False,
    }


def test_build_history_record_summary_mode_minimizes_payload() -> None:
    record = build_history_record(
        session_id="s1",
        query="q",
        document_type="근로계약서",
        contract_text="계약서 원문",
        reference_text="참조 법령",
        response_payload=_sample_response(),
        run_id="run-1",
        storage_mode="summary",
    )
    assert record["storage_mode"] == "summary"
    assert record["record_version"] == 1
    assert record["run_id"] == "run-1"
    assert record["contract_text"] == ""
    assert record["reference_text"] == ""
    assert record["contract_hash"]
    assert record["ref_hash"]
    assert isinstance(record["response"], dict)
    assert "summary" in record["response"]


def test_build_history_record_full_mode_keeps_raw_text() -> None:
    record = build_history_record(
        session_id="s1",
        query="q",
        document_type="근로계약서",
        contract_text="계약서 원문",
        reference_text="참조 법령",
        response_payload=_sample_response(),
        run_id="run-2",
        storage_mode="full",
    )
    assert record["storage_mode"] == "full"
    assert record["record_version"] == 1
    assert record["run_id"] == "run-2"
    assert record["contract_text"] == "계약서 원문"
    assert record["reference_text"] == "참조 법령"
    assert record["response"]["summary"] == "요약"


def test_migrate_history_record_v0_to_v1() -> None:
    legacy = {
        "session_id": "s1",
        "query": "q",
        "document_type": "근로계약서",
        "contract_len": "11",
        "ref_len": "7",
        "storage_mode": "summary",
        "response": "legacy text",
    }
    migrated = migrate_history_record(legacy)
    assert migrated["record_version"] == 1
    assert migrated["contract_len"] == 11
    assert migrated["ref_len"] == 7
    assert migrated["storage_mode"] == "summary"
    assert migrated["run_id"] == ""
    assert isinstance(migrated["response"], dict)
