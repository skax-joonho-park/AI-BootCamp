from src.retrieval import SearchHit, rerank_hits


def test_rerank_prioritizes_uploaded_reference_when_enabled() -> None:
    hits = [
        SearchHit(
            content="일반 지식 문서 내용",
            source="knowledge.md",
            score=0.58,
            metadata={"category": "uncategorized"},
        ),
        SearchHit(
            content="근로기준법 제17조: 임금 구성항목·계산방법·지급방법을 서면 명시해야 합니다.",
            source="uploaded_reference_text",
            score=0.50,
            metadata={"category": "statutes", "source_type": "reference_upload"},
        ),
    ]
    ranked = rerank_hits(
        hits=hits,
        query="근로기준법 임금 명시",
        role_hint="근로계약서, 임금, 근로기준법",
        route_categories={"statutes", "standard_contracts"},
        top_k=2,
        provider="heuristic",
        enabled=True,
    )
    assert ranked[0].source == "uploaded_reference_text"


def test_rerank_disabled_keeps_base_score_priority() -> None:
    hits = [
        SearchHit(
            content="일반 지식 문서 내용",
            source="knowledge.md",
            score=0.58,
            metadata={"category": "uncategorized"},
        ),
        SearchHit(
            content="근로기준법 제17조: 임금 구성항목·계산방법·지급방법을 서면 명시해야 합니다.",
            source="uploaded_reference_text",
            score=0.50,
            metadata={"category": "statutes", "source_type": "reference_upload"},
        ),
    ]
    ranked = rerank_hits(
        hits=hits,
        query="근로기준법 임금 명시",
        role_hint="근로계약서, 임금, 근로기준법",
        route_categories={"statutes", "standard_contracts"},
        top_k=2,
        provider="heuristic",
        enabled=False,
    )
    assert ranked[0].source == "knowledge.md"


def test_rerank_limits_same_source_by_max_per_source() -> None:
    hits = [
        SearchHit(content="문서A-1", source="same.md", score=0.9, metadata={"category": "statutes"}),
        SearchHit(content="문서A-2", source="same.md", score=0.88, metadata={"category": "statutes"}),
        SearchHit(content="문서A-3", source="same.md", score=0.87, metadata={"category": "statutes"}),
        SearchHit(content="문서B-1", source="other.md", score=0.7, metadata={"category": "statutes"}),
    ]
    ranked = rerank_hits(
        hits=hits,
        query="근로기준법 조항",
        role_hint="근로계약서, 임금",
        route_categories={"statutes"},
        top_k=4,
        max_per_source=2,
        provider="heuristic",
        enabled=True,
    )
    same_count = sum(1 for item in ranked if item.source == "same.md")
    assert same_count <= 2


def test_rerank_score_is_clipped_to_unit_interval() -> None:
    hits = [
        SearchHit(
            content="근로기준법 제17조: 임금 구성항목·계산방법·지급방법을 서면 명시해야 합니다.",
            source="uploaded_reference_text",
            score=0.96,
            metadata={"category": "statutes", "source_type": "reference_upload", "score_breakdown": {"fused": 0.96}},
        )
    ]
    ranked = rerank_hits(
        hits=hits,
        query="근로기준법 임금 명시",
        role_hint="근로계약서, 임금, 근로기준법",
        route_categories={"statutes", "standard_contracts"},
        top_k=1,
        provider="heuristic",
        enabled=True,
    )
    assert ranked
    assert 0.0 <= ranked[0].score <= 1.0
    breakdown = ranked[0].metadata.get("score_breakdown", {})
    assert isinstance(breakdown, dict)
    assert "final_post_rerank_raw" in breakdown
