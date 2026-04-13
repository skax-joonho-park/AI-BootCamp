"""Structured output schemas for final response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClauseNotes(BaseModel):
    key_clauses: list[str] = Field(description="핵심 조항 진단 요약")
    problematic_clauses: list[str] = Field(description="우선순위 문제 조항 및 개선 항목")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")
    evidence_map: dict[str, list[int]] = Field(
        default_factory=dict,
        description="조항/진단 문장을 근거 chunk 번호([1],[2]...)와 매핑",
    )


class RiskNotes(BaseModel):
    risk_areas: list[str] = Field(description="위험 조항 및 법적 리스크 항목")
    legal_concerns: list[str] = Field(description="법적 우려 사항 및 검토 포인트")
    mitigation_advice: list[str] = Field(description="리스크 완화 및 대응 방향")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")
    evidence_map: dict[str, list[int]] = Field(
        default_factory=dict,
        description="리스크/우려 항목을 근거 chunk 번호([1],[2]...)와 매핑",
    )


class RevisionNotes(BaseModel):
    priorities: list[str] = Field(description="우선순위 기반 수정 실행 항목")
    revision_steps: list[str] = Field(description="단계별 수정 방향 및 작성 가이드")
    validation_checks: list[str] = Field(description="검토 완료 기준 및 체크포인트")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")
    evidence_map: dict[str, list[int]] = Field(
        default_factory=dict,
        description="수정 항목을 근거 chunk 번호([1],[2]...)와 매핑",
    )


class ReferenceItem(BaseModel):
    rank: int = Field(description="검색 결과 순위")
    source: str = Field(description="출처 파일명")
    chunk_id: int | str | None = Field(default=None, description="원본 문서 내 청크 ID")
    location: str = Field(default="n/a", description="원본 문서 위치")
    score: float = Field(default=0.0, description="검색 점수 (0~1)")
    category: str | None = Field(default=None, description="문서 카테고리")
    snippet: str = Field(default="", description="청크 내용 요약")
    score_breakdown: dict[str, float] | None = Field(
        default=None,
        description="점수 분해 정보(vector/bm25/fused/penalty/rerank boosts)",
    )
    collected_at: str | None = Field(default=None, description="문서 수집/업로드 일시")
    source_url: str | None = Field(default=None, description="원문 출처 URL")
    curator: str | None = Field(default=None, description="문서 정리 담당자")
    license: str | None = Field(default=None, description="문서 사용 라이선스/내부 정책")


class FinalAnswer(BaseModel):
    summary: str = Field(description="요청에 대한 핵심 요약")
    clause_analysis: list[str] = Field(
        default_factory=list,
        description="조항 분석 액션 아이템(라우트에 따라 비워질 수 있음)",
    )
    risk_findings: list[str] = Field(
        default_factory=list,
        description="위험 조항 탐지 액션 아이템(라우트에 따라 비워질 수 있음)",
    )
    revision_plan: list[str] = Field(
        default_factory=list,
        description="수정 계획 아이템(라우트에 따라 비워질 수 있음)",
    )
    input_gap_notice: str | None = Field(
        default=None,
        description="입력 정보 부족 시 사용자에게 추가 제공을 요청하는 안내 문구",
    )
    references: list[ReferenceItem] = Field(
        default_factory=list,
        description="참고한 문서/근거 출처 구조화 목록",
    )
