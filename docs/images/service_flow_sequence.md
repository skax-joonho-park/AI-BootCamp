# LegalPilot AI — 서비스 플로우 시퀀스

## 시퀀스 다이어그램

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Sup as Supervisor Agent
    participant RAG as RAG Agent
    participant Clause as Clause Analyzer
    participant Risk as Risk Detection
    participant Advice as Revision Advisor
    participant Syn as Synthesis

    User->>UI: 질문 + 계약서 유형 선택<br/>+ 계약서 텍스트 + 참조 법령(선택)
    UI->>Sup: ChatRequest(session_id, query,<br/>document_type, contract_text, reference_text)

    Note over Sup: 부정 패턴 탐지 → 라우팅 결정<br/>(clause_only / risk_only /<br/>advice_only / full_review)

    Sup->>RAG: route + query + contract_text + reference_text
    RAG->>RAG: FAISS 벡터 검색 + BM25 검색<br/>+ Ephemeral 청크 혼합 + 리랭크
    RAG-->>Sup: rag_context + rag_refs (rank/score/snippet)

    alt route = full_review
        Sup->>Clause: rag_context + contract_text + document_type
        Clause->>Clause: clause_keyword_match_score() 도구 호출
        Clause-->>Syn: clause_notes (ClauseNotes)

        Sup->>Risk: rag_context + contract_text + document_type
        Risk->>Risk: legal_issue_bank() 도구 호출
        Risk-->>Syn: risk_notes (RiskNotes)

        Sup->>Advice: rag_context + clause_notes + risk_notes
        Advice->>Advice: contract_reference_gap_score() 도구 호출
        Advice-->>Syn: revision_notes (RevisionNotes)

    else route = clause_only
        Sup->>Clause: rag_context + contract_text
        Clause-->>Syn: clause_notes
        Note over Syn: risk_findings = []<br/>revision_plan = []

    else route = risk_only
        Sup->>Risk: rag_context + contract_text
        Risk-->>Syn: risk_notes
        Note over Syn: clause_analysis = []<br/>revision_plan = []

    else route = advice_only
        Sup->>Advice: rag_context + contract_text
        Advice-->>Syn: revision_notes
        Note over Syn: clause_analysis = []<br/>risk_findings = []
    end

    Syn->>Syn: FinalAnswer 조합<br/>+ 면책 고지 삽입("법적 자문이 아닙니다")<br/>+ 응답 캐시 저장 (세션별)
    Syn-->>UI: ChatResponse(summary / clause_analysis /<br/>risk_findings / revision_plan / references)

    UI-->>User: 결과 카드 렌더링<br/>(요약 / 조항 분석 / 위험 조항 탐지 /<br/>수정 계획 / 참고 출처)
    UI->>UI: 실행 기록 사이드바 저장
```

## 라우팅 결정 규칙

| 라우트 | 트리거 패턴 예시 | 실행 에이전트 | 비어 있는 필드 |
|---|---|---|---|
| `clause_only` | "조항 분석만 해줘", "조항 위주로", "조항 검토만" | Clause Analyzer | `risk_findings`, `revision_plan` |
| `risk_only` | "위험 조항만 알려줘", "법적 우려만", "리스크 중심" | Risk Detection | `clause_analysis`, `revision_plan` |
| `advice_only` | "수정 계획만 작성해줘", "개선 방향만" | Revision Advisor | `clause_analysis`, `risk_findings` |
| `full_review` | "전체 검토", "통합 분석", "조항·위험·수정 모두" | Clause + Risk + Advice | (없음) |

## 부정 패턴 탐지 로직

- `CLAUSE_ONLY_MARKERS`: `("조항만", "조항 분석만", "조항 위주로", ...)` 토큰 매칭
- `NEGATION_PATTERNS`: `("아니야", "제외는 아니야", ...)` 패턴 탐지 시 LLM 재판단 위임
- 모호한 질의(예: "계약서 전반을 도와줘")는 `full_review`로 폴백

## 데이터 흐름 요약

```
User Input
    └─► Supervisor (route 결정)
            └─► RAG (rag_context + rag_refs)
                    ├─► Clause Analyzer → ClauseNotes
                    ├─► Risk Detection  → RiskNotes
                    └─► Revision Advisor → RevisionNotes
                                └─► Synthesis → FinalAnswer (JSON)
                                                    └─► UI 렌더링
```
