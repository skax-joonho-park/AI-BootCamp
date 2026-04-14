# [Step3 - 서비스 개발]

## 1) 구현 범위

- LangGraph 기반 Multi-Agent 워크플로우 구현
- RAG 파이프라인(문서 로딩, 청킹, FAISS + BM25 하이브리드 검색) 구현
- Structured Output(Pydantic) 기반 최종 응답 생성
- Streamlit UI + FastAPI API + CLI 실행 경로 제공
- `.env` 기반 Azure OpenAI 설정 로딩 및 모듈 분리
- API 예외 처리(400/500) 기반 안정적 오류 응답 구성
- Streamlit 실행 입력 기록 관리(조회/삭제/다시 불러오기) 및 세션 자동 관리
- `ChatRequest.contract_text`/`ChatRequest.reference_text` 기반 계약서-법령 갭 분석 명시화

## 2) 핵심 파일

- `src/workflow/engine.py`: LangGraph 오케스트레이션, 서비스 진입점(`LegalPilotService`)
- `src/workflow/contracts.py`: `ChatRequest`/`ChatResponse` 스키마 (`document_type`, `contract_text`, `reference_text` 포함)
- `src/retrieval/documents.py`: 데이터 로딩/청킹
- `src/retrieval/hybrid.py`: FAISS + BM25 검색
- `src/agents/tools.py`: Tool Calling 도구 함수(`clause_keyword_match_score`, `legal_issue_bank`, `contract_reference_gap_score`)
- `src/agents/schemas.py`: Structured Output 스키마(`ClauseNotes`, `RiskNotes`, `RevisionNotes`, `FinalAnswer`)
- `src/api/app.py`: FastAPI 엔드포인트
- `src/ui/streamlit_app.py`: Streamlit UI
- `main.py`: CLI 실행 엔트리

## 2-1) 구조도/플로우 산출물

- 아키텍처 다이어그램: `docs/images/system_architecture.png` / `docs/images/system_architecture.md` (Mermaid 소스)
- 서비스 플로우 다이어그램: `docs/images/service_flow_sequence.png` / `docs/images/service_flow_sequence.md` (Mermaid 소스)
- E2E 테스트 체크리스트: `docs/evidence/e2e_test_checklist.md`
- 실행 증빙 로그: `docs/evidence/agent_execution_log.md`
- 최종 응답 JSON: `docs/evidence/agent_final_answer.json`
- 포함 내용:
  - LangGraph 주요 노드/엣지(`supervisor → rag → (clause/risk/advice) → synthesis`)
  - RAG 컨텍스트/참고 출처(`rag_context`, `rag_refs`)가 Supervisor/Synthesis에 주입되는 데이터 흐름

## 3) 실행 방법

### 의존성 설치

```bash
pip install -r requirements.txt
# (선택) Okt 형태소 분석 활성화 시: pip install konlpy==0.6.0
```

### 환경변수 설정

`.env` 파일에 아래 값 설정:

- `AOAI_ENDPOINT`
- `AOAI_API_KEY`
- `AOAI_DEPLOY_GPT4O`
- `AOAI_DEPLOY_EMBED_ADA` (또는 `AOAI_EMBEDDING_DEPLOYMENT`)
- `AOAI_API_VERSION`
- `MEMORY_MAX_SESSIONS` (선택, 기본 200)
- `MEMORY_TTL_SECONDS` (선택, 기본 86400초)
- `SESSION_MEMORY_PERSIST_ENABLED` (선택, 기본 true)
- `SESSION_MEMORY_PII_MASK` (선택, 기본 false)
- `UI_HISTORY_PERSIST_ENABLED` (선택, 기본 true)
- `UI_HISTORY_PII_MASK` (선택, 기본 false)
- `UI_HISTORY_STORAGE_MODE` (선택, 기본 summary / `summary|full`)
- `UI_PAGE_ICON_MODE` (선택, 기본 emoji)
- `UI_PAGE_ICON_EMOJI` (선택, 기본 ⚖️)
- `INDEX_FORCE_REBUILD` (선택, 기본 false)
- `VECTOR_WEIGHT` (선택, 기본 0.6)
- `BM25_WEIGHT` (선택, 기본 0.4)
- `RAG_EVIDENCE_SCORE_THRESHOLD` (선택, 기본 0.45)
- `EPHEMERAL_REF_BASE_SCORE` (선택, 기본 0.42 / 업로드 참조 법령 임시 근거 기본 점수)
- `EPHEMERAL_CONTRACT_BASE_SCORE` (선택, 기본 0.36 / 업로드 계약서 임시 근거 기본 점수)
- `EPHEMERAL_OVERLAP_WEIGHT` (선택, 기본 0.35)
- `FEW_SHOT_MAX_EXAMPLES` (선택, 기본 1)
- `RERANK_ENABLED` (선택, 기본 true)
- `RERANK_PROVIDER` (선택, 기본 heuristic)
- `RERANK_MAX_PER_SOURCE` (선택, 기본 2)
- `RETRIEVAL_MAX_CHUNKS_PER_FILE` (선택, 기본 0)
- `ALLOW_UNCATEGORIZED_IN_FILTER` (선택, 기본 true)
- `FAISS_ALLOW_DANGEROUS_DESERIALIZATION` (선택, 기본 false)
- `FINAL_ANSWER_CACHE_ENABLED` (선택, 기본 true)
- `FINAL_ANSWER_CACHE_BYPASS_CONTEXTUAL` (선택, 기본 true)
- `FINAL_ANSWER_CACHE_MAX_PER_SESSION` (선택, 기본 5)
- `STATE_STORE_BACKEND` (선택, 기본 file / `file|sqlite|redis` 스위치)
- `STATE_STORE_DSN` (선택, 기본 빈값 / SQLite·Redis 확장용 DSN 예약 필드)

### CLI 실행

```bash
python main.py \
  --query "근로계약서의 포괄임금제 조항이 적법한지 검토해줘" \
  --document-type "근로계약서" \
  --contract-text "월 급여 300만 원(포괄임금제 적용), 경업금지 3년" \
  --reference-text "근로기준법 제17조: 임금 구성항목 서면 명시 의무"
```

### FastAPI 실행

```bash
python scripts/run_api.py
```

### Streamlit 실행

```bash
python scripts/run_streamlit.py
```

## 4) 기본 검증 결과

- `pytest` 실행 결과: 52 passed
- 검증 대상:
  - 세션 메모리 동작
  - Clause/Risk Tool 기본 응답
  - 라우팅 로직(clause_only/risk_only/advice_only/full_review)
  - 평가 지표(routing_accuracy/reference_inclusion_rate/plan_quality_rate)
  - UI 입력 기록 스키마 마이그레이션

## 4-1) 차별성 검증 지표(운영 체크)

- **라우팅 정확도**: 의도 라벨 셋(`clause_only`, `risk_only`, `advice_only`, `full_review`) 기준 수동 평가 정확도
- **근거 포함률**: 응답에서 `references` 1개 이상 포함 비율 및 근거-응답 정합성 점검
- **수정 계획 품질**: `revision_plan` 항목의 구체성(행동/우선순위/법령 근거) 5점 척도 평가
- **자동화 실행 경로**:
  ```bash
  python scripts/evaluate_differentiation_metrics.py --cases data/eval/sample_queries.json
  ```

- 리랭크 다양성 비교는 `--compare-rerank-on-off`로 실행하며, 동일 케이스를 `RERANK_ENABLED=true/false`로 각각 돌려 duplicate-source 비율 차이를 확인합니다.
- 현재 샘플 구성은 총 25건(`clause_only` 5, `risk_only` 5, `advice_only` 5, `full_review` 5, 모호 질의 5)
- 샘플 질의셋의 `expected_route`가 있을 경우 Top-1 라우팅 정확도를 계산
- `references >= 1` 비율, `revision_plan >= 4` 비율을 함께 계산
- 임계치 미달 시 non-zero 종료코드로 CI/배치 점검 가능

## 4-2) RAG 범위/안전 정책

- 지식 범위: `data/knowledge` 내 법령 요약, 표준 계약서, 분쟁 사례, 계약서 예시 카테고리 문서 중심
- 근거 부족 모드: 상위 RAG 점수 < `RAG_EVIDENCE_SCORE_THRESHOLD`이면 보수적 표현 전환
- 법률 자문 면책 고지: 모든 응답 summary에 "본 검토 의견은 법적 자문이 아님" 문구 자동 삽입
- 추적성: `references`는 rank/source/chunk/location/score/snippet 포함

## 4-3) 구현 과정에서 발생한 이슈 및 해결 방안

- **이슈**: 라우팅 부정문 오분류 — "조항만 보고 싶은데 위험 제외는 아니야" 같은 부정 문맥에서 `clause_only`로 오분류
  - **해결**: NEGATION_PATTERNS + 부정 문맥 탐지 로직 추가, 부정 탐지 시 LLM 재판단으로 위임

- **이슈**: Structured Output 간헐 파싱 실패
  - **해결**: temperature 하향 + JSON repair 지시 재시도 후 최소 스키마 강제(`enforce_chat_response_contract`)

- **이슈**: 업로드 텍스트가 RAG 검색 결과에 반영되지 않음
  - **해결**: `contract_text`/`reference_text`를 임시 청크(ephemeral evidence)로 생성해 검색 후보에 혼합

- **이슈**: 동일 문서의 청크가 references에 중복 등장
  - **해결**: `RERANK_MAX_PER_SOURCE`로 동일 source 청크 수 상한 적용

## 5) 차별성 지표 자동화

```bash
python scripts/evaluate_differentiation_metrics.py \
  --cases data/eval/sample_queries.json \
  --output docs/evidence/metrics_run_output.txt
```

- 지표: 라우팅 정확도, 근거 포함률, 수정 계획 품질률
- 기준 조정: `--min-routing-accuracy`, `--min-reference-rate`, `--min-plan-quality-rate`
- 분포 검증: `--min-per-labeled-route`, `--min-ambiguous-cases`
- 증빙 파일은 `--output` 옵션으로 UTF-8 저장

## 6) 파일 구조 요약

```
├── main.py                          # CLI 진입점
├── requirements.txt                 # 의존성 (Python 3.12)
├── data/
│   ├── knowledge/
│   │   ├── statutes/                # 법령 요약본
│   │   ├── standard_contracts/      # 표준 계약서
│   │   ├── case_guides/             # 분쟁 사례 가이드
│   │   └── contract_examples/       # 계약서 예시
│   └── eval/
│       └── sample_queries.json      # 평가 질의셋
├── docs/
│   ├── step2_planning_design.md
│   ├── step3_service_development.md
│   └── evidence/
├── scripts/
│   ├── run_api.py
│   ├── run_streamlit.py
│   ├── generate_submission_evidence.py
│   └── evaluate_differentiation_metrics.py
├── src/
│   ├── config/                      # .env 로딩, 모델 클라이언트
│   ├── retrieval/                   # 문서 로딩, 하이브리드 검색, 리랭크
│   ├── agents/                      # 도구 함수, 구조화 스키마
│   ├── workflow/                    # LangGraph 엔진, 계약, 프롬프트
│   ├── api/                         # FastAPI 앱
│   ├── ui/                          # Streamlit UI, 기록 관리
│   └── utils/                       # 메모리, 파일 추출, IO
└── tests/                           # pytest 테스트 (52 passed)
```
