# LegalPilot AI - Final Project

LegalPilot AI는 법률 문서 검토를 위한 End-to-End 멀티 에이전트 서비스입니다.

## 실행/경로 규약

- 모든 명령은 루트를 현재 작업 디렉토리로 두고 실행합니다.
- 문서에 표시된 상대 경로(`scripts/...`, `data/...`, `docs/...`, `src/...`)는 모두 위 루트 기준입니다.

## 처음 실행 3단계

1. `.env` 설정: `AOAI_*` 값 입력
2. 지식 문서 배치: `data/knowledge/statutes|standard_contracts|case_guides|contract_examples` 중 최소 1개 카테고리에 문서 추가
3. 실행: `python scripts/run_streamlit.py` (UI) 또는 `python scripts/run_api.py` (API)

## 구현 완료 항목

- **프롬프트 엔지니어링**: Supervisor/Clause/Risk/Advice 에이전트 역할 기반 프롬프트
- **도메인 안전 정책**: 공통 프롬프트 정책에 법률 자문 면책 문구 + 결과 보장 표현 금지(근거 기반 검토 코칭) 규칙 명시
- **Few-shot 운영성 개선**: Few-shot 예시는 `data/prompts/few_shots/*.md` 외부 파일에서 로드하며, 파일 누락/오류 시 코드 기본값으로 자동 fallback
- **멀티 에이전트(LangGraph)**: Supervisor → RAG → (Clause/Risk/Advice) → Synthesis 라우트 기반 분기 실행
- **요구사항-인터페이스 정합성**: `ChatRequest`에서 `contract_text`/`reference_text`를 명시적으로 받아 계약서-법령 갭 분석 지원
- **에이전트 자율 정책**: 계약서/참조 법령 텍스트 미입력 시 각 노드의 로컬 fallback 정책 적용
- **RAG**: 문서 로딩(`.txt/.md/.csv/.pdf/.docx/.xlsx`), 청킹, FAISS 벡터 검색 + BM25 키워드 검색
- **RAG 품질**: 인덱스 영속화(`data/index/faiss`), 메타데이터 기반 컨텍스트(page/paragraph/sheet/row), route-aware 카테고리 필터 + no-hit 재검색 fallback, 전용 리랭크 레이어
- **중복 제어 책임 분리**: rerank 활성 시 retrieval 단계 source 상한(`RETRIEVAL_MAX_CHUNKS_PER_FILE`)은 비활성화하고, 최종 다양성 제어는 `RERANK_MAX_PER_SOURCE`에서 단일 강제로 적용
- **로더 메타데이터 병합**: `src/retrieval/documents.py`에서 문서별 `*.meta.json` sidecar의 최소 필드(`collected_at/source_url/curator/license`)를 병합 로딩하고, sidecar 누락/오류 시 경고를 출력해 추적성을 유지
- **RAG 점수 안정성**: FAISS distance를 쿼리-독립 변환(`1/(1+d)`)으로 0~1 스케일링해 `RAG_EVIDENCE_SCORE_THRESHOLD` 해석 일관성 강화
- **데이터 거버넌스 가시화**: Streamlit 사이드바에서 `retriever_meta.json`의 카테고리 품질 경고(`uncategorized` 비율/필수 카테고리 누락)를 즉시 노출
- **파일 저장 안정성**: `session_memory.json`/`final_answer_cache.json`/`chunks.json`/`retriever_meta.json`/`ui_input_history.json` 저장 경로에 원자적 쓰기(temp→replace) 적용
- **히스토리 스키마 진화 대응**: `record_version` 기반 `migrate_history_record()`로 구버전 실행 기록 자동 업그레이드
- **RAG 추적성**: references에 rank/source/chunk/location/snippet 정보를 포함해 citation-근거 연결 강화
- **구조화 출력**: Pydantic 스키마 기반 최종 응답 생성
- **Route-aware 출력 정책**: synthesis 단계에서 라우트별 최소 섹션 규칙 적용, 불필요 섹션은 빈 배열로 처리
- **구조화 출력 내구성**: Clause/Risk/Synthesis 파싱 실패 시 노드 단위 degrade fallback
- **근거 연결성 강화**: Clause/Risk notes에 `evidence_map` 포함, 최종 불릿에 citation(`[1][2]`) 스타일 적용
- **그래프 체크포인팅**: LangGraph checkpointer와 `thread_id=session_id` 연동(실행 상태 복원 기반)
- **체크포인터 역할 분리**: `MemorySaver`는 프로세스 내 런타임 복원용, 재시작 이후 영속 복원은 `session_memory.json`/`final_answer_cache.json`이 담당
- **라우팅 안정화**: Supervisor에서 "제외/전용" 키워드 1차 휴리스틱 라우팅 후, 미해당 케이스만 LLM 라우팅으로 처리
- **부정/예외 라우팅 보강**: `~제외는 아니고`, `~제외하지 말고` 같은 부정 문맥을 휴리스틱에서 탐지해 오분류를 줄이고 LLM 재판단으로 위임
- **최종 응답 재사용 캐시**: 파일 기반 invoke 캐시(`final_answer_cache.json` + lock)는 정규화된 최종 `ChatResponse` payload를 재사용
- **계약서-법령 갭 도구**: `contract_reference_gap_score`(필수/우대 항목 매칭률 + 누락 Top-N) 추가 및 Clause Agent tool loop 연동
- **선택형 신뢰도 메타데이터**: `ChatResponse`에 `route/routing_reason/rag_low_confidence/cached_state_hit/node_status` 포함, Streamlit 디버그 토글로 표시 가능
- **안정성**: API 레벨 예외 처리를 통한 사용자 친화적 오류 응답
- **오류 계약 표준화**: API/CLI/UI 공통 `error_code/detail` 페이로드(`LegalPilotError`) 적용

## 폴더 구조

- `data/knowledge`: RAG 원천 문서 (`.txt/.md/.csv/.pdf/.docx/.xlsx`)
  - 권장 카테고리: `statutes/`, `standard_contracts/`, `case_guides/`, `contract_examples/`
- `docs`: 제출용 기획/설계/개발 문서 (`step2_planning_design.md`, `step3_service_development.md`)
- `scripts`: 실행 스크립트
- `src/config`: `.env` 로딩 및 모델 클라이언트
- `src/retrieval`: 데이터 로딩, 청킹, 하이브리드 검색
- `src/agents`: 도구 및 구조화 응답 스키마
- `src/workflow`: LangGraph 워크플로우 및 서비스 레이어
- `src/api`: FastAPI 앱
- `src/ui`: Streamlit 앱

## 제출 문서

- [Step2 - 기획 및 설계](./docs/step2_planning_design.md)
- [Step3 - 서비스 개발](./docs/step3_service_development.md)
- [System Architecture Image](./docs/images/system_architecture.png)
- [Service Flow Image](./docs/images/service_flow_sequence.png)
- [Evidence - Agent Execution Log](./docs/evidence/agent_execution_log.md)
- [Evidence - Final Answer JSON](./docs/evidence/agent_final_answer.json)
- [Evidence - E2E Test Checklist](./docs/evidence/e2e_test_checklist.md)
- [Evidence Generator Script](./scripts/generate_submission_evidence.py)
- [Differentiation Metrics Evaluator](./scripts/evaluate_differentiation_metrics.py)

## 환경 설정

1. 의존성 설치:

```bash
pip install -r requirements.txt
```

선택(한국어 형태소 분석 Okt 사용 시):

```bash
pip install konlpy==0.6.0
```

2. `.env` 생성 후 필수 값 입력:

- `AOAI_ENDPOINT`
- `AOAI_API_KEY`
- `AOAI_DEPLOY_GPT4O`
- `AOAI_DEPLOY_EMBED_ADA` (또는 `AOAI_EMBEDDING_DEPLOYMENT`)
- `AOAI_API_VERSION`
- `MEMORY_MAX_SESSIONS` (선택, 기본 `200`)
- `MEMORY_TTL_SECONDS` (선택, 기본 `86400`)
- `SESSION_MEMORY_PERSIST_ENABLED` (선택, 기본 `true`)
- `SESSION_MEMORY_PII_MASK` (선택, 기본 `false`)
- `UI_HISTORY_PERSIST_ENABLED` (선택, 기본 `true`)
- `UI_HISTORY_PII_MASK` (선택, 기본 `false`)
- `UI_HISTORY_STORAGE_MODE` (선택, 기본 `summary`; `summary|full`)
- `UI_PAGE_ICON_MODE` (선택, 기본 `emoji`)
- `UI_PAGE_ICON_EMOJI` (선택, 기본 `⚖️`)
- `INDEX_FORCE_REBUILD` (선택, 기본 `false`)
- `VECTOR_WEIGHT` (선택, 기본 `0.6`)
- `BM25_WEIGHT` (선택, 기본 `0.4`)
- `RAG_EVIDENCE_SCORE_THRESHOLD` (선택, 기본 `0.45`)
- `EPHEMERAL_REF_BASE_SCORE` (선택, 기본 `0.42`; 업로드 참조 법령 임시 근거 기본 점수)
- `EPHEMERAL_CONTRACT_BASE_SCORE` (선택, 기본 `0.36`; 업로드 계약서 임시 근거 기본 점수)
- `EPHEMERAL_OVERLAP_WEIGHT` (선택, 기본 `0.35`)
- `FEW_SHOT_MAX_EXAMPLES` (선택, 기본 `1`)
- `RERANK_ENABLED` (선택, 기본 `true`)
- `RERANK_PROVIDER` (선택, 기본 `heuristic`)
- `RERANK_MAX_PER_SOURCE` (선택, 기본 `2`)
- `RETRIEVAL_MAX_CHUNKS_PER_FILE` (선택, 기본 `0`)
- `ALLOW_UNCATEGORIZED_IN_FILTER` (선택, 기본 `true`)
- `FAISS_ALLOW_DANGEROUS_DESERIALIZATION` (선택, 기본 `false`)
- `FINAL_ANSWER_CACHE_ENABLED` (선택, 기본 `true`)
- `FINAL_ANSWER_CACHE_MAX_PER_SESSION` (선택, 기본 `5`)

## 빠른 시작

1) `.env` 준비

2) `data/knowledge` 예시 문서 준비
- 최소 1개 이상 문서를 아래 카테고리 중 하나에 넣습니다.
  - `data/knowledge/statutes/`
  - `data/knowledge/standard_contracts/`
  - `data/knowledge/case_guides/`
  - `data/knowledge/contract_examples/`
- 지원 포맷: `.txt/.md/.csv/.pdf/.docx/.xlsx`

### 지식 문서 최소 체크리스트

- [ ] 카테고리별 최소 1개 문서 확보
- [ ] 루트가 아닌 카테고리 하위 폴더에 배치
- [ ] 문서 최신성 기준: 최근 3개월 이내 문서 우선
- [ ] 금지 콘텐츠 제외: 개인정보 원문, 저작권 위반 원문 전문, 근거 불명확 자료

### 지식 문서 메타데이터 최소 필드

| 필드 | 설명 | 예시 |
|---|---|---|
| `collected_at` | 문서 수집/업로드 일시(ISO 권장) | `2026-03-05` |
| `source_url` | 원문 출처 URL | `https://law.go.kr/...` |
| `curator` | 요약/정리 담당자 또는 팀 | `legalpilot-team` |
| `license` | 사용 가능 라이선스/내부 사용 정책 | `CC-BY-4.0`, `internal-use` |

3) 실행
- API: `python scripts/run_api.py`
- UI: `python scripts/run_streamlit.py`
- 접속: `http://localhost:8501`

## 실행 옵션

### 1) CLI (빠른 테스트)

```bash
python main.py \
  --query "근로계약서의 포괄임금제 조항이 적법한지 검토해줘" \
  --document-type "근로계약서" \
  --contract-text "월 급여 300만 원(포괄임금제), 경업금지 3년" \
  --reference-text "근로기준법 제17조: 임금 구성항목 서면 명시 의무"
```

### 2) FastAPI

```bash
python scripts/run_api.py
```

- Health: `http://127.0.0.1:8000/health`
- Chat endpoint: `POST http://127.0.0.1:8000/chat`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

```json
{
  "session_id": "api-demo-1",
  "user_query": "포괄임금제 조항의 위험 요소를 분석해줘.",
  "document_type": "근로계약서",
  "contract_text": "월 급여 300만 원(포괄임금제 적용)",
  "reference_text": "근로기준법 제53조: 연장근로 1주 12시간 한도"
}
```

```json
{
  "summary": "요약...",
  "clause_analysis": ["..."],
  "risk_findings": ["..."],
  "revision_plan": ["..."],
  "references": [{"rank": 1, "source": "근로기준법_요약.md", "score": 0.82}],
  "route": "full_review",
  "node_status": {"clause": {"status": "ok", "error_code": null, "detail": null}}
}
```

### 3) Streamlit

```bash
python scripts/run_streamlit.py
```

- UI는 계약서 파일 업로드를 지원합니다: `.txt`, `.md`, `.csv`, `.pdf`, `.docx`, `.xlsx`
- UI는 참조 법령/표준 계약서 파일 업로드 및 텍스트 입력(선택)을 지원합니다.
- 세션 ID는 내부에서 자동 관리되며, 새 대화는 `새 대화 시작` 버튼으로 생성합니다.
- 사이드바에서 실행 입력 기록 조회/삭제를 제공합니다.
- 사이드바 "디버그/신뢰도 표시"에서 `참고 출처 메타데이터 표시` 토글을 켜면 references의 `collected_at/source_url/curator/license`를 함께 확인할 수 있습니다.

## 문제 해결 (Troubleshooting)

- `ModuleNotFoundError: No module named 'src'`
  - 반드시 루트에서 실행하세요.
- `Missing environment variables: AOAI_ENDPOINT`
  - `.env`와 필수 키를 확인하세요.
- 지식 문서 업데이트 후 인덱스가 오래된 것 같을 때
  - `.env`에 `INDEX_FORCE_REBUILD=true`를 설정하고 1회 재실행하세요.
- FAISS 캐시 로드 안전성
  - 운영 배포에서는 `FAISS_ALLOW_DANGEROUS_DESERIALIZATION=false`로 캐시 로드 대신 재생성을 사용하세요.
- `/chat`에서 API `400` 또는 `500`이 반환될 때
  - `400`: 입력/설정 이슈(예: 환경변수 누락)
  - `500`: 런타임/모델/검색 이슈
  - 에러 계약 예시: `{"error_code":"CONFIG_MISSING_ENV","detail":"Missing environment variables: AOAI_ENDPOINT"}`

## 차별성 지표 자동화

```bash
python scripts/evaluate_differentiation_metrics.py \
  --cases data/eval/sample_queries.json \
  --output docs/evidence/metrics_run_output.txt
```

- 지표: 라우팅 정확도(`expected_route`가 있을 때), 근거 포함률, 수정 계획 품질률
- 샘플 구성: 총 25건(`clause_only` 5, `risk_only` 5, `advice_only` 5, `full_review` 5, 모호 질의 5)
- 기준 조정: `--min-routing-accuracy`, `--min-reference-rate`, `--min-plan-quality-rate`
- 분포 검증: `--min-per-labeled-route`, `--min-ambiguous-cases`
