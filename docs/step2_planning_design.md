[Step2 - 기획 및 설계]

**[서비스명 - LegalPilot AI (리걸파일럿)]**

**1. 프로젝트 개요 – 기획 배경 및 핵심 내용**

### **1.1 프로젝트 기획 배경**

- **어떤 문제를 해결하고자 하는가?**
  법률 문서(근로계약서, 임대차계약서, NDA, 용역계약서 등) 검토 과정에서 법령 기준·표준 계약서·분쟁 사례를 직접 대조하는 작업이 복잡하고 전문성을 요구하는 문제를 해결하고자 합니다.

- **기존 방식의 한계는 무엇인가?**
  비전문가가 법령 원문을 직접 해석하거나, 변호사 자문을 별도로 구해야 하는 비용·접근성 문제가 있습니다. 단순 검색은 계약 맥락을 반영하지 못합니다.

- **Agent 서비스로 해결할 수 있는 Pain Point는 무엇인가?**
  멀티 에이전트가 역할을 분담하여(조항 분석, 위험 탐지, 수정 계획 수립) 계약서와 관련 법령을 대조해 근거 기반 검토 의견을 빠르게 제공합니다.

- **이 프로젝트를 시작하게 된 동기는 무엇인가?**
  계약서 검토는 정보 탐색보다 "위험 식별 + 수정 우선순위"가 핵심인데, 이를 자동화·구조화한 서비스 수요가 높다고 판단했습니다.

### **1.2 핵심 아이디어 및 가치 제안(Value Proposition)**

- **서비스가 제공하는 핵심 기능은 무엇인가?**
  1) 계약서 조항 분석(`contract_text` 기반 핵심/문제 조항 파악), 2) 위험 조항 탐지(법령 위반 가능성·불리한 조건), 3) 수정 계획 수립(우선순위·검토 항목 중심 액션 플랜)

- **사용자에게 제공되는 가치와 기대효과는 무엇인가?**
  계약 체결 전 위험 식별, 수정 우선순위 명확화, 법령 근거 기반 검토 의견으로 계약 리스크 최소화

- **기존 서비스 대비 차별성은 무엇인가?**
  단일 챗봇이 아닌 역할 기반 Multi-Agent + RAG 결합으로 "근거 있는 조항 검토 + 수정 계획"까지 종단 간(End-to-End) 제공

- **차별성 검증 포인트(단일 챗봇 대비)**
  - 라우팅 정확도: 의도 라벨(`clause_only`, `risk_only`, `advice_only`, `full_review`) 기준 샘플 질의셋 Top-1 정확도 측정
  - 근거 포함률: 최종 응답에서 `references`가 1개 이상 포함된 비율과 근거-본문 의미 일치 여부 측정
  - 수정 계획 품질: `revision_plan` 항목의 실행 가능성(구체 행동/우선순위 포함 여부) 점수화 비교
  - 자동화 경로: `scripts/evaluate_differentiation_metrics.py` + `data/eval/sample_queries.json`로 배치 평가(임계치 미달 시 실패 코드 반환)
  - 경계 조건 운영 원칙: 샘플 질의셋은 고정 라벨 케이스(`clause_only/risk_only/advice_only/full_review`)와 무라벨 모호 질의를 혼합해 라우팅 안정성과 실서비스 유사성을 함께 점검
  - 분포 기준: 현재 샘플 구성은 총 25건(`clause_only` 5, `risk_only` 5, `advice_only` 5, `full_review` 5, 모호 질의 5)

### **1.3 대상 사용자 및 기대 사용자 경험(UX)**

- **주요 타겟**
  개인 사업자, 프리랜서, 임차인, 중소기업 담당자 — 법률 전문가 없이 계약서를 직접 검토해야 하는 비전문가

- **사용자에게 어떤 흐름과 경험을 제공할 것인가?**
  질문 입력/계약서 유형 선택 → 계약서 업로드(파일/텍스트) + 참조 법령 입력(선택) → 조항 분석/위험 탐지/수정 계획 확인 → 입력 기록 저장 → 필요 시 "다시 불러오기"로 과거 입력/결과 복원

- **사용자가 서비스에서 얻는 구체적 Benefit은 무엇인가?**
  "어떤 조항이 문제인가", "어떤 법령 위반이 우려되는가", "어떻게 수정해야 하는가"가 명확한 체크리스트와 법령 근거 기반 조언

**2. 기술 구성 – 서비스에 적용할 기술 스택**

### **2.1 Prompt Engineering 전략**

- **역할 기반 프롬프트**
  Supervisor, Clause Agent, Risk Agent, Advice Agent, RAG Agent 각각의 시스템 프롬프트 분리

- **고품질 응답 전략(Few-shot + 근거 기반 요약)**
  계약서 유형별 예시 답변(Few-shot) + 근거 법령/출처 기반 요약 + 금지 규칙(근거 없는 단정 금지, 법률 자문 보장 표현 금지)
  - Tool 활용 실효성 강화: 도구 출력은 JSON(점수/키워드/이슈 배열)으로 표준화하고, Agent가 이를 최소 1회 이상 본문에 반영하도록 지시
  - 운영 비용 제어: Few-shot은 계약서 유형별 최소 예시만 선택 주입(`FEW_SHOT_MAX_EXAMPLES`)

- **출력 구조화 템플릿 정의**
  JSON 스키마 기반 출력(요약, 조항 분석, 위험 조항, 수정 계획, 근거 출처)
  - route-aware 출력 규칙: `clause_only/risk_only/advice_only/full_review` 라우트에 따라 섹션 최소 개수/표시 여부를 다르게 적용
  - 법률 면책 문구 자동 첨부: 모든 응답 summary에 "본 검토 의견은 법적 자문이 아님" 고지를 자동 삽입

### **2.2 LangChain / LangGraph 기반 Agent 구조**

- **Multi-Agent 설계 개념**
  Supervisor가 사용자 요청을 분류/라우팅하고, RAG 근거를 공통으로 확보한 뒤 Clause/Risk/Advice Agent가 분업 처리한 결과를 통합해 최종 응답 생성

- **각 Agent의 역할(Role) 정의**
  - `Supervisor(Planner 역할 포함)`: 요청 분해, 우선순위 결정, 결과 통합
  - `Clause Agent`: 핵심·문제 조항 분석 및 evidence_map 생성
  - `Risk Agent`: 위험 조항 탐지 및 법적 우려 사항 식별
  - `Advice Agent`: 우선순위·수정 방향 중심 액션 플랜 수립
  - `RAG Agent`: 법령/표준 계약서/판례 검색 및 근거 제공

- **Tool Calling, ReAct, Memory 활용 여부**
  Tool Calling(필수), ReAct 스타일 도구 루프(max step 기반 종료), 세션 메모리 + LangGraph Checkpointer 적용

### **2.3 RAG 구성**

- **데이터 수집/전처리 파이프라인**
  법령 요약본, 표준 계약서, 분쟁 사례 가이드, 계약서 예시 문서(TXT/MD/CSV/PDF/DOCX/XLSX) 수집 → 정규화 → 청킹

- **임베딩 모델 및 Vector DB 선택**
  `text-embedding-ada-002`(환경변수 `AOAI_DEPLOY_EMBED_ADA`) + FAISS

- **검색 로직과 응답 생성 방식**
  하이브리드 검색(BM25 + 벡터 유사도) → 상위 문서 재정렬 → 출처 포함 응답 생성
  - 업로드 입력 근거 편입: `contract_text/reference_text`를 임시 청크(ephemeral evidence)로 생성해 검색 후보에 혼합
  - 임시 근거 점수 파라미터화: `EPHEMERAL_REF_BASE_SCORE`, `EPHEMERAL_CONTRACT_BASE_SCORE`, `EPHEMERAL_OVERLAP_WEIGHT`로 분리

- **도메인 지식 범위(출처 유형/라이선스/최신성)**
  - 출처 유형: 법령 요약본, 표준 계약서, 법원 분쟁 사례 가이드, 계약서 예시
  - 저장 경로/카테고리: `data/knowledge/statutes/*`, `data/knowledge/standard_contracts/*`, `data/knowledge/case_guides/*`, `data/knowledge/contract_examples/*`
  - 라이선스 원칙: 공개 법령 자료 중심 구성, 저작권 제약이 있는 원문은 요약/메타데이터만 반영

- **RAG 안전 정책(신뢰성 설계)**
  - 근거 부족 시 전환: "일반 법령 가이드 기반 조언"으로 전환하고 단정형 표현을 제한
  - 법률 자문 면책 고지: 모든 응답에 "본 검토 의견은 법적 자문이 아님" 문구 삽입
  - 도메인 범위 고지: 지식 범위를 벗어나는 질문은 전문 법률가 상담을 안내

### **2.4 서비스 개발 및 패키징 계획**

- **UI 개발 방식**
  Streamlit 기반 대화형 UI(질문/계약서 유형 선택, 계약서 + 참조 법령 파일 업로드, 결과 카드, 실행 입력 기록 조회/삭제/다시 불러오기)

- **BE(API) 및 배포 전략**
  FastAPI로 Agent 실행 API 분리, 예외 처리(400/500)로 사용자 친화적 오류 응답 제공
  - 에러 처리 표준화: `error_code/detail` 공통 계약(`LegalPilotError`)을 API/CLI/UI에 일관 적용

- **설정/환경 관리 계획**
  `.env` 사용, `src/config/settings.py`에서 로드, `requirements.txt`로 의존성 고정

### **2.5 선택적 확장 기능**

- **LLM Fundamentals 기반 Structured Output / Function Calling**
  결과를 JSON으로 강제하여 UI 렌더링 안정화

- **안정성/복원성 확장(운영 관점)**
  Structured Output 실패 시 노드별 fallback(최소 필드 degrade) 적용, 체크포인터 기반 세션 복원으로 재실행 비용 최소화
  - 부분 실패 계약: `ChatResponse.node_status`(ok/degraded/skipped, error_code/detail)로 노드별 상태를 명시

### **2.6 설계-구현 1:1 매핑**

| 설계 정책(문서) | 구현 위치(코드) | 구현 방식 요약 |
|---|---|---|
| Route-aware 출력 정책 | `src/workflow/engine.py` (`route_minimums`, `normalize_final_answer_by_route`, `enforce_final_answer_policy`) | 라우트별 최소 항목/섹션 표시 규칙을 코드 후처리로 강제 |
| 부분 실패 격리 계약 | `src/workflow/engine.py` (`derive_node_status`), `src/workflow/contracts.py` (`ChatResponse.node_status`) | 노드 단위 fallback/degraded를 응답 메타로 노출하고 전체 요청은 유지 |
| Tool Calling 능동 실행 | `src/workflow/engine.py` (`_run_tool_loop_structured_with_trace`), `src/agents/tools.py` | `bind_tools()`로 도구를 모델에 주입하고 `tool_calls`를 모델이 자율 선택 |
| 체크포인터/메모리 역할 분리 | `src/workflow/engine.py` (`MemorySaver`, `final_answer_cache`), `src/utils/memory.py` (`SessionMemory`) | 런타임 그래프 복원 vs 재시작 후 영속 대화/결과 캐시를 분리 운영 |
| RAG 안전/근거 추적 | `src/workflow/engine.py` (`rag_node`), `src/retrieval/hybrid.py`, `src/retrieval/rerank.py` | 하이브리드 검색 + 재정렬 + 저신뢰 안전모드 + 구조화 references/score_breakdown 제공 |
| 차별성 자동평가 루프 | `scripts/evaluate_differentiation_metrics.py`, `data/eval/sample_queries.json` | 라우팅/근거 포함/수정 계획 품질 지표를 배치 실행으로 자동 검증 |

#### 2.6-1 기술별 실제 적용 위치(1페이지 요약)

| 기술 요소 | 실제 적용 위치 | 확인 포인트 |
|---|---|---|
| Supervisor 라우팅 | `src/workflow/engine.py::supervisor_node`, `route_after_supervisor` | 의도 분류 후 `clause/risk/advice/synthesis` 조건 분기 |
| 하이브리드 검색 | `src/retrieval/hybrid.py::HybridRetriever.search` | FAISS + BM25 융합, 쿼리-독립 점수 스케일 변환 |
| 리랭크/다양성 제어 | `src/retrieval/rerank.py::rerank_hits` | category boost + source당 max 청크 제한 |
| RAG 오케스트레이션 | `src/workflow/engine.py::rag_node` | route-aware filter, no-hit fallback, low-confidence 안전모드 |
| 실행 기록 요약 저장 | `src/ui/history_record.py::build_history_record` | `summary/full` 저장 모드, `record_version` 기반 역호환 |
| UI 업로드 파싱 공통화 | `src/utils/file_extract.py` | UI/RAG가 동일 파서를 공유해 포맷별 예외 처리 단일화 |

### **2.7 차별성 검증 데이터셋 확장 가이드**

- 고정 라벨 케이스(`clause_only/risk_only/advice_only/full_review`)와 무라벨 모호 질의를 혼합해 라우팅 안정성과 실서비스 유사성을 함께 점검
- 경계 조건(부정문 라우팅, 복합 의도)을 별도로 추가해 휴리스틱 라우팅 내성을 검증
- `data/eval/sample_queries.json` 필드: `query`, `document_type`, `contract_text`, `reference_text`, `expected_route`

| 필드 | 설명 |
|---|---|
| `collected_at` | 문서 수집/업로드 일시(ISO 권장) |
| `source_url` | 원문 출처 URL |
| `curator` | 요약/정리 담당자 또는 팀 |
| `license` | 사용 가능 라이선스/내부 사용 정책 |
