# E2E 테스트 체크리스트

CLI/FastAPI/Streamlit 통합 검증을 위한 제출용 체크리스트입니다.

## 0) 사전 준비

- [ ] 프로젝트 루트로 이동: `/`
- [ ] 의존성 설치: `pip install -r requirements.txt`
- [ ] `.env`에 필수 AOAI 설정값 입력
- [ ] (선택) 저장/개인정보 관련 변수 확인: `UI_HISTORY_PERSIST_ENABLED`, `UI_HISTORY_PII_MASK`, `SESSION_MEMORY_PERSIST_ENABLED`, `SESSION_MEMORY_PII_MASK`
- [ ] `data/knowledge`에 최소 1개 이상의 문서가 있는지 확인

샘플 `contract_text`:

```text
월 급여 300만 원(연장·야간·휴일 근로수당 포함, 포괄임금제 적용)
수습 기간 3개월(최저임금의 80% 지급)
경업금지: 퇴사 후 3년 동안 동종업계 취업 금지
비밀유지: 퇴사 후 5년간 업무 관련 정보 공개 금지
```

샘플 `reference_text`:

```text
근로기준법 제17조: 임금의 구성항목, 계산방법, 지급방법을 서면으로 명시해야 한다.
근로기준법 제53조: 당사자 합의로도 1주에 12시간을 초과하는 연장근로는 불가하다.
근로기준법 제35조: 수습 사용 중인 근로자는 3개월 이내일 경우 해고 예고 예외.
```

## 1) CLI E2E

### 1-1. 계약서 + 참조 법령 동시 실행 (full_review)

- [ ] 실행:

```bash
python main.py \
  --session-id "cli-e2e-1" \
  --document-type "근로계약서" \
  --query "포괄임금제 조항의 적법성과 위험 요소, 수정 방향을 통합 검토해줘." \
  --contract-text "월 급여 300만 원(포괄임금제 적용), 경업금지 3년, 수습 3개월(80% 지급)" \
  --reference-text "근로기준법 제17조: 임금 구성항목 서면 명시. 제53조: 연장근로 1주 12시간 한도."
```

- [ ] 검증:
  - [ ] JSON 응답이 반환된다.
  - [ ] `summary`, `clause_analysis`, `risk_findings`, `revision_plan`, `references` 키가 존재한다.
  - [ ] `references`가 1개 이상 포함된다.
  - [ ] `revision_plan`이 4개 이상 포함된다.
  - [ ] `route`가 `full_review`로 반환된다.

### 1-2. 조항 분석 전용 라우팅 확인 (clause_only)

- [ ] 실행:

```bash
python main.py \
  --session-id "cli-e2e-2" \
  --document-type "근로계약서" \
  --query "조항 분석만 해줘. 위험 진단과 수정 계획은 제외해." \
  --contract-text "월 급여 300만 원, 주 40시간 근무, 퇴직금 법정 기준"
```

- [ ] 검증:
  - [ ] `route`가 `clause_only`로 반환된다.
  - [ ] `risk_findings`, `revision_plan`이 빈 배열이다.

### 1-3. 수정 계획 전용 라우팅 확인 (advice_only)

- [ ] 실행:

```bash
python main.py \
  --session-id "cli-e2e-3" \
  --document-type "NDA" \
  --query "수정 계획만 간단히 작성해줘." \
  --contract-text "비밀유지 기간 10년, 위반 시 손해배상 5000만 원"
```

- [ ] 검증:
  - [ ] `route`가 `advice_only`로 반환된다.
  - [ ] `clause_analysis`, `risk_findings`가 빈 배열이다.

## 2) FastAPI E2E

### 2-1. 서버 실행

- [ ] 실행: `python scripts/run_api.py`
- [ ] 헬스체크 확인: `http://127.0.0.1:8000/health`

### 2-2. `/chat` 호출 (full_review)

- [ ] 실행:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
    "session_id": "api-e2e-1",
    "user_query": "포괄임금제 조항을 검토하고 위험 요소와 수정 방향을 알려줘.",
    "document_type": "근로계약서",
    "contract_text": "월 급여 300만 원(포괄임금제 적용), 경업금지 3년",
    "reference_text": "근로기준법 제17조: 임금 구성항목 서면 명시. 제53조: 연장근로 1주 12시간 한도."
  }'
```

- [ ] 검증:
  - [ ] HTTP 200 응답
  - [ ] `clause_analysis`, `risk_findings`, `revision_plan`, `references` 필드 존재
  - [ ] 한국어 응답
  - [ ] `references` 포함

### 2-3. 에러 처리 확인 (선택)

- [ ] 설정/입력 오류 시 400/500 안내 메시지가 명확한지 확인한다.
- [ ] 에러 계약 형식: `{"error_code":"...", "detail":"..."}`

## 3) Streamlit E2E

### 3-1. UI 실행

- [ ] 실행: `python scripts/run_streamlit.py`
- [ ] 접속: `http://localhost:8501`

### 3-2. 계약서 검토 입력/실행 시나리오

- [ ] 질문 입력 및 계약서 유형 선택 (`근로계약서` / `임대차계약서` / `NDA` / `용역계약서`)
- [ ] `계약서 텍스트` 입력(또는 계약서 파일 업로드: `.txt/.md/.csv/.pdf/.docx/.xlsx`)
- [ ] `참조 법령/표준 계약서 텍스트` 입력(선택, 또는 파일 업로드)
- [ ] `에이전트 실행` 클릭

- [ ] 검증:
  - [ ] 결과 카드(요약/조항 분석/위험 조항 탐지/수정 계획/참고 출처) 렌더링
  - [ ] summary에 "법적 자문이 아닙니다" 면책 문구 포함
  - [ ] 실행 입력 기록 저장 및 사이드바 표시
  - [ ] `다시 불러오기` 시 질문/계약서유형/계약서/참조법령/결과 복원
  - [ ] `새 대화 시작` 시 새 세션으로 초기화
  - [ ] 입력창 `max_chars` 카운터 정상 동작
  - [ ] 업로드 반영 방식(`덮어쓰기/추가하기`) 전환 시 계약서/참조 법령 텍스트가 의도한 방식으로 반영
  - [ ] 기록 저장 모드(`summary`/`full`) 전환 시 저장 필드가 정책대로 반영
  - [ ] 디버그 메타에서 `node_status`가 노드별(`ok/degraded/skipped`)로 표시
  - [ ] `참고 출처 메타데이터 표시` 토글 ON/OFF에 따라 `collected_at/source_url/curator/license` 노출 전환
  - [ ] 카테고리 품질 경고가 사이드바에 노출되는지 확인

### 3-2-1. 라우트별 검증 시나리오

| 시나리오 | 예상 route | 확인 항목 |
|---|---|---|
| "조항 분석만 해줘. 위험/수정 제외" | `clause_only` | `risk_findings`/`revision_plan` 빈 배열 |
| "위험 조항만 확인해줘" | `risk_only` | `clause_analysis`/`revision_plan` 빈 배열 |
| "수정 계획만 간단히 작성해줘" | `advice_only` | `clause_analysis`/`risk_findings` 빈 배열 |
| "조항·위험·수정 계획 모두 검토해줘" | `full_review` | 3개 섹션 모두 항목 포함 |

### 3-2-2. 기록 저장 모드/마스킹 조합 정책(점검표)

| 저장 모드 | PII 마스킹 | 저장되는 입력 필드 | 저장되는 응답 필드 |
|---|---|---|---|
| `summary` | OFF | `contract_text`/`reference_text` 원문 미저장, `contract_len/ref_len`, `contract_hash/ref_hash`, 미리보기, `run_id` 저장 | 요약 응답(`summary`, route 메타, 상위 항목, compact refs) 저장 |
| `summary` | ON | 위 `summary` 저장 필드 동일 + 저장 직전 이메일/전화번호 마스킹 적용 | 위 `summary` 응답 필드 동일 + 문자열 필드 마스킹 적용 |
| `full` | OFF | `contract_text`/`reference_text` 원문 + 길이/해시 + `run_id` 저장 | 응답 원문(`response`) 전체 저장 |
| `full` | ON | 위 `full` 저장 필드 동일 + 저장 직전 이메일/전화번호 마스킹 적용 | 위 `full` 응답 필드 동일 + 문자열 필드 마스킹 적용 |

### 3-3. 재시작 후 기록 유지 확인

- [ ] Streamlit 종료(`Ctrl+C`) 후 재실행
- [ ] 사이드바 기록이 유지되는지 확인 (기록 저장 옵션 ON 기준)
- [ ] `실행 입력 기록 파일 저장 사용` OFF 시 디스크 로드/저장이 중단되는지 확인

## 4) 차별성 지표 자동 검증

- [ ] 실행:

```bash
python scripts/evaluate_differentiation_metrics.py \
  --cases data/eval/sample_queries.json \
  --output docs/evidence/metrics_run_output.txt
```

- [ ] 검증:
  - [ ] 라우팅 정확도 출력 (`clause_only/risk_only/advice_only/full_review` 기준)
  - [ ] 근거 포함률 출력
  - [ ] 수정 계획 품질률 출력
  - [ ] 임계치 충족 시 `[PASS] All thresholds satisfied.` 출력

## 5) 제출 산출물 점검

- [ ] `docs/evidence/agent_execution_log.md` 최신화 (LegalPilot 실행 기반)
- [ ] `docs/evidence/agent_final_answer.json` 최신화 (법률 문서 검토 결과)
- [ ] `docs/evidence/metrics_run_output.txt` 최신화
- [ ] Streamlit 캡처: `docs/evidence/streamlit_*.png`
- [ ] `.env` 제출 안전 점검: `python scripts/check_env_submission_safety.py --strict` 실행 후 경고가 없거나, `.env`는 제출물에서 제외했는지 확인
- [ ] (선택) 지표 자동검증 실행 결과 캡처 첨부
