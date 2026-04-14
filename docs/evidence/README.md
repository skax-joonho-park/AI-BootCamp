# 제출 산출물(Evidence)

LegalPilot AI — 법률 문서 검토 멀티 에이전트 최종 제출 증빙 파일 목록입니다.

## 파일 목록

| 파일 | 설명 |
|---|---|
| `agent_execution_log.md` | Supervisor 라우팅 + RAG 근거(refs) 포함 실행 로그 |
| `agent_final_answer.json` | 구조화 최종 결과 JSON (`clause_analysis/risk_findings/revision_plan/references`) |
| `metrics_run_output.txt` | 차별성 지표 자동검증 결과 (`[PASS] All thresholds satisfied.`) |
| `e2e_test_checklist.md` | CLI/FastAPI/Streamlit E2E 테스트 체크리스트 |
| `streamlit_legalpilot_initial.png` | Streamlit 초기 UI 화면 (LegalPilot AI 브랜딩, 입력 폼) |
| `streamlit_legalpilot_full_review_result.png` | full_review 실행 결과 화면 (요약/조항 분석/위험 조항/수정 계획 카드) |
| `streamlit_legalpilot_results_closeup.png` | 결과 카드 확대 캡처 |

## 생성 방법

### 1) 에이전트 실행 로그/결과 JSON 자동 생성

프로젝트 루트(`AI-BootCamp/`)에서:

```bash
python scripts/generate_submission_evidence.py
```

필요 시 입력 커스터마이즈:

```bash
python scripts/generate_submission_evidence.py \
  --query "포괄임금제 조항의 적법성과 위험 요소를 통합 검토해줘." \
  --document-type "근로계약서" \
  --contract-text "월 급여 300만 원(포괄임금제 적용), 경업금지 3년" \
  --reference-text "근로기준법 제17조: 임금 구성항목 서면 명시"
```

### 2) Streamlit 화면 캡처

1. `python scripts/run_streamlit.py` 실행
2. `http://localhost:8501/` 접속
3. 계약서 유형/질문/계약서 텍스트 입력 후 `에이전트 실행` 클릭
4. 결과 카드가 표시되는 화면 캡처 → `docs/evidence/streamlit_legalpilot_*.png`로 저장

### 3) 차별성 지표 자동검증 로그

```bash
python scripts/evaluate_differentiation_metrics.py \
  --cases data/eval/sample_queries.json \
  --output docs/evidence/metrics_run_output.txt
```

- 지표: 라우팅 정확도(macro recall), 근거 포함률, 수정 계획 품질률
- 케이스: `clause_only(5)`, `risk_only(5)`, `advice_only(5)`, `full_review(5)`, 모호(5) = 총 25건
- 임계치 충족 시: `[PASS] All thresholds satisfied.`
