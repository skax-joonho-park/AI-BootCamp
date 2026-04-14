# Agent Execution Log

- Generated at: 2026-04-14 09:05:01
- Session ID: evidence-session-legalpilot
- Document type: 근로계약서

## Input
- Query: 근로계약서의 포괄임금제 조항이 적법한지 검토하고 위험 요소와 수정 방향을 알려줘.
- Contract text length: 48
- Reference text length: 60

## Supervisor Routing
- Route: full_review
- Reason: 사용자가 근로계약서의 포괄임금제 조항에 대해 적법성 검토와 위험 요소 및 수정 방향을 요청하였으므로, 조항 분석, 위험 탐지, 수정 계획을 통합적으로 수행하는 것이 적합합니다.

## RAG References
- [1] 근로계약서_문제_예시.md (chunk=42, location=n/a, score=1.0)
- [2] 근로계약서_분쟁사례_가이드.md (chunk=1, location=n/a, score=0.9931)
- [3] 근로계약서_분쟁사례_가이드.md (chunk=4, location=n/a, score=0.97)
- [4] 근로계약서_문제_예시.md (chunk=41, location=n/a, score=0.9324)

## RAG Context Preview
```
[1] (근로계약서_문제_예시.md, score=1.0)
## 예시 1: 포괄임금제 위험 조항

[2] (근로계약서_분쟁사례_가이드.md, score=0.9931)
## 사례 1: 포괄임금제 무효 판결

[3] (근로계약서_분쟁사례_가이드.md, score=0.97)
**실무 시사점**
- 포괄임금제는 근로시간 산정이 곤란한 업무(영업직, 재택근무 등)에만 제한적 유효
- 포괄임금 내 각 수당 항목과 금액을 명세화하지 않으면 무효 위험
- 별도 포괄임금 동의서 작성 권고

[4] (근로계약서_문제_예시.md, score=0.9324)
# 근로계약서 문제 조항 예시 및 수정 방향

[rag-agent-note] route=full_review, source_hint=근로기준법, 포괄임금제, category_filter=none, top_score=1.0000, low_confidence=False
```

## Final Answer Artifact
- JSON file: `docs/evidence/agent_final_answer.json`
