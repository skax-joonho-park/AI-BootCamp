"""Reusable prompt templates and policy lines for workflow nodes."""

from __future__ import annotations

PROMPT_RULE_KOREAN_ONLY = "중요: 답변은 반드시 한국어로만 작성하세요."
PROMPT_RULE_NO_COT = "중요: 생각 과정을 노출하지 말고 결과만 제시하세요."
PROMPT_RULE_PLAN_CITATION = "중요: 근거 번호 citation([1][2])을 수정 계획 항목 끝에 표기하세요."
PROMPT_RULE_NO_GUARANTEE = (
    "중요: 법적 결과를 보장하거나 단정하지 말고, 조건부 표현과 근거 기반 검토 의견으로 작성하세요."
)

PROMPT_RULE_RESULT_ONLY = "생각 과정을 노출하지 말고 결과만 작성하세요."
PROMPT_RULE_ALL_FIELDS_KOREAN = "모든 필드는 한국어로 작성하세요."

TOOL_LOOP_FINALIZATION_INSTRUCTION = (
    "위 대화와 도구 결과를 종합하여 최종 결과를 작성하세요. "
    "더 이상 도구를 호출하지 말고, 생각 과정을 노출하지 말고 결과만 작성하세요. "
    "반드시 한국어로 작성하세요."
)

STRUCTURED_JSON_REPAIR_INSTRUCTION = (
    "직전 응답의 구조화 파싱이 실패했습니다. "
    "스키마 필드를 빠짐없이 채우되, 생각 과정 없이 결과만 한국어 JSON으로 다시 작성하세요."
)

ROUTE_NAME_DESCRIPTIONS = {
    "clause_only": "조항 분석 중심",
    "risk_only": "위험 조항 탐지 중심",
    "full_review": "조항 분석 + 위험 탐지 + 수정 계획 통합",
    "advice_only": "수정 계획 위주(간단 검토)",
}

CLAUSE_ONLY_MARKERS = (
    "조항만",
    "조항 분석만",
    "조항만 봐",
    "조항 위주로",
    "조항 중심으로",
    "조항 검토만",
)
RISK_ONLY_MARKERS = (
    "위험만",
    "위험 조항만",
    "리스크만",
    "위험 탐지만",
)
ADVICE_ONLY_MARKERS = (
    "수정만",
    "수정 계획만",
    "수정안만",
    "개선안만",
    "수정 위주로",
    "수정계획만",
    "수정 방향만",
    "전체 요약 없이",
)

EXCLUSION_PATTERNS = ("제외", "빼", "빼줘", "제외해")
NEGATION_PATTERNS = ("말고", "말아", "말자", "않", "아니", "원치 않", "싶지 않", "필요 없")

EXCLUSION_TERMS = {
    "clause": ("조항", "조항 분석", "조항 검토"),
    "risk": ("위험", "리스크", "위험 조항"),
    "advice": ("수정", "수정 계획", "수정안"),
}

INTENT_KEYWORDS = {
    "clause": ("조항", "조문", "내용", "검토"),
    "risk": ("위험", "리스크", "불리", "독소"),
    "advice": ("수정", "개선", "보완", "바꾸"),
}

SUPERVISOR_ROUTING_RULE_LINES = (
    '사용자가 제외 요청을 명시(예: "위험 제외", "수정 제외", "조항 제외")하면 해당 제외 의도를 강하게 반영하라.',
    "계약서 텍스트가 미제공이면 clause_only는 가능한 한 피하라.",
    "계약서 텍스트가 미제공이고 조항 분석 요청이면 advice_only 또는 full_review 중 더 안전한 쪽을 선택하라.",
    "위험 조항 탐지 요청이 명확하면 risk_only를 우선 검토하라.",
    "참조 법령/표준 계약서가 제공된 경우, 비교 분석 요청은 clause_only 또는 full_review를 우선 검토하라.",
)


def build_common_policy_block(*, include_plan_citation: bool = False) -> str:
    lines = [PROMPT_RULE_KOREAN_ONLY, PROMPT_RULE_NO_COT, PROMPT_RULE_NO_GUARANTEE]
    if include_plan_citation:
        lines.append(PROMPT_RULE_PLAN_CITATION)
    return "\n".join(lines)


def build_specialist_system_prompt(agent_name: str, focus: str) -> str:
    return (
        f"당신은 {agent_name}입니다. {focus} "
        "출력은 반드시 한국어로, 스키마를 준수한 구조화 결과만 반환하세요."
    )


def build_route_options_block() -> str:
    lines = ["선택 가능한 route:"]
    for route, desc in ROUTE_NAME_DESCRIPTIONS.items():
        lines.append(f"- {route}: {desc}")
    return "\n".join(lines)


def build_supervisor_routing_rules_block() -> str:
    lines = ["[라우팅 규칙]"]
    lines.extend(f"- {item}" for item in SUPERVISOR_ROUTING_RULE_LINES)
    lines.append(
        "- 용어 정의(휴리스틱/LLM 공통): "
        f"clause={INTENT_KEYWORDS['clause']}, risk={INTENT_KEYWORDS['risk']}, advice={INTENT_KEYWORDS['advice']}"
    )
    return "\n".join(lines)
