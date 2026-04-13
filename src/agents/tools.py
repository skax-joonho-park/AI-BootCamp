"""Tools used by specialist agents.

These functions are exposed as LangChain tools via `@tool`.
Actual execution is *agentic* (LLM-selected), not manual:
- nodes provide a tool list to `model.bind_tools([...])`
- the model decides whether/which tool to call
- tool outputs are returned as JSON strings for deterministic parsing/reflection checks
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from langchain_core.tools import tool


def _infer_document_keywords(document_type: str) -> tuple[str, list[str], list[str]]:
    doc_required = {
        "근로계약서": ["임금", "근로시간", "휴가", "업무내용", "계약기간", "해고", "퇴직금"],
        "임대차계약서": ["임대료", "보증금", "임대기간", "관리비", "원상복구", "갱신", "해지"],
        "nda": ["비밀정보", "기밀", "공개금지", "유효기간", "손해배상", "반환"],
        "용역계약서": ["용역범위", "납기", "대가", "지적재산권", "하자보증", "해제"],
    }
    doc_preferred = {
        "근로계약서": ["연장근로", "포괄임금", "경업금지", "전직금지", "손해배상"],
        "임대차계약서": ["분쟁조정", "중도해지", "특약사항", "인테리어", "원상회복"],
        "nda": ["위약벌", "유지보수", "제3자", "공개예외", "관할법원"],
        "용역계약서": ["변경계약", "지체상금", "검수기준", "분쟁해결", "원하청"],
    }

    key = "근로계약서"
    lowered = document_type.lower()
    if "임대" in document_type or "임차" in document_type or "전세" in lowered:
        key = "임대차계약서"
    elif "nda" in lowered or "비밀" in document_type or "기밀" in document_type:
        key = "nda"
    elif "용역" in document_type or "도급" in document_type or "위탁" in document_type:
        key = "용역계약서"
    return key, doc_required[key], doc_preferred[key]


_SYNONYM_CANONICAL = {
    "임금": "임금",
    "급여": "임금",
    "월급": "임금",
    "보수": "임금",
    "전세": "보증금",
    "월세": "임대료",
    "해고": "해고",
    "해임": "해고",
    "파면": "해고",
    "nda": "nda",
    "비밀유지": "nda",
    "기밀유지": "nda",
}


def _normalize_token(token: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣]+", "", token.lower()).strip()
    if not cleaned:
        return ""
    return _SYNONYM_CANONICAL.get(cleaned, cleaned)


@lru_cache(maxsize=1)
def _get_kiwi():
    try:
        from kiwipiepy import Kiwi
    except ModuleNotFoundError:
        return None
    return Kiwi()


def _tokenize(text: str) -> set[str]:
    kiwi = _get_kiwi()
    if kiwi is not None:
        tokens: set[str] = set()
        for token in kiwi.tokenize(text):
            norm = _normalize_token(token.form)
            if norm:
                tokens.add(norm)
        return tokens
    rough = re.findall(r"[0-9A-Za-z가-힣]+", text.lower())
    return {norm for token in rough if (norm := _normalize_token(token))}


def _expand_keyword_aliases(keyword: str) -> set[str]:
    base = _normalize_token(keyword)
    if not base:
        return set()
    aliases: dict[str, set[str]] = {
        "임금": {"임금", "급여", "월급", "보수"},
        "보증금": {"보증금", "전세"},
        "임대료": {"임대료", "월세"},
        "해고": {"해고", "해임", "파면"},
        "nda": {"nda", "비밀유지", "기밀유지"},
    }
    for canonical, words in aliases.items():
        if base in words:
            return {_normalize_token(item) for item in words}
    return {base}


@tool
def clause_keyword_match_score(contract_text: str, document_type: str) -> str:
    """Estimate keyword coverage score of a contract for a given document type."""
    key, doc_keywords, _ = _infer_document_keywords(document_type)
    tokens = _tokenize(contract_text)
    matched: list[str] = []
    for keyword in doc_keywords:
        aliases = _expand_keyword_aliases(keyword)
        if aliases & tokens:
            matched.append(keyword)
    score = int((len(matched) / max(len(doc_keywords), 1)) * 100)
    payload = {
        "document_type": document_type,
        "doc_key": key,
        "match_score": score,
        "matched_keywords": matched,
        "missing_keywords": [kw for kw in doc_keywords if kw not in matched],
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def legal_issue_bank(document_type: str) -> str:
    """Return common legal issues and risk areas for a given document type."""
    mapping = {
        "근로계약서": [
            "포괄임금제 적용 시 연장·야간·휴일 수당 미지급 가능성을 확인하세요.",
            "계약기간 및 갱신 거절 요건이 근로기준법과 부합하는지 검토하세요.",
            "경업금지·비밀유지 조항의 기간·범위가 과도하게 설정되었는지 살펴보세요.",
        ],
        "임대차계약서": [
            "임대인의 일방적 계약 해지 조항이 임차인에게 불리하게 설정되어 있는지 확인하세요.",
            "보증금 반환 시기 및 원상복구 범위가 명확히 규정되어 있는지 검토하세요.",
            "묵시적 갱신 거절 통지 기한이 주택임대차보호법 기준에 부합하는지 살펴보세요.",
        ],
        "nda": [
            "비밀정보의 정의 범위가 지나치게 광범위해 사업 활동을 제한할 수 있는지 검토하세요.",
            "비밀유지 의무 유효기간이 계약 종료 후에도 과도하게 지속되는지 확인하세요.",
            "위반 시 손해배상 조항이 실제 손해를 초과하는 위약벌 성격인지 살펴보세요.",
        ],
        "용역계약서": [
            "지체상금 조항의 상한이 없거나 지나치게 높게 설정되어 있는지 확인하세요.",
            "결과물의 지적재산권 귀속 및 사용 범위가 명확히 규정되어 있는지 검토하세요.",
            "하자보수 기간 및 책임 범위가 수급인에게 일방적으로 불리한지 살펴보세요.",
        ],
    }
    key = "근로계약서"
    lowered = document_type.lower()
    if "임대" in document_type or "임차" in document_type or "전세" in lowered:
        key = "임대차계약서"
    elif "nda" in lowered or "비밀" in document_type or "기밀" in document_type:
        key = "nda"
    elif "용역" in document_type or "도급" in document_type or "위탁" in document_type:
        key = "용역계약서"
    payload = {
        "document_type": document_type,
        "doc_key": key,
        "issues": mapping[key],
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def contract_reference_gap_score(reference_text: str, contract_text: str, document_type: str) -> str:
    """Quantify gap between a reference law/standard contract and the target contract."""
    _, required_keywords, preferred_keywords = _infer_document_keywords(document_type)
    ref_tokens = _tokenize(reference_text)
    contract_tokens = _tokenize(contract_text)

    ref_required = [
        kw for kw in required_keywords if _expand_keyword_aliases(kw) & ref_tokens
    ] or required_keywords
    ref_preferred = [kw for kw in preferred_keywords if _expand_keyword_aliases(kw) & ref_tokens]

    matched_required = [kw for kw in ref_required if _expand_keyword_aliases(kw) & contract_tokens]
    missing_required = [kw for kw in ref_required if not (_expand_keyword_aliases(kw) & contract_tokens)]
    matched_preferred = [kw for kw in ref_preferred if _expand_keyword_aliases(kw) & contract_tokens]
    missing_preferred = [kw for kw in ref_preferred if not (_expand_keyword_aliases(kw) & contract_tokens)]

    required_match_rate = round(
        (len(matched_required) / max(len(ref_required), 1)) * 100.0,
        2,
    )
    preferred_match_rate = round(
        (len(matched_preferred) / max(len(ref_preferred), 1)) * 100.0,
        2,
    ) if ref_preferred else 0.0

    payload = {
        "document_type": document_type,
        "ref_required_keywords": ref_required,
        "ref_preferred_keywords": ref_preferred,
        "matched_required_keywords": matched_required,
        "missing_required_top": missing_required[:5],
        "matched_preferred_keywords": matched_preferred,
        "missing_preferred_top": missing_preferred[:5],
        "required_match_rate": required_match_rate,
        "preferred_match_rate": preferred_match_rate,
    }
    return json.dumps(payload, ensure_ascii=False)
