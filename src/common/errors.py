"""Common application error contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ErrorCodes:
    """Centralized error code registry for stable API/UI contracts."""

    CONFIG_MISSING_ENV = "CONFIG_MISSING_ENV"
    KNOWLEDGE_EMPTY = "KNOWLEDGE_EMPTY"
    SERVICE_RUNTIME_ERROR = "SERVICE_RUNTIME_ERROR"
    STRUCTURED_OUTPUT_CONTRACT_ERROR = "STRUCTURED_OUTPUT_CONTRACT_ERROR"
    STRUCTURED_OUTPUT_CLAUSE_FALLBACK = "STRUCTURED_OUTPUT_CLAUSE_FALLBACK"
    STRUCTURED_OUTPUT_RISK_FALLBACK = "STRUCTURED_OUTPUT_RISK_FALLBACK"
    STRUCTURED_OUTPUT_ADVICE_FALLBACK = "STRUCTURED_OUTPUT_ADVICE_FALLBACK"
    STRUCTURED_OUTPUT_SYNTHESIS_FALLBACK = "STRUCTURED_OUTPUT_SYNTHESIS_FALLBACK"


@dataclass
class LegalPilotError(Exception):
    error_code: str
    detail: str
    status_code: int = 400

    def __str__(self) -> str:
        return self.detail

    def to_payload(self) -> dict[str, Any]:
        return {"error_code": self.error_code, "detail": self.detail}
