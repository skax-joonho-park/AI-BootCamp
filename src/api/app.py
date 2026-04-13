"""FastAPI app for LegalPilot service."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.common import LegalPilotError
from src.workflow import ChatRequest, ChatResponse, LegalPilotService


@lru_cache(maxsize=1)
def get_service() -> LegalPilotService:
    return LegalPilotService()


app = FastAPI(title="LegalPilot AI API", version="1.0.0")


@app.exception_handler(LegalPilotError)
def handle_legalpilot_error(_: Request, exc: LegalPilotError) -> JSONResponse:
    # Keep error contract flat for API clients.
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_payload(),
    )


@app.exception_handler(Exception)
def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "detail": f"LegalPilot service failed: {exc}",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    service = get_service()
    return service.run(req)
