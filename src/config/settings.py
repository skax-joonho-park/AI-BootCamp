"""Shared settings loader for the final project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from src.common import LegalPilotError


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    knowledge_dir: Path
    index_dir: Path
    aoai_endpoint: str
    aoai_api_key: str
    aoai_deployment: str
    aoai_embedding_deployment: str
    aoai_api_version: str
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120
    memory_max_sessions: int = 200
    memory_ttl_seconds: int = 60 * 60 * 24
    index_force_rebuild: bool = False
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rag_evidence_score_threshold: float = 0.45
    ephemeral_ref_base_score: float = 0.42
    ephemeral_contract_base_score: float = 0.36
    ephemeral_overlap_weight: float = 0.35
    rerank_enabled: bool = True
    rerank_provider: str = "heuristic"
    rerank_max_per_source: int = 2
    retrieval_max_chunks_per_file: int = 0
    allow_uncategorized_in_filter: bool = True
    uncategorized_ratio_warn_threshold: float = 0.5
    faiss_allow_dangerous_deserialization: bool = False
    graph_state_cache_enabled: bool = True
    graph_state_cache_bypass_contextual: bool = True
    graph_state_cache_max_per_session: int = 5
    state_store_backend: str = "file"  # file | sqlite | redis (scaffold)
    state_store_dsn: str = ""
    session_memory_persist_enabled: bool = True
    session_memory_pii_mask_enabled: bool = False
    ui_history_persist_enabled: bool = True
    ui_history_pii_mask_enabled: bool = False
    ui_history_storage_mode: str = "summary"
    ui_page_icon_mode: str = "emoji"
    ui_page_icon_emoji: str = "⚖️"
    few_shot_max_examples: int = 1

    @property
    def final_answer_cache_enabled(self) -> bool:
        return self.graph_state_cache_enabled

    @property
    def final_answer_cache_bypass_contextual(self) -> bool:
        return self.graph_state_cache_bypass_contextual

    @property
    def final_answer_cache_max_per_session(self) -> int:
        return self.graph_state_cache_max_per_session


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    endpoint = (
        os.getenv("AOAI_ENDPOINT")
        or os.getenv("\ufeffAOAI_ENDPOINT")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    api_key = os.getenv("AOAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AOAI_DEPLOY_GPT4O") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o"
    embedding_deployment = (
        os.getenv("AOAI_EMBEDDING_DEPLOYMENT")
        or os.getenv("AOAI_DEPLOY_EMBED_ADA")
        or "text-embedding-ada-002"
    )
    api_version = os.getenv("AOAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-10-21"
    memory_max_sessions = _int_env("MEMORY_MAX_SESSIONS", 200)
    memory_ttl_seconds = _int_env("MEMORY_TTL_SECONDS", 60 * 60 * 24)
    index_force_rebuild = _bool_env("INDEX_FORCE_REBUILD", False)
    vector_weight = _float_env("VECTOR_WEIGHT", 0.6)
    bm25_weight = _float_env("BM25_WEIGHT", 0.4)
    rag_evidence_score_threshold = _float_env("RAG_EVIDENCE_SCORE_THRESHOLD", 0.45)
    ephemeral_ref_base_score = _float_env("EPHEMERAL_REF_BASE_SCORE", 0.42)
    ephemeral_contract_base_score = _float_env("EPHEMERAL_CONTRACT_BASE_SCORE", 0.36)
    ephemeral_overlap_weight = _float_env("EPHEMERAL_OVERLAP_WEIGHT", 0.35)
    rerank_enabled = _bool_env("RERANK_ENABLED", True)
    rerank_provider = (os.getenv("RERANK_PROVIDER") or "heuristic").strip().lower() or "heuristic"
    rerank_max_per_source = _int_env("RERANK_MAX_PER_SOURCE", 2)
    retrieval_max_chunks_per_file = _int_env("RETRIEVAL_MAX_CHUNKS_PER_FILE", 0)
    allow_uncategorized_in_filter = _bool_env("ALLOW_UNCATEGORIZED_IN_FILTER", True)
    uncategorized_ratio_warn_threshold = _float_env("UNCATEGORIZED_RATIO_WARN_THRESHOLD", 0.5)
    faiss_allow_dangerous_deserialization = _bool_env("FAISS_ALLOW_DANGEROUS_DESERIALIZATION", False)
    graph_state_cache_enabled = _bool_env(
        "FINAL_ANSWER_CACHE_ENABLED",
        _bool_env("GRAPH_STATE_CACHE_ENABLED", True),
    )
    graph_state_cache_bypass_contextual = _bool_env(
        "FINAL_ANSWER_CACHE_BYPASS_CONTEXTUAL",
        _bool_env("GRAPH_STATE_CACHE_BYPASS_CONTEXTUAL", True),
    )
    graph_state_cache_max_per_session = _int_env(
        "FINAL_ANSWER_CACHE_MAX_PER_SESSION",
        _int_env("GRAPH_STATE_CACHE_MAX_PER_SESSION", 5),
    )
    state_store_backend = (os.getenv("STATE_STORE_BACKEND") or "file").strip().lower() or "file"
    state_store_dsn = (os.getenv("STATE_STORE_DSN") or "").strip()
    session_memory_persist_enabled = _bool_env(
        "SESSION_MEMORY_PERSIST_ENABLED",
        _bool_env("MEMORY_PERSIST_ENABLED", True),
    )
    session_memory_pii_mask_enabled = _bool_env(
        "SESSION_MEMORY_PII_MASK",
        _bool_env("PII_MASK_ENABLED", False),
    )
    ui_history_persist_enabled = _bool_env("UI_HISTORY_PERSIST_ENABLED", True)
    ui_history_pii_mask_enabled = _bool_env(
        "UI_HISTORY_PII_MASK",
        _bool_env("PII_MASK_ENABLED", False),
    )
    ui_history_storage_mode = (os.getenv("UI_HISTORY_STORAGE_MODE") or "summary").strip().lower() or "summary"
    ui_page_icon_mode = (os.getenv("UI_PAGE_ICON_MODE") or "emoji").strip().lower() or "emoji"
    ui_page_icon_emoji = (os.getenv("UI_PAGE_ICON_EMOJI") or "⚖️").strip() or "⚖️"
    few_shot_max_examples = _int_env("FEW_SHOT_MAX_EXAMPLES", 1)

    missing = []
    if not endpoint:
        missing.append("AOAI_ENDPOINT")
    if not api_key:
        missing.append("AOAI_API_KEY")

    if missing:
        raise LegalPilotError(
            error_code="CONFIG_MISSING_ENV",
            detail=f"Missing environment variables: {', '.join(missing)}",
            status_code=400,
        )

    data_dir = project_root / "data"
    knowledge_dir = data_dir / "knowledge"
    index_dir = data_dir / "index"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        knowledge_dir=knowledge_dir,
        index_dir=index_dir,
        aoai_endpoint=endpoint,
        aoai_api_key=api_key,
        aoai_deployment=deployment,
        aoai_embedding_deployment=embedding_deployment,
        aoai_api_version=api_version,
        memory_max_sessions=memory_max_sessions,
        memory_ttl_seconds=memory_ttl_seconds,
        index_force_rebuild=index_force_rebuild,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        rag_evidence_score_threshold=rag_evidence_score_threshold,
        ephemeral_ref_base_score=ephemeral_ref_base_score,
        ephemeral_contract_base_score=ephemeral_contract_base_score,
        ephemeral_overlap_weight=ephemeral_overlap_weight,
        rerank_enabled=rerank_enabled,
        rerank_provider=rerank_provider,
        rerank_max_per_source=max(1, min(rerank_max_per_source, 5)),
        retrieval_max_chunks_per_file=max(0, min(retrieval_max_chunks_per_file, 10)),
        allow_uncategorized_in_filter=allow_uncategorized_in_filter,
        uncategorized_ratio_warn_threshold=max(0.0, min(uncategorized_ratio_warn_threshold, 1.0)),
        faiss_allow_dangerous_deserialization=faiss_allow_dangerous_deserialization,
        graph_state_cache_enabled=graph_state_cache_enabled,
        graph_state_cache_bypass_contextual=graph_state_cache_bypass_contextual,
        graph_state_cache_max_per_session=max(1, min(graph_state_cache_max_per_session, 20)),
        state_store_backend=(
            state_store_backend
            if state_store_backend in {"file", "sqlite", "redis"}
            else "file"
        ),
        state_store_dsn=state_store_dsn,
        session_memory_persist_enabled=session_memory_persist_enabled,
        session_memory_pii_mask_enabled=session_memory_pii_mask_enabled,
        ui_history_persist_enabled=ui_history_persist_enabled,
        ui_history_pii_mask_enabled=ui_history_pii_mask_enabled,
        ui_history_storage_mode=(
            ui_history_storage_mode if ui_history_storage_mode in {"summary", "full"} else "summary"
        ),
        ui_page_icon_mode=(ui_page_icon_mode if ui_page_icon_mode in {"emoji", "default"} else "emoji"),
        ui_page_icon_emoji=ui_page_icon_emoji,
        few_shot_max_examples=max(0, min(few_shot_max_examples, 3)),
    )
