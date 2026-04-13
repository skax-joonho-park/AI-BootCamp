"""Hybrid retriever: FAISS vector search + BM25 keyword search."""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.common import LegalPilotError
from src.config import get_embedding_model, load_settings
from src.retrieval.documents import chunk_documents, iter_source_files, load_documents_with_report
from src.utils.io import atomic_write_text


_FALLBACK_STOPWORDS = {
    "그리고",
    "또는",
    "그러나",
    "때문",
    "위해",
    "관련",
    "대한",
    "에서",
    "으로",
    "하다",
    "합니다",
    "있는",
    "없는",
    "경우",
    "사용",
    "기반",
}


def _tokenize(text: str) -> list[str]:
    tokens = _tokenize_with_kiwi(text)
    if tokens:
        return tokens
    tokens = _tokenize_with_okt(text)
    if tokens:
        return tokens
    return _tokenize_fallback(text)


def _tokenize_with_kiwi(text: str) -> list[str]:
    kiwi = _get_kiwi()
    if kiwi is None:
        return []
    stop_tags = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX"}
    result = []
    for token in kiwi.tokenize(text):
        if token.tag in stop_tags:
            continue
        cleaned = token.form.strip().lower()
        if cleaned:
            result.append(cleaned)
    return result


def _tokenize_with_okt(text: str) -> list[str]:
    okt = _get_okt()
    if okt is None:
        return []
    stop_tags = {"Josa"}
    result = []
    for token, tag in okt.pos(text, norm=True, stem=True):
        if tag in stop_tags:
            continue
        cleaned = token.strip().lower()
        if cleaned:
            result.append(cleaned)
    return result


def _strip_korean_particles(token: str) -> str:
    particles = (
        "으로",
        "에서",
        "에게",
        "까지",
        "부터",
        "보다",
        "처럼",
        "하고",
        "이다",
        "습니다",
        "였다",
        "이다",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "의",
        "도",
        "만",
    )
    for particle in particles:
        if token.endswith(particle) and len(token) > len(particle) + 1:
            return token[: -len(particle)]
    return token


def _tokenize_fallback(text: str) -> list[str]:
    rough = re.findall(r"[0-9A-Za-z가-힣]+", text.lower())
    tokens = [_strip_korean_particles(token) for token in rough]
    return [
        token
        for token in tokens
        if token and token not in _FALLBACK_STOPWORDS and len(token) > 1
    ]


@lru_cache(maxsize=1)
def _tokenizer_backend_name() -> str:
    if _get_kiwi() is not None:
        return "kiwi"
    if _get_okt() is not None:
        return "okt"
    return "fallback"


@lru_cache(maxsize=1)
def _get_kiwi():
    try:
        from kiwipiepy import Kiwi
    except ModuleNotFoundError:
        return None
    return Kiwi()


@lru_cache(maxsize=1)
def _get_okt():
    try:
        from konlpy.tag import Okt
    except ModuleNotFoundError:
        return None
    return Okt()


def _file_content_fingerprint(path: Path, sample_size: int = 65536) -> str:
    stat = path.stat()
    hasher = hashlib.sha256()
    hasher.update(f"{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    try:
        with path.open("rb") as file:
            if stat.st_size <= sample_size * 2:
                hasher.update(file.read())
            else:
                hasher.update(file.read(sample_size))
                file.seek(max(stat.st_size - sample_size, 0))
                hasher.update(file.read(sample_size))
    except OSError:
        # Fallback to stat-only fingerprint if file read fails.
        pass
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _faiss_cache_hashes(index_dir: Path) -> dict[str, str] | None:
    index_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    if not (index_file.exists() and pkl_file.exists()):
        return None
    return {
        "index.faiss": _sha256_file(index_file),
        "index.pkl": _sha256_file(pkl_file),
    }


def _cache_hashes_match(expected: dict[str, Any], current: dict[str, str] | None) -> bool:
    if not isinstance(expected, dict) or not current:
        return False
    for key, value in current.items():
        if str(expected.get(key, "")) != value:
            return False
    return True


def _corpus_signature(paths: list[Path], root: Path) -> str:
    records = []
    for path in sorted(paths):
        rel = path.relative_to(root).as_posix()
        fingerprint = _file_content_fingerprint(path)
        records.append(f"{rel}|{fingerprint}")
    return "|".join(records)


def _normalize_weights(vector_weight: float, bm25_weight: float) -> tuple[float, float]:
    vw = max(vector_weight, 0.0)
    bw = max(bm25_weight, 0.0)
    total = vw + bw
    if total <= 1e-8:
        return 0.6, 0.4
    return vw / total, bw / total


def _category_diagnostics(chunks: list[Document], uncategorized_warn_threshold: float = 0.5) -> dict[str, Any]:
    required_categories = {"statutes", "standard_contracts", "case_guides", "contract_examples"}
    distribution: dict[str, int] = {}
    for chunk in chunks:
        category = str(chunk.metadata.get("category", "uncategorized")).strip().lower() or "uncategorized"
        distribution[category] = distribution.get(category, 0) + 1

    total = sum(distribution.values())
    uncategorized_count = distribution.get("uncategorized", 0)
    uncategorized_ratio = round((uncategorized_count / total), 4) if total else 0.0
    present_required = sorted([category for category in required_categories if category in distribution])
    missing_required = sorted(required_categories - set(distribution.keys()))
    threshold = max(0.0, min(float(uncategorized_warn_threshold), 1.0))
    warning = (
        "uncategorized_ratio_high"
        if uncategorized_ratio >= threshold
        else ("missing_required_categories" if missing_required else "")
    )
    return {
        "category_distribution": distribution,
        "uncategorized_ratio": uncategorized_ratio,
        "present_required_categories": present_required,
        "missing_required_categories": missing_required,
        "uncategorized_warn_threshold": threshold,
        "category_quality_warning": warning or None,
    }


@dataclass
class SearchHit:
    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class HybridRetriever:
    """Simple in-memory hybrid retriever."""

    def __init__(
        self,
        chunks: list[Document],
        vector_db: FAISS,
        bm25: BM25Okapi,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        tokenizer_backend: str = "fallback",
    ) -> None:
        self.chunks = chunks
        self.vector_db = vector_db
        self.bm25 = bm25
        self.vector_weight, self.bm25_weight = _normalize_weights(vector_weight, bm25_weight)
        self.tokenizer_backend = tokenizer_backend

    @classmethod
    def build(cls) -> "HybridRetriever":
        settings = load_settings()
        source_files = list(iter_source_files(settings.knowledge_dir))
        signature = _corpus_signature(source_files, settings.knowledge_dir)

        faiss_dir = settings.index_dir / "faiss"
        chunks_path = settings.index_dir / "chunks.json"
        meta_path = settings.index_dir / "retriever_meta.json"

        chunks: list[Document] | None = None
        vector_db: FAISS | None = None
        tokenizer_backend = _tokenizer_backend_name()
        vector_weight = settings.vector_weight
        bm25_weight = settings.bm25_weight
        if (
            not settings.index_force_rebuild
            and faiss_dir.exists()
            and chunks_path.exists()
            and meta_path.exists()
        ):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and meta.get("signature") == signature:
                    expected_hashes = meta.get("cache_hashes", {})
                    current_hashes = _faiss_cache_hashes(faiss_dir)
                    if not _cache_hashes_match(expected_hashes, current_hashes):
                        raise ValueError("FAISS cache hash verification failed.")
                    vector_weight = float(meta.get("vector_weight", settings.vector_weight))
                    bm25_weight = float(meta.get("bm25_weight", settings.bm25_weight))
                    tokenizer_backend = str(meta.get("tokenizer_backend", tokenizer_backend))
                    embeddings = get_embedding_model()
                    vector_db = FAISS.load_local(
                        str(faiss_dir),
                        embeddings,
                        allow_dangerous_deserialization=settings.faiss_allow_dangerous_deserialization,
                    )
                    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
                    if isinstance(raw_chunks, list):
                        chunks = [
                            Document(
                                page_content=str(item.get("page_content", "")),
                                metadata=item.get("metadata", {}),
                            )
                            for item in raw_chunks
                            if isinstance(item, dict)
                        ]
            except Exception:
                chunks = None
                vector_db = None

        if chunks is not None and vector_db is not None:
            tokenized_chunks = [_tokenize(doc.page_content) for doc in chunks]
            bm25 = BM25Okapi(tokenized_chunks)
            diagnostics = _category_diagnostics(
                chunks,
                uncategorized_warn_threshold=settings.uncategorized_ratio_warn_threshold,
            )
            if diagnostics.get("category_quality_warning"):
                print(
                    "[WARN] Knowledge category quality issue:",
                    diagnostics.get("category_quality_warning"),
                    diagnostics.get("category_distribution"),
                )
            return cls(
                chunks=chunks,
                vector_db=vector_db,
                bm25=bm25,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                tokenizer_backend=tokenizer_backend,
            )

        docs, load_failures = load_documents_with_report(settings.knowledge_dir)
        if not docs:
            raise LegalPilotError(
                error_code="KNOWLEDGE_EMPTY",
                detail=(
                    f"No knowledge documents found in {settings.knowledge_dir}. "
                    "Add .txt/.md/.csv/.pdf/.docx/.xlsx files before running."
                ),
                status_code=400,
            )

        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        embeddings = get_embedding_model()
        vector_db = FAISS.from_documents(chunks, embedding=embeddings)
        faiss_dir.mkdir(parents=True, exist_ok=True)
        vector_db.save_local(str(faiss_dir))

        chunks_payload = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
        ]
        atomic_write_text(
            chunks_path,
            json.dumps(chunks_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        diagnostics = _category_diagnostics(
            chunks,
            uncategorized_warn_threshold=settings.uncategorized_ratio_warn_threshold,
        )
        if diagnostics.get("category_quality_warning"):
            print(
                "[WARN] Knowledge category quality issue:",
                diagnostics.get("category_quality_warning"),
                diagnostics.get("category_distribution"),
            )
        atomic_write_text(
            meta_path,
            json.dumps(
                {
                    "signature": signature,
                    "vector_weight": _normalize_weights(
                        settings.vector_weight,
                        settings.bm25_weight,
                    )[0],
                    "bm25_weight": _normalize_weights(
                        settings.vector_weight,
                        settings.bm25_weight,
                    )[1],
                    "tokenizer_backend": tokenizer_backend,
                    "cache_hashes": _faiss_cache_hashes(faiss_dir),
                    "document_load_failure_count": len(load_failures),
                    "document_load_failures": load_failures[:100],
                    **diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tokenized_chunks = [_tokenize(doc.page_content) for doc in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        return cls(
            chunks=chunks,
            vector_db=vector_db,
            bm25=bm25,
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
            tokenizer_backend=tokenizer_backend,
        )

    def _vector_scores(self, query: str, top_k: int) -> dict[int, float]:
        docs_with_scores = self.vector_db.similarity_search_with_score(query, k=top_k)
        if not docs_with_scores:
            return {}
        scores: dict[int, float] = {}
        # Use a query-independent transform for FAISS distances so threshold semantics
        # remain stable across queries: distance -> similarity in [0, 1] via 1/(1+d).
        for doc, distance in docs_with_scores:
            score = 1.0 / (1.0 + max(float(distance), 0.0))
            idx = doc.metadata.get("chunk_id")
            if isinstance(idx, int):
                scores[idx] = float(score)
        return scores

    def _bm25_scores(self, query: str, top_k: int) -> dict[int, float]:
        tokens = _tokenize(query)
        raw_scores = np.array(self.bm25.get_scores(tokens), dtype=float)
        if raw_scores.size == 0:
            return {}
        top_indices = raw_scores.argsort()[::-1][:top_k]
        max_score = raw_scores[top_indices].max() if len(top_indices) else 0.0
        denom = max(max_score, 1e-8)
        return {int(i): float(raw_scores[i] / denom) for i in top_indices}

    def _query_doc_length_penalty(self, query: str, doc_text: str) -> float:
        q_len = max(len(_tokenize(query)), 1)
        d_len = max(len(_tokenize(doc_text)), 1)
        ratio = d_len / q_len
        if ratio <= 8:
            return 1.0
        return max(0.7, 1.0 - min(0.3, (ratio - 8) * 0.01))

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: set[str] | None = None,
        max_chunks_per_file: int = 2,
    ) -> list[SearchHit]:
        settings = load_settings()
        k = top_k or settings.top_k

        vector_scores = self._vector_scores(query, k)
        bm25_scores = self._bm25_scores(query, k * 2)

        merged: dict[int, float] = {}
        for idx, score in vector_scores.items():
            merged[idx] = merged.get(idx, 0.0) + (self.vector_weight * score)
        for idx, score in bm25_scores.items():
            merged[idx] = merged.get(idx, 0.0) + (self.bm25_weight * score)

        rescored: list[tuple[int, float]] = []
        lowered_filter = {item.lower() for item in category_filter} if category_filter else None
        for idx, score in merged.items():
            doc = self.chunks[idx]
            if category_filter:
                category = str(doc.metadata.get("category", "")).lower()
                if lowered_filter and category not in lowered_filter:
                    continue
            penalty = self._query_doc_length_penalty(query, doc.page_content)
            final_score = score * penalty
            doc.metadata["score_breakdown"] = {
                "vector": round(float(vector_scores.get(idx, 0.0)), 4),
                "bm25": round(float(bm25_scores.get(idx, 0.0)), 4),
                "fused": round(float(score), 4),
                "length_penalty": round(float(penalty), 4),
                "final_pre_rerank": round(float(final_score), 4),
            }
            rescored.append((idx, final_score))

        ranked = sorted(rescored, key=lambda x: x[1], reverse=True)
        hits: list[SearchHit] = []
        per_source_count: dict[str, int] = {}
        apply_source_cap = max_chunks_per_file > 0
        for idx, score in ranked:
            doc = self.chunks[idx]
            source = str(doc.metadata.get("filename", "unknown"))
            current_count = per_source_count.get(source, 0)
            if apply_source_cap and current_count >= max_chunks_per_file:
                continue
            per_source_count[source] = current_count + 1
            hits.append(
                SearchHit(
                    content=doc.page_content,
                    source=source,
                    score=round(score, 4),
                    metadata=dict(doc.metadata),
                )
            )
            if len(hits) >= k:
                break
        return hits

    def search_as_context(self, query: str, top_k: int | None = None) -> tuple[str, list[dict[str, Any]]]:
        hits = self.search(query, top_k=top_k)
        context_parts = []
        refs: list[dict[str, Any]] = []
        for i, hit in enumerate(hits, start=1):
            location = ""
            if "page_number" in hit.metadata:
                location = f", page={hit.metadata['page_number']}"
            elif "paragraph_number" in hit.metadata:
                location = f", paragraph={hit.metadata['paragraph_number']}"
            elif "sheet_name" in hit.metadata and "row_number" in hit.metadata:
                location = f", sheet={hit.metadata['sheet_name']}, row={hit.metadata['row_number']}"
            elif "row_number" in hit.metadata:
                location = f", row={hit.metadata['row_number']}"

            context_parts.append(f"[{i}] ({hit.source}, score={hit.score}{location})\n{hit.content}")
            refs.append(
                {
                    "rank": i,
                    "source": hit.source,
                    "score": hit.score,
                    "category": hit.metadata.get("category"),
                    "location": location.lstrip(", "),
                }
            )
        return "\n\n".join(context_parts), refs
