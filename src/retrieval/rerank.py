"""Rerank strategies for retrieval hits."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def _is_role_relevant(text: str, hint: str) -> bool:
    keywords = [token.strip().lower() for token in hint.split(",") if token.strip()]
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords) if keywords else True


def rerank_hits(
    hits: list[Any],
    query: str,
    role_hint: str,
    route_categories: set[str] | None = None,
    top_k: int = 4,
    max_per_source: int = 2,
    provider: str = "heuristic",
    enabled: bool = True,
) -> list[Any]:
    """Rerank retrieval hits with pluggable strategy.

    Notes:
    - `provider=cross_encoder|llm` currently falls back to `heuristic`.
    - Keep this module isolated so higher-quality rerankers can be swapped in later.
    """
    if not hits:
        return []

    ranked = sorted(hits, key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)
    if not enabled or provider.lower() in {"none", "off", "disabled"}:
        return ranked[:top_k]

    provider_key = provider.lower()
    if provider_key in {"cross_encoder", "llm"}:
        provider_key = "heuristic"

    if provider_key != "heuristic":
        return ranked[:top_k]

    query_tokens = [token for token in query.lower().split() if token]
    lowered_categories = {item.lower() for item in route_categories} if route_categories else set()
    rescored: list[Any] = []

    for hit in ranked:
        source = str(getattr(hit, "source", ""))
        content = str(getattr(hit, "content", ""))
        metadata = dict(getattr(hit, "metadata", {}) or {})
        base_score = float(getattr(hit, "score", 0.0))
        category = str(metadata.get("category", "")).lower()
        source_type = str(metadata.get("source_type", "")).lower()

        role_boost = 0.15 if _is_role_relevant(f"{source} {content}", role_hint) else 0.0
        category_boost = 0.08 if lowered_categories and category in lowered_categories else 0.0
        lexical_boost = 0.05 if any(token in content.lower() for token in query_tokens) else 0.0
        # Prioritize uploaded reference law as first-class evidence for gap analysis.
        upload_boost = 0.12 if source_type == "reference_upload" else (0.08 if source_type == "contract_upload" else 0.0)

        raw_new_score = base_score + role_boost + category_boost + lexical_boost + upload_boost
        # Keep post-rerank score scale in 0~1 so threshold semantics stay stable.
        new_score = round(min(1.0, max(0.0, raw_new_score)), 4)
        score_breakdown = metadata.get("score_breakdown")
        if isinstance(score_breakdown, dict):
            metadata["score_breakdown"] = {
                **score_breakdown,
                "rerank_role_boost": round(role_boost, 4),
                "rerank_category_boost": round(category_boost, 4),
                "rerank_lexical_boost": round(lexical_boost, 4),
                "rerank_upload_boost": round(upload_boost, 4),
                "final_post_rerank_raw": round(raw_new_score, 4),
                "final_post_rerank": round(new_score, 4),
            }
        hit_kwargs = asdict(hit)
        hit_kwargs["score"] = new_score
        hit_kwargs["metadata"] = metadata
        rescored.append(type(hit)(**hit_kwargs))

    rescored.sort(key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)

    if max_per_source <= 0:
        max_per_source = 1
    deduped: list[Any] = []
    source_counter: dict[str, int] = {}
    for hit in rescored:
        source = str(getattr(hit, "source", "unknown"))
        count = source_counter.get(source, 0)
        if count >= max_per_source:
            continue
        source_counter[source] = count + 1
        deduped.append(hit)
        if len(deduped) >= top_k:
            break
    return deduped[:top_k]
