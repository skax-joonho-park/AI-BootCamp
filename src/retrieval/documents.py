"""Knowledge document loading and chunk preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.file_extract import extract_sections_from_path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}
REQUIRED_SIDECAR_META_FIELDS = ("collected_at", "source_url", "curator", "license")


def _load_sidecar_metadata(path: Path) -> dict[str, str]:
    """Load optional *.meta.json sidecar and keep only required metadata fields."""
    sidecar = path.with_name(f"{path.name}.meta.json")
    if not sidecar.exists():
        print(f"[WARN] Missing metadata sidecar for {path.name}: {sidecar.name}")
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Invalid metadata sidecar for {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        print(f"[WARN] Invalid metadata sidecar for {path.name}: root must be object")
        return {}

    merged: dict[str, str] = {}
    missing_fields: list[str] = []
    for field in REQUIRED_SIDECAR_META_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            merged[field] = value.strip()
        else:
            missing_fields.append(field)
    if missing_fields:
        missing = ", ".join(missing_fields)
        print(f"[WARN] Incomplete metadata sidecar for {path.name}: missing [{missing}]")
    return merged


def _infer_root_category_from_filename(path: Path) -> str:
    """Infer category for root-level files to avoid uncategorized overuse."""
    stem = path.stem.lower()
    filename = path.name.lower()
    text = f"{stem} {filename}"

    if any(token in text for token in ("statute", "법령", "법률", "시행령", "규정", "조례")):
        return "statutes"
    if any(token in text for token in ("standard", "표준계약", "표준_계약", "model_contract", "표준")):
        return "standard_contracts"
    if any(token in text for token in ("case", "판례", "사례", "가이드", "guide", "해설")):
        return "case_guides"
    if any(token in text for token in ("example", "예시", "sample", "계약서_예시", "contract_example")):
        return "contract_examples"
    return "uncategorized"


def iter_source_files(knowledge_dir: Path) -> Iterable[Path]:
    for path in sorted(knowledge_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents_with_report(knowledge_dir: Path) -> tuple[list[Document], list[dict[str, str]]]:
    docs: list[Document] = []
    failures: list[dict[str, str]] = []
    for path in iter_source_files(knowledge_dir):
        try:
            sections = extract_sections_from_path(path)
        except Exception as exc:
            # Keep service available even if one file is malformed or parser dependency is missing.
            print(f"[WARN] Failed to load {path.name}: {exc}")
            relative = path.relative_to(knowledge_dir).as_posix()
            failures.append({"file": relative, "error": str(exc)})
            continue

        if not sections:
            continue

        relative = path.relative_to(knowledge_dir)
        category = (
            relative.parts[0]
            if len(relative.parts) > 1
            else _infer_root_category_from_filename(path)
        )
        sidecar_meta = _load_sidecar_metadata(path)
        for section in sections:
            content = str(section.get("content", "")).strip()
            if not content:
                continue
            section_meta = section.get("metadata", {})
            metadata = {
                "source": str(path),
                "filename": path.name,
                "relative_path": str(relative).replace("\\", "/"),
                "category": category,
            }
            if isinstance(section_meta, dict):
                metadata.update(section_meta)
            metadata.update(sidecar_meta)

            docs.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )
    return docs, failures


def load_documents(knowledge_dir: Path) -> list[Document]:
    docs, _ = load_documents_with_report(knowledge_dir)
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    per_source_counter: dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        source = str(chunk.metadata.get("source", "unknown"))
        per_source_counter[source] = per_source_counter.get(source, 0) + 1
        chunk.metadata["chunk_index_in_source"] = per_source_counter[source]
    return chunks
