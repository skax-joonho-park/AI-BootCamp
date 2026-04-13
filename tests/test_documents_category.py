from pathlib import Path

from langchain_core.documents import Document

from src.retrieval.documents import _infer_root_category_from_filename, load_documents_with_report
from src.retrieval.hybrid import _category_diagnostics


def test_infer_root_category_from_filename_statutes() -> None:
    assert _infer_root_category_from_filename(Path("civil_statute_overview.md")) == "statutes"
    assert _infer_root_category_from_filename(Path("근로기준_법령_요약.md")) == "statutes"


def test_infer_root_category_from_filename_standard_and_case() -> None:
    assert _infer_root_category_from_filename(Path("표준계약_NDA.md")) == "standard_contracts"
    assert _infer_root_category_from_filename(Path("분쟁사례_가이드.md")) == "case_guides"


def test_infer_root_category_from_filename_example_fallback() -> None:
    assert _infer_root_category_from_filename(Path("근로계약서_예시.md")) == "contract_examples"
    assert _infer_root_category_from_filename(Path("notes.md")) == "uncategorized"


def test_load_documents_merges_sidecar_metadata(tmp_path: Path) -> None:
    doc_path = tmp_path / "statutes" / "근로기준법_요약.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("근로기준법 제17조 임금 명시 의무", encoding="utf-8")
    sidecar_path = tmp_path / "statutes" / "근로기준법_요약.md.meta.json"
    sidecar_path.write_text(
        """
{
  "collected_at": "2026-03-09",
  "source_url": "https://example.com/statutes",
  "curator": "legalpilot-team",
  "license": "CC-BY-4.0"
}
""".strip(),
        encoding="utf-8",
    )

    docs, failures = load_documents_with_report(tmp_path)
    assert failures == []
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["collected_at"] == "2026-03-09"
    assert meta["source_url"] == "https://example.com/statutes"
    assert meta["curator"] == "legalpilot-team"
    assert meta["license"] == "CC-BY-4.0"


def test_load_documents_warns_when_sidecar_missing(tmp_path: Path, capsys) -> None:
    doc_path = tmp_path / "contract_tips.md"
    doc_path.write_text("조항 작성 가이드라인", encoding="utf-8")

    docs, _ = load_documents_with_report(tmp_path)
    captured = capsys.readouterr()

    assert len(docs) == 1
    assert "Missing metadata sidecar for contract_tips.md" in captured.out


def test_category_diagnostics_respects_warn_threshold() -> None:
    chunks = [
        Document(page_content="a", metadata={"category": "uncategorized"}),
        Document(page_content="b", metadata={"category": "uncategorized"}),
        Document(page_content="c", metadata={"category": "statutes"}),
        Document(page_content="d", metadata={"category": "standard_contracts"}),
    ]
    relaxed = _category_diagnostics(chunks, uncategorized_warn_threshold=0.6)
    strict = _category_diagnostics(chunks, uncategorized_warn_threshold=0.4)
    assert relaxed["uncategorized_ratio"] == 0.5
    assert relaxed["category_quality_warning"] != "uncategorized_ratio_high"
    assert strict["category_quality_warning"] == "uncategorized_ratio_high"
