from pathlib import Path

from src.utils.file_extract import extract_sections_from_path, extract_text_from_upload


def test_extract_sections_from_path_markdown(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text("첫 문단\n\n둘째 문단", encoding="utf-8")

    sections = extract_sections_from_path(sample)
    assert len(sections) == 2
    assert sections[0]["metadata"]["section_type"] == "paragraph"


def test_extract_text_from_upload_txt() -> None:
    text = "계약서 핵심 조항"
    extracted = extract_text_from_upload("contract.txt", text.encode("utf-8"))
    assert extracted == text


def test_extract_text_from_upload_unsupported() -> None:
    try:
        extract_text_from_upload("sample.exe", b"binary")
    except RuntimeError as exc:
        assert "지원하지 않는 파일 형식" in str(exc)
        return
    raise AssertionError("RuntimeError was expected for unsupported format.")

