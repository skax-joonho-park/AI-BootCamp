from src.ui.input_merge import merge_uploaded_text


def test_merge_uploaded_text_overwrite_mode_replaces_existing() -> None:
    merged = merge_uploaded_text("기존 계약서 내용", "업로드 계약서 내용", "덮어쓰기")
    assert merged == "업로드 계약서 내용"


def test_merge_uploaded_text_append_mode_combines_with_separator() -> None:
    merged = merge_uploaded_text("기존 계약서 내용", "업로드 계약서 내용", "추가하기")
    assert merged == "기존 계약서 내용\n\n업로드 계약서 내용"


def test_merge_uploaded_text_append_mode_avoids_dup_when_same_text() -> None:
    # 동일 파일 재업로드 시 streamlit 쪽에서 signature로 차단하지만,
    # 병합 함수도 중복 누적 회귀를 막기 위해 동일 텍스트는 1회만 유지한다.
    merged = merge_uploaded_text("동일 내용", "동일 내용", "추가하기")
    assert merged == "동일 내용"


def test_merge_uploaded_text_with_empty_upload_keeps_existing() -> None:
    merged = merge_uploaded_text("기존 텍스트", "   ", "추가하기")
    assert merged == "기존 텍스트"
