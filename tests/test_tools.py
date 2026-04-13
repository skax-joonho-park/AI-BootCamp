from src.agents.tools import clause_keyword_match_score, legal_issue_bank, contract_reference_gap_score


def test_clause_keyword_match_score() -> None:
    result = clause_keyword_match_score.invoke(
        {
            "contract_text": "월 급여 300만 원, 포괄임금제 적용, 연장근로 포함, 경업금지 3년",
            "document_type": "근로계약서",
        }
    )
    assert '"match_score":' in result


def test_legal_issue_bank() -> None:
    result = legal_issue_bank.invoke({"document_type": "근로계약서"})
    assert '"issues":' in result
    assert "임금" in result or "근로" in result


def test_contract_reference_gap_score() -> None:
    result = contract_reference_gap_score.invoke(
        {
            "reference_text": "근로기준법 제17조: 임금 구성항목·계산방법·지급방법을 서면 명시. 제53조: 연장근로 1주 12시간 한도.",
            "contract_text": "월 급여 300만 원, 포괄임금제 적용.",
            "document_type": "근로계약서",
        }
    )
    assert '"required_match_rate":' in result
    assert '"missing_required_top":' in result
