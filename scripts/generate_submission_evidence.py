"""Generate submission evidence files for Step2/Step3 documents.

This script creates:
1) Agent execution log with routing + RAG evidence
2) Final structured response JSON

Streamlit UI screenshot is intentionally collected manually by the user because
it requires an interactive browser view.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workflow import LegalPilotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission evidence artifacts")
    parser.add_argument(
        "--query",
        default="근로계약서의 포괄임금제 조항이 적법한지 검토하고 위험 요소와 수정 방향을 알려줘.",
        help="User query to run for evidence generation",
    )
    parser.add_argument("--document-type", default="근로계약서")
    parser.add_argument(
        "--contract-text",
        default="월 급여 300만 원(연장·야간·휴일 근로수당 포함), 포괄임금제 적용, 경업금지 3년",
    )
    parser.add_argument(
        "--reference-text",
        default="근로기준법 제17조: 임금 구성항목·계산방법·지급방법을 서면 명시. 제53조: 연장근로 1주 12시간 한도.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/evidence",
        help="Directory where artifacts are saved",
    )
    parser.add_argument("--session-id", default="evidence-session")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    service = LegalPilotService()
    state = {
        "session_id": args.session_id,
        "user_query": args.query,
        "document_type": args.document_type,
        "contract_text": args.contract_text,
        "reference_text": args.reference_text,
    }
    result = service.graph.invoke(
        state,
        config={"configurable": {"thread_id": args.session_id}},
    )
    final_answer = result.get("final_answer", {})

    json_path = output_dir / "agent_final_answer.json"
    json_path.write_text(json.dumps(final_answer, ensure_ascii=False, indent=2), encoding="utf-8")

    route = result.get("route", "unknown")
    routing_reason = result.get("routing_reason", "unknown")
    rag_refs = result.get("rag_refs", [])
    rag_context = result.get("rag_context", "")
    rag_preview = rag_context[:800] + ("..." if len(rag_context) > 800 else "")

    rag_lines: list[str] = []
    for ref in rag_refs if isinstance(rag_refs, list) else []:
        if isinstance(ref, dict):
            rag_lines.append(
                f"- [{ref.get('rank', '?')}] {ref.get('source', 'unknown')} "
                f"(chunk={ref.get('chunk_id', 'na')}, location={ref.get('location', 'n/a')}, score={ref.get('score', 0.0)})"
            )
        else:
            rag_lines.append(f"- {ref}")
    rag_refs_text = "\n".join(rag_lines) if rag_lines else "- (없음)"

    log_path = output_dir / "agent_execution_log.md"
    log_text = f"""# Agent Execution Log

- Generated at: {timestamp}
- Session ID: {args.session_id}
- Document type: {args.document_type}

## Input
- Query: {args.query}
- Contract text length: {len(args.contract_text)}
- Reference text length: {len(args.reference_text)}

## Supervisor Routing
- Route: {route}
- Reason: {routing_reason}

## RAG References
{rag_refs_text}

## RAG Context Preview
```
{rag_preview}
```

## Final Answer Artifact
- JSON file: `{json_path.as_posix()}`
"""
    log_path.write_text(log_text, encoding="utf-8")

    print(f"[OK] Created: {log_path.as_posix()}")
    print(f"[OK] Created: {json_path.as_posix()}")


if __name__ == "__main__":
    main()
