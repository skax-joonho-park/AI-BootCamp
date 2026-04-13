"""CLI entry point for LegalPilot AI."""

from __future__ import annotations

import argparse
import json
import sys

from src.common import LegalPilotError
from src.workflow import ChatRequest, LegalPilotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LegalPilot AI from CLI")
    parser.add_argument("--session-id", default="cli-default")
    parser.add_argument("--document-type", default="근로계약서")
    parser.add_argument("--query", required=True, help="User request/question")
    parser.add_argument("--contract-text", default="", help="Optional contract plain text")
    parser.add_argument(
        "--reference-text",
        default="",
        help="Optional reference law/standard contract plain text for gap analysis",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        service = LegalPilotService()
        response = service.run(
            ChatRequest(
                session_id=args.session_id,
                user_query=args.query,
                document_type=args.document_type,
                contract_text=args.contract_text,
                reference_text=args.reference_text,
            )
        )
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    except LegalPilotError as exc:
        print(json.dumps(exc.to_payload(), ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
