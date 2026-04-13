"""Run FastAPI server for LegalPilot."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(project_root)
    )
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(project_root),
        reload_dirs=[str(project_root)],
    )


if __name__ == "__main__":
    main()
