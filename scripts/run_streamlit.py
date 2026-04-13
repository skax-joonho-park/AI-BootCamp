"""Run Streamlit app for LegalPilot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "src" / "ui" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, cwd=project_root, env=env)


if __name__ == "__main__":
    main()
