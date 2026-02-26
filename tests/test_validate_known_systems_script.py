"""Integration checks for the known-systems validation entrypoint."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def test_validate_known_systems_script_reports_reasonable_raw_radius_error() -> None:
    """Guard against path/import regressions that inflate calibration corrections."""
    repo_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "scripts/validate_known_systems.py", "--max-error-percent", "5.0"]
    process = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    combined_output = f"{process.stdout}\n{process.stderr}"
    assert process.returncode == 0, combined_output
    assert "Result: PASS" in combined_output

    raw_radius_errors = [
        float(match.group(1))
        for match in re.finditer(r"Raw error:\s*([0-9.]+)%", combined_output)
    ]
    assert raw_radius_errors, combined_output
    assert max(raw_radius_errors) < 10.0, combined_output
