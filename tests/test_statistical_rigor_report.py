"""Tests for the statistical rigor reporting script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_statistical_rigor_report_generation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "statistical_rigor_report.json"
    output_md = tmp_path / "statistical_rigor_report.md"

    command = [
        sys.executable,
        "scripts/statistical_rigor_report.py",
        "--output",
        str(output_json),
        "--markdown-output",
        str(output_md),
        "--n-bootstrap",
        "1000",
    ]
    process = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, f"{process.stdout}\n{process.stderr}"

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert "slacs_validation" in report
    assert "ablation_study" in report
    assert "sota_comparison" in report
    assert "overall_warnings" in report
    assert output_md.exists()

