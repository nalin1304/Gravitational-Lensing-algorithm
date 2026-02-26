#!/usr/bin/env python3
"""
Publication readiness gate for journal submissions.

This script executes reproducibility-critical checks and writes a machine-readable
report under results/. It is intended to be the single command reviewers and
authors can run to verify the code artifact baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from api.main import app

MAX_RAW_RADIUS_ERROR_PERCENT = 10.0
MAX_RAW_MASS_ERROR_PERCENT = 35.0
MAX_RADIUS_CALIBRATION_FACTOR = 1.5
MAX_MASS_CALIBRATION_FACTOR = 2.0


@dataclass
class CheckResult:
    name: str
    command: str
    passed: bool
    return_code: int
    stdout_tail: list[str]
    stderr_tail: list[str]


def _run_command(name: str, command: Sequence[str]) -> CheckResult:
    proc = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_lines = proc.stdout.strip().splitlines()[-20:]
    stderr_lines = proc.stderr.strip().splitlines()[-20:]
    return CheckResult(
        name=name,
        command=" ".join(command),
        passed=proc.returncode == 0,
        return_code=proc.returncode,
        stdout_tail=stdout_lines,
        stderr_tail=stderr_lines,
    )


def _check_files_exist(paths: Sequence[Path]) -> list[str]:
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
    return missing


def _openapi_summary() -> dict[str, object]:
    client = TestClient(app)
    response = client.get("/openapi.json")
    if response.status_code != 200:
        return {"ok": False, "status_code": response.status_code, "endpoint_count": 0}

    payload = response.json()
    endpoint_count = 0
    for path_item in payload.get("paths", {}).values():
        endpoint_count += len(
            [method for method in path_item.keys() if method.lower() in {"get", "post", "put", "patch", "delete"}]
        )

    required_paths = ["/api/v1/synthetic", "/api/v1/inference", "/api/v1/auth/login", "/api/v1/analyses"]
    available_paths = set(payload.get("paths", {}).keys())
    missing_paths = [path for path in required_paths if path not in available_paths]

    return {
        "ok": True,
        "status_code": response.status_code,
        "endpoint_count": endpoint_count,
        "missing_required_paths": missing_paths,
    }


def _calibration_quality_summary() -> dict[str, object]:
    """Validate calibration quality with raw-error and correction-factor gates."""
    from src.validation.calibration import SyntheticDataCalibrator

    calibrator = SyntheticDataCalibrator()
    eligible_systems = [
        key
        for key, lit in calibrator.literature.items()
        # Restrict to galaxy-scale systems resolved by the default 10" FOV.
        if float(lit.get("einstein_radius_arcsec", 0.0)) <= 5.0
    ]
    for system_key in eligible_systems:
        calibrator.calibrate_system(system_key)

    if not calibrator.calibration_results:
        return {"ok": False, "reason": "No calibration results produced."}

    results = list(calibrator.calibration_results.values())
    max_raw_radius_error = float(max(abs(result.raw_radius_error_percent) for result in results))
    max_raw_mass_error = float(max(abs(result.raw_mass_error_percent) for result in results))
    max_radius_factor = float(max(abs(result.radius_calibration_factor) for result in results))
    max_mass_factor = float(max(abs(result.mass_calibration_factor) for result in results))

    passed = bool(
        max_raw_radius_error <= MAX_RAW_RADIUS_ERROR_PERCENT
        and max_raw_mass_error <= MAX_RAW_MASS_ERROR_PERCENT
        and max_radius_factor <= MAX_RADIUS_CALIBRATION_FACTOR
        and max_mass_factor <= MAX_MASS_CALIBRATION_FACTOR
    )

    return {
        "ok": passed,
        "max_raw_radius_error_percent": max_raw_radius_error,
        "max_raw_mass_error_percent": max_raw_mass_error,
        "max_radius_calibration_factor": max_radius_factor,
        "max_mass_calibration_factor": max_mass_factor,
        "thresholds": {
            "max_raw_radius_error_percent": MAX_RAW_RADIUS_ERROR_PERCENT,
            "max_raw_mass_error_percent": MAX_RAW_MASS_ERROR_PERCENT,
            "max_radius_calibration_factor": MAX_RADIUS_CALIBRATION_FACTOR,
            "max_mass_calibration_factor": MAX_MASS_CALIBRATION_FACTOR,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run publication readiness checks.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced set of checks (no full test suite).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/publication_gate_report.json"),
        help="Path to JSON output report.",
    )
    args = parser.parse_args()

    required_files = [
        Path("README.md"),
        Path("AGENTS.md"),
        Path("CITATION.cff"),
        Path("IEEE_SUBMISSION_CHECKLIST.md"),
        Path("JOURNAL_PUBLICATION_READINESS.md"),
        Path("paper/TIER1_TOPIC_AND_RIGOR.md"),
        Path("scripts/statistical_rigor_report.py"),
        Path("scripts/validate_known_systems.py"),
    ]
    missing_files = _check_files_exist(required_files)

    checks: list[CheckResult] = []
    checks.append(_run_command("UI smoke", ["python3", "-m", "pytest", "tests/test_next_ui.py", "-q"]))
    checks.append(_run_command("API core", ["python3", "-m", "pytest", "tests/test_api.py", "-q"]))
    checks.append(_run_command("Type check", ["python3", "-m", "mypy", "src/", "--ignore-missing-imports"]))
    checks.append(_run_command("Known systems", ["python3", "scripts/validate_known_systems.py"]))
    checks.append(
        _run_command(
            "Statistical rigor report",
            [
                "python3",
                "scripts/statistical_rigor_report.py",
                "--output",
                "results/statistical_rigor_report.json",
                "--markdown-output",
                "results/statistical_rigor_report.md",
            ],
        )
    )

    if not args.quick:
        checks.append(_run_command("Full regression", ["python3", "-m", "pytest", "tests/", "-q"]))

    openapi_info = _openapi_summary()
    openapi_passed = bool(openapi_info.get("ok")) and int(openapi_info.get("endpoint_count", 0)) > 0
    if openapi_info.get("missing_required_paths"):
        openapi_passed = False

    calibration_quality = _calibration_quality_summary()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "mode": "quick" if args.quick else "full",
        "missing_required_files": missing_files,
        "openapi": openapi_info,
        "calibration_quality": calibration_quality,
        "checks": [asdict(check) for check in checks],
        "all_checks_passed": all(check.passed for check in checks),
    }
    report["publication_gate_passed"] = (
        not missing_files
        and report["all_checks_passed"]
        and bool(openapi_passed)
        and bool(calibration_quality.get("ok"))
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status_label = "PASS" if report["publication_gate_passed"] else "FAIL"
    print(f"Publication Gate: {status_label}")
    print(f"Report: {args.output}")
    print(f"OpenAPI endpoints: {openapi_info.get('endpoint_count', 0)}")
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
    if openapi_info.get("missing_required_paths"):
        missing = ", ".join(openapi_info["missing_required_paths"])
        print(f"Missing required OpenAPI paths: {missing}")
    if not calibration_quality.get("ok"):
        print("Calibration quality check failed.")
        print(json.dumps(calibration_quality, indent=2))

    return 0 if report["publication_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
