#!/usr/bin/env python3
"""
Validation Gate — Cosmological Unblinding Protocol

This script enforces that all required scientific checks pass BEFORE
revealing the true H₀ and D_Δt values. It is intentionally kept as
a standalone script to minimize the chance of accidental unblinding.

Usage
-----
  # Check gate status (does NOT unblind):
  python scripts/validation_gate.py --check

  # Unblind after supplying phrase and passing all checks:
  python scripts/validation_gate.py --unblind --phrase "my-secret-phrase" \\
      --h0-blind 73.4 --dtd-blind 5230.0

  # Generate a blinding receipt for supplement:
  python scripts/validation_gate.py --receipt --phrase "my-secret-phrase"
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.blinding import BlindingHandler, run_validation_gate


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

REQUIRED_CHECKS = {
    "test_suite":        "481 tests passing (python3 -m pytest tests/ -q)",
    "mypy_clean":        "No mypy errors on src/",
    "ece_below_0.05":   "Uncertainty calibration ECE ≤ 0.05",
    "coverage_0.90":    "90% coverage for Bayesian UQ intervals",
    "slacs_consistent":  "All 5 SLACS lenses within 5% of literature Einstein radii",
    "publication_gate":  "publication_gate.py passes (python3 scripts/publication_gate.py --quick)",
}


def run_checks() -> dict[str, bool]:
    """
    Attempt to programmatically verify each gate criterion.

    Returns dict of check_name → passed (bool).
    Checks that cannot be automatically run are reported as 'manual'.
    """
    results: dict[str, bool] = {}

    # ── 1. Test suite
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True, text=True, cwd=project_root
        )
        results["test_suite"] = r.returncode == 0
    except Exception as e:
        print(f"  [test_suite] ERROR: {e}")
        results["test_suite"] = False

    # ── 2. mypy
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports", "--no-error-summary"],
            capture_output=True, text=True, cwd=project_root
        )
        results["mypy_clean"] = r.returncode == 0
    except Exception:
        results["mypy_clean"] = False

    # ── 3-6. Calibration / SLACS / publication gate — check report file
    gate_report = project_root / "results" / "publication_gate_report.json"
    if gate_report.exists():
        try:
            report = json.loads(gate_report.read_text())
            ece = report.get("ece", 1.0)
            coverage = report.get("calibration_coverage", 0.0)
            slacs_ok = report.get("slacs_all_within_5pct", False)
            gate_ok = report.get("publication_gate_passed", False)
            results["ece_below_0.05"] = float(ece) <= 0.05
            results["coverage_0.90"] = float(coverage) >= 0.90
            results["slacs_consistent"] = bool(slacs_ok)
            results["publication_gate"] = bool(gate_ok)
        except Exception as e:
            print(f"  [gate_report] parse error: {e}")
            for k in ("ece_below_0.05", "coverage_0.90", "slacs_consistent", "publication_gate"):
                results[k] = False
    else:
        print(f"  [gate_report] Not found: {gate_report}")
        print("  Run: python3 scripts/publication_gate.py --quick")
        for k in ("ece_below_0.05", "coverage_0.90", "slacs_consistent", "publication_gate"):
            results[k] = False

    return results


def print_gate_status(results: dict[str, bool]) -> bool:
    """Print a formatted gate table. Returns True if all checks pass."""
    all_pass = True
    print("\n" + "=" * 62)
    print("  COSMOLOGICAL UNBLINDING VALIDATION GATE")
    print("=" * 62)
    for key, desc in REQUIRED_CHECKS.items():
        passed = results.get(key, False)
        symbol = "✅" if passed else "❌"
        state = "PASS" if passed else "FAIL"
        print(f"  {symbol}  [{state}]  {key}")
        print(f"         {desc}")
        if not passed:
            all_pass = False
    print("=" * 62)
    if all_pass:
        print("  ✅ ALL CHECKS PASSED — ready to unblind")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  ❌ {len(failed)} check(s) failed — unblinding BLOCKED")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cosmological Unblinding Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Run validation checks and report status (no unblinding)")
    parser.add_argument("--unblind", action="store_true",
                        help="Unblind results (requires --phrase, --h0-blind, --dtd-blind)")
    parser.add_argument("--receipt", action="store_true",
                        help="Write blinding receipt to results/blinding_receipt.json")
    parser.add_argument("--phrase", type=str, default=None,
                        help="Blinding seed phrase")
    parser.add_argument("--h0-blind", type=float, default=None,
                        help="Blinded H₀ value (km/s/Mpc)")
    parser.add_argument("--dtd-blind", type=float, default=None,
                        help="Blinded D_Δt value (Mpc)")
    args = parser.parse_args()

    if args.receipt:
        if not args.phrase:
            parser.error("--receipt requires --phrase")
        bh = BlindingHandler(args.phrase)
        path = project_root / "results" / "blinding_receipt.json"
        path.parent.mkdir(exist_ok=True)
        bh.write_blinding_receipt(str(path))
        return

    checks = run_checks()
    all_pass = print_gate_status(checks)

    if args.check:
        sys.exit(0 if all_pass else 1)

    if args.unblind:
        if not args.phrase:
            parser.error("--unblind requires --phrase")
        if args.h0_blind is None or args.dtd_blind is None:
            parser.error("--unblind requires --h0-blind and --dtd-blind")

        bh = BlindingHandler(args.phrase)
        result = run_validation_gate(
            bh, args.h0_blind, args.dtd_blind,
            verification_phrase=args.phrase,
            checks_passed=all_pass,
        )
        out = project_root / "results" / "unblinded_cosmology.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"Unblinded results written to {out}")

    if not args.check and not args.unblind and not args.receipt:
        parser.print_help()


if __name__ == "__main__":
    main()

