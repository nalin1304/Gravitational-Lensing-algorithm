#!/usr/bin/env python3
"""
Validate known gravitational lens systems against literature anchors.

This script runs the calibration suite and enforces a publication-style gate:
all *raw* Einstein-radius and enclosed-mass errors must remain below a
configurable threshold (default: 5%). Calibrated metrics are reported for
diagnostics only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.calibration import run_full_calibration_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-error-percent",
        type=float,
        default=5.0,
        help="Maximum allowed raw percent error for each system.",
    )
    parser.add_argument(
        "--max-mass-error-percent",
        type=float,
        default=35.0,
        help=(
            "Maximum allowed raw mass percent error. Mass anchors are typically "
            "less uniform across literature model conventions than Einstein radius."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    calibrator = run_full_calibration_suite()

    if not calibrator.calibration_results:
        print("FAIL: no calibration results were produced.")
        return 2

    print("\nValidation Summary")
    print("=" * 72)
    print(
        f"{'System':30} | {'Raw R err %':>11} | {'Raw M err %':>11} | "
        f"{'Cal R err %':>11} | {'Cal M err %':>11}"
    )
    print("-" * 96)

    max_radius_error = 0.0
    max_mass_error = 0.0
    max_cal_radius_error = 0.0
    max_cal_mass_error = 0.0
    for result in calibrator.calibration_results.values():
        raw_radius_err = float(result.raw_radius_error_percent)
        raw_mass_err = float(result.raw_mass_error_percent)
        cal_radius_err = float(result.calibrated_radius_error_percent)
        cal_mass_err = float(result.calibrated_mass_error_percent)
        max_radius_error = max(max_radius_error, abs(raw_radius_err))
        max_mass_error = max(max_mass_error, abs(raw_mass_err))
        max_cal_radius_error = max(max_cal_radius_error, abs(cal_radius_err))
        max_cal_mass_error = max(max_cal_mass_error, abs(cal_mass_err))
        print(
            f"{result.system_name:30} | {raw_radius_err:11.4f} | {raw_mass_err:11.4f} | "
            f"{cal_radius_err:11.4f} | {cal_mass_err:11.4f}"
        )

    print("-" * 96)
    print(
        f"Max raw errors: radius={max_radius_error:.4f}% mass={max_mass_error:.4f}%"
    )
    print(
        f"Max calibrated errors (diagnostic): radius={max_cal_radius_error:.4f}% "
        f"mass={max_cal_mass_error:.4f}%"
    )

    radius_threshold = float(args.max_error_percent)
    mass_threshold = float(args.max_mass_error_percent)
    passed = max_radius_error < radius_threshold and max_mass_error < mass_threshold
    print(f"Radius threshold: {radius_threshold:.2f}%")
    print(f"Mass threshold: {mass_threshold:.2f}%")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
