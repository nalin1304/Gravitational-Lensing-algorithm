#!/usr/bin/env python3
"""Generate statistical-rigor summary from validation and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_mean_ci(
    values: list[float],
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    if arr.size == 1:
        val = float(arr[0])
        return {"n": 1.0, "mean": val, "std": 0.0, "ci95_low": val, "ci95_high": val}

    sample_idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    boot_means = arr[sample_idx].mean(axis=1)
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(boot_means, [low_q, high_q])
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def _slacs_rigor_summary(
    slacs_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    metric_names = ["rmse", "mae", "ssim", "psnr", "mass_conservation"]
    stats = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in slacs_rows if metric in row]
        stats[metric] = _bootstrap_mean_ci(values, rng, n_bootstrap)

    thresholds = {
        "rmse_max": 0.006,
        "ssim_min": 0.90,
        "mass_conservation_abs_tolerance": 0.02,
    }
    per_system = []
    joint_pass_count = 0
    for row in slacs_rows:
        rmse = float(row["rmse"])
        ssim = float(row["ssim"])
        mass_ratio = float(row["mass_conservation"])
        is_pass = (
            rmse <= thresholds["rmse_max"]
            and ssim >= thresholds["ssim_min"]
            and abs(mass_ratio - 1.0) <= thresholds["mass_conservation_abs_tolerance"]
        )
        if is_pass:
            joint_pass_count += 1
        per_system.append(
            {
                "name": str(row.get("name", "unknown")),
                "rmse": rmse,
                "ssim": ssim,
                "mass_conservation": mass_ratio,
                "joint_pass": is_pass,
            }
        )

    n_systems = len(slacs_rows)
    pass_rate = float(joint_pass_count / n_systems) if n_systems else 0.0
    prediction_modes = sorted(
        {str(row.get("prediction_mode", "unknown")) for row in slacs_rows}
    )
    data_sources = sorted(
        {str(row.get("data_source", "unknown")) for row in slacs_rows}
    )
    return {
        "n_systems": n_systems,
        "metrics": stats,
        "thresholds": thresholds,
        "joint_pass_count": joint_pass_count,
        "joint_pass_rate": pass_rate,
        "prediction_modes": prediction_modes,
        "data_sources": data_sources,
        "systems": per_system,
    }


def _ablation_rigor_summary(ablation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = next((row for row in ablation_rows if row.get("config") == "Full Pipeline"), None)
    if full is None:
        return {"error": "Missing 'Full Pipeline' row in ablation results."}

    full_rmse = float(full["rmse_mean"])
    full_ssim = float(full["ssim_mean"])
    comparisons = []
    for row in ablation_rows:
        cfg = str(row.get("config", "unknown"))
        if cfg == "Full Pipeline":
            continue
        rmse = float(row["rmse_mean"])
        ssim = float(row["ssim_mean"])
        comparisons.append(
            {
                "config": cfg,
                "delta_rmse_vs_full": rmse - full_rmse,
                "delta_ssim_vs_full": ssim - full_ssim,
                "rmse_ratio_vs_full": (rmse / full_rmse) if full_rmse > 0 else float("nan"),
            }
        )

    comparisons.sort(key=lambda item: item["delta_rmse_vs_full"], reverse=True)
    evaluation_modes = sorted(
        {str(row.get("evaluation_mode", "unknown")) for row in ablation_rows}
    )
    return {
        "full_pipeline": {
            "rmse_mean": full_rmse,
            "ssim_mean": full_ssim,
            "pass_rate": float(full.get("pass_rate", 0.0)),
        },
        "evaluation_modes": evaluation_modes,
        "comparisons": comparisons,
    }


def _sota_rigor_summary(sota_results: dict[str, Any]) -> dict[str, Any]:
    score_rows = []
    for method, metrics in sota_results.items():
        evaluation_mode = str(metrics.get("evaluation_mode", "unknown"))
        score_rows.append(
            {
                "method": method,
                "rmse_mean": float(metrics["rmse"]["mean"]),
                "ssim_mean": float(metrics["ssim"]["mean"]),
                "time_ms_mean": float(metrics["time_s"]["mean"]) * 1000.0,
                "evaluation_mode": evaluation_mode,
            }
        )
    score_rows.sort(key=lambda row: row["rmse_mean"])

    our_method = "Ours (Full PINN)"
    our_index = next((i for i, row in enumerate(score_rows) if row["method"] == our_method), None)
    our_rank = int(our_index + 1) if our_index is not None else None

    proxy_methods_count = sum(
        1 for row in score_rows if "proxy" in str(row.get("evaluation_mode", "")).lower()
    )
    return {
        "n_methods": len(score_rows),
        "proxy_methods_count": proxy_methods_count,
        "rank_by_rmse": score_rows,
        "our_method_rank_by_rmse": our_rank,
    }


def _build_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    slacs = report.get("slacs_validation", {})
    if slacs.get("joint_pass_rate", 0.0) < 0.6:
        warnings.append(
            "Fewer than 60% of SLACS systems pass joint RMSE/SSIM/mass-conservation thresholds."
        )
    rmse_stats = slacs.get("metrics", {}).get("rmse", {})
    if rmse_stats.get("ci95_high", 0.0) > 0.006:
        warnings.append("SLACS RMSE 95% CI upper bound exceeds 0.006 threshold.")
    ssim_stats = slacs.get("metrics", {}).get("ssim", {})
    if ssim_stats.get("ci95_low", 1.0) < 0.90:
        warnings.append("SLACS SSIM 95% CI lower bound is below 0.90.")
    if any("proxy" in mode.lower() for mode in slacs.get("prediction_modes", [])):
        warnings.append(
            "SLACS validation includes proxy prediction modes; treat as sensitivity analysis."
        )

    sota = report.get("sota_comparison", {})
    our_rank = sota.get("our_method_rank_by_rmse")
    if our_rank is None:
        warnings.append("Could not determine RMSE rank for 'Ours (Full PINN)'.")
    elif int(our_rank) > 2:
        warnings.append("Our method ranks lower than top-2 by RMSE in current SOTA table.")
    if int(sota.get("proxy_methods_count", 0)) > 0:
        warnings.append(
            "SOTA comparison contains proxy-simulation methods, not direct checkpoint inference."
        )

    ablation = report.get("ablation_study", {})
    full = ablation.get("full_pipeline", {})
    if float(full.get("pass_rate", 0.0)) < 0.8:
        warnings.append("Full pipeline ablation pass_rate is below 0.80.")
    if any("proxy" in mode.lower() for mode in ablation.get("evaluation_modes", [])):
        warnings.append("Ablation results are proxy sensitivity experiments.")

    return warnings


def _to_markdown(report: dict[str, Any]) -> str:
    slacs = report.get("slacs_validation", {})
    ablation = report.get("ablation_study", {})
    sota = report.get("sota_comparison", {})
    warnings = report.get("overall_warnings", [])

    lines = [
        "# Statistical Rigor Report",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## SLACS Validation Summary",
        f"- Systems: {slacs.get('n_systems', 0)}",
        f"- Joint pass rate: {slacs.get('joint_pass_rate', 0.0):.2%}",
    ]
    for metric in ("rmse", "ssim", "mass_conservation"):
        stats = slacs.get("metrics", {}).get(metric, {})
        lines.append(
            f"- {metric}: mean={stats.get('mean', float('nan')):.6f}, "
            f"95% CI=[{stats.get('ci95_low', float('nan')):.6f}, {stats.get('ci95_high', float('nan')):.6f}]"
        )

    lines.extend(
        [
            "",
            "## Ablation Effect Summary",
            f"- Full pipeline RMSE mean: {ablation.get('full_pipeline', {}).get('rmse_mean', float('nan')):.6f}",
            f"- Full pipeline pass rate: {ablation.get('full_pipeline', {}).get('pass_rate', 0.0):.2%}",
            "",
            "## SOTA Table Summary",
            f"- Methods compared: {sota.get('n_methods', 0)}",
            f"- Our RMSE rank: {sota.get('our_method_rank_by_rmse', 'N/A')}",
            "",
            "## Warnings",
        ]
    )
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slacs-json",
        type=Path,
        default=Path("results/real_data/slacs_validation_results.json"),
    )
    parser.add_argument(
        "--ablation-json",
        type=Path,
        default=Path("results/ablation_results.json"),
    )
    parser.add_argument(
        "--sota-json",
        type=Path,
        default=Path("results/sota_comparison_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/statistical_rigor_report.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/statistical_rigor_report.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if warnings are present.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    slacs_rows = _read_json(args.slacs_json)
    ablation_rows = _read_json(args.ablation_json)
    sota_rows = _read_json(args.sota_json)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "slacs_json": str(args.slacs_json),
            "ablation_json": str(args.ablation_json),
            "sota_json": str(args.sota_json),
        },
        "bootstrap_samples": args.n_bootstrap,
        "slacs_validation": _slacs_rigor_summary(slacs_rows, rng, args.n_bootstrap),
        "ablation_study": _ablation_rigor_summary(ablation_rows),
        "sota_comparison": _sota_rigor_summary(sota_rows),
    }
    report["overall_warnings"] = _build_warnings(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_to_markdown(report), encoding="utf-8")

    print("Statistical rigor report generated.")
    print(f"JSON: {args.output}")
    print(f"Markdown: {args.markdown_output}")
    if report["overall_warnings"]:
        print("Warnings:")
        for warning in report["overall_warnings"]:
            print(f"- {warning}")

    if args.strict and report["overall_warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
