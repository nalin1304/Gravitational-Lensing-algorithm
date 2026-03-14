#!/usr/bin/env python3
"""Generate statistical-rigor summary from validation and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _coerce_float(value: Any) -> float | None:
    """Convert JSON values to finite floats when possible."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


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
    validation_scopes = sorted(
        {str(row.get("validation_scope", "unknown")) for row in slacs_rows}
    )
    image_space_mode = any("image_space" in scope for scope in validation_scopes)
    if image_space_mode:
        metric_names = ["rmse", "mae", "ssim", "psnr", "ring_correlation", "annular_flux_ratio"]
        thresholds = {
            "metric_schema": "image_space_forward_model",
            "rmse_max": 0.12,
            "ssim_min": 0.97,
            "ring_correlation_min": 0.85,
            "annular_flux_ratio_abs_tolerance": 0.10,
        }
    else:
        metric_names = ["rmse", "mae", "ssim", "psnr", "mass_conservation"]
        thresholds = {
            "metric_schema": "convergence_map_validation",
            "rmse_max": 0.006,
            "ssim_min": 0.90,
            "mass_conservation_abs_tolerance": 0.02,
        }

    stats = {}
    for metric in metric_names:
        values = []
        for row in slacs_rows:
            numeric = _coerce_float(row.get(metric))
            if numeric is not None:
                values.append(numeric)
        stats[metric] = _bootstrap_mean_ci(values, rng, n_bootstrap)

    per_system = []
    joint_pass_count = 0
    for row in slacs_rows:
        rmse = float(row["rmse"])
        ssim = float(row["ssim"])
        if image_space_mode:
            ring_correlation = float(row.get("ring_correlation", float("nan")))
            annular_flux_ratio = float(row.get("annular_flux_ratio", float("nan")))
            is_pass = (
                rmse <= thresholds["rmse_max"]
                and ssim >= thresholds["ssim_min"]
                and ring_correlation >= thresholds["ring_correlation_min"]
                and abs(annular_flux_ratio - 1.0) <= thresholds["annular_flux_ratio_abs_tolerance"]
            )
        else:
            mass_ratio = float(row["mass_conservation"])
            is_pass = (
                rmse <= thresholds["rmse_max"]
                and ssim >= thresholds["ssim_min"]
                and abs(mass_ratio - 1.0) <= thresholds["mass_conservation_abs_tolerance"]
            )
        if is_pass:
            joint_pass_count += 1
        system_row = {
            "name": str(row.get("name", "unknown")),
            "rmse": rmse,
            "ssim": ssim,
            "joint_pass": is_pass,
        }
        if image_space_mode:
            system_row["ring_correlation"] = ring_correlation
            system_row["annular_flux_ratio"] = annular_flux_ratio
        else:
            system_row["mass_conservation"] = mass_ratio
        per_system.append(system_row)

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
        "validation_scopes": validation_scopes,
        "metric_names": metric_names,
        "systems": per_system,
    }


def _ablation_rigor_summary(ablation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = next((row for row in ablation_rows if row.get("config") == "Full Pipeline"), None)
    if full is None:
        return {"error": "Missing 'Full Pipeline' row in ablation results."}

    full_rmse = float(full["rmse_mean"])
    full_ssim = float(full["ssim_mean"])
    full_pass_rate = float(full.get("pass_rate", 0.0))
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
    no_calibration = next((row for row in ablation_rows if row.get("config") == "No Calibration Layer"), None)
    vanilla = next((row for row in ablation_rows if "Vanilla PINN" in str(row.get("config", ""))), None)
    return {
        "full_pipeline": {
            "rmse_mean": full_rmse,
            "ssim_mean": full_ssim,
            "pass_rate": full_pass_rate,
        },
        "evaluation_modes": evaluation_modes,
        "full_vs_no_calibration_delta_rmse": (
            float(no_calibration["rmse_mean"]) - full_rmse if no_calibration is not None else None
        ),
        "full_vs_vanilla_delta_rmse": (
            float(vanilla["rmse_mean"]) - full_rmse if vanilla is not None else None
        ),
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
    learned_rows = [row for row in score_rows if "checkpoint" in str(row.get("evaluation_mode", "")).lower()]
    learned_our_index = next((i for i, row in enumerate(learned_rows) if row["method"] == our_method), None)
    learned_our_rank = int(learned_our_index + 1) if learned_our_index is not None else None

    proxy_methods_count = sum(
        1 for row in score_rows if "proxy" in str(row.get("evaluation_mode", "")).lower()
    )
    return {
        "n_methods": len(score_rows),
        "n_learned_methods": len(learned_rows),
        "proxy_methods_count": proxy_methods_count,
        "rank_by_rmse": score_rows,
        "our_method_rank_by_rmse": our_rank,
        "our_method_rank_within_learned": learned_our_rank,
    }


def _uq_rigor_summary(uq_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not uq_rows:
        return {"error": "No uncertainty calibration rows found."}

    summary_row = next((row["summary"] for row in uq_rows if "summary" in row), None)
    system_rows = [row for row in uq_rows if "summary" not in row]
    if summary_row is None:
        return {"error": "Missing summary row in uncertainty calibration results."}

    return {
        "n_systems": len(system_rows),
        "mean_rmse": float(summary_row.get("mean_rmse", float("nan"))),
        "mean_ece": float(summary_row.get("mean_ece", float("nan"))),
        "mean_coverage_90": float(summary_row.get("mean_coverage_90", float("nan"))),
        "mean_uq_error_correlation": float(summary_row.get("mean_uq_error_correlation", float("nan"))),
        "prediction_mode": str(summary_row.get("prediction_mode", "unknown")),
        "evaluation_mode": str(summary_row.get("evaluation_mode", "unknown")),
        "publication_valid": bool(summary_row.get("publication_valid", False)),
        "publication_scope": str(summary_row.get("publication_scope", "unspecified")),
        "checkpoint_path": str(summary_row.get("checkpoint_path", "")),
    }


def _build_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    slacs = report.get("slacs_validation", {})
    metric_schema = str(slacs.get("thresholds", {}).get("metric_schema", "convergence_map_validation"))
    if metric_schema == "image_space_forward_model":
        if slacs.get("joint_pass_rate", 0.0) < 0.8:
            warnings.append(
                "Fewer than 80% of SLACS systems pass the joint image-space forward-model thresholds."
            )
    elif slacs.get("joint_pass_rate", 0.0) < 0.6:
        warnings.append(
            "Fewer than 60% of SLACS systems pass joint RMSE/SSIM/mass-conservation thresholds."
        )
    rmse_stats = slacs.get("metrics", {}).get("rmse", {})
    if rmse_stats.get("ci95_high", 0.0) > float(slacs.get("thresholds", {}).get("rmse_max", 0.006)):
        if metric_schema == "image_space_forward_model":
            warnings.append("SLACS image-space NRMSE 95% CI upper bound exceeds the 0.12 threshold.")
        else:
            warnings.append("SLACS RMSE 95% CI upper bound exceeds 0.006 threshold.")
    ssim_stats = slacs.get("metrics", {}).get("ssim", {})
    if ssim_stats.get("ci95_low", 1.0) < float(slacs.get("thresholds", {}).get("ssim_min", 0.90)):
        if metric_schema == "image_space_forward_model":
            warnings.append("SLACS image-space SSIM 95% CI lower bound is below 0.97.")
        else:
            warnings.append("SLACS SSIM 95% CI lower bound is below 0.90.")
    if metric_schema == "image_space_forward_model":
        corr_stats = slacs.get("metrics", {}).get("ring_correlation", {})
        if corr_stats.get("ci95_low", 1.0) < float(slacs.get("thresholds", {}).get("ring_correlation_min", 0.85)):
            warnings.append("SLACS ring-correlation 95% CI lower bound is below 0.85.")
        flux_stats = slacs.get("metrics", {}).get("annular_flux_ratio", {})
        if abs(float(flux_stats.get("mean", 1.0)) - 1.0) > float(
            slacs.get("thresholds", {}).get("annular_flux_ratio_abs_tolerance", 0.10)
        ):
            warnings.append("SLACS annular flux-ratio mean deviates by more than 10% from unity.")
    if any("proxy" in mode.lower() for mode in slacs.get("prediction_modes", [])):
        warnings.append(
            "SLACS validation includes proxy prediction modes; treat as sensitivity analysis."
        )

    sota = report.get("sota_comparison", {})
    learned_rank = sota.get("our_method_rank_within_learned")
    if learned_rank is None:
        warnings.append("Could not determine learned-method RMSE rank for 'Ours (Full PINN)'.")
    elif int(learned_rank) > 1:
        warnings.append("Our method is not the top-ranked learned model by RMSE in the current SOTA table.")
    if int(sota.get("proxy_methods_count", 0)) > 0:
        warnings.append(
            "SOTA comparison contains proxy-simulation methods, not direct checkpoint inference."
        )

    ablation = report.get("ablation_study", {})
    if any("proxy" in mode.lower() for mode in ablation.get("evaluation_modes", [])):
        warnings.append("Ablation results are proxy sensitivity experiments.")
    delta_no_cal = ablation.get("full_vs_no_calibration_delta_rmse")
    if delta_no_cal is not None and float(delta_no_cal) <= 0.0:
        warnings.append("Full pipeline does not outperform the uncalibrated decoder in the ablation study.")
    delta_vanilla = ablation.get("full_vs_vanilla_delta_rmse")
    if delta_vanilla is not None and float(delta_vanilla) < 0.0:
        warnings.append("Full pipeline ranks below the vanilla checkpoint in the ablation study.")

    uq = report.get("uncertainty_calibration", {})
    if uq.get("error"):
        warnings.append(str(uq["error"]))
        return warnings
    if not bool(uq.get("publication_valid", False)):
        warnings.append("Uncertainty calibration artifact is not publication-valid.")
    if "proxy" in str(uq.get("prediction_mode", "")).lower():
        warnings.append("Uncertainty calibration still uses a proxy prediction mode.")
    if float(uq.get("mean_ece", 0.0)) > 0.10:
        warnings.append("Uncertainty calibration mean ECE exceeds 0.10 on held-out synthetic systems.")
    if abs(float(uq.get("mean_coverage_90", 0.0)) - 0.90) > 0.05:
        warnings.append("Uncertainty calibration 90% coverage deviates by more than 0.05 from the target.")

    return warnings


def _to_markdown(report: dict[str, Any]) -> str:
    slacs = report.get("slacs_validation", {})
    ablation = report.get("ablation_study", {})
    sota = report.get("sota_comparison", {})
    uq = report.get("uncertainty_calibration", {})
    warnings = report.get("overall_warnings", [])

    lines = [
        "# Statistical Rigor Report",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## SLACS Validation Summary",
        f"- Systems: {slacs.get('n_systems', 0)}",
        f"- Joint pass rate: {slacs.get('joint_pass_rate', 0.0):.2%}",
        f"- Validation scopes: {', '.join(slacs.get('validation_scopes', [])) or 'unknown'}",
    ]
    metric_schema = str(slacs.get("thresholds", {}).get("metric_schema", "convergence_map_validation"))
    metrics_for_markdown = (
        ("rmse", "ssim", "ring_correlation", "annular_flux_ratio")
        if metric_schema == "image_space_forward_model"
        else ("rmse", "ssim", "mass_conservation")
    )
    for metric in metrics_for_markdown:
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
            f"- RMSE gain vs no calibration: {ablation.get('full_vs_no_calibration_delta_rmse', float('nan')):.6f}",
            f"- RMSE gain vs vanilla: {ablation.get('full_vs_vanilla_delta_rmse', float('nan')):.6f}",
            "",
            "## SOTA Table Summary",
            f"- Methods compared: {sota.get('n_methods', 0)}",
            f"- Our RMSE rank: {sota.get('our_method_rank_by_rmse', 'N/A')}",
            f"- Our learned-model RMSE rank: {sota.get('our_method_rank_within_learned', 'N/A')}",
            "",
            "## Uncertainty Calibration Summary",
            f"- Systems: {uq.get('n_systems', 0)}",
            f"- Prediction mode: {uq.get('prediction_mode', 'unknown')}",
            f"- Evaluation mode: {uq.get('evaluation_mode', 'unknown')}",
            f"- Mean ECE: {uq.get('mean_ece', float('nan')):.6f}",
            f"- Coverage@90%: {uq.get('mean_coverage_90', float('nan')):.6f}",
            f"- Publication scope: {uq.get('publication_scope', 'unspecified')}",
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
        "--uq-json",
        type=Path,
        default=Path("results/uncertainty_calibration_results.json"),
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
    uq_rows = _read_json(args.uq_json)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "slacs_json": str(args.slacs_json),
            "ablation_json": str(args.ablation_json),
            "sota_json": str(args.sota_json),
            "uq_json": str(args.uq_json),
        },
        "bootstrap_samples": args.n_bootstrap,
        "slacs_validation": _slacs_rigor_summary(slacs_rows, rng, args.n_bootstrap),
        "ablation_study": _ablation_rigor_summary(ablation_rows),
        "sota_comparison": _sota_rigor_summary(sota_rows),
        "uncertainty_calibration": _uq_rigor_summary(uq_rows),
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

