"""Checkpoint-backed comparison against analytic gravitational-lens baselines.

This benchmark evaluates the released neural checkpoints and explicit analytic
profile fits on the same synthetic NFW test suite. It does not inject proxy
noise or fabricate method outputs.

Methods
-------
1. Ours (Full PINN): released checkpoint with affine decoder calibration
   fitted on a disjoint synthetic calibration split.
2. Vanilla PINN: released ablated checkpoint with the same calibration
   protocol.
3. Parametric NFW Fit: coarse-to-fine grid search over ``(M_vir, c)`` with
   known redshifts.
4. SIE Approximation: thin-lens SIE-like fit with second-moment ellipticity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mpl_cache_dir = Path(tempfile.gettempdir()) / "gravitational_lensing_matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile
from src.ml.checkpoint_benchmarks import (
    DEFAULT_ABLATION_CHECKPOINTS,
    fit_affine_decoder_calibration,
    fit_parametric_nfw_profile,
    fit_sie_like_profile,
    generate_sie_like_convergence,
    load_ablation_checkpoint_model,
    predict_calibrated_decoder_map,
)
from src.ml.generate_dataset import generate_convergence_map_vectorized
from src.validation import ScientificValidator, ValidationLevel


@dataclass(frozen=True)
class BenchmarkCase:
    """One synthetic benchmark system."""

    system_id: int
    mass_msun: float
    concentration: float
    z_lens: float
    z_source: float
    extent_arcsec: float
    ground_truth: np.ndarray


def generate_benchmark_cases(
    n_cases: int,
    grid_size: int,
    rng: np.random.Generator,
) -> list[BenchmarkCase]:
    """Generate deterministic NFW benchmark cases spanning galaxy-scale lenses."""
    cases: list[BenchmarkCase] = []
    for system_id in range(n_cases):
        mass_msun = float(10.0 ** rng.uniform(11.5, 13.4))
        concentration = float(rng.uniform(5.0, 14.0))
        z_lens = float(rng.uniform(0.15, 0.6))
        z_source = float(min(z_lens + rng.uniform(0.35, 1.6), 3.0))
        lens_system = LensSystem(z_lens=z_lens, z_source=z_source)
        extent_arcsec = float(max(2.0, 2.2 * lens_system.einstein_radius_scale(mass_msun)))
        lens_model = NFWProfile(
            M_vir=mass_msun,
            concentration=concentration,
            lens_system=lens_system,
        )
        ground_truth = generate_convergence_map_vectorized(
            lens_model=lens_model,
            grid_size=grid_size,
            extent=extent_arcsec,
        ).astype(np.float64)
        cases.append(
            BenchmarkCase(
                system_id=system_id,
                mass_msun=mass_msun,
                concentration=concentration,
                z_lens=z_lens,
                z_source=z_source,
                extent_arcsec=extent_arcsec,
                ground_truth=ground_truth,
            )
        )
    return cases


def _evaluate_prediction(
    validator: ScientificValidator,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    result = validator.validate_convergence_map(
        predicted=predicted,
        ground_truth=ground_truth,
        profile_type="NFW",
        verbose=False,
    )
    return {
        "rmse": float(result.metrics.get("rmse", float("nan"))),
        "mae": float(result.metrics.get("mae", float("nan"))),
        "ssim": float(result.metrics.get("ssim", float("nan"))),
        "psnr": float(result.metrics.get("psnr", float("nan"))),
        "mass_conservation": float(result.metrics.get("mass_conservation_ratio", float("nan"))),
    }


def _checkpoint_method(
    checkpoint_key: str,
    calibration_cases: list[BenchmarkCase],
    mc_samples: int = 1,
) -> Callable[[BenchmarkCase], tuple[np.ndarray, float]]:
    checkpoint_path = DEFAULT_ABLATION_CHECKPOINTS[checkpoint_key]
    model, device = load_ablation_checkpoint_model(checkpoint_path)
    calibration = fit_affine_decoder_calibration(
        model=model,
        calibration_maps=[case.ground_truth for case in calibration_cases],
        device=device,
        mc_samples=mc_samples,
    )

    def _predict(case: BenchmarkCase) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        predicted = predict_calibrated_decoder_map(
            model=model,
            convergence_map=case.ground_truth,
            calibration=calibration,
            device=device,
            mc_samples=mc_samples,
        )
        elapsed = time.perf_counter() - start
        return predicted, elapsed

    _predict.metadata = {  # type: ignore[attr-defined]
        "evaluation_mode": "checkpoint_backed_affine_decoder",
        "inference_backend": str(checkpoint_path),
        "calibration_cases": len(calibration_cases),
        "mc_samples": mc_samples,
    }
    return _predict


def _parametric_nfw_method() -> Callable[[BenchmarkCase], tuple[np.ndarray, float]]:
    def _predict(case: BenchmarkCase) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        fit_result = fit_parametric_nfw_profile(
            convergence_map=case.ground_truth,
            z_lens=case.z_lens,
            z_source=case.z_source,
            extent_arcsec=case.extent_arcsec,
        )
        lens_system = LensSystem(z_lens=case.z_lens, z_source=case.z_source)
        fitted_lens = NFWProfile(
            M_vir=fit_result.mass_msun,
            concentration=fit_result.concentration,
            lens_system=lens_system,
        )
        predicted = generate_convergence_map_vectorized(
            lens_model=fitted_lens,
            grid_size=case.ground_truth.shape[0],
            extent=case.extent_arcsec,
        )
        elapsed = time.perf_counter() - start
        return predicted, elapsed

    _predict.metadata = {  # type: ignore[attr-defined]
        "evaluation_mode": "analytic_profile_refit",
        "inference_backend": "coarse_to_fine_nfw_grid_search",
    }
    return _predict


def _sie_method() -> Callable[[BenchmarkCase], tuple[np.ndarray, float]]:
    def _predict(case: BenchmarkCase) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        fit_result = fit_sie_like_profile(
            convergence_map=case.ground_truth,
            extent_arcsec=case.extent_arcsec,
        )
        predicted = generate_sie_like_convergence(
            grid_size=case.ground_truth.shape[0],
            extent_arcsec=case.extent_arcsec,
            einstein_radius_arcsec=fit_result.einstein_radius_arcsec,
            axis_ratio=fit_result.axis_ratio,
            position_angle_deg=fit_result.position_angle_deg,
        )
        elapsed = time.perf_counter() - start
        return predicted, elapsed

    _predict.metadata = {  # type: ignore[attr-defined]
        "evaluation_mode": "analytic_profile_refit",
        "inference_backend": "sie_like_moment_fit",
    }
    return _predict


def evaluate_method(
    method_name: str,
    method: Callable[[BenchmarkCase], tuple[np.ndarray, float]],
    test_cases: list[BenchmarkCase],
) -> dict[str, object]:
    """Evaluate one method across all test cases."""
    validator = ScientificValidator(level=ValidationLevel.RIGOROUS)
    metric_store: dict[str, list[float]] = {
        "rmse": [],
        "mae": [],
        "ssim": [],
        "psnr": [],
        "mass_conservation": [],
        "time_s": [],
    }

    for case in test_cases:
        predicted, elapsed = method(case)
        metrics = _evaluate_prediction(validator, predicted=predicted, ground_truth=case.ground_truth)
        for metric_name in ("rmse", "mae", "ssim", "psnr", "mass_conservation"):
            metric_store[metric_name].append(metrics[metric_name])
        metric_store["time_s"].append(float(elapsed))

    summary: dict[str, object] = {
        metric_name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for metric_name, values in metric_store.items()
    }
    metadata = getattr(method, "metadata", {})
    summary.update(metadata)
    summary["n_test_cases"] = len(test_cases)
    print(
        f"  RMSE: {summary['rmse']['mean']:.6f} ± {summary['rmse']['std']:.6f} | "
        f"SSIM: {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}"
    )
    print(f"  Mode: {summary.get('evaluation_mode', 'unknown')} | Backend: {summary.get('inference_backend', 'unknown')}")
    return summary


def generate_latex_table(all_results: dict[str, dict[str, object]], outdir: Path) -> None:
    """Generate publication-ready LaTeX comparison table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Checkpoint-backed comparison with analytic baseline methods}",
        r"\label{tab:sota}",
        r"\small",
        r"\begin{tabular}{l|cccc|c}",
        r"\toprule",
        r"\textbf{Method} & \textbf{RMSE}$\downarrow$ & \textbf{MAE}$\downarrow$ "
        r"& \textbf{SSIM}$\uparrow$ & \textbf{PSNR}$\uparrow$ & \textbf{Time (ms)} \\",
        r"\midrule",
    ]

    for method_name, metrics in all_results.items():
        is_ours = method_name == "Ours (Full PINN)"
        prefix = r"\textbf{" if is_ours else ""
        suffix = "}" if is_ours else ""
        rmse = metrics["rmse"]
        mae = metrics["mae"]
        ssim = metrics["ssim"]
        psnr = metrics["psnr"]
        time_s = metrics["time_s"]
        lines.append(
            f"  {prefix}{method_name}{suffix} & "
            f"{rmse['mean']:.4f} ± {rmse['std']:.4f} & "
            f"{mae['mean']:.4f} ± {mae['std']:.4f} & "
            f"{ssim['mean']:.4f} ± {ssim['std']:.4f} & "
            f"{psnr['mean']:.1f} ± {psnr['std']:.1f} & "
            f"{time_s['mean'] * 1000.0:.1f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\parbox{\columnwidth}{\footnotesize "
            r"All methods are executed directly: released neural checkpoints use a disjoint affine decoder calibration, "
            r"while analytic baselines refit their profile parameters on each test map.}",
            r"\end{table}",
        ]
    )

    output_path = outdir / "sota_comparison_table.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLaTeX table saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64, help="Grid size.")
    parser.add_argument("--n-lenses", type=int, default=10, help="Number of evaluation systems.")
    parser.add_argument("--n-calibration", type=int, default=6, help="Number of calibration systems for checkpoint decoding.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.n_lenses < 1:
        raise ValueError("--n-lenses must be >= 1")
    if args.n_calibration < 2:
        raise ValueError("--n-calibration must be >= 2")

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  SOTA COMPARISON BENCHMARK")
    print("=" * 80)
    print("  Mode: checkpoint-backed + analytic profile fitting")
    print(f"  Grid: {args.grid}×{args.grid} | Calibration: {args.n_calibration} | Test: {args.n_lenses}")
    print("=" * 80)

    calibration_cases = generate_benchmark_cases(args.n_calibration, args.grid, rng)
    test_cases = generate_benchmark_cases(args.n_lenses, args.grid, rng)

    methods: dict[str, Callable[[BenchmarkCase], tuple[np.ndarray, float]]] = {
        "Ours (Full PINN)": _checkpoint_method("full", calibration_cases),
        "Vanilla PINN": _checkpoint_method("vanilla", calibration_cases),
        "Parametric NFW": _parametric_nfw_method(),
        "SIE Approximation": _sie_method(),
    }

    all_results: dict[str, dict[str, object]] = {}
    for method_name, method in methods.items():
        print(f"\n▶ Evaluating: {method_name}")
        all_results[method_name] = evaluate_method(method_name, method, test_cases)

    print(f"\n{'=' * 90}")
    print(f"{'Method':<24} {'RMSE':>14} {'MAE':>14} {'SSIM':>12} {'PSNR (dB)':>12}")
    print(f"{'-' * 90}")
    for method_name, metrics in all_results.items():
        marker = "★" if method_name == "Ours (Full PINN)" else " "
        print(
            f"{marker} {method_name:<22} "
            f"{metrics['rmse']['mean']:.4f}±{metrics['rmse']['std']:.4f}  "
            f"{metrics['mae']['mean']:.4f}±{metrics['mae']['std']:.4f}  "
            f"{metrics['ssim']['mean']:.4f}±{metrics['ssim']['std']:.4f}  "
            f"{metrics['psnr']['mean']:.1f}±{metrics['psnr']['std']:.1f}"
        )
    print(f"{'=' * 90}")

    generate_latex_table(all_results, outdir)
    json_path = outdir / "sota_comparison_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"JSON results saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

