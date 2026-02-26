"""
Uncertainty Calibration Analysis

Evaluates the calibration of Bayesian uncertainty estimates (MC Dropout)
by computing coverage probability, expected calibration error (ECE),
and generating reliability diagrams.

These metrics are essential for publication in any journal that requires
uncertainty quantification (IEEE TCI, A&A, ApJ, MNRAS).

The script REQUIRES a trained PINN model to generate real MC Dropout samples.
If no model checkpoint is found it falls back to a Gaussian-noise proxy ---
this proxy is ONLY for standalone smoke-testing and MUST NOT be reported in
any publication.

Usage:
    # With real model (required for publication):
    python scripts/uncertainty_calibration.py --model models/pinn_best.pt

    # Smoke-test without model (not publication-valid):
    python scripts/uncertainty_calibration.py

References:
    Gal & Ghahramani (2016) - Dropout as a Bayesian Approximation
    Gal & Ghahramani (2016b) - Uncertainty in Deep Learning (thesis)
    Lard et al. (2024) - Substructure Power Spectrum with CNN+UQ
    Varma et al. (2024) - MVE + UDA for strong lensing UQ

Author: Gravitational Lensing Research Platform
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.lens_models.mass_profiles import NFWProfile
from src.lens_models.lens_system import LensSystem
from src.ml.generate_dataset import generate_convergence_map_vectorized

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def run_mc_dropout_predictions(
    convergence_map: np.ndarray,
    model,
    device,
    n_forward_passes: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run real MC Dropout forward passes through the trained PINN.

    With dropout enabled at inference time (Gal & Ghahramani 2016),
    each forward pass produces a different stochastic realisation of the
    convergence map.  The mean and std across T passes are the predictive
    mean and epistemic uncertainty.

    Parameters
    ----------
    convergence_map : np.ndarray
        True convergence map [H, W] used as network input.
    model : PhysicsInformedNN
        Trained PINN.  Dropout layers are set to train() internally.
    device : torch.device
        Target device.
    n_forward_passes : int
        Number of stochastic forward passes T.

    Returns
    -------
    mean_pred : np.ndarray  [H, W]
    epistemic_std : np.ndarray  [H, W]
    """
    import torch
    from src.utils.common import prepare_model_input

    # Keep BatchNorm in eval mode, but enable Dropout stochasticity
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

    # Prepare input tensor [1, 1, H, W]
    input_tensor = prepare_model_input(convergence_map, target_size=convergence_map.shape[0])
    input_tensor = input_tensor.to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(n_forward_passes):
            # Model returns (params, classification) — we need the decoded κ map
            # Use the convergence map itself as proxy if model has no decoder output
            params, _ = model(input_tensor)
            # Reconstruct κ map from predicted NFW parameters
            from src.lens_models.mass_profiles import NFWProfile
            from src.lens_models.lens_system import LensSystem
            from src.ml.generate_dataset import generate_convergence_map_vectorized

            p = params.cpu().numpy()[0]
            # params: [log10_Mvir, concentration, z_l, z_s, ellipticity]
            M_vir = 10 ** float(p[0])
            conc = max(float(p[1]), 1.0)
            z_l = max(float(p[2]), 0.05)
            z_s = max(float(p[3]), z_l + 0.05)

            lens_sys = LensSystem(z_lens=z_l, z_source=z_s)
            lens = NFWProfile(M_vir=M_vir, concentration=conc, lens_system=lens_sys)
            kappa = generate_convergence_map_vectorized(
                lens_model=lens,
                grid_size=convergence_map.shape[0],
                extent=max(2.0, 1.5 * 1.0),
            )
            predictions.append(kappa)

    predictions = np.array(predictions)  # [T, H, W]
    return predictions.mean(axis=0), predictions.std(axis=0)


def _fallback_gaussian_predictions(
    ground_truth: np.ndarray,
    rng: np.random.RandomState,
    n_forward_passes: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-noise proxy for MC Dropout.

    ⚠️  WARNING: This is NOT a real uncertainty estimate.  It is only valid
    for smoke-testing the calibration pipeline when no trained model is
    available.  Results produced by this function MUST NOT appear in any
    publication or scientific report.
    """
    import warnings
    warnings.warn(
        "\n\n"
        "[uncertainty_calibration] SMOKE-TEST MODE: using Gaussian-noise proxy\n"
        "instead of real MC Dropout forward passes. These results are INVALID\n"
        "for publication. Provide --model <checkpoint.pt> for real UQ.\n",
        UserWarning,
        stacklevel=2,
    )
    predictions = []
    for _ in range(n_forward_passes):
        dropout_noise = rng.normal(0, 0.008, ground_truth.shape)
        base_noise = rng.normal(0, 0.003, ground_truth.shape)
        pred = np.maximum(ground_truth + base_noise + dropout_noise, 0)
        predictions.append(pred)
    predictions = np.array(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0)


def compute_coverage(
    ground_truth: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    confidence_levels: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical coverage probability at various confidence levels.

    Coverage = fraction of pixels where |κ_true - κ_pred| < z * σ_pred

    A well-calibrated model should have coverage ≈ confidence level.
    """
    if confidence_levels is None:
        confidence_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

    from scipy.stats import norm
    z_scores = norm.ppf((1 + confidence_levels) / 2)

    residuals = np.abs(ground_truth - predicted_mean)
    coverages = []

    for z in z_scores:
        within = (residuals < z * predicted_std).mean()
        coverages.append(within)

    return confidence_levels, np.array(coverages)


def compute_ece(
    confidence_levels: np.ndarray,
    empirical_coverages: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = mean(|empirical_coverage - expected_coverage|)
    """
    return np.mean(np.abs(empirical_coverages - confidence_levels))


def generate_calibration_figure(
    all_results: List[Dict],
    outdir: Path,
):
    """Generate publication-ready reliability diagram."""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ---- Panel (a): Reliability Diagram ----
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration', alpha=0.7)

    for res in all_results:
        ax.plot(res["confidence_levels"], res["coverages"],
                'o-', ms=3, lw=1.2, alpha=0.5)

    # Mean across lenses
    mean_coverages = np.mean([r["coverages"] for r in all_results], axis=0)
    ax.plot(all_results[0]["confidence_levels"], mean_coverages,
            's-', color='#e74c3c', markersize=5, lw=2, label='Mean (all lenses)')

    ax.set_xlabel('Expected Coverage', fontsize=11)
    ax.set_ylabel('Empirical Coverage', fontsize=11)
    ax.set_title('(a) Reliability Diagram', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # ---- Panel (b): ECE per lens ----
    ax = axes[1]
    names = [r["name"].replace("SDSS ", "") for r in all_results]
    eces = [r["ece"] for r in all_results]
    colors = ['#2ecc71' if e < 0.05 else '#e74c3c' for e in eces]
    ax.barh(names, eces, color=colors, edgecolor='#333', height=0.6)
    ax.axvline(0.05, color='gray', linestyle='--', lw=1.2, alpha=0.7, label='ECE < 0.05')
    ax.set_xlabel('Expected Calibration Error (ECE)', fontsize=11)
    ax.set_title('(b) ECE per Lens', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    # ---- Panel (c): Uncertainty vs Error scatter ----
    ax = axes[2]
    for res in all_results:
        ax.scatter(res["pixel_uncertainties"], res["pixel_errors"],
                   alpha=0.02, s=1, c='steelblue')

    # Add perfect correlation line
    max_val = max(max(r["pixel_uncertainties"].max() for r in all_results),
                  max(r["pixel_errors"].max() for r in all_results))
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='σ = |error|')
    ax.set_xlabel('Predicted σ (epistemic)', fontsize=11)
    ax.set_ylabel('|Prediction Error|', fontsize=11)
    ax.set_title('(c) Uncertainty vs Error', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val * 1.1])
    ax.set_ylim([0, max_val * 1.1])

    plt.tight_layout()
    fig_path = outdir / "uncertainty_calibration.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📊 Calibration figure saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Uncertainty calibration analysis")
    parser.add_argument("--grid", type=int, default=64, help="Grid size")
    parser.add_argument("--n-samples", type=int, default=30, help="MC Dropout forward passes")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PINN checkpoint (.pt). Required for publication-valid UQ."
    )
    args = parser.parse_args()

    # --- Load real model if available ---
    model = None
    device = None
    if args.model:
        try:
            import torch
            from src.utils.common import load_pretrained_model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = load_pretrained_model(args.model, device=str(device))
            model.eval()
            print(f"✅ Loaded PINN model from {args.model} (real MC Dropout)")
        except Exception as e:
            print(f"⚠️  Could not load model ({e}). Falling back to smoke-test proxy.")
            model = None

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # SLACS lenses for calibration analysis
    LENSES = [
        {"name": "SDSS J0946+1006", "z_l": 0.222, "z_s": 0.609, "sigma_v": 263.0, "theta_E": 1.38},
        {"name": "SDSS J1250+0523", "z_l": 0.232, "z_s": 0.795, "sigma_v": 252.0, "theta_E": 1.13},
        {"name": "SDSS J1402+6321", "z_l": 0.205, "z_s": 0.481, "sigma_v": 267.0, "theta_E": 1.35},
        {"name": "SDSS J0252+0039", "z_l": 0.280, "z_s": 0.982, "sigma_v": 164.0, "theta_E": 1.04},
        {"name": "SDSS J0037-0942", "z_l": 0.195, "z_s": 0.632, "sigma_v": 279.0, "theta_E": 1.53},
    ]

    print("\n" + "=" * 75)
    print("  UNCERTAINTY CALIBRATION ANALYSIS — MC Dropout")
    print("=" * 75)
    print(f"  Grid: {args.grid}×{args.grid} | T={args.n_samples} forward passes")
    print("=" * 75)

    all_results = []

    for entry in LENSES:
        print(f"\n▶ {entry['name']}")

        # Generate ground truth
        M_vir = 1e12 * (entry["sigma_v"] / 200.0) ** 4
        lens_sys = LensSystem(z_lens=entry["z_l"], z_source=entry["z_s"])
        lens = NFWProfile(M_vir=M_vir, concentration=10.0, lens_system=lens_sys)
        extent = max(2.0 * entry["theta_E"], 2.0)
        ground_truth = generate_convergence_map_vectorized(
            lens_model=lens, grid_size=args.grid, extent=extent
        )

        # Run MC Dropout (real PINN if available, else smoke-test proxy)
        if model is not None and device is not None:
            mean_pred, epistemic_std = run_mc_dropout_predictions(
                ground_truth, model, device, n_forward_passes=args.n_samples
            )
        else:
            mean_pred, epistemic_std = _fallback_gaussian_predictions(
                ground_truth, rng, n_forward_passes=args.n_samples
            )

        # Coverage analysis
        confidence_levels, coverages = compute_coverage(
            ground_truth, mean_pred, epistemic_std
        )

        # ECE
        ece = compute_ece(confidence_levels, coverages)

        # Per-pixel uncertainty vs error (for scatter plot)
        pixel_errors = np.abs(ground_truth.ravel() - mean_pred.ravel())
        pixel_uncertainties = epistemic_std.ravel()

        # Pearson correlation (uncertainty-error alignment)
        corr = np.corrcoef(pixel_uncertainties, pixel_errors)[0, 1]

        print(f"  ECE = {ece:.4f}  |  Coverage@90% = {coverages[8]:.3f}  |  "
              f"UQ-Error Corr = {corr:.3f}")

        all_results.append({
            "name": entry["name"],
            "ece": ece,
            "confidence_levels": confidence_levels.tolist(),
            "coverages": coverages.tolist(),
            "coverage_90": coverages[8],
            "uq_error_correlation": corr,
            "pixel_uncertainties": pixel_uncertainties,
            "pixel_errors": pixel_errors,
        })

    # Summary
    mean_ece = np.mean([r["ece"] for r in all_results])
    mean_cov90 = np.mean([r["coverage_90"] for r in all_results])
    mean_corr = np.mean([r["uq_error_correlation"] for r in all_results])

    print(f"\n{'=' * 75}")
    print(f"  SUMMARY")
    print(f"  Mean ECE:            {mean_ece:.4f} {'✅ Well-calibrated' if mean_ece < 0.05 else '⚠️  Needs improvement'}")
    print(f"  Mean Coverage@90%:   {mean_cov90:.3f} (target: 0.900)")
    print(f"  Mean UQ-Error Corr:  {mean_corr:.3f}")
    print(f"{'=' * 75}")

    # Generate figure
    generate_calibration_figure(all_results, outdir)

    # Save JSON (without large arrays)
    json_results = [{k: v for k, v in r.items()
                     if k not in ("pixel_uncertainties", "pixel_errors")}
                    for r in all_results]
    json_results.append({
        "summary": {
            "mean_ece": mean_ece,
            "mean_coverage_90": mean_cov90,
            "mean_uq_error_correlation": mean_corr,
        }
    })
    json_path = outdir / "uncertainty_calibration_results.json"
    json_path.write_text(json.dumps(json_results, indent=2, default=str))
    print(f"📊 JSON results saved: {json_path}")

    print("\n✓ Uncertainty calibration analysis complete.")


if __name__ == "__main__":
    main()
