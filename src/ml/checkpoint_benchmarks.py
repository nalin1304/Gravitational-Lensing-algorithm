"""Checkpoint-backed benchmark utilities for publication artifacts.

This module centralizes the benchmark inference path used by the publication
scripts. It provides:

1. Loading the shipped PyTorch ablation checkpoints.
2. Calibrating the decoder output into convergence-map units using a disjoint
   calibration split.
3. Analytic NFW and SIE-like profile fitting for non-neural baselines.

Notes
-----
The checkpoint architecture is reconstructed from the released state dicts.
The affine decoder calibration is fitted only on a disjoint calibration split;
the test split is never used to determine the affine parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile
from src.ml.generate_dataset import generate_convergence_map_vectorized
from src.utils.common import prepare_model_input

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


DEFAULT_ABLATION_CHECKPOINTS = {
    "full": Path("models/ablation_pinn.pth"),
    "vanilla": Path("models/ablation_vanilla.pth"),
}


@dataclass(frozen=True)
class DecoderCalibration:
    """Affine calibration for decoder outputs."""

    slope: float
    intercept: float
    calibration_cases: int
    mc_samples: int


@dataclass(frozen=True)
class NFWFitResult:
    """Best-fit NFW parameters for a convergence map."""

    mass_msun: float
    concentration: float
    rmse: float


@dataclass(frozen=True)
class SIEApproximation:
    """Best-fit SIE-like approximation for a convergence map."""

    einstein_radius_arcsec: float
    axis_ratio: float
    position_angle_deg: float
    rmse: float


if _HAS_TORCH:

    class AblationCheckpointNet(nn.Module):
        """PyTorch architecture reconstructed from the released ablation checkpoints."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
            )
            self.dense = nn.Sequential(
                nn.Linear(128 * 8 * 8, 1024),
                nn.BatchNorm1d(1024),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),
            )
            self.param_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 6),
            )
            self.class_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3),
            )
            self.source_decoder = nn.Sequential(
                nn.Linear(256, 64 * 4 * 4),
                nn.BatchNorm1d(64 * 4 * 4),
                nn.GELU(),
                nn.Unflatten(1, (64, 4, 4)),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            )

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return parameter logits, class logits, and decoder output."""
            features = self.encoder(inputs)
            flattened = features.view(inputs.size(0), -1)
            dense_features = self.dense(flattened)
            return (
                self.param_head(dense_features),
                self.class_head(dense_features),
                self.source_decoder(dense_features),
            )

else:

    class AblationCheckpointNet:  # pragma: no cover - optional dependency path
        """Dummy class when torch is unavailable."""

        def __init__(self) -> None:
            raise RuntimeError("PyTorch is required to load benchmark checkpoints.")


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for checkpoint-backed benchmark inference.")


def _resolve_device(device: str | None = None) -> "torch.device":
    _require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_dropout_sampling_mode(model: "AblationCheckpointNet", enabled: bool) -> None:
    model.eval()
    if not enabled:
        return
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def load_ablation_checkpoint_model(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple["AblationCheckpointNet", "torch.device"]:
    """Load one of the released ablation checkpoints.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the checkpoint state dict.
    device : str or None, optional
        Explicit torch device string. Defaults to CUDA when available,
        otherwise CPU.

    Returns
    -------
    model : AblationCheckpointNet
        Loaded model in evaluation mode.
    device : torch.device
        Device used for the model.
    """
    _require_torch()
    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

    resolved_device = _resolve_device(device)
    model = AblationCheckpointNet().to(resolved_device)
    state_dict = torch.load(resolved_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, resolved_device


def infer_decoder_output(
    model: "AblationCheckpointNet",
    convergence_map: np.ndarray,
    device: "torch.device",
    mc_samples: int = 1,
) -> np.ndarray:
    """Return the raw decoder output for one convergence map.

    Parameters
    ----------
    model : AblationCheckpointNet
        Loaded checkpoint model.
    convergence_map : np.ndarray
        Input convergence map with shape ``(N, N)``.
    device : torch.device
        Device for inference.
    mc_samples : int, optional
        Number of dropout samples to average. ``1`` uses deterministic
        evaluation mode.

    Returns
    -------
    decoder_output : np.ndarray
        Raw decoder image with the same spatial shape as the input.
    """
    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1")

    input_tensor = prepare_model_input(convergence_map).to(device)
    _set_dropout_sampling_mode(model, enabled=mc_samples > 1)

    decoder_outputs: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(mc_samples):
            _, _, decoded = model(input_tensor)
            decoder_outputs.append(decoded.detach().cpu().numpy()[0, 0])

    model.eval()
    return np.mean(decoder_outputs, axis=0)


def fit_affine_decoder_calibration(
    model: "AblationCheckpointNet",
    calibration_maps: Sequence[np.ndarray],
    device: "torch.device",
    mc_samples: int = 1,
) -> DecoderCalibration:
    """Fit a global affine decoder calibration on a disjoint split.

    Parameters
    ----------
    model : AblationCheckpointNet
        Loaded checkpoint model.
    calibration_maps : sequence of np.ndarray
        Calibration convergence maps. These maps must be disjoint from the
        later evaluation/test cases.
    device : torch.device
        Torch device for inference.
    mc_samples : int, optional
        Number of decoder samples to average before fitting.

    Returns
    -------
    DecoderCalibration
        Affine coefficients ``kappa = slope * raw + intercept``.
    """
    if not calibration_maps:
        raise ValueError("At least one calibration map is required.")

    raw_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    for calibration_map in calibration_maps:
        raw_decoder = infer_decoder_output(
            model=model,
            convergence_map=calibration_map,
            device=device,
            mc_samples=mc_samples,
        )
        raw_blocks.append(raw_decoder.reshape(-1, 1))
        truth_blocks.append(np.asarray(calibration_map, dtype=np.float64).reshape(-1, 1))

    design_matrix = np.hstack(
        [np.vstack(raw_blocks), np.ones((sum(block.shape[0] for block in raw_blocks), 1), dtype=np.float64)]
    )
    targets = np.vstack(truth_blocks)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, targets, rcond=None)
    return DecoderCalibration(
        slope=float(coefficients[0, 0]),
        intercept=float(coefficients[1, 0]),
        calibration_cases=len(calibration_maps),
        mc_samples=mc_samples,
    )


def apply_affine_decoder_calibration(
    raw_decoder_output: np.ndarray,
    calibration: DecoderCalibration,
) -> np.ndarray:
    """Convert raw decoder output into convergence-map units.

    Parameters
    ----------
    raw_decoder_output : np.ndarray
        Raw decoder image.
    calibration : DecoderCalibration
        Affine calibration coefficients.

    Returns
    -------
    calibrated_convergence_map : np.ndarray
        Non-negative convergence map.
    """
    calibrated = calibration.slope * np.asarray(raw_decoder_output, dtype=np.float64) + calibration.intercept
    return np.maximum(calibrated, 0.0)


def predict_calibrated_decoder_map(
    model: "AblationCheckpointNet",
    convergence_map: np.ndarray,
    calibration: DecoderCalibration,
    device: "torch.device",
    mc_samples: int = 1,
) -> np.ndarray:
    """Run checkpoint inference and return the calibrated convergence map."""
    raw_decoder = infer_decoder_output(
        model=model,
        convergence_map=convergence_map,
        device=device,
        mc_samples=mc_samples,
    )
    return apply_affine_decoder_calibration(raw_decoder, calibration)


def _rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((lhs - rhs) ** 2)))


def fit_parametric_nfw_profile(
    convergence_map: np.ndarray,
    z_lens: float,
    z_source: float,
    extent_arcsec: float,
    log10_mass_bounds: tuple[float, float] = (11.2, 13.8),
    concentration_bounds: tuple[float, float] = (4.0, 16.0),
    grid_points: int = 12,
    refinement_steps: int = 2,
) -> NFWFitResult:
    """Fit an NFW profile directly to a convergence map by grid search.

    Parameters
    ----------
    convergence_map : np.ndarray
        Target convergence map.
    z_lens : float
        Lens redshift.
    z_source : float
        Source redshift.
    extent_arcsec : float
        Map extent used to generate the convergence grid.
    log10_mass_bounds : tuple of float, optional
        Search bounds for ``log10(M_vir / M_sun)``.
    concentration_bounds : tuple of float, optional
        Search bounds for the concentration parameter.
    grid_points : int, optional
        Number of grid points per dimension at each search stage.
    refinement_steps : int, optional
        Number of coarse-to-fine refinement stages.

    Returns
    -------
    NFWFitResult
        Best-fit mass, concentration, and RMSE.
    """
    grid_size = int(convergence_map.shape[0])
    mass_low, mass_high = log10_mass_bounds
    conc_low, conc_high = concentration_bounds
    best_result: NFWFitResult | None = None
    lens_system = LensSystem(z_lens=z_lens, z_source=z_source)

    for _ in range(refinement_steps + 1):
        log_masses = np.linspace(mass_low, mass_high, grid_points)
        concentrations = np.linspace(conc_low, conc_high, grid_points)
        for log_mass in log_masses:
            for concentration in concentrations:
                lens_model = NFWProfile(
                    M_vir=float(10.0**log_mass),
                    concentration=float(concentration),
                    lens_system=lens_system,
                )
                candidate_map = generate_convergence_map_vectorized(
                    lens_model=lens_model,
                    grid_size=grid_size,
                    extent=extent_arcsec,
                )
                candidate_rmse = _rmse(candidate_map, convergence_map)
                if best_result is None or candidate_rmse < best_result.rmse:
                    best_result = NFWFitResult(
                        mass_msun=float(10.0**log_mass),
                        concentration=float(concentration),
                        rmse=candidate_rmse,
                    )

        assert best_result is not None
        mass_center = np.log10(best_result.mass_msun)
        concentration_center = best_result.concentration
        mass_span = max((mass_high - mass_low) * 0.35, 0.05)
        concentration_span = max((conc_high - conc_low) * 0.35, 0.5)
        mass_low = max(log10_mass_bounds[0], mass_center - mass_span / 2.0)
        mass_high = min(log10_mass_bounds[1], mass_center + mass_span / 2.0)
        conc_low = max(concentration_bounds[0], concentration_center - concentration_span / 2.0)
        conc_high = min(concentration_bounds[1], concentration_center + concentration_span / 2.0)

    return best_result


def generate_sie_like_convergence(
    grid_size: int,
    extent_arcsec: float,
    einstein_radius_arcsec: float,
    axis_ratio: float,
    position_angle_deg: float,
) -> np.ndarray:
    """Generate an SIE-like convergence map.

    Notes
    -----
    This uses a thin-lens elliptical isothermal approximation of the form

    ``kappa(x, y) = theta_E / (2 * R_ell)``,

    with ``R_ell`` defined in a rotated pseudo-elliptical coordinate system.
    It is intended for baseline comparison, not precision strong-lensing
    parameter inference.
    """
    coordinates = np.linspace(-extent_arcsec, extent_arcsec, grid_size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    angle_rad = np.deg2rad(position_angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    x_rot = x_grid * cos_angle + y_grid * sin_angle
    y_rot = -x_grid * sin_angle + y_grid * cos_angle
    clipped_axis_ratio = float(np.clip(axis_ratio, 0.35, 1.0))
    elliptical_radius = np.sqrt(
        clipped_axis_ratio * x_rot**2 + (y_rot**2) / clipped_axis_ratio + 1.0e-6
    )
    return 0.5 * float(einstein_radius_arcsec) / elliptical_radius


def estimate_axis_ratio_and_angle(convergence_map: np.ndarray, extent_arcsec: float) -> tuple[float, float]:
    """Estimate axis ratio and position angle from weighted second moments."""
    grid_size = int(convergence_map.shape[0])
    coordinates = np.linspace(-extent_arcsec, extent_arcsec, grid_size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    weights = np.maximum(np.asarray(convergence_map, dtype=np.float64), 0.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return 1.0, 0.0

    x_center = float(np.sum(weights * x_grid) / total_weight)
    y_center = float(np.sum(weights * y_grid) / total_weight)
    x_shifted = x_grid - x_center
    y_shifted = y_grid - y_center
    covariance = np.array(
        [
            [
                np.sum(weights * x_shifted * x_shifted) / total_weight,
                np.sum(weights * x_shifted * y_shifted) / total_weight,
            ],
            [
                np.sum(weights * x_shifted * y_shifted) / total_weight,
                np.sum(weights * y_shifted * y_shifted) / total_weight,
            ],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = float(np.sqrt(max(eigenvalues[1], 1.0e-12)))
    minor = float(np.sqrt(max(eigenvalues[0], 1.0e-12)))
    axis_ratio = float(np.clip(minor / major, 0.35, 1.0))
    major_vector = eigenvectors[:, 1]
    position_angle_deg = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])))
    return axis_ratio, position_angle_deg


def fit_sie_like_profile(
    convergence_map: np.ndarray,
    extent_arcsec: float,
    theta_e_bounds: tuple[float, float] | None = None,
    grid_points: int = 72,
) -> SIEApproximation:
    """Fit an SIE-like profile to a convergence map.

    Parameters
    ----------
    convergence_map : np.ndarray
        Target convergence map.
    extent_arcsec : float
        Map extent used to generate the convergence grid.
    theta_e_bounds : tuple of float or None, optional
        Search bounds for the Einstein radius. When omitted, the bounds are
        tied to the field of view.
    grid_points : int, optional
        Number of Einstein-radius samples.

    Returns
    -------
    SIEApproximation
        Best-fit SIE-like approximation parameters.
    """
    grid_size = int(convergence_map.shape[0])
    axis_ratio, position_angle_deg = estimate_axis_ratio_and_angle(convergence_map, extent_arcsec)
    if theta_e_bounds is None:
        theta_e_bounds = (0.05 * extent_arcsec, 1.8 * extent_arcsec)

    best_result: SIEApproximation | None = None
    for theta_e in np.linspace(theta_e_bounds[0], theta_e_bounds[1], grid_points):
        candidate_map = generate_sie_like_convergence(
            grid_size=grid_size,
            extent_arcsec=extent_arcsec,
            einstein_radius_arcsec=float(theta_e),
            axis_ratio=axis_ratio,
            position_angle_deg=position_angle_deg,
        )
        candidate_rmse = _rmse(candidate_map, convergence_map)
        if best_result is None or candidate_rmse < best_result.rmse:
            best_result = SIEApproximation(
                einstein_radius_arcsec=float(theta_e),
                axis_ratio=axis_ratio,
                position_angle_deg=position_angle_deg,
                rmse=candidate_rmse,
            )

    assert best_result is not None
    return best_result
