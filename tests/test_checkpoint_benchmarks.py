"""Tests for checkpoint-backed benchmark utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    TORCH_AVAILABLE = False


def _synthetic_nfw_case(
    mass_msun: float = 1.6e12,
    concentration: float = 8.5,
    z_lens: float = 0.23,
    z_source: float = 0.89,
    grid_size: int = 64,
) -> tuple[np.ndarray, float, float, float]:
    lens_system = LensSystem(z_lens=z_lens, z_source=z_source)
    extent_arcsec = float(max(2.0, 2.2 * lens_system.einstein_radius_scale(mass_msun)))
    lens_model = NFWProfile(
        M_vir=mass_msun,
        concentration=concentration,
        lens_system=lens_system,
    )
    convergence_map = generate_convergence_map_vectorized(
        lens_model=lens_model,
        grid_size=grid_size,
        extent=extent_arcsec,
    )
    return convergence_map, z_lens, z_source, extent_arcsec


def test_parametric_nfw_fit_recovers_low_error_map() -> None:
    convergence_map, z_lens, z_source, extent_arcsec = _synthetic_nfw_case(grid_size=48)
    fit = fit_parametric_nfw_profile(
        convergence_map=convergence_map,
        z_lens=z_lens,
        z_source=z_source,
        extent_arcsec=extent_arcsec,
        grid_points=10,
        refinement_steps=1,
    )
    assert fit.mass_msun > 0.0
    assert fit.concentration > 0.0
    assert fit.rmse < 0.02


def test_sie_like_fit_returns_physical_parameters() -> None:
    convergence_map, _, _, extent_arcsec = _synthetic_nfw_case(grid_size=48)
    fit = fit_sie_like_profile(convergence_map=convergence_map, extent_arcsec=extent_arcsec)
    fitted_map = generate_sie_like_convergence(
        grid_size=convergence_map.shape[0],
        extent_arcsec=extent_arcsec,
        einstein_radius_arcsec=fit.einstein_radius_arcsec,
        axis_ratio=fit.axis_ratio,
        position_angle_deg=fit.position_angle_deg,
    )
    assert 0.35 <= fit.axis_ratio <= 1.0
    assert np.isfinite(fit.rmse)
    assert np.all(fitted_map >= 0.0)


@pytest.mark.skipif(
    not TORCH_AVAILABLE or not DEFAULT_ABLATION_CHECKPOINTS["full"].exists(),
    reason="PyTorch benchmark checkpoint not available.",
)
def test_checkpoint_calibration_produces_non_negative_prediction() -> None:
    calibration_maps = [_synthetic_nfw_case(mass_msun=mass)[0] for mass in (1.2e12, 1.6e12)]
    test_map = _synthetic_nfw_case(mass_msun=2.0e12)[0]
    model, device = load_ablation_checkpoint_model(DEFAULT_ABLATION_CHECKPOINTS["full"])
    calibration = fit_affine_decoder_calibration(
        model=model,
        calibration_maps=calibration_maps,
        device=device,
    )
    prediction = predict_calibrated_decoder_map(
        model=model,
        convergence_map=test_map,
        calibration=calibration,
        device=device,
    )
    assert prediction.shape == test_map.shape
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction >= 0.0)

