"""Tests for observational image-space diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile
from src.validation.observational_diagnostics import (
    build_hst_psf_kernel,
    elliptical_sersic_source,
    fit_lensed_host_observation,
    subtract_smooth_foreground,
)


def _sigma_v_to_virial_mass(sigma_v_kms: float) -> float:
    return 1.0e12 * (sigma_v_kms / 200.0) ** 4


def test_subtract_smooth_foreground_preserves_ring_signal() -> None:
    grid_size = 64
    y_idx, x_idx = np.mgrid[:grid_size, :grid_size]
    center = 0.5 * (grid_size - 1)
    radius_pixels = np.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2)

    smooth_component = np.exp(-radius_pixels / 10.0)
    ring_component = 0.4 * np.exp(-0.5 * ((radius_pixels - 18.0) / 2.0) ** 2)
    image = smooth_component + ring_component

    residual, smooth = subtract_smooth_foreground(image, radius_pixels)

    assert residual.shape == image.shape
    assert smooth.shape == image.shape
    assert float(np.max(residual)) > 0.05
    assert float(np.mean(residual[radius_pixels < 6.0])) < 0.05


def test_fit_lensed_host_observation_recovers_synthetic_ring() -> None:
    entry = {
        "z_lens": 0.222,
        "z_source": 0.609,
        "sigma_v": 263.0,
        "einstein_radius": 1.38,
    }
    grid_size = 32
    extent_arcsec = max(2.0 * entry["einstein_radius"], 2.0)
    pixel_scale_arcsec = 2.0 * extent_arcsec / grid_size

    x_arcsec = np.linspace(-extent_arcsec, extent_arcsec, grid_size)
    y_arcsec = np.linspace(-extent_arcsec, extent_arcsec, grid_size)
    image_x_arcsec, image_y_arcsec = np.meshgrid(x_arcsec, y_arcsec)
    radius_arcsec = np.sqrt(image_x_arcsec**2 + image_y_arcsec**2)

    lens_system = LensSystem(z_lens=entry["z_lens"], z_source=entry["z_source"])
    lens_model = NFWProfile(
        M_vir=_sigma_v_to_virial_mass(entry["sigma_v"]),
        concentration=10.0,
        lens_system=lens_system,
    )
    alpha_x_arcsec, alpha_y_arcsec = lens_model.deflection_angle(image_x_arcsec, image_y_arcsec)
    beta_x_arcsec = image_x_arcsec - alpha_x_arcsec
    beta_y_arcsec = image_y_arcsec - alpha_y_arcsec

    source_image = elliptical_sersic_source(
        beta_x_arcsec,
        beta_y_arcsec,
        center_x_arcsec=0.12,
        center_y_arcsec=-0.04,
        effective_radius_arcsec=0.16,
        sersic_index=1.8,
        amplitude=1.0,
        axis_ratio=0.72,
        position_angle_rad=0.35,
        background_level=0.01,
    )
    psf_kernel = build_hst_psf_kernel(pixel_scale_arcsec=pixel_scale_arcsec)
    lensed_image = fftconvolve(source_image, psf_kernel, mode="same")

    foreground_light = 0.25 * np.exp(-radius_arcsec / 0.45)
    observed_image = np.clip(lensed_image + foreground_light, 0.0, None)
    weight_map = np.ones_like(observed_image) / (0.02**2)

    fit_result = fit_lensed_host_observation(
        observed_image=observed_image,
        weight_map=weight_map,
        lens_model=lens_model,
        pixel_scale_arcsec=pixel_scale_arcsec,
        einstein_radius_arcsec=entry["einstein_radius"],
    )

    assert fit_result.passed
    assert fit_result.metrics["ring_ssim"] > 0.97
    assert fit_result.metrics["ring_correlation"] > 0.90
    assert abs(fit_result.metrics["annular_flux_ratio"] - 1.0) < 0.08

