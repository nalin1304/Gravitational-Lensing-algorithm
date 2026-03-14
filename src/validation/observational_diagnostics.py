"""
Observational image-space diagnostics for strong-lens validation.

This module fits a simple forward model directly to observed HST cutouts
instead of comparing surface-brightness images against convergence maps.
The intended use is annular ring diagnostics on galaxy-galaxy lenses where
the lens mass model is fixed from literature constraints and the source-light
distribution is optimized in image space.

References
----------
Sersic, J. L. 1963, Boletin de la Asociacion Argentina de Astronomia, 6, 41
Schneider, P., Ehlers, J., & Falco, E. E. 1992, Gravitational Lenses
Anderson, J., & King, I. R. 2000, PASP, 112, 1360
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.optics.epsf_model import ePSFModel


IMAGE_SPACE_THRESHOLDS: dict[str, float] = {
    "ring_nrmse_max": 0.12,
    "ring_ssim_min": 0.97,
    "ring_correlation_min": 0.85,
    "annular_flux_ratio_abs_tolerance": 0.10,
}


@dataclass(frozen=True)
class ObservationalFitResult:
    """Container for observational image-space diagnostics."""

    passed: bool
    metrics: dict[str, float]
    best_fit_parameters: dict[str, float]
    model_image: np.ndarray
    processed_observed_image: np.ndarray
    normalized_weight_map: np.ndarray
    annular_mask: np.ndarray
    psf_kernel: np.ndarray


def build_hst_psf_kernel(
    pixel_scale_arcsec: float,
    empirical_psf: np.ndarray | None = None,
    kernel_size: int = 21,
    wavelength_micron: float = 0.80,
    detector_position: tuple[float, float] = (2048.0, 2048.0),
) -> np.ndarray:
    """
    Build an HST-like PSF kernel.

    Empirical kernels are used when present. Otherwise this function uses the
    physically motivated ePSF model already shipped in the repository rather
    than a Gaussian approximation.
    """
    if empirical_psf is not None:
        kernel = np.asarray(empirical_psf, dtype=np.float64)
        kernel = np.clip(kernel, 0.0, None)
        kernel_sum = float(kernel.sum())
        if kernel_sum <= 0.0:
            raise ValueError("Empirical PSF kernel must have positive total flux.")
        return kernel / kernel_sum

    epsf_model = ePSFModel(
        pixel_scale=pixel_scale_arcsec,
        kernel_size=kernel_size,
        wavelength_micron=wavelength_micron,
        aperture_diameter_m=2.4,
        detector_shape=(4096, 4096),
        include_charge_diffusion=True,
        seed=0,
    )
    return epsf_model.evaluate(*detector_position)


def subtract_smooth_foreground(
    image: np.ndarray,
    radial_coordinate_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Suppress smooth foreground light with a radial-median profile.

    The median profile is intentionally robust to localized arc structure and
    acts as a diagnostic foreground subtraction, not a full lens-galaxy
    photometric decomposition.
    """
    radial_bins = np.floor(radial_coordinate_pixels).astype(int)
    max_bin = int(radial_bins.max())
    profile_radius: list[float] = []
    profile_value: list[float] = []
    for radial_bin in range(max_bin + 1):
        annulus_values = image[radial_bins == radial_bin]
        if annulus_values.size == 0:
            continue
        profile_radius.append(float(radial_bin))
        profile_value.append(float(np.median(annulus_values)))

    if not profile_radius:
        smooth_component = np.zeros_like(image, dtype=np.float64)
    else:
        smooth_component = np.interp(
            radial_coordinate_pixels.ravel(),
            np.asarray(profile_radius, dtype=np.float64),
            np.asarray(profile_value, dtype=np.float64),
        ).reshape(image.shape)
    ring_component = np.clip(np.asarray(image, dtype=np.float64) - smooth_component, 0.0, None)
    return ring_component, smooth_component


def elliptical_sersic_source(
    beta_x_arcsec: np.ndarray,
    beta_y_arcsec: np.ndarray,
    *,
    center_x_arcsec: float,
    center_y_arcsec: float,
    effective_radius_arcsec: float,
    sersic_index: float,
    amplitude: float,
    axis_ratio: float,
    position_angle_rad: float,
    background_level: float = 0.0,
) -> np.ndarray:
    """
    Evaluate an elliptical Sersic source profile in the source plane.
    """
    dx_arcsec = np.asarray(beta_x_arcsec, dtype=np.float64) - center_x_arcsec
    dy_arcsec = np.asarray(beta_y_arcsec, dtype=np.float64) - center_y_arcsec

    cos_phi = float(np.cos(position_angle_rad))
    sin_phi = float(np.sin(position_angle_rad))
    major_axis = cos_phi * dx_arcsec + sin_phi * dy_arcsec
    minor_axis = -sin_phi * dx_arcsec + cos_phi * dy_arcsec

    safe_axis_ratio = max(float(axis_ratio), 0.3)
    safe_radius = max(float(effective_radius_arcsec), 1.0e-3)
    safe_index = max(float(sersic_index), 0.5)
    elliptical_radius = np.sqrt(major_axis**2 + (minor_axis / safe_axis_ratio) ** 2)
    sersic_bn = 1.9992 * safe_index - 0.3271
    surface_brightness = float(amplitude) * np.exp(
        -sersic_bn * ((elliptical_radius / safe_radius) ** (1.0 / safe_index) - 1.0)
    )
    return np.clip(surface_brightness + float(background_level), 0.0, None)


def fit_lensed_host_observation(
    observed_image: np.ndarray,
    weight_map: np.ndarray,
    lens_model: Any,
    pixel_scale_arcsec: float,
    einstein_radius_arcsec: float,
    empirical_psf: np.ndarray | None = None,
    detector_position: tuple[float, float] = (2048.0, 2048.0),
) -> ObservationalFitResult:
    """
    Fit a fixed-lens forward model to an observed HST annulus.

    Parameters
    ----------
    observed_image : ndarray
        HST science cutout, already resized to the working grid.
    weight_map : ndarray
        Matching inverse-variance weight map.
    lens_model : object
        Lens model implementing ``deflection_angle(x, y)`` in arcseconds.
    pixel_scale_arcsec : float
        Output pixel scale in arcseconds per pixel.
    einstein_radius_arcsec : float
        Literature Einstein radius defining the diagnostic annulus.
    empirical_psf : ndarray, optional
        Empirical PSF kernel if available.
    detector_position : tuple, optional
        Detector position used by the analytic ePSF model.
    """
    observed_image = np.asarray(observed_image, dtype=np.float64)
    weight_map = np.asarray(weight_map, dtype=np.float64)
    if observed_image.shape != weight_map.shape:
        raise ValueError("observed_image and weight_map must have identical shapes.")

    grid_size_y, grid_size_x = observed_image.shape
    half_extent_arcsec_x = 0.5 * pixel_scale_arcsec * grid_size_x
    half_extent_arcsec_y = 0.5 * pixel_scale_arcsec * grid_size_y
    x_arcsec = np.linspace(-half_extent_arcsec_x, half_extent_arcsec_x, grid_size_x)
    y_arcsec = np.linspace(-half_extent_arcsec_y, half_extent_arcsec_y, grid_size_y)
    image_x_arcsec, image_y_arcsec = np.meshgrid(x_arcsec, y_arcsec)
    radius_arcsec = np.sqrt(image_x_arcsec**2 + image_y_arcsec**2)
    radius_pixels = radius_arcsec / max(pixel_scale_arcsec, 1.0e-12)

    annular_mask = (
        (radius_arcsec >= 0.6 * einstein_radius_arcsec)
        & (radius_arcsec <= 1.8 * einstein_radius_arcsec)
    )
    if not np.any(annular_mask):
        raise ValueError("Annular diagnostic mask is empty.")

    background_mask = (
        (radius_arcsec >= 2.0 * einstein_radius_arcsec)
        & (radius_arcsec <= 2.8 * einstein_radius_arcsec)
    )
    if np.any(background_mask):
        background_level = float(np.median(observed_image[background_mask]))
    else:
        background_level = float(np.median(observed_image))
    processed_observed = np.clip(observed_image - background_level, 0.0, None)
    annular_peak = float(np.max(processed_observed[annular_mask]))
    if annular_peak <= 0.0:
        raise ValueError("Observed annulus has no positive signal after foreground suppression.")
    processed_observed /= annular_peak
    normalized_weight_map = weight_map * (annular_peak**2)
    normalized_sigma = np.sqrt(1.0 / (normalized_weight_map + 1.0e-12))
    median_sigma = float(np.median(normalized_sigma[annular_mask]))
    normalized_sigma = normalized_sigma / max(median_sigma, 1.0e-12)
    normalized_sigma = np.clip(normalized_sigma, 0.25, 6.0)

    psf_kernel = build_hst_psf_kernel(
        pixel_scale_arcsec=pixel_scale_arcsec,
        empirical_psf=empirical_psf,
        detector_position=detector_position,
    )

    alpha_x_arcsec, alpha_y_arcsec = lens_model.deflection_angle(image_x_arcsec, image_y_arcsec)
    beta_x_arcsec = image_x_arcsec - np.asarray(alpha_x_arcsec, dtype=np.float64)
    beta_y_arcsec = image_y_arcsec - np.asarray(alpha_y_arcsec, dtype=np.float64)

    parameter_names = (
        "center_x_arcsec",
        "center_y_arcsec",
        "effective_radius_arcsec",
        "sersic_index",
        "amplitude",
        "axis_ratio",
        "position_angle_rad",
        "background_level",
    )

    def render_model(parameter_vector: np.ndarray) -> np.ndarray:
        source_image = elliptical_sersic_source(
            beta_x_arcsec,
            beta_y_arcsec,
            center_x_arcsec=float(parameter_vector[0]),
            center_y_arcsec=float(parameter_vector[1]),
            effective_radius_arcsec=float(parameter_vector[2]),
            sersic_index=float(parameter_vector[3]),
            amplitude=float(parameter_vector[4]),
            axis_ratio=float(parameter_vector[5]),
            position_angle_rad=float(parameter_vector[6]),
            background_level=float(parameter_vector[7]),
        )
        model_image = fftconvolve(source_image, psf_kernel, mode="same")
        model_image = np.clip(model_image, 0.0, None)
        model_peak = float(np.max(model_image[annular_mask]))
        if model_peak > 0.0:
            model_image = model_image / model_peak
        return model_image

    def objective(parameter_vector: np.ndarray) -> float:
        center_x_arcsec, center_y_arcsec, effective_radius_arcsec, sersic_index, amplitude, axis_ratio, position_angle_rad, background_level = parameter_vector
        if not (
            -0.7 <= center_x_arcsec <= 0.7
            and -0.7 <= center_y_arcsec <= 0.7
            and 0.03 <= effective_radius_arcsec <= 1.8
            and 0.5 <= sersic_index <= 5.0
            and 0.0 < amplitude <= 5.0
            and 0.3 <= axis_ratio <= 1.0
            and -0.5 * np.pi <= position_angle_rad <= 0.5 * np.pi
            and 0.0 <= background_level <= 0.3
        ):
            return 1.0e9

        model_image = render_model(parameter_vector)
        residual = ((model_image - processed_observed) / normalized_sigma)[annular_mask]
        return float(np.mean(residual**2))

    start_vectors = (
        np.array([0.0, 0.0, 0.12, 1.0, 1.0, 0.8, 0.0, 0.01], dtype=np.float64),
        np.array([0.15, 0.0, 0.18, 2.0, 1.0, 0.6, 0.5, 0.02], dtype=np.float64),
        np.array([-0.15, 0.1, 0.08, 3.0, 0.8, 0.7, -0.4, 0.01], dtype=np.float64),
        np.array([0.22, -0.08, 0.24, 1.6, 1.2, 0.55, 0.95, 0.01], dtype=np.float64),
        np.array([-0.24, 0.12, 0.10, 2.8, 0.9, 0.65, -1.10, 0.01], dtype=np.float64),
    )
    best_result = None
    for start_vector in start_vectors:
        result = minimize(
            objective,
            start_vector,
            method="Nelder-Mead",
            options={"maxiter": 350, "xatol": 1.0e-3, "fatol": 1.0e-3},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError("Image-space source optimization did not produce a result.")

    best_fit_vector = np.asarray(best_result.x, dtype=np.float64)
    model_image = render_model(best_fit_vector)
    residual_image = model_image - processed_observed

    ring_nrmse = float(np.sqrt(np.mean(residual_image[annular_mask] ** 2)))
    ring_mae = float(np.mean(np.abs(residual_image[annular_mask])))
    ring_flux_ratio = float(
        np.sum(model_image[annular_mask]) / (np.sum(processed_observed[annular_mask]) + 1.0e-12)
    )
    model_annulus = model_image[annular_mask].ravel()
    observed_annulus = processed_observed[annular_mask].ravel()
    if np.std(model_annulus) <= 1.0e-12 or np.std(observed_annulus) <= 1.0e-12:
        ring_correlation = 0.0
    else:
        ring_correlation = float(np.corrcoef(model_annulus, observed_annulus)[0, 1])
    # SSIM on bounding-box crop of the annular mask region (avoids zero-inflation
    # from multiplying by a binary mask which zeros most pixels).
    _rows = np.any(annular_mask, axis=1)
    _cols = np.any(annular_mask, axis=0)
    _rmin, _rmax = int(np.where(_rows)[0][0]), int(np.where(_rows)[0][-1])
    _cmin, _cmax = int(np.where(_cols)[0][0]), int(np.where(_cols)[0][-1])
    _model_crop = model_image[_rmin:_rmax + 1, _cmin:_cmax + 1]
    _obs_crop = processed_observed[_rmin:_rmax + 1, _cmin:_cmax + 1]
    _ssim_win = min(7, _model_crop.shape[0], _model_crop.shape[1])
    if _ssim_win % 2 == 0:
        _ssim_win = max(_ssim_win - 1, 1)
    ring_ssim = float(
        structural_similarity(
            _model_crop,
            _obs_crop,
            win_size=_ssim_win,
            data_range=float(
                _obs_crop.max() - _obs_crop.min() + 1.0e-6
            ),
        )
    )
    ring_psnr = float(
        peak_signal_noise_ratio(
            processed_observed,
            model_image,
            data_range=float(
                max(model_image.max(), processed_observed.max())
                - min(model_image.min(), processed_observed.min())
                + 1.0e-6
            ),
        )
    )
    chi2 = float(np.sum((residual_image**2) * normalized_weight_map * annular_mask))
    dof = max(int(np.sum(annular_mask)) - len(parameter_names), 1)
    reduced_chi2 = chi2 / dof

    radial_bins = np.floor(radius_pixels).astype(int)
    radial_model: list[float] = []
    radial_observed: list[float] = []
    for radial_bin in range(int(radial_bins[annular_mask].max()) + 1):
        annulus = (radial_bins == radial_bin) & annular_mask
        if not np.any(annulus):
            continue
        radial_model.append(float(np.mean(model_image[annulus])))
        radial_observed.append(float(np.mean(processed_observed[annulus])))
    radial_model_arr = np.asarray(radial_model, dtype=np.float64)
    radial_observed_arr = np.asarray(radial_observed, dtype=np.float64)
    radial_rmse = float(np.sqrt(np.mean((radial_model_arr - radial_observed_arr) ** 2)))

    metrics = {
        "ring_nrmse": ring_nrmse,
        "ring_mae": ring_mae,
        "ring_ssim": ring_ssim,
        "ring_psnr": ring_psnr,
        "ring_correlation": ring_correlation,
        "annular_flux_ratio": ring_flux_ratio,
        "reduced_chi2": reduced_chi2,
        "fit_objective": float(best_result.fun),
        "radial_rmse": radial_rmse,
    }
    passed = bool(
        ring_nrmse <= IMAGE_SPACE_THRESHOLDS["ring_nrmse_max"]
        and ring_ssim >= IMAGE_SPACE_THRESHOLDS["ring_ssim_min"]
        and ring_correlation >= IMAGE_SPACE_THRESHOLDS["ring_correlation_min"]
        and abs(ring_flux_ratio - 1.0) <= IMAGE_SPACE_THRESHOLDS["annular_flux_ratio_abs_tolerance"]
    )

    best_fit_parameters = {
        parameter_name: float(value)
        for parameter_name, value in zip(parameter_names, best_fit_vector)
    }

    return ObservationalFitResult(
        passed=passed,
        metrics=metrics,
        best_fit_parameters=best_fit_parameters,
        model_image=model_image,
        processed_observed_image=processed_observed,
        normalized_weight_map=normalized_weight_map,
        annular_mask=annular_mask.astype(np.float64),
        psf_kernel=psf_kernel,
    )

