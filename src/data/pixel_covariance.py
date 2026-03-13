"""
Pixel-Level Covariance Matrix for Drizzled FITS Imaging

Drizzled HST/JWST images produced by AstroDrizzle do NOT have independent
pixel noise. The sub-pixel resampling (drizzle) algorithm convolves the
native-pixel noise with an overlap kernel, introducing cross-pixel
correlations that can span 2–5 pixels for typical pixfrac parameters.

Ignoring these correlations leads to:
  - Underestimated parameter uncertainties (over-confident posteriors)
  - Biased chi-squared goodness-of-fit statistics
  - Invalid calibration curves (ECE too low)

This module provides:
  1. build_drizzle_covariance — assemble the pixel-covariance matrix Σ from
     the exposure-time map, RMS map, and drizzle kernel overlap
  2. apply_covariance_whitening — whiten (χ-transform) a residual image using
     the Cholesky factor L of Σ (Σ = L Lᵀ)
  3. effective_noise_correlation_length — estimate the empirical pixel
     correlation length for diagnostic reporting

References
----------
Fruchter & Hook (2002), PASP 114, 144
  — Original drizzle algorithm and noise model
Casertano et al. (2000), AJ 120, 2747
  — ACS data-quality pipeline and RMS map construction
Häussler et al. (2004), AJ 128, 2177
  — Correlated noise bias in morphological measurements
Rowe et al. (2015), A&C 10, 121
  — GalSim correlated noise model (inverse covariance approach)

Usage
-----
    from src.data.pixel_covariance import (
        build_drizzle_covariance,
        apply_covariance_whitening,
        effective_noise_correlation_length,
    )

    # Build covariance from drizzle kernel params + rms_map
    cov = build_drizzle_covariance(
        rms_map=rms,          # (N, M) float32 RMS image from HST pipeline
        pixfrac=0.8,          # drizzle pixfrac parameter
        scale=0.5,            # output/input pixel scale ratio
        kernel="turbo",       # drizzle kernel type
    )

    # Whiten a model residual with the covariance
    chi = apply_covariance_whitening(residual=obs - model, cov=cov)

    # Report effective correlation length (in pixels)
    xi = effective_noise_correlation_length(cov, shape=rms.shape)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Drizzle kernel overlap functions
# ---------------------------------------------------------------------------

def _square_kernel_overlap(
    sep: np.ndarray, pixfrac: float, scale: float
) -> np.ndarray:
    """
    Compute the overlap integral between two square drizzle kernels
    separated by `sep` pixels in the output frame.

    The overlap fraction for a 1-D pair of square kernels of half-width
    w = pixfrac * scale * 0.5 (in output-pixel units) separated by dx is:

        A(dx) = max(0, 2w - |dx|) / (2w)        for |dx| <= 2w

    For 2-D the product A(dx) * A(dy) gives the fractional area overlap.

    Ref: Fruchter & Hook (2002), Eq. 4
    """
    w = pixfrac * scale  # decorrelation length: kernel reaches zero at separation = w (Fruchter & Hook 2002)
    w = max(w, 1e-6)
    overlap_1d = np.maximum(0.0, 1.0 - np.abs(sep) / w)
    return overlap_1d


def _gaussian_kernel_overlap(
    sep: np.ndarray, pixfrac: float, scale: float
) -> np.ndarray:
    """
    Approximate Gaussian drizzle kernel overlap.
    σ is set so the Gaussian FWHM equals pixfrac * scale.
    """
    sigma = pixfrac * scale / (2 * np.sqrt(2 * np.log(2)))
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * (sep / sigma) ** 2)


def _lanczos_kernel_overlap(
    sep: np.ndarray, pixfrac: float, scale: float, n: int = 3
) -> np.ndarray:
    """Lanczos-n kernel overlap (approximate via convolution width)."""
    sigma = pixfrac * scale * n / 2.0
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * (sep / sigma) ** 2)


_KERNEL_FUNCTIONS = {
    "square": _square_kernel_overlap,
    "gaussian": _gaussian_kernel_overlap,
    "turbo": _square_kernel_overlap,      # turbo ≈ square
    "lanczos2": lambda s, p, sc: _lanczos_kernel_overlap(s, p, sc, n=2),
    "lanczos3": lambda s, p, sc: _lanczos_kernel_overlap(s, p, sc, n=3),
}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def build_drizzle_covariance(
    rms_map: np.ndarray,
    pixfrac: float = 0.8,
    scale: float = 0.5,
    kernel: Literal["square", "gaussian", "turbo", "lanczos2", "lanczos3"] = "square",
    max_lag: int = 5,
    regularize_eps: float = 1e-6,
) -> np.ndarray:
    """
    Build the full pixel covariance matrix for a drizzled FITS image.

    The covariance between output pixel (i, j) and output pixel (i', j') is:

        Σ[(i,j),(i',j')] = σᵢⱼ · σᵢ'ⱼ' · K(|i-i'|, |j-j'|)

    where σᵢⱼ is the per-pixel RMS and K is the normalized drizzle overlap
    kernel evaluated at the angular separation between the two pixels.

    Only pairs within `max_lag` pixels are coupled (sparse approximation).
    This band-approximation is accurate because drizzle correlations decay
    to < 1% beyond ~5 pixels for typical pixfrac/scale combinations.

    Parameters
    ----------
    rms_map : (N, M) ndarray
        Per-pixel RMS noise, as provided by the HST/JWST pipeline
        (*_rms.fits or weight map with WHT = 1/σ²).
    pixfrac : float
        Drizzle pixfrac parameter (0 < pixfrac ≤ 1).
        Smaller values → less correlation; pixfrac=1 → full overlap.
    scale : float
        Output/input pixel scale ratio (0 < scale ≤ 1).
    kernel : str
        Drizzle kernel type: "square" | "gaussian" | "turbo" |
        "lanczos2" | "lanczos3".
    max_lag : int
        Maximum pixel lag to include in the covariance matrix.
        Pixels farther apart than this are assumed uncorrelated.
    regularize_eps : float
        Small diagonal added to Σ before Cholesky decomposition to
        ensure positive-definiteness (Tikhonov regularization).
        Units: fraction of median σ².

    Returns
    -------
    Sigma : (N*M, N*M) ndarray  [for small images]
        Full covariance matrix. Large images must be downsampled to avoid
        prohibitive memory costs.

    Notes
    -----
    For a 64×64 image the full Σ is 4096×4096 (128 MB float32).
    For a 128×128 image the full Σ is 16384×16384 (1 GB float32). Larger
    images must be downsampled before calling this function.
    """
    if rms_map.ndim != 2:
        raise ValueError(f"rms_map must be 2-D, got shape {rms_map.shape}")
    rms_map = np.asarray(rms_map, dtype=np.float64)

    N, M = rms_map.shape
    n_pix = N * M

    if pixfrac <= 0 or pixfrac > 1:
        raise ValueError(f"pixfrac must be in (0, 1], got {pixfrac}")
    if scale <= 0 or scale > 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")

    kernel_fn = _KERNEL_FUNCTIONS.get(kernel)
    if kernel_fn is None:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {list(_KERNEL_FUNCTIONS)}")

    if n_pix > 128 * 128:
        raise ValueError(
            f"Image size {N}×{M} = {n_pix} pixels is too large for a dense covariance "
            "matrix. Downsample the image or reduce the grid size to avoid fallback "
            "approximations."
        )

    # ── Full dense covariance matrix for manageable image sizes
    sigma_flat = rms_map.ravel()                 # (N*M,)
    Sigma = np.empty((n_pix, n_pix), dtype=np.float64)

    rows = np.arange(n_pix)
    row_i = rows // M
    row_j = rows % M

    for idx in range(n_pix):
        di = row_i - row_i[idx]    # (N*M,) pixel row separations
        dj = row_j - row_j[idx]    # (N*M,) pixel col separations
        mask = (np.abs(di) <= max_lag) & (np.abs(dj) <= max_lag)
        K = np.zeros(n_pix, dtype=np.float64)
        if mask.any():
            K[mask] = (
                kernel_fn(np.abs(di[mask]).astype(np.float64), pixfrac, scale)
                * kernel_fn(np.abs(dj[mask]).astype(np.float64), pixfrac, scale)
            )
        Sigma[idx, :] = sigma_flat[idx] * sigma_flat * K

    # Regularize for positive-definiteness
    median_var = float(np.median(sigma_flat ** 2))
    Sigma += np.eye(n_pix) * regularize_eps * median_var

    return Sigma


def _sparse_covariance_profile(
    rms_map: np.ndarray,
    pixfrac: float,
    scale: float,
    kernel_fn,
    max_lag: int,
) -> dict:
    """Return a sparse dict with variance array + 2-D correlation kernel."""
    lags = np.arange(0, max_lag + 1, dtype=np.float64)
    corr_1d = kernel_fn(lags, pixfrac, scale)
    corr_1d /= corr_1d[0]   # normalize to 1 at zero lag
    corr_2d = np.outer(corr_1d, corr_1d)   # 2-D separable approximation
    return {
        "variance": rms_map ** 2,            # (N, M) per-pixel variance
        "correlation_kernel": corr_2d,       # (max_lag+1, max_lag+1)
        "pixfrac": pixfrac,
        "scale": scale,
        "kernel": kernel_fn.__name__ if hasattr(kernel_fn, "__name__") else "custom",
        "max_lag": max_lag,
    }


def apply_covariance_whitening(
    residual: np.ndarray,
    cov: np.ndarray,
    return_chisq: bool = True,
) -> np.ndarray | float:
    """
    Whiten a residual image using the Cholesky factorization of Σ.

    Computes the generalized whitened residual:

        χ = L⁻¹ r        (L from Σ = L Lᵀ)

    and optionally the scalar chi-squared statistic:

        χ² = rᵀ Σ⁻¹ r = ||χ||²

    This accounts for the full pixel-covariance structure introduced by
    drizzle processing.

    Ref: Press et al. (2007) "Numerical Recipes", §15.6

    Parameters
    ----------
    residual : ndarray
        Model residual image (observed - model), shape (N, M) or (N*M,).
    cov : (N*M, N*M) ndarray
        Full covariance matrix from `build_drizzle_covariance`.
    return_chisq : bool
        If True, return the scalar χ² value.
        If False, return the whitened residual vector χ.

    Returns
    -------
    float or (N*M,) ndarray
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    if cov.shape != (len(r), len(r)):
        raise ValueError(
            f"Covariance matrix shape {cov.shape} does not match "
            f"residual length {len(r)}"
        )

    c, low = cho_factor(cov, lower=True)
    if return_chisq:
        # r^T Σ⁻¹ r — cho_solve gives Σ⁻¹r, correct for chi-squared
        chi_inv = cho_solve((c, low), r)
        return float(np.dot(r, chi_inv))
    # Whitened residual L⁻¹r where Σ = L L^T
    # c from cho_factor (lower=True) is the lower-triangular Cholesky factor L
    whitened = solve_triangular(c, r, lower=True)
    return whitened


def effective_noise_correlation_length(
    cov: np.ndarray | dict,
    shape: tuple[int, int],
) -> float:
    """
    Estimate the effective pixel-to-pixel noise correlation length.

    For a full covariance matrix, reads the first row and measures the
    e-folding scale of the correlation function ρ(Δ).

    For a sparse dict (large images), reads the precomputed 1-D kernel.

    Returns
    -------
    xi : float
        Effective correlation length in pixels (1/e decay half-width).
        For uncorrelated noise this is 0.

    Ref: Rowe et al. (2015), GalSim §3.3 — definition of correlation length
    """
    if isinstance(cov, dict):
        C1d = cov["correlation_kernel"][:, 0]
    elif isinstance(cov, np.ndarray):
        N, M = shape
        sigma0 = np.sqrt(float(cov[0, 0]))
        if sigma0 < 1e-30:
            return 0.0
        row0 = cov[0, :M].copy()
        row0 /= row0[0]
        C1d = row0
    else:
        return 0.0

    # Find the 1/e crossing of the correlation profile
    xi = 0.0
    for k, v in enumerate(C1d):
        if v < np.exp(-1.0):
            xi = float(k - 1 + (np.exp(-1.0) - C1d[k - 1]) /
                       (v - C1d[k - 1])) if k > 0 else 0.0
            break
    else:
        xi = float(len(C1d))  # fully correlated within max_lag
    return xi


def whiten_image_block_diagonal(
    residual: np.ndarray,
    rms_map: np.ndarray,
    pixfrac: float = 0.8,
    scale: float = 0.5,
    kernel: str = "square",
    block_size: int = 16,
) -> Tuple[np.ndarray, float]:
    """
    Block-diagonal covariance whitening for large images.

    Partitions the image into non-overlapping `block_size × block_size`
    tiles, builds a dense covariance for each tile, and whitens within
    each block. The global chi-squared is the sum of per-block chi-squares.

    This is the recommended approach for images larger than 128×128 pixels
    where building the full covariance is prohibitive.

    Parameters
    ----------
    residual, rms_map : (N, M) ndarray
    pixfrac, scale, kernel : drizzle parameters
    block_size : int
        Tile size in pixels (power of 2 preferred for FFT alignment).

    Returns
    -------
    whitened : (N, M) ndarray
        Whitened residual image (each block independently whitened).
    chisq : float
        Total chi-squared statistic across all blocks.
    """
    residual = np.asarray(residual, dtype=np.float64)
    rms_map = np.asarray(rms_map, dtype=np.float64)
    N, M = residual.shape
    whitened = np.zeros_like(residual)
    total_chisq = 0.0

    for i0 in range(0, N, block_size):
        for j0 in range(0, M, block_size):
            i1 = min(i0 + block_size, N)
            j1 = min(j0 + block_size, M)
            r_block = residual[i0:i1, j0:j1]
            s_block = rms_map[i0:i1, j0:j1]
            cov_block = build_drizzle_covariance(
                s_block, pixfrac=pixfrac, scale=scale, kernel=kernel
            )
            r_flat = r_block.ravel()
            chi_flat = apply_covariance_whitening(
                r_flat, cov_block, return_chisq=False
            )
            whitened[i0:i1, j0:j1] = chi_flat.reshape(r_block.shape)
            total_chisq += float(np.dot(r_flat, chi_flat))

    return whitened, total_chisq
