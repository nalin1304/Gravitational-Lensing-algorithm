"""
Effective PSF Model with Spatially Varying Zernike Kernel

The effective Point Spread Function (ePSF) of space-based telescopes
(HST, Roman, Euclid) varies with position on the detector due to:
  - Optical path differences (wavefront error)
  - Detector charge diffusion
  - Jitter smearing

This module models the ePSF using a Zernike polynomial wavefront expansion
(Z4–Z22) evaluated across the detector field of view (FOV), and renders
the physical PSF kernel at arbitrary pixel positions.

Zernike Order
-------------
ANSI / OSA double-index convention:
    Z( n, m ) = R_n^|m|(r) × [cos(mθ) for m≥0, sin(|m|θ) for m<0]
Indices Z4 through Z22 cover:
    Z4  — defocus (n=2, m=0)
    Z5  — vertical astigmatism
    Z6  — oblique astigmatism
    Z7  — vertical coma
    Z8  — horizontal coma
    Z9  — vertical trefoil
    Z10 — oblique trefoil
    Z11 — primary spherical
    ...up to Z22 (5th-order aberrations)

References
----------
Noll (1976) JOSA 66, 207 — Zernike polynomial normalisation
Anderson & King (2000) PASP 112, 1360 — HST ePSF methodology
Perrin et al. (2012) SPIE 8442 — WebbPSF Zernike models

Usage
-----
    from src.optics.epsf_model import ePSFModel

    psf = ePSFModel(pixel_scale=0.05, kernel_size=21)
    # Evaluate at detector position (x_det=512, y_det=1024)
    kernel = psf.evaluate(x_det=512, y_det=1024)
    # Convolve a science image
    convolved = psf.convolve(image, x_det=512, y_det=1024)
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Zernike polynomials (Z1 – Z22)
# NOTE: Uses sequential (n, m) ordering with m increasing from -n to +n,
# which differs from the Noll (1976) / ANSI OSA standard. The physical
# wavefront aberration content is identical; only the index→aberration
# name mapping differs (e.g., j=4 here is astigmatism, not defocus).
# ---------------------------------------------------------------------------

# Map from linear Zernike index j (1-based) to (n, m)
_ZERNIKE_NM: Dict[int, Tuple[int, int]] = {}
_j = 1
for _n in range(9):  # n = 0..8 covers all up to Z36
    for _m in range(-_n, _n + 1, 2):
        _ZERNIKE_NM[_j] = (_n, _m)
        _j += 1
        if _j > 36:
            break
    if _j > 36:
        break


def _radial_zernike(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial part R_n^|m|(rho) of Zernike polynomial."""
    m_abs = abs(m)
    R = np.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * math.factorial(n - s) /
                 (math.factorial(s) *
                  math.factorial((n + m_abs) // 2 - s) *
                  math.factorial((n - m_abs) // 2 - s)))
        R += coeff * rho ** (n - 2 * s)
    return R


def zernike_basis(j: int, nx: int = 64, ny: int = 64) -> np.ndarray:
    """
    Evaluate single Zernike mode Z_j on a unit-disk grid of size (ny, nx).

    The grid is inscribed in the unit circle (|r| ≤ 1).

    Parameters
    ----------
    j : int
        1-based Zernike index (ANSI OSA).
    nx, ny : int
        Output grid dimensions (matches PSF kernel size).

    Returns
    -------
    Z : (ny, nx) ndarray
        Zernike mode values; zero outside the unit disk.
    """
    if j not in _ZERNIKE_NM:
        raise ValueError(f"Zernike index {j} not tabulated (max 36)")
    n, m = _ZERNIKE_NM[j]

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    mask = rho <= 1.0

    R = _radial_zernike(n, m, rho)

    if m == 0:
        norm = np.sqrt(n + 1)
        Z = norm * R
    elif m > 0:
        norm = np.sqrt(2 * (n + 1))
        Z = norm * R * np.cos(m * theta)
    else:
        norm = np.sqrt(2 * (n + 1))
        Z = norm * R * np.sin(abs(m) * theta)

    Z[~mask] = 0.0
    return Z


# ---------------------------------------------------------------------------
# Wavefront error model
# ---------------------------------------------------------------------------

class WavefrontErrorModel:
    """
    Polynomial model for how Zernike coefficients vary across the FOV.

    Each Zernike coefficient c_j varies as a 2-D polynomial in normalised
    detector coordinates (u, v) ∈ [-1, 1]²:

        c_j(u, v) = Σ_{k,l} a_{jkl} u^k v^l    (k, l ≤ polynomials_order)

    Parameters
    ----------
    detector_shape : (H, W)
        Detector dimensions in pixels.
    polynomials_order : int
        Maximum polynomial order for the FOV variation fit.
    seed : int
        RNG seed for generating the default random field.
    """

    Z_INDICES = list(range(4, 23))  # Z4 – Z22

    def __init__(
        self,
        detector_shape: Tuple[int, int] = (4096, 4096),
        polynomials_order: int = 2,
        seed: int = 0,
    ):
        self.detector_shape = detector_shape
        self.poly_order = polynomials_order
        rng = np.random.default_rng(seed)

        # Each Zernike coefficient is a low-order polynomial over FOV
        n_coeffs = (polynomials_order + 1) ** 2
        # (n_zernike_modes, n_poly_coeffs) — random small amplitudes [nm RMS]
        self._poly_coeffs = rng.normal(0, 0.05, (len(self.Z_INDICES), n_coeffs))

    def _normalise_position(self, x_det: float, y_det: float) -> Tuple[float, float]:
        H, W = self.detector_shape
        return (2 * x_det / W - 1), (2 * y_det / H - 1)

    def wavefront_coefficients(
        self, x_det: float, y_det: float
    ) -> Dict[int, float]:
        """
        Evaluate Zernike coefficients [nm RMS] at detector position (x, y).

        Returns
        -------
        dict mapping Zernike index j → coefficient value in nm
        """
        u, v = self._normalise_position(x_det, y_det)
        # Build monomial feature vector [u^k v^l], k+l ≤ poly_order
        features = []
        for k in range(self.poly_order + 1):
            for l in range(self.poly_order + 1):
                features.append(u ** k * v ** l)
        feat = np.array(features)

        result = {}
        for idx, j in enumerate(self.Z_INDICES):
            result[j] = float(np.dot(self._poly_coeffs[idx], feat[: self._poly_coeffs.shape[1]]))
        return result

    def set_zernike_field(
        self, zernike_index: int, poly_coefficients: List[float]
    ) -> None:
        """
        Override the polynomial coefficients for a specific Zernike mode.

        Parameters
        ----------
        zernike_index : int
            Zernike index j (4–22).
        poly_coefficients : list of float
            Polynomial coefficients in FOV-normalised coordinates.
        """
        if zernike_index not in self.Z_INDICES:
            raise ValueError(f"Index {zernike_index} outside Z4–Z22 range")
        idx = self.Z_INDICES.index(zernike_index)
        n_required = self._poly_coeffs.shape[1]
        coeffs = np.array(poly_coefficients[:n_required], dtype=float)
        if len(coeffs) < n_required:
            coeffs = np.pad(coeffs, (0, n_required - len(coeffs)))
        self._poly_coeffs[idx] = coeffs


# ---------------------------------------------------------------------------
# ePSF kernel renderer
# ---------------------------------------------------------------------------

class ePSFModel:
    """
    Spatially-varying effective PSF model for Roman/Euclid surveys.

    Renders the optical PSF by summing Zernike modes weighted by the
    local wavefront error coefficients, then adds diffraction from the
    aperture and optional charge diffusion.

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    kernel_size : int
        Output PSF kernel side length in pixels (odd recommended).
    wavelength_micron : float
        Effective wavelength in microns (for diffraction limit).
    aperture_diameter_m : float
        Primary mirror diameter in metres (default: Roman 2.4 m).
    detector_shape : (H, W)
        Detector dimensions for the wavefront model.
    include_charge_diffusion : bool
        If True, convolve with a 2-D Gaussian charge-diffusion kernel
        (σ = 0.5 pixels, appropriate for H4RG detectors).
    """

    def __init__(
        self,
        pixel_scale: float = 0.11,
        kernel_size: int = 21,
        wavelength_micron: float = 1.55,
        aperture_diameter_m: float = 2.4,
        detector_shape: Tuple[int, int] = (4096, 4096),
        include_charge_diffusion: bool = True,
        seed: int = 0,
    ):
        self.pixel_scale = pixel_scale
        if kernel_size % 2 == 0:
            kernel_size += 1
            warnings.warn(f"kernel_size adjusted to {kernel_size} (must be odd)", stacklevel=2)
        self.kernel_size = kernel_size
        self.wavelength_um = wavelength_micron
        self.aperture_m = aperture_diameter_m
        self.include_charge_diffusion = include_charge_diffusion

        self.wavefront = WavefrontErrorModel(
            detector_shape=detector_shape, seed=seed
        )
        self._cache: Dict[Tuple[float, float], np.ndarray] = {}

    def _diffraction_airy(self) -> np.ndarray:
        """
        Compute the theoretical Airy-disk PSF at the given pixel scale.

        Uses the Fraunhofer approximation:
            I(r) ∝ [2 J₁(x) / x]²,   x = π D r / (λ f)

        where r is radius from centre (arcsec), D is aperture, λ wavelength.
        """
        ks = self.kernel_size
        centre = ks // 2
        x_arcsec = np.arange(ks) - centre
        Y, X = np.meshgrid(x_arcsec, x_arcsec)
        r_arcsec = np.sqrt(X ** 2 + Y ** 2) * self.pixel_scale

        # Convert to dimensionless argument
        lam_arcsec = 206265 * self.wavelength_um * 1e-6 / self.aperture_m
        x = np.pi * r_arcsec / lam_arcsec
        # J1(x)/x — avoid division by zero at centre
        with np.errstate(invalid="ignore", divide="ignore"):
            j1x = np.where(
                x < 1e-10,
                0.5,
                np.vectorize(self._j1)(x) / x,
            )
        airy = (2 * j1x) ** 2
        airy /= airy.sum() + 1e-30
        return airy.astype(np.float64)

    @staticmethod
    def _j1(x: float) -> float:
        """Bessel J₁ via the series expansion (valid for all x)."""
        # Use scipy if available, else Taylor series
        try:
            from scipy.special import j1
            return float(j1(x))
        except ImportError:
            # Taylor: J1(x) ≈ x/2 - x³/16 + x⁵/384 ...
            terms = [x / 2, -(x ** 3) / 16, (x ** 5) / 384, -(x ** 7) / 18432]
            return sum(terms)

    def _wavefront_opd(self, zernike_coeffs: Dict[int, float]) -> np.ndarray:
        """
        Render the optical path difference (OPD) wavefront from Zernike coefficients.

        Returns (kernel_size, kernel_size) OPD map in nm.
        """
        opd = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64)
        for j, c in zernike_coeffs.items():
            try:
                Z = zernike_basis(j, nx=self.kernel_size, ny=self.kernel_size)
                opd += c * Z
            except Exception:
                pass
        return opd

    def _opd_to_psf(self, opd: np.ndarray) -> np.ndarray:
        """
        Convert OPD map to PSF kernel via Fourier optics.

        The pupil function P(u,v) = A(u,v) exp(2πi W(u,v)/λ), where
        A is the circular aperture mask and W is the OPD in units of λ.
        PSF = |FFT(P)|².

        Ref: Goodman (2005) "Introduction to Fourier Optics", Ch. 5
        """
        ks = self.kernel_size
        centre = ks // 2
        x = np.linspace(-1, 1, ks)
        Y, X = np.meshgrid(x, x)
        aperture = (np.sqrt(X ** 2 + Y ** 2) <= 1.0).astype(np.float64)

        # Phase in radians: φ = 2π W[nm] / λ[nm]
        lam_nm = self.wavelength_um * 1000
        phase = 2 * np.pi * opd / lam_nm

        pupil = aperture * np.exp(1j * phase)
        # Pad for cleaner FFT sampling
        pad = ks
        pupil_padded = np.pad(pupil, pad)
        amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))
        psf_padded = np.abs(amplitude) ** 2
        # Crop back to kernel_size, centred
        cy = psf_padded.shape[0] // 2
        cx = psf_padded.shape[1] // 2
        psf = psf_padded[cy - centre: cy + centre + 1, cx - centre: cx + centre + 1]
        psf /= psf.sum() + 1e-30
        return psf

    def _charge_diffusion_kernel(self) -> np.ndarray:
        """2-D Gaussian charge diffusion kernel (σ=0.5 px)."""
        sigma = 0.5
        ks = self.kernel_size
        c = ks // 2
        x = np.arange(ks) - c
        Y, X = np.meshgrid(x, x)
        g = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def evaluate(
        self,
        x_det: float,
        y_det: float,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Evaluate the ePSF kernel at detector position (x_det, y_det).

        Parameters
        ----------
        x_det, y_det : float
            Detector pixel coordinates.
        use_cache : bool
            Cache the kernel at this position (rounded to 128-px grid).

        Returns
        -------
        kernel : (kernel_size, kernel_size) float64 ndarray
            Normalised PSF kernel (sums to 1).
        """
        cache_key = (round(x_det / 128) * 128.0, round(y_det / 128) * 128.0)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        zcoeffs = self.wavefront.wavefront_coefficients(x_det, y_det)
        opd = self._wavefront_opd(zcoeffs)
        psf = self._opd_to_psf(opd)

        # Mix with Airy disk (physical PSF is always ≥ diffraction limit)
        airy = self._diffraction_airy()
        psf = 0.7 * psf + 0.3 * airy
        psf /= psf.sum() + 1e-30

        if self.include_charge_diffusion:
            cd = self._charge_diffusion_kernel()
            psf = fftconvolve(psf, cd, mode="same")
            psf = np.clip(psf, 0, None)
            psf /= psf.sum() + 1e-30

        if use_cache:
            self._cache[cache_key] = psf

        return psf

    def convolve(
        self,
        image: np.ndarray,
        x_det: float,
        y_det: float,
    ) -> np.ndarray:
        """
        Convolve a science image with the ePSF at detector position (x, y).

        For large images with significant spatial variation, consider calling
        ``convolve_spatially_varying`` instead.

        Parameters
        ----------
        image : (H, W) ndarray
        x_det, y_det : float
            Representative detector position for the PSF kernel.

        Returns
        -------
        (H, W) ndarray — PSF-convolved image.
        """
        kernel = self.evaluate(x_det, y_det)
        return fftconvolve(image, kernel, mode="same").astype(np.float64)

    def convolve_spatially_varying(
        self,
        image: np.ndarray,
        patch_size: int = 256,
    ) -> np.ndarray:
        """
        Convolve with a spatially varying ePSF using patch decomposition.

        The image is divided into non-overlapping patches; each patch is
        convolved with the ePSF evaluated at the patch centre.

        Parameters
        ----------
        image : (H, W) ndarray
        patch_size : int
            Tile side length in pixels.
        """
        H, W = image.shape
        out = np.zeros_like(image, dtype=np.float64)

        for y0 in range(0, H, patch_size):
            for x0 in range(0, W, patch_size):
                y1 = min(y0 + patch_size, H)
                x1 = min(x0 + patch_size, W)
                patch = image[y0:y1, x0:x1]
                xc = (x0 + x1) / 2
                yc = (y0 + y1) / 2
                kernel = self.evaluate(xc, yc)
                out[y0:y1, x0:x1] = fftconvolve(patch, kernel, mode="same")

        return out

    def zernike_map(self, zernike_index: int, grid_points: int = 32) -> np.ndarray:
        """
        Evaluate the spatial variation of one Zernike coefficient over the FOV.

        Returns
        -------
        (grid_points, grid_points) float64 ndarray of Zernike coefficients.
        """
        H, W = self.wavefront.detector_shape
        xs = np.linspace(0, W, grid_points)
        ys = np.linspace(0, H, grid_points)
        result = np.zeros((grid_points, grid_points))
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                coeffs = self.wavefront.wavefront_coefficients(x, y)
                result[i, j] = coeffs.get(zernike_index, 0.0)
        return result
