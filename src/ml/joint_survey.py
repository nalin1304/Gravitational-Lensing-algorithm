"""
Joint-Survey Deblending — Multi-Resolution Likelihood Module

Simultaneously models the same gravitational lens system using data from
multiple telescope tiers with different pixel scales:
  - Ground-based (Rubin/LSST-like): ~0.2 arcsec/pixel, wide field
  - Space-based (Roman-like): ~0.11 arcsec/pixel, high resolution

The joint likelihood treats the two datasets as independent measurements
of the same underlying lens model, with per-survey PSF, noise, and
covariance handled separately via a shared source-plane representation.

Physics
-------
The joint log-likelihood is:

    ln L = ln L_ground + ln L_space

where each term is either Gaussian (uncorrelated) or correlated via the
pixel-level covariance (Σ) from drizzle processing:

    ln L = -0.5 [ r⊤ Σ⁻¹ r + ln |Σ| + N ln(2π) ]

For computational efficiency, the correlated term uses block-diagonal
Cholesky decomposition from src.data.pixel_covariance.

Reference
---------
Birrer et al. (2022) A&A 657, L15 — joint Rubin+Euclid lensing
Galan et al. (2022) A&A — pixelised source with multi-resolution likelihood

Usage
-----
    from src.ml.joint_survey import JointSurveyLikelihood, SurveyObservation

    ground = SurveyObservation(image=img_rubin, sigma=sigma_rubin, pixel_scale=0.2)
    space  = SurveyObservation(image=img_roman, sigma=sigma_roman, pixel_scale=0.11)
    jll = JointSurveyLikelihood(ground=ground, space=space)
    logL = jll.log_likelihood(model_ground, model_space)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.signal import fftconvolve
    from scipy.linalg import cho_factor, cho_solve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SurveyObservation:
    """
    Single-survey observation container.

    Parameters
    ----------
    image : (H, W) ndarray
        Observed flux image.
    sigma : (H, W) ndarray or float
        Per-pixel RMS noise (from pipeline weight map: σ = 1/√w).
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    psf_kernel : (K, K) ndarray or None
        Normalised PSF kernel. If None, delta (no blurring) is assumed.
    weight_map : (H, W) ndarray or None
        Pipeline weight map (w = 1/σ²). If provided, overrides ``sigma``.
    mask : (H, W) bool ndarray or None
        True where pixels are bad / saturated (excluded from likelihood).
    survey_name : str
        Identifier for logging ('rubin', 'roman', 'hst', …).
    """
    image: np.ndarray
    sigma: object  # float or (H, W)
    pixel_scale: float
    psf_kernel: Optional[np.ndarray] = None
    weight_map: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    survey_name: str = "survey"

    def __post_init__(self):
        self.image = np.asarray(self.image, dtype=np.float64)
        if self.weight_map is not None:
            wm = np.asarray(self.weight_map, dtype=np.float64)
            wm = np.where(wm > 0, wm, 1e-30)
            self.sigma = 1.0 / np.sqrt(wm)
        self.sigma = np.broadcast_to(
            np.asarray(self.sigma, dtype=np.float64), self.image.shape
        ).copy()
        if self.mask is None:
            self.mask = np.zeros(self.image.shape, dtype=bool)

    def convolve_model(self, model: np.ndarray) -> np.ndarray:
        """Convolve a model image with the survey PSF."""
        if self.psf_kernel is None:
            return model
        if not _HAS_SCIPY:
            raise ImportError(
                "SciPy is required for PSF convolution. "
                "Install scipy to use non-delta PSF kernels."
            )
        return fftconvolve(model, self.psf_kernel, mode="same")


# ---------------------------------------------------------------------------
# Likelihood functions
# ---------------------------------------------------------------------------

def _gaussian_log_likelihood(
    observed: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Independent-pixel Gaussian log-likelihood:
        ln L = -0.5 Σ [(obs - model)² / σ² + ln(2π σ²)]
    """
    valid = ~mask
    r = (observed[valid] - model[valid]) / sigma[valid]
    return float(-0.5 * (np.dot(r, r) + np.sum(np.log(2 * np.pi * sigma[valid] ** 2))))


def _correlated_log_likelihood(
    observed: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    covariance: Optional[np.ndarray] = None,
) -> float:
    """
    Correlated-pixel log-likelihood using explicit covariance matrix.
        ln L = -0.5 [r⊤ Σ⁻¹ r + ln|Σ| + N ln(2π)]

    Falls back to independent-pixel if covariance is None or scipy missing.
    """
    if covariance is None or not _HAS_SCIPY:
        return _gaussian_log_likelihood(observed, model, sigma, mask)

    valid = ~mask
    r = (observed - model).ravel()
    # Select valid pixels only (block-diagonal approximation applied externally)
    valid_idx = np.where(valid.ravel())[0]
    r_valid = r[valid_idx]

    Sigma_sub = covariance[np.ix_(valid_idx, valid_idx)]
    try:
        c, low = cho_factor(Sigma_sub, lower=True)
        chisq = float(np.dot(r_valid, cho_solve((c, low), r_valid)))
        sign, logdet = np.linalg.slogdet(Sigma_sub)
        N = len(r_valid)
        return -0.5 * (chisq + logdet + N * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        warnings.warn(
            "Covariance matrix not positive-definite; falling back to diagonal.",
            RuntimeWarning, stacklevel=3,
        )
        return _gaussian_log_likelihood(observed, model, sigma, mask)


# ---------------------------------------------------------------------------
# Joint likelihood engine
# ---------------------------------------------------------------------------

class JointSurveyLikelihood:
    """
    Multi-resolution joint likelihood for simultaneous ground + space analysis.

    The two survey images are assumed to contain the same lens system at
    different angular resolutions. Each dataset contributes an independent
    log-likelihood term:

        ln L_joint = Σ_s  ln L_s(obs_s | model_s)

    where model_s is the source-plane model rendered at survey s's pixel scale
    and convolved with the survey's PSF.

    Parameters
    ----------
    *observations : SurveyObservation
        One or more survey observations (typically 2: ground + space).
    use_correlated_likelihood : bool
        If True and covariance matrices are provided, use the full
        pixel-covariance likelihood (Cholesky solver).
    """

    def __init__(
        self,
        *observations: SurveyObservation,
        use_correlated_likelihood: bool = False,
    ):
        if len(observations) < 1:
            raise ValueError("At least one SurveyObservation required.")
        self.observations = list(observations)
        self.use_correlated = use_correlated_likelihood
        self._covariances: dict[str, Optional[np.ndarray]] = {
            o.survey_name: None for o in observations
        }

    def set_covariance(self, survey_name: str, covariance: np.ndarray) -> None:
        """
        Set a pre-computed pixel covariance matrix for a survey.

        Parameters
        ----------
        survey_name : str
        covariance : (N², N²) ndarray
            Full pixel covariance for the image of that survey.
        """
        if survey_name not in self._covariances:
            raise KeyError(f"Survey '{survey_name}' not registered")
        self._covariances[survey_name] = covariance

    def log_likelihood(
        self,
        *models: np.ndarray,
    ) -> float:
        """
        Evaluate the joint log-likelihood.

        Parameters
        ----------
        *models : ndarray
            Predicted model images for each survey observation, in the same
            order as the observations passed to the constructor. Each model
            is convolved internally with the survey PSF before comparison.

        Returns
        -------
        float  — joint log-likelihood (sum across surveys)
        """
        if len(models) != len(self.observations):
            raise ValueError(
                f"Expected {len(self.observations)} model images, got {len(models)}"
            )

        log_L = 0.0
        for obs, model in zip(self.observations, models):
            model_convolved = obs.convolve_model(np.asarray(model, dtype=np.float64))
            cov = self._covariances.get(obs.survey_name)

            if self.use_correlated and cov is not None:
                log_L += _correlated_log_likelihood(
                    obs.image, model_convolved, obs.sigma, obs.mask, cov
                )
            else:
                log_L += _gaussian_log_likelihood(
                    obs.image, model_convolved, obs.sigma, obs.mask
                )

        return log_L

    def chi_squared(self, *models: np.ndarray) -> dict:
        """
        Compute per-survey and total reduced chi-squared.

        Returns
        -------
        dict with keys: per-survey χ²/dof and 'total_chisq_dof'
        """
        result = {}
        total_chisq = 0.0
        total_dof = 0

        for obs, model in zip(self.observations, models):
            model_conv = obs.convolve_model(np.asarray(model, dtype=np.float64))
            valid = ~obs.mask
            r = (obs.image[valid] - model_conv[valid]) / obs.sigma[valid]
            chisq = float(np.dot(r, r))
            dof = int(valid.sum())
            result[obs.survey_name] = {"chisq": chisq, "dof": dof, "chisq_dof": chisq / max(dof, 1)}
            total_chisq += chisq
            total_dof += dof

        result["total_chisq_dof"] = total_chisq / max(total_dof, 1)
        return result

    def covariance_from_weight_map(
        self,
        survey_name: str,
        pixfrac: float = 0.8,
        scale: float = 0.5,
        kernel: str = "square",
    ) -> np.ndarray:
        """
        Estimate the pixel covariance matrix from the survey weight map.

        Delegates to src.data.pixel_covariance.build_drizzle_covariance.
        Automatically stored for subsequent correlated likelihood calls.

        Parameters
        ----------
        survey_name : str
        pixfrac, scale, kernel : drizzle parameters

        Returns
        -------
        covariance : ndarray
        """
        obs = next((o for o in self.observations if o.survey_name == survey_name), None)
        if obs is None:
            raise KeyError(f"Survey '{survey_name}' not found")

        try:
            from src.data.pixel_covariance import build_drizzle_covariance
        except ImportError:
            raise ImportError("src.data.pixel_covariance required for covariance estimation")

        cov = build_drizzle_covariance(obs.sigma, pixfrac=pixfrac, scale=scale, kernel=kernel)
        self.set_covariance(survey_name, cov)
        return cov

    @classmethod
    def make_synthetic(
        cls,
        grid_size_ground: int = 32,
        grid_size_space: int = 64,
        seed: int = 42,
    ) -> "JointSurveyLikelihood":
        """
        Create a synthetic joint-survey setup for testing / CI.

        Returns a JointSurveyLikelihood with two SurveyObservations
        containing injected Einstein-ring signals at different resolutions.
        """
        rng = np.random.default_rng(seed)

        def _ring_image(size, r_e, noise_level=0.02):
            Y, X = np.ogrid[:size, :size]
            c = size // 2
            dist = np.sqrt((X - c) ** 2 + (Y - c) ** 2)
            ring = np.exp(-0.5 * ((dist - r_e) / 1.5) ** 2)
            return ring + rng.normal(0, noise_level, (size, size))

        ground = SurveyObservation(
            image=_ring_image(grid_size_ground, r_e=grid_size_ground * 0.25),
            sigma=0.02,
            pixel_scale=0.2,
            survey_name="rubin",
        )
        space = SurveyObservation(
            image=_ring_image(grid_size_space, r_e=grid_size_space * 0.25),
            sigma=0.01,
            pixel_scale=0.11,
            survey_name="roman",
        )
        return cls(ground, space)
