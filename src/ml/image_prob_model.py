"""
Probabilistic image likelihood model for lens/source inference.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except ImportError as exc:  # pragma: no cover - hard requirement
    raise ImportError(
        "JAX is required for ImageProbModel. "
        "Install JAX to enable probabilistic image likelihoods."
    ) from exc

from src.ml.pinn import compute_nfw_deflection

ArrayLike = Any


def sersic_source(beta_x: ArrayLike, beta_y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    """
    Evaluate a Sersic source-light profile.
    """
    radial_norm = jnp.sqrt((beta_x - params["center_x"]) ** 2 + (beta_y - params["center_y"]) ** 2)
    effective_radius = params["R_sersic"] + 1e-8
    sersic_index = params["n_sersic"] + 1e-8
    b_n = 1.9992 * sersic_index - 0.3271
    return params["amp"] * jnp.exp(-b_n * ((radial_norm / effective_radius) ** (1.0 / sersic_index) - 1.0))


class ImageProbModel:
    """
    Compute log-likelihood of observed images under lens/source parameters.
    """

    def __init__(self, observed_image: ArrayLike, noise_map: ArrayLike, grid_extent: float = 3.0):
        self.observed_image = observed_image
        self.noise_map = noise_map

        res_y, res_x = observed_image.shape
        x = jnp.linspace(-grid_extent, grid_extent, res_x)
        y = jnp.linspace(-grid_extent, grid_extent, res_y)
        xx, yy = jnp.meshgrid(x, y)
        self.grid_x = xx
        self.grid_y = yy

    def log_likelihood_single(
        self,
        lens_params: Dict[str, float],
        source_params: Dict[str, float],
    ) -> ArrayLike:
        """
        Compute the log-likelihood for one parameter realization.
        """
        alpha_x, alpha_y = compute_nfw_deflection(
            M_vir=jnp.array([lens_params["M_vir"]]),
            r_s=jnp.array([lens_params["r_s"]]),
            theta_x=self.grid_x.ravel()[None, :],
            theta_y=self.grid_y.ravel()[None, :],
        )

        alpha_x = alpha_x.reshape(self.grid_x.shape)
        alpha_y = alpha_y.reshape(self.grid_y.shape)

        beta_x = self.grid_x - alpha_x
        beta_y = self.grid_y - alpha_y

        model_image = sersic_source(beta_x, beta_y, source_params)
        chi2 = jnp.sum(((self.observed_image - model_image) / (self.noise_map + 1e-8)) ** 2)
        return -0.5 * chi2

    def log_likelihood_batch(
        self,
        batch_lens: Dict[str, ArrayLike],
        batch_source: Dict[str, ArrayLike],
    ) -> ArrayLike:
        """
        Vectorized batched log-likelihood evaluation.
        """
        vmapped_ll = jax.vmap(self.log_likelihood_single)
        return vmapped_ll(batch_lens, batch_source)
