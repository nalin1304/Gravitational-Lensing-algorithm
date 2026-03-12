"""
Non-parametric source modeling with Gaussian-process regularization.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except ImportError as exc:  # pragma: no cover - hard requirement
    raise ImportError(
        "JAX is required for PixelizedSourceModel. "
        "Install JAX to enable GP-regularized inversion."
    ) from exc


ArrayLike = Any


def rbf_kernel(x: ArrayLike, y: ArrayLike, length_scale: float) -> ArrayLike:
    """Squared Exponential (RBF) kernel."""
    diff = x[:, None, :] - y[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    return jnp.exp(-0.5 * dist_sq / (length_scale**2 + 1e-8))


def matern_kernel(
    x: ArrayLike,
    y: ArrayLike,
    length_scale: float,
    nu: float = 1.5,
) -> ArrayLike:
    """Matern kernel (common smoothness: nu=1.5 or 2.5)."""
    diff = x[:, None, :] - y[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)

    if nu == 1.5:
        factor = jnp.sqrt(3.0) * dist / length_scale
        return (1.0 + factor) * jnp.exp(-factor)
    if nu == 2.5:
        factor = jnp.sqrt(5.0) * dist / length_scale
        return (1.0 + factor + (factor**2) / 3.0) * jnp.exp(-factor)
    return jnp.exp(-0.5 * (dist / length_scale) ** 2)


class PixelizedSourceModel:
    """
    Pixelized source plane with GP-style regularized linear inversion.
    """

    def __init__(self, resolution: int = 50, extent: float = 2.0):
        x = jnp.linspace(-extent, extent, resolution)
        y = jnp.linspace(-extent, extent, resolution)
        self.grid_x, self.grid_y = jnp.meshgrid(x, y)
        self.n_pixels = resolution * resolution
        self.coords = jnp.stack([self.grid_x.ravel(), self.grid_y.ravel()], axis=-1)
        self.pixel_scale = 2.0 * extent / resolution

    def covariance_matrix(self, length_scale: float, kernel: str = "matern") -> ArrayLike:
        if kernel == "matern":
            return matern_kernel(self.coords, self.coords, length_scale)
        return rbf_kernel(self.coords, self.coords, length_scale)

    def build_lensing_operator(self, mapped_coords: ArrayLike) -> ArrayLike:
        """
        Build observation operator mapping source pixels to image-plane samples.
        """
        diff = mapped_coords[:, None, :] - self.coords[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        operator = jnp.exp(-0.5 * dist_sq / (self.pixel_scale**2))
        operator = operator / (jnp.sum(operator, axis=1, keepdims=True) + 1e-8)
        return operator

    def solve_linear_inversion(
        self,
        mapped_coords: ArrayLike,
        observed_image: ArrayLike,
        noise_var: float,
        length_scale: float,
        lambda_reg: float,
        kernel: str = "matern",
    ) -> Dict[str, ArrayLike]:
        """
        Solve linear inversion for source intensities under GP regularization.
        """
        operator = self.build_lensing_operator(mapped_coords)
        source_cov = self.covariance_matrix(length_scale, kernel=kernel)
        source_cov_inv = jnp.linalg.inv(source_cov + jnp.eye(self.n_pixels) * 1e-6)

        design_matrix = (operator.T @ operator) / noise_var + lambda_reg * source_cov_inv
        rhs = (operator.T @ observed_image.ravel()) / noise_var

        source_intensity = jax.scipy.linalg.solve(design_matrix, rhs, assume_a="pos")

        source_intensity = jnp.clip(source_intensity, 0.0, None)
        model_image = (operator @ source_intensity).reshape(observed_image.shape)

        chi_sq = jnp.sum((observed_image.ravel() - model_image.ravel()) ** 2) / noise_var
        reg_penalty = lambda_reg * (source_intensity.T @ source_cov_inv @ source_intensity)

        sign, logdet = jnp.linalg.slogdet(design_matrix)
        sign_prior, logdet_prior = jnp.linalg.slogdet(lambda_reg * source_cov_inv)
        del sign, sign_prior

        log_evidence = -0.5 * chi_sq - 0.5 * reg_penalty - 0.5 * logdet + 0.5 * logdet_prior

        return {
            "source_intensity": source_intensity.reshape(self.grid_x.shape),
            "model_image": model_image,
            "log_evidence": log_evidence,
            "chi_sq": chi_sq,
        }
