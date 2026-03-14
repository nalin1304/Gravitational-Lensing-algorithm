"""
mu-GLANCE: Magnification residual diagnostics for lensing validation.

This module computes model-independent flux anomaly statistics and spatially
smoothed residual maps to detect unresolved structure.
"""

from __future__ import annotations

from typing import Any

try:
    import jax.numpy as jnp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    import numpy as jnp  # type: ignore


ArrayLike = Any


class MuGlanceValidator:
    """
    Statistical validator for magnification anomaly structure.
    """

    def evaluate_flux_anomalies(
        self,
        obs_fluxes: ArrayLike,
        pred_fluxes: ArrayLike,
    ) -> ArrayLike:
        """
        Compute the bounded fractional anomaly vector.

        Formula:
            (F_obs - F_pred) / (F_obs + F_pred + eps)
        """
        return (obs_fluxes - pred_fluxes) / (obs_fluxes + pred_fluxes + 1e-8)

    def aggregate_anomaly_score(
        self,
        obs_fluxes: ArrayLike,
        pred_fluxes: ArrayLike,
        noise_sigma: float = 0.05,
    ) -> float:
        """
        Compute a chi-square-like aggregate anomaly statistic.
        """
        anomalies = self.evaluate_flux_anomalies(obs_fluxes, pred_fluxes)
        return float(jnp.sum((anomalies / noise_sigma) ** 2))

    def map_spatial_residuals(
        self,
        image_coords_x: ArrayLike,
        image_coords_y: ArrayLike,
        anomalies: ArrayLike,
        grid_size: int = 100,
        extent: float = 3.0,
        correlation_length: float = 0.5,
    ) -> ArrayLike:
        """
        Interpolate sparse image anomalies to a dense residual field using RBFs.
        """
        x_lin = jnp.linspace(-extent, extent, grid_size)
        y_lin = jnp.linspace(-extent, extent, grid_size)
        X, Y = jnp.meshgrid(x_lin, y_lin)

        grid_coords_x = X.ravel()
        grid_coords_y = Y.ravel()

        diff_x = grid_coords_x[:, None] - image_coords_x[None, :]
        diff_y = grid_coords_y[:, None] - image_coords_y[None, :]
        dist_sq = diff_x**2 + diff_y**2

        scale_sq = correlation_length**2
        weights = jnp.exp(-0.5 * dist_sq / scale_sq)
        weight_sum = jnp.sum(weights, axis=1) + 1e-5
        residual_field = jnp.sum(weights * anomalies[None, :], axis=1) / weight_sum

        return residual_field.reshape((grid_size, grid_size))

