"""
Bayesian evidence diagnostics for model comparison.
"""

from __future__ import annotations

from typing import Any

try:
    import jax.numpy as jnp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    import numpy as jnp  # type: ignore


ArrayLike = Any


class BayesFactorComparator:
    """
    Utilities for Bayes-factor style model selection summaries.
    """

    def log_bayes_factor(self, log_evidence_1: float, log_evidence_2: float) -> float:
        """
        Compute log Bayes factor ln(K) = ln(Z1) - ln(Z2).
        """
        return float(log_evidence_1 - log_evidence_2)

    def savage_dickey_ratio(
        self,
        prior_density_null: float,
        posterior_density_null: float,
    ) -> float:
        """
        Compute Savage-Dickey density ratio for nested-model comparison.
        """
        return float(prior_density_null / (posterior_density_null + 1e-12))

    def track_eccentricity_correlation(
        self,
        eccentricities: ArrayLike,
        microlensing_signals: ArrayLike,
    ) -> float:
        """
        Compute Pearson-style correlation between eccentricity and signal.
        """
        e_mean = jnp.mean(eccentricities)
        s_mean = jnp.mean(microlensing_signals)

        cov = jnp.mean((eccentricities - e_mean) * (microlensing_signals - s_mean))
        var_e = jnp.std(eccentricities)
        var_s = jnp.std(microlensing_signals)
        correlation = cov / (var_e * var_s + 1e-8)
        return float(correlation)
