"""
Unit tests for Bayesian Factor evidence validation routines ().
"""

import pytest

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="requires jax")
if not HAS_JAX:
    pytest.skip("requires jax", allow_module_level=True)

from src.validation.bayes_factor import BayesFactorComparator

class TestBayesFactorComparator_JAX:

    def test_log_bayes_factor(self):
        comparator = BayesFactorComparator()
        # Evaluate simple evidence ratio
        assert jnp.allclose(comparator.log_bayes_factor(10.0, 5.0), 5.0)
        assert jnp.allclose(comparator.log_bayes_factor(2.5, 4.0), -1.5)

    def test_savage_dickey_ratio(self):
        comparator = BayesFactorComparator()
        # p(theta=0 | M1) = 0.5, p(theta=0 | M1, D) = 0.1
        ratio = comparator.savage_dickey_ratio(0.5, 0.1)
        assert jnp.allclose(ratio, 5.0, atol=1e-4)

    def test_track_eccentricity_correlation(self):
        comparator = BayesFactorComparator()
        # Perfect positive linear correlation
        e_perfect = jnp.array([0.1, 0.2, 0.3, 0.4])
        s_perfect = jnp.array([1.0, 2.0, 3.0, 4.0])
        corr_perfect = comparator.track_eccentricity_correlation(e_perfect, s_perfect)
        assert jnp.allclose(corr_perfect, 1.0, atol=1e-4)
        
        # Perfect negative correlation
        s_negative = jnp.array([4.0, 3.0, 2.0, 1.0])
        corr_negative = comparator.track_eccentricity_correlation(e_perfect, s_negative)
        assert jnp.allclose(corr_negative, -1.0, atol=1e-4)
        
        # Zero correlation symmetric
        e_zero = jnp.array([-1.0, 1.0, -1.0, 1.0])
        s_zero = jnp.array([1.0, 1.0, -1.0, -1.0])
        corr_zero = comparator.track_eccentricity_correlation(e_zero, s_zero)
        assert jnp.allclose(corr_zero, 0.0, atol=1e-4)

