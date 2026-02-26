"""
Unit tests for the μ-GLANCE statistical validation tools (Phase 32).
Ensures robustness of non-parametric flux anomaly evaluations against physical boundaries.
"""

import pytest

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="requires jax")
from src.validation.mu_glance import MuGlanceValidator

class TestMuGlanceValidator_JAX:
    """Verifies bounds and projections of the μ-GLANCE anomaly metric."""

    def test_evaluate_flux_anomalies(self):
        validator = MuGlanceValidator()
        
        # Simulate observed vs PINN predicted macroscopic fluxes
        obs_fluxes = jnp.array([10.0, 5.0, 0.0])
        pred_fluxes = jnp.array([12.0, 4.0, 2.0])
        
        anomalies = validator.evaluate_flux_anomalies(obs_fluxes, pred_fluxes)
        
        assert anomalies.shape == (3,)
        
        # Validation of the normalized divergence (10-12)/(22)
        assert jnp.allclose(anomalies[0], -0.090909, atol=1e-4)
        # Validation of (5-4)/(9)
        assert jnp.allclose(anomalies[1], 0.111111, atol=1e-4)
        # Verification of complete suppression bounds
        assert jnp.allclose(anomalies[2], -1.0, atol=1e-4)
        
    def test_aggregate_anomaly_score(self):
        validator = MuGlanceValidator()
        obs_fluxes = jnp.array([10.0, 5.0])
        pred_fluxes = jnp.array([12.0, 4.0])
        
        score = validator.aggregate_anomaly_score(obs_fluxes, pred_fluxes, noise_sigma=0.1)
        
        assert isinstance(score, jnp.ndarray) or isinstance(score, float)
        assert score > 0.0
        assert jnp.isfinite(score)

    def test_map_spatial_residuals(self):
        validator = MuGlanceValidator()
        
        # Map 3 generic quad/cross image configuration anomalies
        img_x = jnp.array([0.5, -0.5, 0.0])
        img_y = jnp.array([0.5, -0.5, -0.5])
        anomalies = jnp.array([0.6, -0.4, 0.1])
        
        res_map = validator.map_spatial_residuals(
            img_x, img_y, anomalies, grid_size=60, extent=2.0, correlation_length=0.4
        )
        
        assert res_map.shape == (60, 60)
        assert jnp.isfinite(res_map).all()
        # Confirm absolute stability bounds derived from pure interpolation logic
        assert jnp.max(jnp.abs(res_map)) <= 1.0
