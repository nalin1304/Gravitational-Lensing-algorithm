"""
Unit tests for the Neural Ordinary Differential Equations ().
Verifies the integration of JAX/Diffrax with Equinox structural parameters.
"""

import pytest
try:
    import jax
    import jax.numpy as jnp
    import equinox  # noqa: F401
    import diffrax  # noqa: F401
    HAS_NODE_DEPS = True
except ImportError:
    HAS_NODE_DEPS = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]

from src.ml.neural_ode import DynamicsNODE, AnalyticFusingNODE

pytestmark = pytest.mark.skipif(
    not HAS_NODE_DEPS,
    reason="requires jax, equinox, and diffrax",
)

class TestNeuralODE_JAX:
    """Validate Diffrax integration mechanics for N-body simulation components."""

    def test_dynamics_node_forward(self):
        """Test pure neural network ODE integration."""
        model = DynamicsNODE(state_dim=12, hidden_dim=32, layers=2, seed=0)
        
        y0 = jnp.zeros(12)
        ts = jnp.linspace(0.0, 1.0, 10)
        
        ys = model(y0, ts)
        
        assert ys.shape == (10, 12)
        assert jnp.isfinite(ys).all()

    def test_analytic_fusing_node_forward(self):
        """Test the Analytic Fusing paradigm integrating Newtonian gravity + Neural perturbation."""
        model = AnalyticFusingNODE(state_dim=12, hidden_dim=32, layers=2, seed=42)
        
        # Initial positions and velocities simulating Milky Way and LMC
        r1 = jnp.array([0.0, 0.0, 0.0])
        v1 = jnp.array([0.0, 0.0, 0.0])
        r2 = jnp.array([50.0, 0.0, 0.0])  # 50 kpc separation
        v2 = jnp.array([0.0, 200.0, 0.0]) # 200 km/s transverse velocity
        
        y0 = jnp.concatenate([r1, v1, r2, v2])
        ts = jnp.linspace(0.0, 5.0, 20)
        
        ys = model(y0, ts)
        
        assert ys.shape == (20, 12)
        assert jnp.isfinite(ys).all()
        
        # Ensuring positions evolved due to velocities
        assert not jnp.allclose(ys[0], ys[-1])
