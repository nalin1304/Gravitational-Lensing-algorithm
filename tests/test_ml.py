"""
Unit tests for ML module (JAX Migration)
"""

import pytest
try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax  # noqa: F401
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    eqx = None  # type: ignore[assignment]

import numpy as np
from pathlib import Path
import tempfile
import h5py

if HAS_ML_DEPS:
    from src.ml.pinn import PhysicsInformedNN, physics_informed_loss
    from src.ml.pinn_models import LensingPINN, NFW_PINN, PhysicsLoss
else:  # pragma: no cover - skipped when deps unavailable
    PhysicsInformedNN = None  # type: ignore[assignment]
    physics_informed_loss = None  # type: ignore[assignment]
    LensingPINN = None  # type: ignore[assignment]
    NFW_PINN = None  # type: ignore[assignment]
    PhysicsLoss = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not HAS_ML_DEPS,
    reason="requires jax, equinox, and optax",
)


class TestPhysicsInformedNN_JAX:
    """Tests for JAX PhysicsInformedNN architecture"""

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        key = jax.random.PRNGKey(0)
        model = PhysicsInformedNN(key=key)
        assert model is not None
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'param_fc1')

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes"""
        key = jax.random.PRNGKey(0)
        model = PhysicsInformedNN(key=key)
        batch_size = 8

        x = jax.random.normal(key, (batch_size, 1, 64, 64))

        # Batched forward
        batch_forward = jax.vmap(model)
        params, class_logits = batch_forward(x)

        assert params.shape == (batch_size, 5)
        assert class_logits.shape == (batch_size, 3)

    def test_predict_method(self):
        """Test predict method returns interpretable outputs"""
        key = jax.random.PRNGKey(0)
        model = PhysicsInformedNN(key=key)

        x = jax.random.normal(key, (4, 1, 64, 64))
        results = model.predict(x)

        assert 'params' in results
        assert 'M_vir' in results
        assert 'class_probs' in results

        assert results['params'].shape == (4, 5)
        assert results['class_probs'].shape == (4, 3)
        assert results['class_labels'].shape == (4,)
        
        # Check sum to 1
        prob_sums = jnp.sum(results['class_probs'], axis=1)
        assert jnp.allclose(prob_sums, jnp.ones(4), atol=1e-5)


class TestPhysicsInformedLoss_JAX:
    """Tests for physics-informed loss function in JAX"""

    def test_loss_computation(self):
        """Test that loss computes without errors"""
        batch_size = 4
        key = jax.random.PRNGKey(42)
        model = PhysicsInformedNN(key=key)

        images = jax.random.normal(key, (batch_size, 1, 64, 64))
        true_params = jax.random.normal(key, (batch_size, 5))
        true_classes = jax.random.randint(key, (batch_size,), 0, 3)

        total_loss, losses = physics_informed_loss(
            model=model,
            images=images,
            true_params=true_params,
            true_classes=true_classes,
            key=key,
            lambda_physics=0.1
        )

        assert 'total' in losses
        assert 'mse_params' in losses
        assert 'ce_class' in losses
        assert 'physics_residual' in losses

        assert float(losses['total']) >= 0
        assert float(losses['mse_params']) >= 0
        assert float(losses['ce_class']) >= 0
        assert float(losses['physics_residual']) >= 0


class TestLensingPINN_JAX:
    """Tests for LensingPINN equinox architecture"""

    def test_lensing_pinn_forward(self):
        key = jax.random.PRNGKey(0)
        model = LensingPINN(input_dim=5, output_dim=4, key=key)
        batch_size = 10
        
        # [x, y, M, r_s, c]
        x = jax.random.normal(key, (batch_size, 5))
        
        vmodel = jax.vmap(model)
        out = vmodel(x)
        
        assert out.shape == (batch_size, 4)

    def test_nfw_pinn_forward(self):
        key = jax.random.PRNGKey(0)
        model = NFW_PINN(key=key)
        batch_size = 10
        
        # [r, log_mass, c]
        x = jax.random.normal(key, (batch_size, 3))
        
        vmodel = jax.vmap(model)
        out = vmodel(x)
        
        assert out.shape == (batch_size, 4)
        # Convergence constraint check
        assert jnp.all(out[:, 0] >= 0.0) 

