"""
Unit tests for the non-parametric source models (Phase 30).
"""

import pytest

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="requires jax")
if not HAS_JAX:
    pytest.skip("requires jax", allow_module_level=True)

from src.ml.source_models import PixelizedSourceModel, matern_kernel, rbf_kernel


class TestSourceModels_JAX:
    """Validate GP Matern/RBF Kernels and ParamU NNLS inversion architecture."""

    def test_gp_kernels(self):
        """Test the Matern and RBF covariance evaluations."""
        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        
        K_matern = matern_kernel(x, y, 1.0)
        K_rbf = rbf_kernel(x, y, 1.0)
        
        assert K_matern.shape == (2, 2)
        assert K_rbf.shape == (2, 2)
        # Identity diagonals
        assert jnp.allclose(jnp.diag(K_matern), 1.0)
        assert jnp.allclose(jnp.diag(K_rbf), 1.0)
        
    def test_pixelized_model_init(self):
        """Confirm resolution maps exactly onto N^2 pixels setup."""
        model = PixelizedSourceModel(resolution=30, extent=1.5)
        assert model.n_pixels == 900
        assert model.coords.shape == (900, 2)
        assert model.pixel_scale == 3.0 / 30.0
        
    def test_linear_inversion_shapes(self):
        """Test the heart of the ParamU linear decoupled model."""
        model = PixelizedSourceModel(resolution=20, extent=1.0) # 400 pixels
        
        img_res = 50
        x = jnp.linspace(-1., 1., img_res)
        y = jnp.linspace(-1., 1., img_res)
        X, Y = jnp.meshgrid(x, y)
        mapped_coords = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        
        obs_image = jnp.ones((img_res, img_res))
        
        # Test NNLS solver + model evidence calculator
        res = model.solve_linear_inversion(
            mapped_coords=mapped_coords,
            observed_image=obs_image,
            noise_var=0.1,
            length_scale=0.5,
            lambda_reg=1e-2,
            kernel='matern'
        )
        
        assert res['source_intensity'].shape == (20, 20)
        assert res['model_image'].shape == (img_res, img_res)
        
        assert jnp.isfinite(res['log_evidence'])
        assert jnp.isfinite(res['chi_sq'])
