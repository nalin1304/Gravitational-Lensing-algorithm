"""
Test PINN model with adaptive pooling for variable input sizes.
Tests Step 6 of todo list.
"""
import pytest
try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax  # noqa: F401
    HAS_PINN_DEPS = True
except ImportError:
    HAS_PINN_DEPS = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    eqx = None  # type: ignore[assignment]
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if HAS_PINN_DEPS:
    from src.ml.pinn import PhysicsInformedNN
else:  # pragma: no cover - skipped when deps unavailable
    PhysicsInformedNN = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not HAS_PINN_DEPS,
    reason="requires jax, equinox, and optax",
)

@pytest.fixture
def pinn_model():
    """Create PINN model instance."""
    key = jax.random.PRNGKey(42)
    model = PhysicsInformedNN(key=key)
    return model


def test_pinn_accepts_64x64(pinn_model):
    """Test PINN with standard 64x64 input."""
    batch_size = 4
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, 1, 64, 64))
    
    vmodel = jax.vmap(pinn_model)
    params, logits = vmodel(x)
    
    assert params.shape == (batch_size, 5), f"Expected (4, 5), got {params.shape}"
    assert logits.shape == (batch_size, 3), f"Expected (4, 3), got {logits.shape}"
    print(f"✅ 64x64 input: params={params.shape}, logits={logits.shape}")


def test_pinn_accepts_128x128(pinn_model):
    """Test PINN with 128x128 input (larger than training size)."""
    batch_size = 2
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, 1, 128, 128))
    
    vmodel = jax.vmap(pinn_model)
    params, logits = vmodel(x)
    
    assert params.shape == (batch_size, 5), f"Expected (2, 5), got {params.shape}"
    assert logits.shape == (batch_size, 3), f"Expected (2, 3), got {logits.shape}"
    print(f"✅ 128x128 input: params={params.shape}, logits={logits.shape}")


def test_pinn_accepts_256x256(pinn_model):
    """Test PINN with 256x256 input (much larger than training size)."""
    batch_size = 1
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, 1, 256, 256))
    
    vmodel = jax.vmap(pinn_model)
    params, logits = vmodel(x)
    
    assert params.shape == (batch_size, 5), f"Expected (1, 5), got {params.shape}"
    assert logits.shape == (batch_size, 3), f"Expected (1, 3), got {logits.shape}"
    print(f"✅ 256x256 input: params={params.shape}, logits={logits.shape}")


def test_pinn_batch_consistency(pinn_model):
    """Test that different batch sizes produce consistent output shapes."""
    input_sizes = [64, 128, 256]
    batch_sizes = [1, 2, 4]
    
    results = []
    key = jax.random.PRNGKey(0)
    vmodel = jax.vmap(pinn_model)
    
    for size in input_sizes:
        for batch in batch_sizes:
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (batch, 1, size, size))
            params, logits = vmodel(x)
            
            assert params.shape == (batch, 5), f"Inconsistent params shape at size={size}, batch={batch}"
            assert logits.shape == (batch, 3), f"Inconsistent logits shape at size={size}, batch={batch}"
            results.append((size, batch, params.shape, logits.shape))
    
    print(f"\n✅ All {len(results)} size/batch combinations passed:")
    for size, batch, p_shape, l_shape in results:
        print(f"   {size}x{size}, batch={batch}: params={p_shape}, logits={l_shape}")


def test_pinn_output_range(pinn_model):
    """Test that model outputs are in reasonable ranges."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 1, 128, 128))
    
    vmodel = jax.vmap(pinn_model)
    params, logits = vmodel(x)
    
    # Parameters should be finite
    assert jnp.all(jnp.isfinite(params)), "Parameters contain NaN or Inf"
    assert jnp.all(jnp.isfinite(logits)), "Logits contain NaN or Inf"
    
    # Logits should have some variation (not all zeros)
    assert jnp.std(logits) > 0.001, "Logits have no variation"
