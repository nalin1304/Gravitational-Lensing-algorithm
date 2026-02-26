import pytest
import numpy as np
try:
    import jax
    import jax.numpy as jnp
    from src.ml.sbi_npe import NeuralPosteriorEstimator, JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX/distrax not available")
def test_npe_initialization_and_forward():
    """Test that the NPE flow can be initialized and a forward pass computed."""
    param_dim = 4
    obs_dim = 10
    
    npe = NeuralPosteriorEstimator(
        param_dim=param_dim,
        obs_dim=obs_dim,
        hidden_dims=(32, 32),
        num_flow_layers=2
    )
    
    # Dummy data
    theta = jnp.zeros((5, param_dim))
    obs = jnp.zeros((5, obs_dim))
    
    # Calculate log prob
    rng = jax.random.PRNGKey(0)
    log_prob = npe._flow_model.apply(npe.params, rng, theta, obs)
    
    assert log_prob.shape == (5,)
    assert not jnp.any(jnp.isnan(log_prob))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX/distrax not available")
def test_npe_training_step():
    """Test that the NPE can take a step of optimization without crashing."""
    param_dim = 2
    obs_dim = 4
    
    npe = NeuralPosteriorEstimator(
        param_dim=param_dim,
        obs_dim=obs_dim,
        hidden_dims=(16, 16),
        num_flow_layers=2,
        learning_rate=1e-2
    )
    
    def dummy_generator():
        while True:
            # Random parameters
            theta = np.random.normal(size=(16, param_dim))
            # Observation is just a linear projection to test learning
            obs = theta @ np.ones((param_dim, obs_dim))
            yield theta, obs
            
    # Train for 5 steps
    npe.train(dummy_generator(), steps=5)
    
    # State should be updated (no NaNs)
    assert not jnp.any(jnp.isnan(npe.params['linear']['w']))

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX/distrax not available")
def test_npe_sampling():
    """Test that the NPE flow can generate posterior samples given an observation."""
    param_dim = 2
    obs_dim = 4
    
    npe = NeuralPosteriorEstimator(
        param_dim=param_dim,
        obs_dim=obs_dim,
        hidden_dims=(16, 16),
        num_flow_layers=2
    )
    
    # Target observation
    obs = jnp.ones((obs_dim,))
    
    # Generate 500 samples
    samples = npe.sample(obs, num_samples=500)
    
    assert samples.shape == (500, param_dim)
    assert not jnp.any(jnp.isnan(samples))
