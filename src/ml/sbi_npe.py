import jax
import jax.numpy as jnp
import optax
import distrax
import haiku as hk
import numpy as np
from typing import Tuple, Dict, Any, Optional, Iterator

JAX_AVAILABLE = True

class NeuralPosteriorEstimator:
    """
    Simulation-Based Inference (SBI) / Neural Posterior Estimation (NPE) Engine.
    
    Uses Normalizing Flows (Masked Autoregressive Flows via distrax and haiku)
    to perform likelihood-free rapid posterior estimation, bypassing Nested Sampling.
    """
    
    def __init__(
        self, 
        param_dim: int, 
        obs_dim: int, 
        hidden_dims: Tuple[int, ...] = (128, 128),
        num_flow_layers: int = 4,
        learning_rate: float = 1e-3,
        random_seed: int = 42
    ):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX, distrax, optax, and dm-haiku are required for SBI/NPE. "
            )
            
        self.param_dim = param_dim
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.num_flow_layers = num_flow_layers
        self.learning_rate = learning_rate
        
        self.rng = jax.random.PRNGKey(random_seed)
        
        # Haiku transform
        self._flow_model = hk.transform(self._flow_fn)
        
        # We need a dummy input to initialize the network parameters
        dummy_params = jnp.zeros((1, param_dim))
        dummy_context = jnp.zeros((1, obs_dim))
        
        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self._flow_model.init(init_rng, dummy_params, dummy_context)
        
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
    def _flow_fn(self, params: jnp.ndarray, context: jnp.ndarray, inverse: bool = False) -> jnp.ndarray:
        """
        Forward function for Haiku to compute the log-probability of params given context.
        Uses native RealNVP affine coupling layers.
        """
        # Embed context
        embedded_context = hk.Sequential([
            hk.Linear(self.hidden_dims[0]),
            jax.nn.relu,
            hk.Linear(self.hidden_dims[1]),
            jax.nn.relu
        ])(context)
        
        split_index = self.param_dim // 2
        
        z = params
        log_det_jacobian = jnp.zeros(params.shape[0] if params.ndim > 1 else 1)
        
        # We need a stable flow. Since we want log_prob(params | context),
        # we compute the forward mapping from data x (params) to base distribution z.
        # This is the Normalizing Flow standard convention.
        
        for i in range(self.num_flow_layers):
            # Alternate which half gets masked
            if i % 2 == 0:
                z1, z2 = z[..., :split_index], z[..., split_index:]
            else:
                z1, z2 = z[..., split_index:], z[..., :split_index]
                
            mlp_input = jnp.concatenate([z1, embedded_context], axis=-1)
            h = hk.Sequential([
                hk.Linear(self.hidden_dims[0]), jax.nn.relu,
                hk.Linear(self.hidden_dims[1]), jax.nn.relu,
            ])(mlp_input)
            
            out = hk.Linear(z2.shape[-1] * 2,
                            w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                            b_init=jnp.zeros)(h)
            shift, log_scale = jnp.split(out, 2, axis=-1)
            
            # Forward transform: x -> z (Data to Latent)
            if not inverse:
                z2 = (z2 - shift) * jnp.exp(-log_scale)
                log_det_jacobian -= jnp.sum(log_scale, axis=-1)
            else:
                # Inverse transform: z -> x (Latent to Data, for sampling)
                z2 = z2 * jnp.exp(log_scale) + shift
                log_det_jacobian += jnp.sum(log_scale, axis=-1)
                
            if i % 2 == 0:
                z = jnp.concatenate([z1, z2], axis=-1)
            else:
                z = jnp.concatenate([z2, z1], axis=-1)
                
        if not inverse:
            # log P_X(x) = log P_Z(f(x)) + log |det J|
            log_prob_base = -0.5 * jnp.sum(z**2 + jnp.log(2 * np.pi), axis=-1)
            return log_prob_base + log_det_jacobian
        else:
            return z
        
    def _loss_fn(self, params: hk.Params, rng: jnp.ndarray, x: jnp.ndarray, context: jnp.ndarray) -> Tuple[jnp.ndarray, Any]:
        """Negative Log-Likelihood Loss. x is the parameter sample, context is the observation."""
        log_prob = self._flow_model.apply(params, rng, x, context, inverse=False)
        return -jnp.mean(log_prob), ()
        
    def train(self, data_generator: Iterator[Tuple[jnp.ndarray, jnp.ndarray]], steps: int = 1000):
        """
        Train the NPE on simulated pairs of (parameters, observations).
        """
        @jax.jit
        def update_step(params, opt_state, rng, x_batch, c_batch):
            loss, grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, rng, x_batch, c_batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        import tqdm
        pbar = tqdm.tqdm(range(steps), desc="Training NPE Flow")
        for step in pbar:
            theta_batch, obs_batch = next(data_generator)
            
            self.rng, rng_step = jax.random.split(self.rng)
            self.params, self.opt_state, loss = update_step(
                self.params, self.opt_state, rng_step, theta_batch, obs_batch
            )
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': float(loss[0])})


    def sample(self, observation: jnp.ndarray, num_samples: int = 1000) -> jnp.ndarray:
        """
        Sample from the posterior given a target observation.
        """
        def sample_fn(context_single):
            # Tile context to match batch size
            ctx_batch = jnp.tile(context_single, (num_samples, 1))
            
            # Sample from base distribution (Standard Normal)
            z = jax.random.normal(hk.next_rng_key(), (num_samples, self.param_dim))
            
            # Request inverse transform (Latent -> Data)
            return self._flow_fn(z, ctx_batch, inverse=True)
            
        sampler = hk.transform(sample_fn)
        
        self.rng, rng_samp = jax.random.split(self.rng)
        
        if observation.ndim > 1:
            observation = observation.flatten()
            
        samples = sampler.apply(self.params, rng_samp, observation)
        return samples
