"""
Physics-Informed Neural Networks (PINNs) for Gravitational Lensing

Implements PINNs that learn to solve the lensing equation while respecting
physical constraints (Poisson equation, symmetries, boundary conditions).

Refactored to JAX/Equinox for hardware acceleration.
"""

try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
except ImportError:
    jax = None
    jnp = None
    eqx = None

from typing import Tuple, Optional, Dict, List, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LensingPINN(eqx.Module):
    """
    Physics-Informed Neural Network for gravitational lensing in JAX.
    """
    layers: list
    skip_layers: list
    activation: Callable
    use_skip_connections: bool
    use_5d_spherical: bool
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [64, 64, 64],
        output_dim: int = 4,
        activation: str = 'tanh',
        use_skip_connections: bool = True,
        use_5d_spherical: bool = True,
        key: jax.random.PRNGKey = None,
        seed: int = 42
    ):
        """Initialize LensingPINN.

        Functional Randomness Control: weight initialization uses an
        explicit ``seed`` (or user-supplied ``key``) to guarantee
        bit-for-bit reproducibility across platforms.
        """
        if key is None:
            key = jax.random.PRNGKey(seed)
        self.use_skip_connections = use_skip_connections
        self.use_5d_spherical = use_5d_spherical
        
        # In 5D spherical mapping, the 2D spatial inputs (x,y) become 5D. 
        # Net dimension grows by 3.
        net_input_dim = input_dim + 3 if use_5d_spherical else input_dim
        
        if activation == 'tanh':
            self.activation = jnp.tanh
        elif activation == 'sin':
            self.activation = jnp.sin
        elif activation == 'relu':
            self.activation = jax.nn.relu
        elif activation == 'gelu':
            self.activation = jax.nn.gelu
        else:
            self.activation = jnp.tanh
            
        keys = jax.random.split(key, len(hidden_dims) + 1 + len(hidden_dims))
        
        self.layers = []
        self.skip_layers = []
        
        # Input layer
        self.layers.append(eqx.nn.Linear(net_input_dim, hidden_dims[0], key=keys[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(eqx.nn.Linear(hidden_dims[i], hidden_dims[i + 1], key=keys[i+1]))
            
        # Output layer
        self.layers.append(eqx.nn.Linear(hidden_dims[-1], output_dim, key=keys[len(hidden_dims)]))
        
        # Skip layers
        if use_skip_connections:
            for i in range(len(hidden_dims)):
                self.skip_layers.append(eqx.nn.Linear(net_input_dim, hidden_dims[i], key=keys[len(hidden_dims) + 1 + i]))
            
    def _map_5d_spherical(self, x: jax.Array) -> jax.Array:
        """
        Maps (x, y) spatial coordinates to a 5D boundary-free periodic manifold
        to eliminate edge divergences in gravitational gradient learning.
        (Martin and Schaub 2025 formulation substitute).
        """
        sx = x[0]
        sy = x[1]
        
        L = 100.0 # Bounding projection scale
        
        s1 = jnp.sin(jnp.pi * sx / L)
        c1 = jnp.cos(jnp.pi * sx / L)
        s2 = jnp.sin(jnp.pi * sy / L)
        c2 = jnp.cos(jnp.pi * sy / L)
        r  = jnp.tanh(jnp.sqrt(sx**2 + sy**2) / L)
        
        # Recombine 5D spatial + physical parameters
        return jnp.concatenate([jnp.array([s1, c1, s2, c2, r]), x[2:]])

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through network (for a single input vector).
        
        Args:
            x: Input tensor of shape (input_dim,)
        """
        if self.use_5d_spherical:
            x = self._map_5d_spherical(x)
            
        x_input = x
        
        # First hidden layer (with skip connection if enabled)
        x = self.layers[0](x)
        if self.use_skip_connections and len(self.skip_layers) > 0:
            x = x + self.skip_layers[0](x_input)
        x = self.activation(x)
        
        for i in range(1, len(self.layers) - 1):
            if self.use_skip_connections and i < len(self.skip_layers):
                skip = self.skip_layers[i](x_input)
                x = self.activation(self.layers[i](x) + skip)
            else:
                x = self.activation(self.layers[i](x))
                
        output = self.layers[-1](x)
        return output

    def predict_convergence(self, x: float, y: float, lens_params: Optional[jax.Array] = None) -> float:
        """Predict convergence κ for a single point."""
        coords = jnp.array([x, y])
        if lens_params is not None:
            inputs = jnp.concatenate([coords, lens_params])
        else:
            inputs = coords
        
        outputs = self(inputs)
        return outputs[0]


class NFW_PINN(LensingPINN):
    """
    PINN specialized for NFW (Navarro-Frenk-White) mass profiles in JAX.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 64, 64],
        activation: str = 'tanh',
        key: jax.random.PRNGKey = None,
        seed: int = 42
    ):
        if key is None:
            key = jax.random.PRNGKey(seed)
        super().__init__(
            input_dim=5,
            hidden_dims=hidden_dims,
            output_dim=4,
            activation=activation,
            use_skip_connections=True,
            use_5d_spherical=False,
            key=key
        )
        
    def __call__(self, x_in: jax.Array) -> jax.Array:
        """
        Args:
            x_in: Input tensor (3,) = [r, log_mass, concentration]
                Internally expanded to 5 features: [r_norm, log(1+r_norm), 1/(1+r_norm), log_mass, conc]
        """
        r = jnp.abs(x_in[0])  # radius is non-negative
        log_mass = x_in[1]
        conc = jnp.clip(x_in[2], 1.0, 100.0)  # physical concentration range
        
        # r_vir [kpc] from M_vir [M_sun] at z=0 Planck 2018 cosmology
        # ρ_crit(z=0) = 3 H0² / (8πG) ≈ 126 M_sun/kpc³  (H0=67.4 km/s/Mpc)
        # Ref: Planck Collaboration (2018), arXiv:1807.06209, Table 2
        M_vir_msun = 10.0 ** log_mass
        rho_crit0 = 126.0   # M_sun/kpc³  (Planck 2018, H0=67.4)
        r_vir = (M_vir_msun / (4.0 * jnp.pi / 3.0 * 200.0 * rho_crit0)) ** (1.0 / 3.0)
        r_s = r_vir / conc
        
        r_norm = r / (r_s + 1e-10)
        
        features = jnp.array([
            r_norm,
            jnp.log1p(r_norm),  # log1p is more stable than log(1+x) near x≈0
            1 / (1 + r_norm),
            log_mass,
            conc
        ])
        
        output = super().__call__(features)
        
        kappa = jax.nn.relu(output[0:1])
        psi = jnp.exp(log_mass / 10.0) * output[1:2]
        alpha_r = output[2:3]
        dalpha_dr = output[3:4]
        
        return jnp.concatenate([kappa, psi, alpha_r, dalpha_dr])


class PhysicsLoss:
    
    def __init__(
        self,
        lambda_physics: float = 1.0,
        lambda_boundary: float = 0.1,
        lambda_symmetry: float = 0.1
    ):
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary
        self.lambda_symmetry = lambda_symmetry
        
    def data_loss(self, pred: jax.Array, target: jax.Array) -> jax.Array:
        return jnp.mean((pred - target) ** 2)
        
    def poisson_residual(
        self,
        model: eqx.Module,
        x: jax.Array,
        y: jax.Array,
        lens_params: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Compute Poisson equation residual ∀ points in batch: |∇²ψ - 2κ|
        using jax.hessian
        """
        def get_psi_and_kappa(x_scalar, y_scalar, params_scalar):
            coords = jnp.array([x_scalar, y_scalar])
            if params_scalar is not None:
                inputs = jnp.concatenate([coords, params_scalar])
            else:
                inputs = coords
            outputs = model(inputs)
            kappa, psi = outputs[0], outputs[1]
            return psi, kappa

        def psi_fn(x_s, y_s, p_s):
            return get_psi_and_kappa(x_s, y_s, p_s)[0]

        def compute_point_residual(x_val, y_val, p_val):
            hessian_fn = jax.hessian(psi_fn, argnums=(0, 1))
            hessian = hessian_fn(x_val, y_val, p_val)
            d2_psi_dx2 = hessian[0][0]
            d2_psi_dy2 = hessian[1][1]
            laplacian = d2_psi_dx2 + d2_psi_dy2
            
            _, kappa = get_psi_and_kappa(x_val, y_val, p_val)
            residual = laplacian - 2 * kappa
            return residual**2
            
        if lens_params is not None:
            batch_residual = jax.vmap(compute_point_residual)(x, y, lens_params)
        else:
            p_val = jnp.zeros_like(x)
            batch_residual = jax.vmap(lambda x_v, y_v: compute_point_residual(x_v, y_v, None))(x, y)
            
        return jnp.mean(batch_residual)
        
    def boundary_loss(self, kappa: jax.Array, r: jax.Array) -> jax.Array:
        r_threshold = jnp.percentile(r, 90.0)
        mask = r > r_threshold
        boundary_kappa = jnp.where(mask, kappa**2, 0.0)
        count = jnp.sum(mask) + 1e-10
        return jnp.sum(boundary_kappa) / count
        
    def total_loss(
        self,
        model: eqx.Module,
        x: jax.Array,
        y: jax.Array,
        true_kappa: jax.Array,
        lens_params: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, Dict[str, float]]:
        
        def forward_point(x_s, y_s, p_s):
            coords = jnp.array([x_s, y_s])
            inputs = jnp.concatenate([coords, p_s]) if p_s is not None else coords
            return model(inputs)
            
        if lens_params is not None:
            outputs = jax.vmap(forward_point)(x, y, lens_params)
        else:
            outputs = jax.vmap(lambda x_v, y_v: forward_point(x_v, y_v, None))(x, y)
            
        pred_kappa = outputs[:, 0]
        
        loss_data = self.data_loss(pred_kappa, true_kappa)
        loss_physics = self.poisson_residual(model, x, y, lens_params)
        
        r = jnp.sqrt(x**2 + y**2)
        loss_boundary = self.boundary_loss(pred_kappa, r)
        
        total = loss_data + self.lambda_physics * loss_physics + self.lambda_boundary * loss_boundary
        
        loss_dict = {
            'total': float(total),
            'data': float(loss_data),
            'physics': float(loss_physics),
            'boundary': float(loss_boundary)
        }
        
        return total, loss_dict


def create_lensing_pinn(model_type: str = 'general', key: jax.random.PRNGKey = None, *, seed: int = 42, **kwargs):
    """Factory for LensingPINN variants.

    Parameters
    ----------
    seed : int
        Reproducibility seed (used when ``key`` is None).
    """
    if key is None:
        key = jax.random.PRNGKey(seed)
    if model_type == 'general':
        return LensingPINN(key=key, **kwargs)
    elif model_type == 'nfw':
        return NFW_PINN(key=key, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == '__main__':
    logger.info("Testing LensingPINN in JAX...")

    key = jax.random.PRNGKey(42)  # Explicit seed for reproducibility
    model = create_lensing_pinn(model_type='nfw', key=key)
    
    batch_size = 100
    r = jax.random.uniform(key, (batch_size, 1)) * 2.0
    log_mass = jnp.ones((batch_size, 1)) * 12.0
    conc = jnp.ones((batch_size, 1)) * 5.0
    
    inputs = jnp.concatenate([r, log_mass, conc], axis=1)
    
    outputs = jax.vmap(model)(inputs)
    
    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Convergence range: [{outputs[:, 0].min():.6f}, {outputs[:, 0].max():.6f}]")
    logger.info("✅ Equinox PINN test successful!")

