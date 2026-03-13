"""
Physics-Informed Neural Network for Gravitational Lens Parameter Inference

Refactored to JAX/Equinox for hardware acceleration.
"""

try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    
    # Fallback classes for type hints and graceful degradation
    class eqx: # type: ignore
        class Module: pass
        def filter_jit(f): return f
        
    class jax: # type: ignore
        class Array: pass
        class random:
            PRNGKey = type('PRNGKey', (), {})
            
    class jnp: # type: ignore
        pass
        
    class optax: # type: ignore
        GradientTransformation = type('GradientTransformation', (), {})
        OptState = type('OptState', (), {})

from typing import Dict, Tuple, Optional, Any
from functools import partial

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import logging

from src.utils.constants import (
    C_LIGHT,
    G_CONST,
    H0_PLANCK,
    KPC,
    M_SUN_KG,
    OMEGA_M_PLANCK,
    RAD_TO_ARCSEC,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: Astropy is kept for the initialization of constants; 
# in the hot loop, we use pure JAX arrays.

# Convert SI constants to kpc-based units used in JAX kernels.
_G_KPC3_PER_MSUN_S2 = G_CONST * M_SUN_KG / (KPC ** 3)
_C_KPC_PER_S = C_LIGHT / KPC

def _angular_diameter_distances_kpc(z_l: float, z_s: float, H0: float, Omega_m: float) -> Tuple[float, float, float]:
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)
    D_l = cosmo.angular_diameter_distance(z_l).to(u.kpc).value
    D_s = cosmo.angular_diameter_distance(z_s).to(u.kpc).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.kpc).value
    return float(D_l), float(D_s), float(D_ls)


if JAX_AVAILABLE:
    class PhysicsInformedNN(eqx.Module):
        """
        Physics-Informed Neural Network for lens parameter inference in JAX/Equinox.
        
        Architecture:
        - Input: 64x64 image (1, 64, 64)
        - Encoder: Conv2d blocks
        - Dense heads: Params (5), Classes (3)
        """
        conv1: eqx.nn.Conv2d
        conv2: eqx.nn.Conv2d
        conv3: eqx.nn.Conv2d
        pool: eqx.nn.MaxPool2d
        
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear
        fc3: eqx.nn.Linear
        
        param_fc1: eqx.nn.Linear
        param_fc2: eqx.nn.Linear
        
        class_fc1: eqx.nn.Linear
        class_fc2: eqx.nn.Linear
        
        def __init__(self, key: jax.random.PRNGKey):
            keys = jax.random.split(key, 10)
            
            # Conv blocks (in_channels, out_channels, kernel_size, padding)
            self.conv1 = eqx.nn.Conv2d(1, 32, 3, padding=1, key=keys[0])
            self.conv2 = eqx.nn.Conv2d(32, 64, 3, padding=1, key=keys[1])
            self.conv3 = eqx.nn.Conv2d(64, 128, 3, padding=1, key=keys[2])
            self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 64x64 -> pool -> 32x32 -> pool -> 16x16 -> pool -> 8x8
            # Flattened size: 128 * 8 * 8 = 8192
            encoded_size = 128 * 8 * 8
            
            self.fc1 = eqx.nn.Linear(encoded_size, 1024, key=keys[3])
            self.fc2 = eqx.nn.Linear(1024, 512, key=keys[4])
            self.fc3 = eqx.nn.Linear(512, 256, key=keys[5])
            
            self.param_fc1 = eqx.nn.Linear(256, 128, key=keys[6])
            self.param_fc2 = eqx.nn.Linear(128, 5, key=keys[7])
            
            self.class_fc1 = eqx.nn.Linear(256, 128, key=keys[8])
            self.class_fc2 = eqx.nn.Linear(128, 3, key=keys[9])

        def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
            """
            Forward pass for a single image.
            x shape: (1, H, W) -> resized to (1, 64, 64)
            """
            # Adaptive pooling equivalent: Resize all incoming shapes to 64x64
            x = jax.image.resize(x, (1, 64, 64), method='bilinear')
            
            # Encoder
            x = jax.nn.relu(self.conv1(x))
            x = self.pool(x)
            
            x = jax.nn.relu(self.conv2(x))
            x = self.pool(x)
            
            x = jax.nn.relu(self.conv3(x))
            x = self.pool(x)
            
            # Flatten
            x = x.reshape(-1)
            
            # Dense features
            x = jax.nn.relu(self.fc1(x))
            x = jax.nn.relu(self.fc2(x))
            x = jax.nn.relu(self.fc3(x))
            
            # Dual heads
            p = jax.nn.relu(self.param_fc1(x))
            params = self.param_fc2(p)
            
            c = jax.nn.relu(self.class_fc1(x))
            classes = self.class_fc2(c)
            
            return params, classes

        def predict(self, x: jax.Array) -> Dict[str, jax.Array]:
            """Batched prediction"""
            batch_forward = jax.vmap(self.__call__)
            params, class_logits = batch_forward(x)
            
            class_probs = jax.nn.softmax(class_logits, axis=-1)
            class_labels = jnp.argmax(class_probs, axis=-1)
            
            return {
                'params': params,
                'M_vir': params[:, 0],
                'r_s': params[:, 1],
                'beta_x': params[:, 2],
                'beta_y': params[:, 3],
                'H0': params[:, 4],
                'class_probs': class_probs,
                'class_labels': class_labels
            }
else:
    class PhysicsInformedNN(eqx.Module): # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("JAX and Equinox are required to use PhysicsInformedNN.")
        def __call__(self, x: "jax.Array") -> Tuple["jax.Array", "jax.Array"]: # type: ignore
            raise NotImplementedError
        def predict(self, x: "jax.Array") -> Dict[str, "jax.Array"]: # type: ignore
            raise NotImplementedError



def compute_nfw_deflection(
    M_vir: "jax.Array",
    r_s: "jax.Array",
    theta_x: "jax.Array",
    theta_y: "jax.Array",
    z_l: float = 0.5,
    z_s: float = 2.0,
    H0: float = H0_PLANCK,
    Omega_m: float = OMEGA_M_PLANCK
) -> Tuple["jax.Array", "jax.Array"]:
    """
    Compute NFW deflection angle using differentiable JAX operations.

    Notes
    -----
    Uses the Bartelmann (1996) / Wright & Brainerd (2000) reduced-deflection
    kernel α(x) ∝ h(x)/x with cosmological distance scaling through Σ_crit.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for compute_nfw_deflection.")

    G = _G_KPC3_PER_MSUN_S2  # kpc^3 / (M_sun s^2)
    c_kpc = _C_KPC_PER_S  # kpc / s
    
    # M_vir is in raw solar masses (as stored/predicted by the training pipeline).
    # No unit rescaling — training data in generate_dataset.py samples M_vir in
    # [1e11, 5e12] M_sun directly.
    M_vir_solar = M_vir
    
    D_l_val, D_s_val, D_ls_val = _angular_diameter_distances_kpc(z_l, z_s, H0, Omega_m)
    D_l = jnp.array(D_l_val)
    D_s = jnp.array(D_s_val)
    D_ls = jnp.array(D_ls_val)
    
    arcsec_to_rad = jnp.pi / 180.0 / 3600.0
    r_x_kpc = theta_x * D_l * arcsec_to_rad
    r_y_kpc = theta_y * D_l * arcsec_to_rad
    
    r_kpc = jnp.sqrt(r_x_kpc**2 + r_y_kpc**2 + 1e-8)
    x = r_kpc / (r_s + 1e-8)
    
    # h(x) piecewise for deflection
    def h_less(x_val):
        x_val = jnp.clip(x_val, 1e-6, 0.999999)
        arctanh_term = jnp.arctanh(jnp.sqrt((1.0 - x_val) / (1.0 + x_val)))
        return jnp.log(x_val / 2.0) + (2.0 / jnp.sqrt(1.0 - x_val**2)) * arctanh_term
        
    def h_greater(x_val):
        x_val = jnp.clip(x_val, 1.000001, None)
        arctan_term = jnp.arctan(jnp.sqrt((x_val - 1.0) / (x_val + 1.0)))
        return jnp.log(x_val / 2.0) + (2.0 / jnp.sqrt(x_val**2 - 1.0)) * arctan_term
        
    x_safe_less = jnp.clip(x, 1e-6, 0.99)
    x_safe_greater = jnp.clip(x, 1.01, None)
    
    h_x = jnp.where(
        x < 0.99,
        h_less(x_safe_less),
        jnp.where(
            x > 1.01,
            h_greater(x_safe_greater),
            1.0 + jnp.log(0.5)
        )
    )
    
    Sigma_crit = (c_kpc**2 / (4.0 * jnp.pi * G)) * (D_s / (D_l * D_ls + 1e-8))
    
    # Compute the NFW concentration from M_vir and r_s.
    # The virial radius r_vir is defined by M(r_vir) = (4π/3)×200×ρ_crit(z_l)×r_vir³.
    # From M_vir = 4π ρ_s r_s³ f(c) and r_vir = c × r_s we derive:
    #   ρ_crit(z_l) [M_sun/kpc³] using the Friedmann equation at z_l.
    # H(z_l) = H0 × sqrt(Omega_m(1+z_l)³ + (1-Omega_m))  [km/s/Mpc]
    H_z_kpc_s = (H0 * jnp.sqrt(Omega_m * (1.0 + z_l)**3 + (1.0 - Omega_m))
                 / 3085.677581e+16)   # km/s/Mpc → 1/s; 1 Mpc = 3.0857e+19 km → in kpc: /3085.677581e16
    rho_crit_kpc = 3.0 * H_z_kpc_s**2 / (8.0 * jnp.pi * G)   # M_sun / kpc³
    # r_vir [kpc]: (M_vir / (4π/3 × 200 × ρ_crit))^(1/3)
    r_vir_kpc = (M_vir_solar / (4.0 * jnp.pi / 3.0 * 200.0 * rho_crit_kpc + 1e-30)) ** (1.0 / 3.0)
    c_nfw = r_vir_kpc / (r_s + 1e-8)
    # Clamp to physically reasonable range (concentration 2–100)
    c_nfw = jnp.clip(c_nfw, 2.0, 100.0)
    f_c = jnp.log(1.0 + c_nfw) - c_nfw / (1.0 + c_nfw)
    rho_s = M_vir_solar / (4.0 * jnp.pi * r_s**3 * f_c + 1e-8)
    
    kappa_s = (rho_s * r_s) / (Sigma_crit + 1e-8)
    
    # alpha_mag_rad = 4 * kappa_s * theta_s * h(x) / x
    # where theta_s = r_s / D_l (in radians)
    theta_s_rad = r_s / (D_l + 1e-8)
    alpha_mag_rad = 4.0 * kappa_s * theta_s_rad * h_x / (x + 1e-8)
    
    alpha_mag_arcsec = alpha_mag_rad * RAD_TO_ARCSEC
    
    r_kpc_safe = r_kpc + 1e-8
    alpha_x = alpha_mag_arcsec * r_x_kpc / r_kpc_safe
    alpha_y = alpha_mag_arcsec * r_y_kpc / r_kpc_safe
    
    return alpha_x, alpha_y



if JAX_AVAILABLE:
    def physics_informed_loss(
        model: eqx.Module,
        images: jax.Array,
        true_params: jax.Array,
        true_classes: jax.Array,
        key: jax.random.PRNGKey,
        lambda_physics: float = 0.1
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """
        Variational physics optimization loss.

        Minimizes the Lagrangian of the lens system by combining data-fidelity
        (MSE on lens parameters, CE on morphological class) with a physics
        penalty enforcing the lens equation residual:

            β = θ − α(θ)   (Schneider et al. 1992, Eq. 1.8)

        via sampled image-plane points.

        The stochastic sampling of θ-plane points (lines below) approximates
        Monte Carlo integration over the Lagrangian density.

        Ref: Schneider, Ehlers & Falco (1992) "Gravitational Lenses", §1.2
        Ref: Kodi Ramanah et al. (2020) MNRAS 499 — PINN for mass mapping
        """
        batch_forward = jax.vmap(model)
        pred_params, pred_classes = batch_forward(images)
        
        mse_params = jnp.mean((pred_params - true_params)**2)
        ce_class = optax.softmax_cross_entropy_with_integer_labels(pred_classes, true_classes).mean()
        
        batch_size = images.shape[0]
        n_sample_points = 16  # Monte Carlo samples for stochastic Lagrangian integration
        
        k1, k2 = jax.random.split(key)
        theta_x = jax.random.uniform(k1, (batch_size, n_sample_points)) * 2 - 1
        theta_y = jax.random.uniform(k2, (batch_size, n_sample_points)) * 2 - 1
        
        beta_x = pred_params[:, 2:3]
        beta_y = pred_params[:, 3:4]
        M_vir_raw = pred_params[:, 0:1]
        r_s_raw   = pred_params[:, 1:2]

        # Physics bounds penalty on raw (pre-clip) predictions so the gradient
        # signal is non-zero even when the network is out-of-range.
        M_vir_penalty = jax.nn.relu(1e9  - M_vir_raw) + jax.nn.relu(M_vir_raw - 1e14)
        r_s_penalty   = jax.nn.relu(0.1  - r_s_raw)   + jax.nn.relu(r_s_raw   - 500.0)

        # Clip for physical forward pass (jnp.clip uses positional a_min/a_max)
        M_vir  = jnp.clip(M_vir_raw, 1e9, 1e14)    # galaxy-to-cluster scale
        r_s    = jnp.clip(r_s_raw,   0.1, 500.0)   # kpc
        beta_x = jnp.clip(beta_x, -10.0, 10.0)
        beta_y = jnp.clip(beta_y, -10.0, 10.0)

        alpha_x, alpha_y = compute_nfw_deflection(
            M_vir=M_vir,
            r_s=r_s,
            theta_x=theta_x,
            theta_y=theta_y
        )
        
        residual_x = theta_x - beta_x - alpha_x
        residual_y = theta_y - beta_y - alpha_y
        
        raw_physics_residual = jnp.mean(residual_x**2 + residual_y**2)
        regularization = jnp.mean(M_vir_penalty**2) + jnp.mean(r_s_penalty**2)
        
        physics_loss = raw_physics_residual + regularization
        
        total_loss = mse_params + ce_class + lambda_physics * physics_loss
        
        losses = {
            'total': total_loss,
            'mse_params': mse_params,
            'ce_class': ce_class,
            'physics_residual': physics_loss,
            'raw_physics_residual': raw_physics_residual,
            'regularization': regularization
        }
        
        return total_loss, losses

    @eqx.filter_jit
    def train_step(
        model: eqx.Module,
        images: jax.Array,
        true_params: jax.Array,
        true_classes: jax.Array,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        key: jax.random.PRNGKey,
        lambda_physics: float = 0.1
    ) -> Tuple[eqx.Module, optax.OptState, Dict[str, jax.Array]]:
        
        loss_fn = eqx.filter_value_and_grad(physics_informed_loss, has_aux=True)
        
        (total_loss, aux_losses), grads = loss_fn(model, images, true_params, true_classes, key, lambda_physics)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, aux_losses
else:
    def physics_informed_loss(*args, **kwargs): # type: ignore
        raise RuntimeError("JAX/Equinox required.")
    def train_step(*args, **kwargs): # type: ignore
        raise RuntimeError("JAX/Equinox required.")

if __name__ == '__main__':
    if not JAX_AVAILABLE:
        logger.error("JAX/Equinox not available for testing")
        exit(1)
    logger.info("Testing PINN image inference in JAX...")
    key = jax.random.PRNGKey(42)
    model = PhysicsInformedNN(key=key)
    
    images = jax.random.normal(key, (32, 1, 64, 64))
    true_params = jax.random.normal(key, (32, 5))
    true_classes = jax.random.randint(key, (32,), 0, 3)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    new_model, new_opt_state, losses = train_step(
        model, images, true_params, true_classes, optimizer, opt_state, key
    )
    
    logger.info(f"Losses extracted: {losses['mse_params']}")
    logger.info("✅ CNN Inference JAX Test Passed!")
