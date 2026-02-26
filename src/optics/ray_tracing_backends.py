"""
Dual Ray-Tracing Backends for Gravitational Lensing

This module implements two physically distinct ray-tracing methods:
1. thin_lens: Born approximation in cosmological FLRW spacetime (default)
2. schwarzschild_geodesic: Numerical GR integration in static Schwarzschild metric

CRITICAL DISTINCTION:
- thin_lens: For galaxy-scale lensing at cosmological distances (z > 0.1)
  Uses angular diameter distances in expanding universe
- schwarzschild_geodesic: For strong-field regime near black holes (z ≈ 0)
  Assumes flat, static spacetime - NOT compatible with cosmology

Author: IEEE TCI - Scientific Refinement
References:
    - Schneider, Ehlers & Falco (1992): "Gravitational Lenses"
    - Misner, Thorne & Wheeler (1973): "Gravitation"
"""

try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
except ImportError:
    jax = None
    import numpy as np
    jnp = np
    eqx = None

import numpy as np
from typing import Tuple, Optional, Dict, Literal, Callable
from scipy.ndimage import label
import warnings
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Import constants
from ..utils.constants import (
    G_CONST, C_LIGHT, M_SUN_KG, ARCSEC_TO_RAD, RAD_TO_ARCSEC
)


# ============================================================================
# Backend Selection and Validation
# ============================================================================

from enum import Enum

class RayTracingMode(str, Enum):
    """
    Physical regimes for gravitational lensing ray tracing.
    
    THIN_LENS: Standard cosmological lensing (RECOMMENDED for z > 0.05)
        - Uses GR-derived thin-lens formalism on FLRW background
        - Includes proper angular diameter distances with cosmic expansion
        - Valid for: galaxies, clusters, multi-plane systems
        - Based on weak-field approximation of Einstein equations
        
    SCHWARZSCHILD: Strong-field geodesic integration (z ≈ 0 ONLY)
        - Solves null geodesic ODE in Schwarzschild metric
        - Assumes static, asymptotically flat spacetime
        - Valid for: black hole shadows, extreme strong lensing near horizons
        - INVALID for cosmological distances (ignores expansion)
        
    Scientific Guidance:
    -------------------
    - For HST/JWST galaxy lenses: ALWAYS use THIN_LENS
    - For multi-plane lensing: ONLY THIN_LENS is supported
    - For validation against literature: THIN_LENS reproduces Schneider+ results
    - For black hole simulations at z≈0: SCHWARZSCHILD is appropriate
    """
    THIN_LENS = "thin_lens"
    SCHWARZSCHILD = "schwarzschild_geodesic"


RayTracingMethod = Literal["thin_lens", "schwarzschild_geodesic"]


def _as_scalar(value) -> float:
    """Convert scalar/array-like values from profile APIs to python float."""
    return float(np.asarray(value).reshape(-1)[0])


def validate_method_compatibility(
    method: RayTracingMethod,
    redshift_lens: float,
    redshift_source: float
) -> None:
    """
    Validate that ray-tracing method is appropriate for given redshifts.
    
    ENFORCES SCIENTIFIC VALIDITY:
    - Schwarzschild mode RAISES ERROR for z_lens > 0.05
    - Thin-lens mode is ALWAYS safe for any redshift
    
    Parameters
    ----------
    method : {"thin_lens", "schwarzschild_geodesic"}
        Ray-tracing method
    redshift_lens : float
        Lens redshift
    redshift_source : float
        Source redshift
        
    Raises
    ------
    ValueError
        If schwarzschild mode is used with z_lens > 0.05
        
    Warns
    -----
    UserWarning
        If method is suboptimal for given redshift regime
        
    Examples
    --------
    >>> validate_method_compatibility("schwarzschild_geodesic", 0.5, 1.0)
    ValueError: Schwarzschild mode only valid for z_lens ≤ 0.05
    
    >>> validate_method_compatibility("thin_lens", 0.5, 1.0)
    # No error - thin_lens is always valid
    """
    if method == "schwarzschild_geodesic" or method == RayTracingMode.SCHWARZSCHILD:
        # HARD CONSTRAINT: Schwarzschild requires essentially local spacetime
        if redshift_lens > 0.05:
            raise ValueError(
                f"Schwarzschild geodesic mode is ONLY valid for local, "
                f"non-cosmological lenses (z_lens ≤ 0.05).\n"
                f"Current z_lens = {redshift_lens:.4f} violates flat-spacetime assumption.\n"
                f"\n"
                f"For galaxy-scale lensing at cosmological distances:\n"
                f"  → Use mode='thin_lens' (GR-derived formalism on FLRW background)\n"
                f"\n"
                f"Schwarzschild mode ignores:\n"
                f"  - Cosmic expansion (H(z) ≠ 0)\n"
                f"  - Angular diameter distance scaling: D_A ∝ 1/(1+z)\n"
                f"  - Time dilation factors\n"
                f"\n"
                f"Scientific reference: Schneider, Ehlers & Falco (1992), §4.1"
            )
        
        if redshift_source > 0.05:
            logger.warning(
                f"Schwarzschild mode used with z_source={redshift_source:.4f}. "
                f"This assumes static spacetime - not appropriate for cosmological sources."
            )
    
    elif method == "thin_lens" or method == RayTracingMode.THIN_LENS:
        # Thin-lens is ALWAYS scientifically valid
        # It reduces to correct limits at all redshifts
        if redshift_lens < 0.01 and redshift_source < 0.01:
            logger.info(
                f"Using thin_lens for very low redshifts (z_l={redshift_lens:.3f}). "
                f"For strong-field tests near compact objects (e.g., photon rings), "
                f"schwarzschild_geodesic may provide higher accuracy, but thin_lens "
                f"is still scientifically valid."
            )


def schwarzschild_radius(mass_kg: float) -> float:
    """
    Calculate Schwarzschild radius.
    
    r_s = 2GM/c²
    
    Parameters
    ----------
    mass_kg : float
        Mass in kilograms
        
    Returns
    -------
    r_s : float
        Schwarzschild radius in meters
        
    Examples
    --------
    >>> M_bh = 4.3e6 * M_SUN_KG  # Sgr A*
    >>> r_s = schwarzschild_radius(M_bh)
    >>> print(f"r_s = {r_s/1e3:.2f} km")
    """
    return 2.0 * G_CONST * mass_kg / (C_LIGHT**2)


# ============================================================================
# Method 1: Thin-Lens Approximation (Born Approximation + FLRW Cosmology)
# ============================================================================

def thin_lens_ray_trace(
    source_position: Tuple[float, float],
    lens_model,
    grid_extent: float = 3.0,
    grid_resolution: int = 300,
    threshold: float = 0.05,
    return_maps: bool = True
) -> Dict:
    """
    Ray tracing using thin-lens (Born) approximation in FLRW cosmology.
    
    This is the STANDARD method for galaxy-scale gravitational lensing.
    
    Physical Assumptions:
    - Deflection angles small (weak-field approximation)
    - Lens is thin compared to D_l, D_s (thin-screen approximation)
    - FLRW cosmology with angular diameter distances
    - No strong-field effects
    
    Lens Equation (Born Approximation):
        β = θ - α(θ)
        
    where:
        β: source position (arcsec)
        θ: image position (arcsec)
        α: deflection angle = (4GM/c²) × (D_LS/D_L D_S) × (θ/|θ|²)
    
    Parameters
    ----------
    source_position : tuple of float
        Source position (x, y) in arcseconds on source plane
    lens_model : MassProfile
        Lens model (must have .deflection_angle() method)
    grid_extent : float, optional
        Half-width of image plane grid in Einstein radii (default: 3.0)
    grid_resolution : int, optional
        Grid points per dimension (default: 300)
    threshold : float, optional
        Distance threshold for image identification in arcsec (default: 0.05)
    return_maps : bool, optional
        Return full convergence/caustic maps (default: True)
        
    Returns
    -------
    results : dict
        - 'image_positions': ndarray (N, 2), image positions in arcsec
        - 'magnifications': ndarray (N,), magnifications
        - 'convergence_map': 2D array (if return_maps=True)
        - 'source_plane_map': 2D array (if return_maps=True)
        - 'beta_x', 'beta_y': Source plane mappings
        - 'grid_x', 'grid_y': Image plane coordinates
        - 'method': 'thin_lens'
        
    Notes
    -----
    Validity range: Impact parameter b >> r_s (Schwarzschild radius)
    Typical applicability: Galaxy lenses (z ~ 0.1-3.0)
    
    References
    ----------
    Schneider, Ehlers & Falco (1992), Section 3.2
    """
    source_x, source_y = source_position
    
    # Validate cosmological setup
    if hasattr(lens_model, 'lens_system'):
        z_l = lens_model.lens_system.z_l
        z_s = lens_model.lens_system.z_s
        validate_method_compatibility("thin_lens", z_l, z_s)
    
    # Step 1: Create image plane grid
    x = jnp.linspace(-grid_extent, grid_extent, grid_resolution)
    y = jnp.linspace(-grid_extent, grid_extent, grid_resolution)
    xx, yy = jnp.meshgrid(x, y)
    
    # Step 2: Compute deflection angles (vectorized)
    # Uses cosmological distances from lens_model.lens_system
    alpha_x, alpha_y = lens_model.deflection_angle(xx.ravel(), yy.ravel())
    alpha_x = alpha_x.reshape(xx.shape)
    alpha_y = alpha_y.reshape(yy.shape)
    
    # Step 3: Map to source plane via Born approximation: β = θ - α
    beta_x = xx - alpha_x
    beta_y = yy - alpha_y
    
    # Step 4: Find pixels where |β - β_source| < threshold
    distance_to_source = jnp.sqrt((beta_x - source_x)**2 + (beta_y - source_y)**2)
    image_mask = distance_to_source < threshold
    
    # Step 5: Identify connected regions (separate images)
    labeled_array, num_images = label(image_mask)
    
    image_positions = []
    magnifications = []
    
    if num_images > 0:
        dx = x[1] - x[0]
        for img_id in range(1, num_images + 1):
            img_pixels = labeled_array == img_id
            y_indices, x_indices = jnp.where(img_pixels)
            
            if len(x_indices) > 0:
                # Weighted centroid
                weights = 1.0 / (distance_to_source[img_pixels] + 1e-10)
                weights /= weights.sum()
                
                x_centroid = jnp.sum(x[x_indices] * weights)
                y_centroid = jnp.sum(y[y_indices] * weights)
                r_weighted = jnp.sum(jnp.sqrt(x[x_indices]**2 + y[y_indices]**2) * weights)
                r_centroid = jnp.sqrt(x_centroid**2 + y_centroid**2)
                # Annular solutions (Einstein rings) have centroid ~0 even when
                # physically located at finite radius; keep a representative point
                # on the ring to avoid collapsing to the origin.
                if r_weighted > 0 and r_centroid < 0.5 * r_weighted:
                    if np.hypot(source_x, source_y) < max(threshold, 1e-6):
                        angle = 0.0
                    else:
                        angle = np.arctan2(source_y, source_x)
                    x_centroid = r_weighted * jnp.cos(angle)
                    y_centroid = r_weighted * jnp.sin(angle)
                
                # Compute magnification via Jacobian
                mag = _compute_magnification_jacobian(
                    x_centroid, y_centroid, lens_model, dx
                )
                
                image_positions.append([x_centroid, y_centroid])
                magnifications.append(mag)
    
    # Convert to arrays
    if len(image_positions) > 0:
        image_positions = jnp.array(image_positions)
        magnifications = jnp.array(magnifications)
    else:
        image_positions = jnp.array([]).reshape(0, 2)
        magnifications = jnp.array([])
    
    # Assemble results
    results = {
        'image_positions': image_positions,
        'magnifications': magnifications,
        'grid_x': x,
        'grid_y': y,
        'method': 'thin_lens'
    }
    
    if return_maps:
        convergence_map = lens_model.convergence(xx.ravel(), yy.ravel()).reshape(xx.shape)
        results['convergence_map'] = convergence_map
        results['source_plane_map'] = distance_to_source
        results['beta_x'] = beta_x
        results['beta_y'] = beta_y
    
    logger.info(f"Thin-lens ray trace: found {num_images} images for source at {source_position}")
    
    return results


def _compute_magnification_jacobian(
    x: float, 
    y: float, 
    lens_model, 
    dx: float = 0.01
) -> float:
    """
    Compute magnification via Jacobian of lens mapping.
    
    Magnification μ = 1/det(A), where A = ∂β/∂θ = I - ∂α/∂θ
    
    det(A) = (1 - κ - γ₁)(1 - κ + γ₁) - γ₂²
           = (1 - κ)² - γ²
    
    Parameters
    ----------
    x, y : float
        Position in arcseconds
    lens_model : MassProfile
        Lens model
    dx : float
        Finite difference step size (arcsec)
        
    Returns
    -------
    mu : float
        Signed magnification (|μ| > 1 for magnified images)
    """
    # Central differences for Jacobian
    alpha_x0, alpha_y0 = lens_model.deflection_angle(x, y)
    alpha_x0 = _as_scalar(alpha_x0)
    alpha_y0 = _as_scalar(alpha_y0)
    
    alpha_xp, _ = lens_model.deflection_angle(x + dx, y)
    alpha_xm, _ = lens_model.deflection_angle(x - dx, y)
    dalpha_x_dx = (_as_scalar(alpha_xp) - _as_scalar(alpha_xm)) / (2 * dx)
    
    alpha_xp, _ = lens_model.deflection_angle(x, y + dx)
    alpha_xm, _ = lens_model.deflection_angle(x, y - dx)
    dalpha_x_dy = (_as_scalar(alpha_xp) - _as_scalar(alpha_xm)) / (2 * dx)
    
    _, alpha_yp = lens_model.deflection_angle(x + dx, y)
    _, alpha_ym = lens_model.deflection_angle(x - dx, y)
    dalpha_y_dx = (_as_scalar(alpha_yp) - _as_scalar(alpha_ym)) / (2 * dx)
    
    _, alpha_yp = lens_model.deflection_angle(x, y + dx)
    _, alpha_ym = lens_model.deflection_angle(x, y - dx)
    dalpha_y_dy = (_as_scalar(alpha_yp) - _as_scalar(alpha_ym)) / (2 * dx)
    
    # Jacobian: A = I - ∂α/∂θ
    A11 = 1 - dalpha_x_dx
    A12 = -dalpha_x_dy
    A21 = -dalpha_y_dx
    A22 = 1 - dalpha_y_dy
    
    det_A = A11 * A22 - A12 * A21
    
    # Handle critical curves
    if jnp.abs(det_A) < 1e-10:
        mu = jnp.sign(det_A) * 1000.0
    else:
        mu = 1.0 / det_A
    
    return float(np.clip(mu, -1000.0, 1000.0))


# ============================================================================
# Method 2: Schwarzschild Geodesic Integration (Full GR, No Cosmology)
# ============================================================================

def schwarzschild_geodesic_trace(
    impact_parameter: float,
    mass_kg: float,
    max_radius: float = 1000.0,
    rtol: float = 1e-9
) -> Dict:
    """
    Trace null geodesic in Schwarzschild spacetime (STATIC, ASYMPTOTICALLY FLAT).
    
    **WARNING**: This is NOT for cosmological lensing!
    Use ONLY for:
    - Strong-field validation near black holes
    - Testing in flat spacetime
    - Single compact object at z ≈ 0
    
    DO NOT MIX with multi-plane lensing or cosmological distances.
    
    Physical Setup:
    - Schwarzschild metric: ds² = -(1-r_s/r)dt² + (1-r_s/r)⁻¹dr² + r²dΩ²
    - Null geodesic equations integrated numerically
    - Impact parameter b defines initial conditions
    - No expansion, no cosmology
    
    Geodesic Equations:
        d²r/dλ² = -Γ^r_μν (dx^μ/dλ)(dx^ν/dλ)
        d²φ/dλ² = -Γ^φ_μν (dx^μ/dλ)(dx^ν/dλ)
        
    Parameters
    ----------
    impact_parameter : float
        Impact parameter in meters (b = |r × k|)
    mass_kg : float
        Lens mass in kilograms (NOT solar masses)
    max_radius : float, optional
        Integration cutoff in Schwarzschild radii (default: 1000)
    rtol : float, optional
        Relative tolerance for ODE solver (default: 1e-9)
        
    Returns
    -------
    results : dict
        - 'deflection_angle': Total deflection in radians
        - 'trajectory': (r, phi) arrays of photon path
        - 'schwarzschild_radius': r_s in meters
        - 'closest_approach': Minimum radius in meters
        - 'method': 'schwarzschild_geodesic'
        
    Notes
    -----
    Validity: Strong-field regime (b ~ r_s)
    Invalid for: Cosmological lenses, multi-plane systems
    
    Weak-field limit (b >> r_s):
        α ≈ 4GM/(c²b) (recovers Newtonian deflection)
        
    References
    ----------
    Misner, Thorne & Wheeler (1973), Chapter 25
    Chandrasekhar (1983): "The Mathematical Theory of Black Holes"
    """
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if impact_parameter <= 0:
        raise ValueError("impact_parameter must be positive")

    r_s = schwarzschild_radius(mass_kg)
    b_over_rs = impact_parameter / r_s
    r_initial = max_radius * r_s
    b_crit_over_rs = 1.5 * jnp.sqrt(3.0)  # Photon-capture threshold

    logger.info(
        f"Schwarzschild geodesic: M={mass_kg/M_SUN_KG:.2e} M☉, "
        f"b={b_over_rs:.2f} r_s"
    )

    from scipy.integrate import solve_ivp
    
    # We work in dimensionless units x = r_s / r to ensure ODE scale invariance
    def deriv(phi, y):
        # y[0] = x = r_s/r, y[1] = dx/dphi
        return [y[1], 1.5 * y[0]**2 - y[0]]

    def periapsis(phi, y):
        return y[1]
    periapsis.terminal = True
    periapsis.direction = -1

    def capture(phi, y):
        return y[0] - 1.0
    capture.terminal = True

    y0 = [0.0, r_s / impact_parameter]

    sol = solve_ivp(
        deriv, [0.0, 10.0 * np.pi], y0,
        events=[periapsis, capture],
        rtol=1e-11, atol=1e-13,
        dense_output=True
    )

    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Photon escaped, reached periapsis and goes back to infinity
        phi_max = sol.t_events[0][0]
        closest_approach = r_s / sol.y_events[0][0][0]
        deflection_angle = float(2.0 * phi_max - np.pi)

        # Build full symmetric trajectory around periapsis
        phi_half = np.linspace(0, phi_max, 300)
        x_half = sol.sol(phi_half)[0]
        
        # Center the trajectory so phi=0 is periapsis
        phi_trajectory = np.concatenate([-phi_half[::-1], phi_half[1:]])
        x_trajectory = np.concatenate([x_half[::-1], x_half[1:]])
        
        r_trajectory = r_s / np.maximum(x_trajectory, 1e-15)
        r_trajectory = np.clip(r_trajectory, closest_approach, r_initial)
        
    elif sol.status == 1 and len(sol.t_events[1]) > 0:
        # Photon captured by horizon
        deflection_angle = float(np.pi)
        closest_approach = 1.5 * r_s
        
        phi_trap = sol.t_events[1][0]
        phi_trajectory = np.linspace(0.0, phi_trap, 600)
        x_trajectory = sol.sol(phi_trajectory)[0]
        
        r_trajectory = r_s / np.maximum(x_trajectory, 1e-15)
        r_trajectory = np.clip(r_trajectory, 1.01 * r_s, r_initial)
        
    else:
        # Fallback to weak field assumption
        deflection_angle = float(4.0 * G_CONST * mass_kg / (C_LIGHT**2 * impact_parameter))
        closest_approach = impact_parameter
        
        phi_max = 0.5 * (np.pi + deflection_angle)
        phi_trajectory = np.linspace(-phi_max, phi_max, 600)
        sin_phi = np.maximum(np.abs(np.sin(phi_trajectory)), 1e-8)
        r_trajectory = closest_approach / sin_phi
        r_trajectory = np.clip(r_trajectory, closest_approach, r_initial)

    logger.info(
        f"Geodesic complete: α={deflection_angle:.6e} rad, "
        f"r_min={closest_approach / r_s:.2f} r_s"
    )

    return {
        'deflection_angle': float(abs(deflection_angle)),
        'trajectory': (r_trajectory, phi_trajectory),
        'schwarzschild_radius': r_s,
        'closest_approach': float(closest_approach),
        'method': 'schwarzschild_geodesic',
        'impact_parameter': float(impact_parameter),
        'mass': float(mass_kg)
    }


def schwarzschild_deflection_angle(
    impact_parameter_meters: float,
    mass_kg: float
) -> float:
    """
    Compute deflection angle in Schwarzschild spacetime.
    
    Weak-field limit (b >> r_s):
        α ≈ 4GM/(c²b)
        
    Full numerical integration for arbitrary b.
    
    Parameters
    ----------
    impact_parameter_meters : float
        Impact parameter in meters
    mass_kg : float
        Mass in kilograms
        
    Returns
    -------
    alpha : float
        Deflection angle in radians
        
    Examples
    --------
    >>> M_sun = M_SUN_KG
    >>> b = 1e9  # 1e9 meters ~ 7 R_sun
    >>> alpha = schwarzschild_deflection_angle(b, M_sun)
    >>> print(f"α = {alpha:.3e} rad = {alpha * RAD_TO_ARCSEC:.3f} arcsec")
    """
    r_s = schwarzschild_radius(mass_kg)
    
    # Weak-field approximation is reliable for b/r_s >= 10 in this code path.
    if impact_parameter_meters >= 10.0 * r_s:
        # Use analytical weak-field formula
        alpha_weak = 4 * G_CONST * mass_kg / (C_LIGHT**2 * impact_parameter_meters)
        logger.debug(f"Using weak-field approximation: b/r_s = {impact_parameter_meters/r_s:.1f}")
        return alpha_weak
    else:
        # Full numerical integration
        logger.debug(f"Using numerical geodesic integration: b/r_s = {impact_parameter_meters/r_s:.1f}")
        result = schwarzschild_geodesic_trace(impact_parameter_meters, mass_kg)
        return result['deflection_angle']


# ============================================================================
# Unified Interface
# ============================================================================

def ray_trace(
    source_position: Tuple[float, float],
    lens_model,
    method: RayTracingMethod = "thin_lens",
    **kwargs
) -> Dict:
    """
    Unified ray-tracing interface with method selection.
    
    Parameters
    ----------
    source_position : tuple of float
        Source position (x, y) in arcseconds
    lens_model : MassProfile
        Lens model
    method : {"thin_lens", "schwarzschild_geodesic"}, optional
        Ray-tracing method (default: "thin_lens")
    **kwargs : dict
        Method-specific parameters
        
    Returns
    -------
    results : dict
        Ray-tracing results (format depends on method)
        
    Raises
    ------
    ValueError
        If method is not recognized
        
    Examples
    --------
    >>> # Cosmological lensing (standard use case)
    >>> results = ray_trace((0.5, 0.0), lens, method="thin_lens")
    >>> 
    >>> # Strong-field validation (specialized)
    >>> results = ray_trace((0.0, 0.0), lens, method="schwarzschild_geodesic")
    """
    if method == "thin_lens":
        return thin_lens_ray_trace(source_position, lens_model, **kwargs)
    
    elif method == "schwarzschild_geodesic":
        # Schwarzschild requires different parameters
        warnings.warn(
            "schwarzschild_geodesic method requires impact parameter and mass. "
            "Use schwarzschild_geodesic_trace() directly for full control.",
            UserWarning
        )
        raise NotImplementedError(
            "For Schwarzschild geodesics, use schwarzschild_geodesic_trace() directly. "
            "This method requires different parameters than thin-lens ray tracing."
        )
    
    else:
        raise ValueError(
            f"Unknown ray-tracing method: {method}. "
            f"Choose from: 'thin_lens', 'schwarzschild_geodesic'"
        )


# ============================================================================
# Comparison and Validation
# ============================================================================

def compare_methods_weak_field(
    impact_parameter_arcsec: float,
    lens_model,
    mass_kg: Optional[float] = None
) -> Dict:
    """
    Compare thin-lens and Schwarzschild methods in weak-field limit.
    
    Both should agree when b >> r_s.
    
    Parameters
    ----------
    impact_parameter_arcsec : float
        Impact parameter in arcseconds
    lens_model : MassProfile
        Lens model (for thin-lens)
    mass_kg : float, optional
        Mass for Schwarzschild (if None, uses lens_model mass)
        
    Returns
    -------
    comparison : dict
        - 'thin_lens_alpha': Deflection from thin-lens (arcsec)
        - 'schwarzschild_alpha': Deflection from Schwarzschild (arcsec)
        - 'relative_difference': Fractional difference
        - 'agreement': Whether methods agree within tolerance
    """
    # Thin-lens deflection
    alpha_x_thin, alpha_y_thin = lens_model.deflection_angle(
        impact_parameter_arcsec, 0.0
    )
    alpha_thin_arcsec = _as_scalar(jnp.sqrt(alpha_x_thin**2 + alpha_y_thin**2))
    
    # Schwarzschild deflection
    if mass_kg is None:
        if hasattr(lens_model, 'M'):
            mass_kg = lens_model.M * M_SUN_KG
        elif hasattr(lens_model, 'M_vir'):
            mass_kg = lens_model.M_vir * M_SUN_KG
        else:
            raise ValueError("Cannot determine mass for Schwarzschild calculation")
    
    # Convert impact parameter to meters
    if hasattr(lens_model, 'lens_system'):
        D_l = lens_model.lens_system.angular_diameter_distance_lens().to('m').value
        impact_parameter_m = impact_parameter_arcsec * ARCSEC_TO_RAD * D_l
        D_s = lens_model.lens_system.angular_diameter_distance_source().to('m').value
        D_ls = lens_model.lens_system.angular_diameter_distance_lens_source().to('m').value
        # Thin-lens deflection is the reduced angle α̂ × D_ls / D_s.
        reduction_factor = D_ls / D_s if D_s > 0 else 1.0
    else:
        # Assume local (D ~ 1 Mpc for scaling)
        impact_parameter_m = impact_parameter_arcsec * ARCSEC_TO_RAD * 3e22
        reduction_factor = 1.0
    
    alpha_schw_rad = schwarzschild_deflection_angle(impact_parameter_m, mass_kg)
    alpha_schw_arcsec = float(alpha_schw_rad * reduction_factor * RAD_TO_ARCSEC)
    
    # Compare
    if abs(alpha_schw_arcsec) < 1e-20:
        relative_diff = float('inf')
    else:
        relative_diff = float(
            abs(alpha_schw_arcsec - alpha_thin_arcsec) / abs(alpha_schw_arcsec)
        )
    agreement = relative_diff < 0.05  # 5% tolerance
    
    return {
        'thin_lens_alpha': alpha_thin_arcsec,
        'schwarzschild_alpha': alpha_schw_arcsec,
        'relative_difference': relative_diff,
        'agreement': agreement,
        'impact_parameter_arcsec': impact_parameter_arcsec,
        'impact_parameter_r_s': impact_parameter_m / schwarzschild_radius(mass_kg)
    }
