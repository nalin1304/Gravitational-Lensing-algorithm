"""
Exact Schwarzschild Geodesic Integration Module

This module computes null geodesic deflection in Schwarzschild spacetime by
directly integrating the orbital (Binet) equation:
    u'' + u = 1.5 r_s u²   (u = 1/r, dimensionless x = r_s/r)

using scipy.integrate.solve_ivp with RK45 at double-precision tolerance
(rtol=1e-11, atol=1e-13). This yields full-GR deflection angles valid from
the photon sphere out to the weak-field regime.

Key Features:
- Exact numerical Schwarzschild geodesic ODE integration
- Dimensionless scaling (x = r_s/r) for numerical stability
- Periapsis detection via event-based solver termination
- Strong- vs weak-field accuracy comparisons against Born term
- Explicit failure reporting when the GR solver cannot produce a valid solution

Physics Background:
The geodesic equation in curved spacetime:
    d²xᵘ/dλ² + Γᵘᵥσ (dxᵛ/dλ)(dxσ/dλ) = 0

For Schwarzschild metric:
    ds² = -(1 - rs/r)c²dt² + (1 - rs/r)⁻¹dr² + r²(dθ² + sin²θ dφ²)
    where rs = 2GM/c² is the Schwarzschild radius.

References:
- Chandrasekhar (1983): The Mathematical Theory of Black Holes
- Misner, Thorne & Wheeler (1973): Gravitation
- Carroll (2004): Spacetime and Geometry
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

try:
    import einsteinpy  # type: ignore  # noqa: F401
    EINSTEINPY_AVAILABLE = True
except ImportError:
    EINSTEINPY_AVAILABLE = False
    warnings.warn(
        "EinsteinPy not installed. Trajectory-export helpers are unavailable, "
        "but the internal Schwarzschild Binet solver remains usable.",
        ImportWarning
    )

from astropy import units as u
from astropy import constants as const


class GeodesicIntegrator:
    """
    Exact Schwarzschild null geodesic integrator for gravitational deflection.

    Integrates the orbital Binet equation for photon trajectories:
        u'' + u = 1.5 r_s u²   (where u = 1/r, scaled by x = r_s/r)
    using scipy.integrate.solve_ivp (RK45) at double-precision tolerances,
    covering both weak-field (b >> r_s) and strong-field (b ~ r_s) regimes.

    Parameters
    ----------
    mass : float
        Mass of the lens in solar masses (M☉)

    Attributes
    ----------
    M : float
        Mass in solar masses
    rs : float
        Schwarzschild radius in meters
    M_geom : float
        Mass in geometric units (GM/c²) in meters
    metric : None
        Reserved for optional EinsteinPy-backed trajectory output.

    Examples
    --------
    >>> integrator = GeodesicIntegrator(mass=1e12)
    >>> result = integrator.integrate_deflection(impact_parameter=1.5e10)
    >>> print(f"GR deflection: {result['deflection_angle_rad']:.6e} rad")

    Notes
    -----
    The comparison baseline uses the first-order Born term:
        α_Born = 4GM/(c²b)
    The ODE result exceeds the Born term in strong-field regimes.
    """
    
    def __init__(self, mass: float):
        """Initialize geodesic integrator with lens mass."""
        if not EINSTEINPY_AVAILABLE:
            warnings.warn(
                "EinsteinPy not available; trajectory-export helpers are disabled.",
                RuntimeWarning,
            )

        self.M = mass  # Solar masses
        
        # Calculate Schwarzschild radius: rs = 2GM/c²
        M_kg = (mass * u.Msun).to(u.kg).value
        G = const.G.value  # m³ kg⁻¹ s⁻²
        c = const.c.value  # m/s
        
        self.rs = 2 * G * M_kg / (c**2)  # meters
        
        # Create Schwarzschild metric object
        # EinsteinPy uses geometric units (G=c=1)
        # Need to pass mass in meters: M_geom = GM/c²
        self.M_geom = G * M_kg / (c**2)  # meters (geometric units)
        
        # Store but don't create full EinsteinPy metric (complex API).
        # Current solver is PN-only and does not depend on EinsteinPy objects.
        self.metric = None  # We'll integrate manually
    
    def integrate_deflection(
        self,
        impact_parameter: float,
        r_init: Optional[float] = None,
        lambda_steps: int = 10000,
        return_trajectory: bool = False
    ) -> Dict:
        """
        Calculate Schwarzschild deflection angle from explicit geodesic integration.
        
        The solver integrates the null-orbit Binet equation numerically and
        reports the exact Schwarzschild deflection for the chosen impact
        parameter. The first-order Born term is returned only as a comparison
        baseline in the output dictionary.
        
        Parameters
        ----------
        impact_parameter : float
            Impact parameter in meters (closest approach distance)
        r_init : float, optional
            Initial radial position in meters (default: 100 * rs, far from lens)
        lambda_steps : int, optional
            Number of integration steps (default: 10000)
        return_trajectory : bool, optional
            Whether to return full trajectory (default: False)
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'deflection_angle_rad': GR deflection angle in radians
            - 'deflection_angle_arcsec': Deflection angle in arcseconds
            - 'impact_parameter': Input impact parameter (m)
            - 'impact_parameter_rs': b/rs (dimensionless)
            - 'regime': 'strong-field' or 'weak-field'
            - 'trajectory': Reserved for future trajectory export
            - 'simplified_angle_rad': First-order (Born) comparison angle
            - 'relative_error': |α_GR - α_Born|/|α_GR|
        
        Notes
        -----
        Method summary:
        1. Construct geometric-unit mass scale M = GM/c².
        2. Integrate the Schwarzschild null-orbit equation for the chosen `b`.
        3. Compare the GR result with the first-order Born expression.
        
        Notes on scope:
        - Weak-field results converge toward the Born limit as `b/rs` grows.
        - Near the capture boundary the solver may fail cleanly and raises an
          explicit exception rather than substituting a simplified formula.
        
        Examples
        --------
        >>> integrator = GeodesicIntegrator(mass=1e12)
        >>> # Strong field: b ~ 5 rs
        >>> result_strong = integrator.integrate_deflection(5 * integrator.rs)
        >>> # Weak field: b ~ 100 rs
        >>> result_weak = integrator.integrate_deflection(100 * integrator.rs)
        >>> print(f"Strong field error: {result_strong['relative_error']:.1%}")
        >>> print(f"Weak field error: {result_weak['relative_error']:.1%}")
        """
        b = impact_parameter
        
        # Determine regime
        b_over_rs = b / self.rs
        regime = "strong-field" if b_over_rs < 20 else "weak-field"
        
        # Set initial position (far from lens)
        if r_init is None:
            r_init = max(100 * self.rs, 10 * b)  # Start far away
        
        # Initial conditions for null geodesic
        # Start at large r, moving inward with impact parameter b
        
        # In spherical coordinates (t, r, θ, φ):
        # For equatorial plane (θ = π/2):
        # - Position: (t=0, r=r_init, θ=π/2, φ=0)
        # - 4-velocity normalized for null geodesic
        
        # For photon moving in equatorial plane:
        # E/m = (1 - 2M/r) dt/dτ
        # L/m = r² dφ/dτ
        # For null geodesic: ds² = 0
        
        # Energy normalization (set E = 1 in geometric units)
        E = 1.0
        
        # Angular momentum from impact parameter
        L = b * E
        
        # From null condition and conserved quantities:
        # (dr/dλ)² = E² - (1 - 2M/r)(L²/r² + 1)
        # At r = r_init:
        metric_factor = 1 - 2 * self.M_geom / r_init
        term2 = (L**2 / r_init**2)
        dr_dlambda_sq = E**2 - metric_factor * (term2 + 0)  # Last term is for massive particles
        
        if dr_dlambda_sq < 0:
            raise ValueError(
                f"Geodesic cannot reach r={r_init/self.rs:.2f}rs with b={b_over_rs:.2f}rs. "
                "Photon captured or invalid initial conditions."
            )
        
        dr_dlambda = -np.sqrt(dr_dlambda_sq)  # Negative = moving inward
        
        # dφ/dλ from angular momentum
        dphi_dlambda = L / (r_init**2)
        
        # dt/dλ from energy
        dt_dlambda = E / metric_factor
        
        # Initial 4-velocity (geodesic parameter derivatives)
        # In Schwarzschild coordinates: (dt/dλ, dr/dλ, dθ/dλ, dφ/dλ)
        initial_velocity = np.array([
            dt_dlambda,
            dr_dlambda,
            0.0,  # dθ/dλ = 0 (stay in equatorial plane)
            dphi_dlambda
        ])
        
        # Initial position
        initial_position = np.array([
            0.0,  # t
            r_init,  # r
            np.pi / 2,  # θ (equatorial plane)
            0.0  # φ
        ])
        
        alpha_gr = self._integrate_schwarzschild_orbit(
            r_init, b, lambda_steps
        )
        
        # Calculate simplified formula for comparison
        G = const.G.value
        c = const.c.value
        M_kg = (self.M * u.Msun).to(u.kg).value
        alpha_simplified = 4 * G * M_kg / (c**2 * b)  # radians
        
        # Relative error
        relative_error = abs(alpha_gr - alpha_simplified) / alpha_gr if alpha_gr != 0 else 0
        
        # Convert to arcseconds
        alpha_gr_arcsec = (alpha_gr * u.rad).to(u.arcsec).value
        alpha_simp_arcsec = (alpha_simplified * u.rad).to(u.arcsec).value
        
        result = {
            'deflection_angle_rad': alpha_gr,
            'deflection_angle_arcsec': alpha_gr_arcsec,
            'simplified_angle_rad': alpha_simplified,
            'simplified_angle_arcsec': alpha_simp_arcsec,
            'impact_parameter': b,
            'impact_parameter_rs': b_over_rs,
            'schwarzschild_radius': self.rs,
            'regime': regime,
            'relative_error': relative_error,
            'percent_difference': relative_error * 100,
            'gr_exceeds_simplified': alpha_gr > alpha_simplified
        }
        
        return result
    
    def _integrate_schwarzschild_orbit(
        self,
        r_init: float,
        b: float,
        steps: int
    ) -> float:
        """
        Calculate deflection using exact numerical Schwarzschild geodesic integration.
        
        This integrates the orbital equation for null geodesics:
        u'' + u = 1.5 * r_s * u^2 (where u = 1/r)
        
        Parameters
        ----------
        r_init : float
            Starting radius (meters) - not used, integration starts exactly at infinity
        b : float
            Impact parameter (meters)
        steps : int
            Not used, for API compatibility
        
        Returns
        -------
        alpha : float
            Deflection angle in radians
        """
        from scipy.integrate import solve_ivp
        import warnings
        
        r_s = self.rs
        
        # We work in dimensionless units x = r_s / r to ensure ODE scale invariance
        # The orbital equation is x'' + x = 1.5 x^2
        def deriv(phi, y):
            # y[0] = x, y[1] = dx/dphi
            return [y[1], 1.5 * y[0]**2 - y[0]]

        def periapsis(phi, y):
            return y[1]
        periapsis.terminal = True
        periapsis.direction = -1

        def capture(phi, y):
            return y[0] - 1.0
        capture.terminal = True
        
        # Initial conditions exactly at infinity: x = 0, x' = r_s / b
        y0 = [0.0, r_s / b]
        
        # Integrate forward in the angle phi
        sol = solve_ivp(
            deriv, [0.0, 10.0 * np.pi], y0,
            events=[periapsis, capture],
            rtol=1e-11, atol=1e-13
        )
        
        if sol.status == 1 and len(sol.t_events[0]) > 0:
            # Reached periapsis
            phi_max = sol.t_events[0][0]
            # Total angle swept is 2 * phi_max due to symmetry
            alpha_rad = float(2.0 * phi_max - np.pi)
            return alpha_rad
        elif sol.status == 1 and len(sol.t_events[1]) > 0:
            # Photon captured by the black hole (crossed the event horizon).
            # A captured photon has no asymptotic deflection angle — returning
            # a finite value would silently corrupt downstream analyses.
            return float('nan')
        else:
            raise RuntimeError("Geodesic integration failed to converge or find periapsis.")
    
    def compare_strong_vs_weak_field(
        self,
        b_min_rs: float = 1.5,
        b_max_rs: float = 100,
        n_points: int = 20
    ) -> Dict:
        """
        Compare GR vs simplified across strong and weak field regimes.
        
        This generates the accuracy comparison table from Paper Section 4.3.
        
        Parameters
        ----------
        b_min_rs : float, optional
            Minimum impact parameter in Schwarzschild radii (default: 1.5)
        b_max_rs : float, optional
            Maximum impact parameter in Schwarzschild radii (default: 100)
        n_points : int, optional
            Number of sample points (default: 20)
        
        Returns
        -------
        comparison : dict
            Dictionary containing:
            - 'impact_parameters_rs': Array of b/rs values
            - 'gr_deflections': Array of PN deflections (rad)
            - 'simplified_deflections': Array of simplified deflections (rad)
            - 'relative_errors': Array of relative errors
            - 'mean_error_strong': Mean error for b < 20rs
            - 'mean_error_weak': Mean error for b >= 20rs
        
        Examples
        --------
        >>> integrator = GeodesicIntegrator(mass=1e12)
        >>> comparison = integrator.compare_strong_vs_weak_field()
        >>> print(f"Strong field avg error: {comparison['mean_error_strong']:.1%}")
        >>> print(f"Weak field avg error: {comparison['mean_error_weak']:.1%}")
        """
        # Logarithmic spacing for better coverage
        b_rs_values = np.logspace(np.log10(b_min_rs), np.log10(b_max_rs), n_points)
        
        gr_deflections = []
        simp_deflections = []
        rel_errors = []
        
        for b_rs in b_rs_values:
            b = b_rs * self.rs
            result = self.integrate_deflection(b, lambda_steps=5000)
            
            gr_deflections.append(result['deflection_angle_rad'])
            simp_deflections.append(result['simplified_angle_rad'])
            rel_errors.append(result['relative_error'])
        
        gr_deflections = np.array(gr_deflections)
        simp_deflections = np.array(simp_deflections)
        rel_errors = np.array(rel_errors)
        
        # Separate strong (b < 20rs) and weak (b >= 20rs) field
        strong_mask = b_rs_values < 20
        weak_mask = b_rs_values >= 20
        
        mean_error_strong = float(np.mean(rel_errors[strong_mask])) if np.any(strong_mask) else 0.0
        mean_error_weak = float(np.mean(rel_errors[weak_mask])) if np.any(weak_mask) else 0.0
        
        return {
            'impact_parameters_rs': b_rs_values,
            'impact_parameters_m': b_rs_values * self.rs,
            'gr_deflections': gr_deflections,
            'simplified_deflections': simp_deflections,
            'relative_errors': rel_errors,
            'percent_errors': rel_errors * 100,
            'mean_error_strong': mean_error_strong,
            'mean_error_weak': mean_error_weak,
            'strong_field_regime': b_rs_values < 20,
            'weak_field_regime': b_rs_values >= 20
        }


def validate_paper_accuracy_table(mass: float = 1e12) -> Dict:
    """
    Generate a PN-versus-Born comparison table for representative b/rs values.

    This validation intentionally avoids hard-coding absolute % thresholds from
    narrative claims. Instead, it checks method-consistent diagnostics:
    1. finite errors,
    2. strong-field average error > weak-field average error,
    3. non-increasing trend of error with increasing impact parameter.
    
    Parameters
    ----------
    mass : float, optional
        Lens mass in solar masses (default: 1e12)
    
    Returns
    -------
    validation : dict
        Validation results matching a reproducible comparison table:
        | b/rs | Simplified α | GR α | Relative Error | Regime |
    
    Examples
    --------
    >>> validation = validate_paper_accuracy_table()
    >>> for row in validation['table_rows']:
    ...     print(f"{row['b_rs']:6.1f} | {row['simp']:8.4f} | {row['gr']:8.4f} | "
    ...           f"{row['error']:6.1%} | {row['regime']}")
    """
    integrator = GeodesicIntegrator(mass=mass)
    
    # Paper's table: b/rs values
    b_rs_values = [1.5, 5, 20, 50, 100]
    
    table_rows = []
    
    for b_rs in b_rs_values:
        b = b_rs * integrator.rs
        result = integrator.integrate_deflection(b)
        
        row = {
            'b_rs': b_rs,
            'simp': result['simplified_angle_rad'],
            'gr': result['deflection_angle_rad'],
            'error': result['relative_error'],
            'regime': result['regime']
        }
        table_rows.append(row)
    
    strong_errors = np.array([row['error'] for row in table_rows if row['b_rs'] <= 5.0], dtype=float)
    weak_errors = np.array([row['error'] for row in table_rows if row['b_rs'] >= 50.0], dtype=float)
    all_errors = np.array([row['error'] for row in table_rows], dtype=float)

    finite_pass = np.all(np.isfinite(all_errors))
    regime_separation_pass = (
        strong_errors.size > 0
        and weak_errors.size > 0
        and float(np.mean(strong_errors)) > float(np.mean(weak_errors))
    )
    monotonic_pass = np.all(np.diff(all_errors) <= 1e-12)
    
    return {
        'table_rows': table_rows,
        'integrator': integrator,
        'mean_error_strong': float(np.mean(strong_errors)) if strong_errors.size else 0.0,
        'mean_error_weak': float(np.mean(weak_errors)) if weak_errors.size else 0.0,
        'validation_checks': {
            'finite_errors': bool(finite_pass),
            'strong_greater_than_weak': bool(regime_separation_pass),
            'nonincreasing_with_impact_parameter': bool(monotonic_pass),
        },
        'validation_passed': bool(finite_pass and regime_separation_pass and monotonic_pass),
    }


# Convenience function for quick testing
def quick_deflection_comparison(
    mass: float,
    impact_parameter_rs: float
) -> None:
    """
    Quick comparison of GR vs simplified deflection.
    
    Parameters
    ----------
    mass : float
        Lens mass in solar masses
    impact_parameter_rs : float
        Impact parameter in units of Schwarzschild radii
    
    Examples
    --------
    >>> quick_deflection_comparison(1e12, 5.0)
    """
    integrator = GeodesicIntegrator(mass=mass)
    b = impact_parameter_rs * integrator.rs
    result = integrator.integrate_deflection(b)
    
    print(f"\n=== Gravitational Deflection Comparison ===")
    print(f"Lens mass: {mass:.2e} M☉")
    print(f"Schwarzschild radius: {integrator.rs:.3e} m")
    print(f"Impact parameter: {impact_parameter_rs:.2f} rs = {b:.3e} m")
    print(f"Regime: {result['regime']}")
    print(f"\nPN model:       {result['deflection_angle_arcsec']:.6f} arcsec")
    print(f"Born model:     {result['simplified_angle_arcsec']:.6f} arcsec")
    print(f"Relative error: {result['relative_error']:.2%}")
    print(f"PN > Born:      {result['gr_exceeds_simplified']}")
    print("=" * 45)


if __name__ == "__main__":
    # Test the implementation
    print("Testing PN Schwarzschild Deflection Module...")
    
    quick_deflection_comparison(mass=1e12, impact_parameter_rs=5.0)

    print("\nValidating PN vs Born Comparison Table...")
    validation = validate_paper_accuracy_table()

    print("\n" + "=" * 70)
    print("PN VS BORN DEFLECTION COMPARISON TABLE")
    print("=" * 70)
    print(f"{'b/rs':>8} | {'Born α':>12} | {'PN α':>12} | {'Error':>8} | {'Regime':>15}")
    print("-" * 70)

    for row in validation['table_rows']:
        print(
            f"{row['b_rs']:>8.1f} | {row['simp']:>12.6f} | {row['gr']:>12.6f} | "
            f"{row['error']:>7.1%} | {row['regime']:>15}"
        )

    checks = validation["validation_checks"]
    print("=" * 70)
    print(f"Checks: finite={checks['finite_errors']}, strong>weak={checks['strong_greater_than_weak']}, "
          f"monotonic={checks['nonincreasing_with_impact_parameter']}")
    print(f"Validation: {'PASSED' if validation['validation_passed'] else 'FAILED'}")
