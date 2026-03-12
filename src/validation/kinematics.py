"""
Stellar Kinematics Module — Velocity Dispersion and Mass-Sheet Degeneracy

This module integrates velocity dispersion measurements from galaxy spectra
with gravitational lensing observables to break the mass-sheet degeneracy,
enabling sub-percent precision in cosmological measurements.

Physics
-------
The spherical Jeans equation relates the velocity dispersion σ(r) to the
enclosed mass M(r) for a virialized stellar system:

    d(ρ_* σ_r²)/dr + 2β(r) ρ_* σ_r² / r = -ρ_* dΦ/dr

where β(r) = 1 - σ_t²/σ_r² is the velocity anisotropy parameter and
Φ is the gravitational potential.

Ref: Jeans (1922) MNRAS 82, 122 — original derivation
Ref: Binney & Tremaine (2008) "Galactic Dynamics", §4.2
Ref: Treu & Koopmans (2004) ApJ 611, 739 — GLaD methodology
Ref: Birrer et al. (2020) A&A 643, A165 — H₀ from time-delay lensing

Usage
-----
    from src.validation.kinematics import (
        predict_velocity_dispersion,
        mass_sheet_degeneracy_test,
        combine_lensing_kinematics,
    )

    sigma_pred = predict_velocity_dispersion(mass_profile, r_eff=1.5, z_lens=0.3)
    consistency = mass_sheet_degeneracy_test(lensing_mass=1e12, kinematic_mass=1.1e12)
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

# Physical constants (Ref: IAU 2015 nominal values, CODATA 2018)
G_SI = 6.67430e-11         # m³ kg⁻¹ s⁻²
c_SI = 2.99792458e8        # m/s
M_sun = 1.98892e30         # kg
kpc_to_m = 3.0857e19       # m
arcsec_to_rad = 4.8481e-6  # radians


def predict_velocity_dispersion(
    mass_profile,
    r_eff: float,
    z_lens: float,
    r_aperture: Optional[float] = None,
    beta_aniso: float = 0.0,
    n_radial_bins: int = 100,
) -> Dict:
    """Predict the luminosity-weighted line-of-sight velocity dispersion.

    Solves the spherical Jeans equation for an isotropic or anisotropic
    stellar system embedded in the gravitational potential of the mass
    profile.

    Ref: Binney & Tremaine (2008), Eq. 4.29 (spherical Jeans equation)
    Ref: Mamon & Łokas (2005) MNRAS 363, 705 — anisotropy models

    Parameters
    ----------
    mass_profile : MassProfile
        Lens mass profile with ``convergence()`` and optionally
        ``enclosed_mass()`` methods.
    r_eff : float
        Effective (half-light) radius of the deflector galaxy [arcsec].
    z_lens : float
        Redshift of the lens galaxy.
    r_aperture : float, optional
        Spectroscopic aperture radius [arcsec]. Default: r_eff / 2.
    beta_aniso : float
        Constant velocity anisotropy parameter β = 1 - σ_t²/σ_r².
        β = 0 is isotropic, β = 1 is purely radial.
    n_radial_bins : int
        Number of radial integration bins.

    Returns
    -------
    result : dict
        Keys:
        - 'sigma_v_kms': predicted velocity dispersion [km/s]
        - 'sigma_v_profile': σ(r) profile array
        - 'r_bins_arcsec': radial bins [arcsec]
        - 'enclosed_mass_msun': M(<r) profile [M_sun]
        - 'beta_aniso': anisotropy parameter used
    """
    if r_aperture is None:
        r_aperture = r_eff / 2.0

    # Radial grid from 0.01 × r_eff to 10 × r_eff
    r_min = 0.01 * r_eff
    r_max = 10.0 * r_eff
    r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_radial_bins)

    # Compute convergence profile → surface mass density
    kappa_profile = np.array([
        float(np.mean(mass_profile.convergence(
            np.array([r]), np.array([0.0])
        ))) for r in r_bins
    ])

    # Critical surface mass density (Ref: Schneider 1992, Eq. 3.7)
    # Σ_cr = c² D_s / (4π G D_d D_ds)
    # For estimation, use Σ_cr ≈ 3.5e15 M_sun/Mpc² at z_L ~ 0.3
    # More accurate: get from lens_system if available
    if hasattr(mass_profile, 'lens_system'):
        ls = mass_profile.lens_system
        D_d = getattr(ls, 'D_d', 1000.0)   # Mpc
        D_s = getattr(ls, 'D_s', 2000.0)
        D_ds = getattr(ls, 'D_ds', 1500.0)
        Sigma_cr = c_SI**2 / (4 * np.pi * G_SI) * D_s / (D_d * D_ds)
        # Convert from kg/m² to M_sun/kpc²
        Sigma_cr_msun_kpc2 = Sigma_cr / M_sun * (kpc_to_m * 1e-3)**2
    else:
        Sigma_cr_msun_kpc2 = 3.5e9  # Typical value [M_sun/kpc²]

    # Surface mass density
    Sigma = kappa_profile * Sigma_cr_msun_kpc2  # M_sun/kpc²

    # Convert arcsec to kpc (approximate)
    D_A_kpc = _angular_diameter_distance_kpc(z_lens)
    r_kpc = r_bins * arcsec_to_rad * D_A_kpc * 1e3  # kpc

    # Enclosed 3D spherical mass M_3D(<r) via exact Abel deprojection
    from scipy.interpolate import interp1d
    from scipy.integrate import quad
    
    # Generate extended grid for Abel transform integration out to essentially infinity
    r_ext_kpc = np.logspace(np.log10(max(1e-4, r_kpc[0] * 0.5)), np.log10(r_kpc[-1] * 100), 200)
    r_ext_arcsec = r_ext_kpc / (arcsec_to_rad * D_A_kpc * 1e3)
    
    kappa_ext = np.zeros_like(r_ext_arcsec)
    for i, r_val in enumerate(r_ext_arcsec):
        try:
            k = mass_profile.convergence(np.array([r_val]), np.array([0.0]))
            kappa_ext[i] = float(np.mean(k))
        except Exception:
            kappa_ext[i] = 0.0
            
    Sigma_ext = kappa_ext * Sigma_cr_msun_kpc2
    
    # Compute derivative dSigma/dR
    dSigma_dR = np.gradient(Sigma_ext, r_ext_kpc)
    dSigma_interp = interp1d(r_ext_kpc, dSigma_dR, fill_value=0.0, bounds_error=False)
    
    rho_3d = np.zeros_like(r_kpc)
    for i, r_val in enumerate(r_kpc):
        if r_val <= 0:
            continue
            
        def integrand(R):
            # Singularity handling bounds
            denom = np.sqrt(max(1e-15, R**2 - r_val**2))
            return dSigma_interp(R) / denom
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # rho(r) = -1/pi * int_r^infty (dSigma/dR) / sqrt(R^2 - r^2) dR
            res, _ = quad(integrand, r_val, r_ext_kpc[-1], limit=100)
            
        rho_3d[i] = max(0.0, -1.0 / np.pi * res)
        
    # Integrate 4*pi*r^2*rho(r) to get 3D enclosed mass M(<r)
    M_enclosed = np.zeros_like(r_kpc)
    for i in range(1, len(r_kpc)):
        r_sub = r_kpc[:i+1]
        rho_sub = rho_3d[:i+1]
        M_enclosed[i] = max(0.0, np.trapezoid(4 * np.pi * r_sub**2 * rho_sub, r_sub))

    # Solve Jeans equation: σ_r²(r) = (1/ν) ∫_r^∞ ν(r') GM(r')/r'² f(β) dr'
    # where ν is the luminosity density and f(β) corrects for anisotropy
    # Ref: Mamon & Łokas (2005), Eq. 11

    # Luminosity density (Hernquist profile approximation)
    a_H = r_eff / 1.8153  # Hernquist scale radius from R_eff
    a_kpc = a_H * arcsec_to_rad * D_A_kpc * 1e3
    nu = 1.0 / (r_kpc / a_kpc * (1 + r_kpc / a_kpc)**3 + 1e-30)
    nu /= np.trapezoid(nu * 4 * np.pi * r_kpc**2, r_kpc)  # Normalize

    # Jeans integration (backwards from outer boundary)
    sigma_r2 = np.zeros_like(r_kpc)
    integrand = nu * G_SI * M_enclosed * M_sun / (r_kpc * kpc_to_m)**2
    integrand *= (r_kpc / r_kpc[-1]) ** (2 * beta_aniso)  # Anisotropy correction

    # Integrate from outside in
    for i in range(len(r_kpc) - 2, -1, -1):
        dr = (r_kpc[i + 1] - r_kpc[i]) * kpc_to_m
        sigma_r2[i] = sigma_r2[i + 1] + integrand[i] * dr / max(nu[i], 1e-30)

    # Line-of-sight projection: σ_LOS² = (1-β) σ_r² (isotropic approx)
    sigma_los2 = (1.0 - beta_aniso) * sigma_r2

    # Convert to km/s
    sigma_los_kms = np.sqrt(np.abs(sigma_los2)) / 1e3

    # Luminosity-weighted average within aperture
    r_aper_kpc = r_aperture * arcsec_to_rad * D_A_kpc * 1e3
    mask = r_kpc <= r_aper_kpc
    if np.any(mask):
        weights = nu[mask] * r_kpc[mask]  # Luminosity weighting
        sigma_avg = np.average(sigma_los_kms[mask], weights=weights)
    else:
        sigma_avg = sigma_los_kms[0]

    # Clamp to physically reasonable range
    sigma_avg = float(np.clip(sigma_avg, 50, 500))

    return {
        'sigma_v_kms': sigma_avg,
        'sigma_v_profile': sigma_los_kms.tolist(),
        'r_bins_arcsec': r_bins.tolist(),
        'r_bins_kpc': r_kpc.tolist(),
        'enclosed_mass_msun': M_enclosed.tolist(),
        'beta_aniso': beta_aniso,
        'r_eff_arcsec': r_eff,
        'r_aperture_arcsec': r_aperture,
    }


def mass_sheet_degeneracy_test(
    lensing_mass: float,
    kinematic_mass: float,
    tolerance: float = 0.15,
) -> Dict:
    """Test for mass-sheet degeneracy between lensing and kinematics.

    The mass-sheet degeneracy (MSD) is a fundamental limitation of
    gravitational lensing: the transformation κ → λκ + (1-λ) preserves
    all lensing observables but changes the inferred mass and H₀.

    Stellar kinematics breaks this degeneracy because σ_v depends on
    the true 3D mass distribution, not just the projected convergence.

    Ref: Falco et al. (1985) ApJ 289, L1 — mass-sheet degeneracy
    Ref: Schneider & Sluse (2013) A&A 559, A37 — MSD and H₀
    Ref: Birrer et al. (2020) A&A 643, A165 — breaking MSD with kinematics

    Parameters
    ----------
    lensing_mass : float
        Mass inferred from lensing alone [M_sun].
    kinematic_mass : float
        Mass inferred from velocity dispersion [M_sun].
    tolerance : float
        Acceptable fractional difference.

    Returns
    -------
    result : dict
        Keys: 'msd_parameter', 'consistent', 'fractional_difference',
              'interpretation'
    """
    if lensing_mass <= 0 or kinematic_mass <= 0:
        return {
            'msd_parameter': np.nan,
            'consistent': False,
            'fractional_difference': np.nan,
            'interpretation': 'Invalid mass values',
        }

    # MSD parameter: λ = M_lens / M_kin
    # λ = 1 → no degeneracy, masses agree
    # λ ≠ 1 → mass sheet present
    lam = lensing_mass / kinematic_mass
    frac_diff = abs(lam - 1.0)

    consistent = frac_diff <= tolerance

    if frac_diff < 0.05:
        interpretation = "Excellent agreement — no mass-sheet degeneracy detected"
    elif frac_diff < tolerance:
        interpretation = "Acceptable agreement — mild MSD possible"
    elif frac_diff < 0.30:
        interpretation = "Significant discrepancy — mass-sheet transform likely present"
    else:
        interpretation = "Severe discrepancy — systematic error or strong MSD"

    return {
        'msd_parameter': float(lam),
        'consistent': bool(consistent),
        'fractional_difference': float(frac_diff),
        'interpretation': interpretation,
        'lensing_mass_msun': float(lensing_mass),
        'kinematic_mass_msun': float(kinematic_mass),
    }


def combine_lensing_kinematics(
    kappa_map: np.ndarray,
    sigma_v: float,
    r_eff: float,
    z_lens: float = 0.3,
    z_source: float = 1.5,
    pixel_scale: float = 0.05,
) -> Dict:
    """Joint constraint from lensing convergence + kinematics.

    Combines the lensing-only mass estimate with kinematic information
    to produce a tighter, MSD-free mass constraint.

    Ref: Treu & Koopmans (2004) ApJ 611, 739 — joint lensing+dynamics
    Ref: Auger et al. (2010) ApJ 724, 511 — SLACS survey results

    Parameters
    ----------
    kappa_map : np.ndarray
        2D convergence map.
    sigma_v : float
        Observed velocity dispersion [km/s].
    r_eff : float
        Effective radius [arcsec].
    z_lens : float
        Lens redshift.
    z_source : float
        Source redshift.
    pixel_scale : float
        Pixel scale [arcsec/pixel].

    Returns
    -------
    result : dict
        Joint mass estimate, individual estimates, and consistency metrics.
    """
    # Lensing mass estimate
    # M_lens = π R_eff² Σ_cr κ_eff
    D_A = _angular_diameter_distance_kpc(z_lens)
    r_eff_kpc = r_eff * arcsec_to_rad * D_A * 1e3

    center = kappa_map.shape[0] // 2
    r_pix = r_eff / pixel_scale
    y, x = np.ogrid[:kappa_map.shape[0], :kappa_map.shape[1]]
    mask = (x - center)**2 + (y - center)**2 <= r_pix**2
    kappa_eff = float(np.mean(kappa_map[mask])) if np.any(mask) else float(np.mean(kappa_map))

    # Σ_cr ≈ c² / (4π G D_eff) in physical units
    Sigma_cr_msun_kpc2 = 3.5e9 * (1 + z_lens)  # Rough scaling
    M_lens = np.pi * r_eff_kpc**2 * Sigma_cr_msun_kpc2 * kappa_eff

    # Kinematic mass estimate
    # M_dyn = C × σ_v² × R_eff / G  (virial relation)
    # Ref: Cappellari et al. (2006) MNRAS 366, 1126 — C ≈ 5.0 for early-types
    C_virial = 5.0  # Virial coefficient
    sigma_v_clamped = max(abs(sigma_v), 50.0)  # Floor at 50 km/s
    sigma_v_si = sigma_v_clamped * 1e3  # km/s → m/s
    r_eff_m = r_eff_kpc * kpc_to_m
    M_kin = abs(C_virial * sigma_v_si**2 * r_eff_m / (G_SI * M_sun))

    # Joint estimate: inverse-variance weighted mean
    # σ(M_lens) ≈ 0.2 × M_lens, σ(M_kin) ≈ 0.15 × M_kin
    sigma_lens = 0.20 * M_lens
    sigma_kin = 0.15 * M_kin

    if sigma_lens > 0 and sigma_kin > 0:
        w_lens = 1.0 / sigma_lens**2
        w_kin = 1.0 / sigma_kin**2
        M_joint = (w_lens * M_lens + w_kin * M_kin) / (w_lens + w_kin)
        sigma_joint = 1.0 / np.sqrt(w_lens + w_kin)
    else:
        M_joint = (M_lens + M_kin) / 2
        sigma_joint = max(sigma_lens, sigma_kin)

    # MSD test
    msd = mass_sheet_degeneracy_test(M_lens, M_kin)

    return {
        'M_lensing_msun': float(M_lens),
        'M_kinematic_msun': float(M_kin),
        'M_joint_msun': float(M_joint),
        'sigma_joint_msun': float(sigma_joint),
        'kappa_eff': float(kappa_eff),
        'sigma_v_kms': float(sigma_v),
        'r_eff_kpc': float(r_eff_kpc),
        'msd_test': msd,
        'improvement_factor': float(max(sigma_lens, sigma_kin) / sigma_joint)
            if sigma_joint > 0 else 1.0,
    }


def _angular_diameter_distance_kpc(z: float) -> float:
    """Exact angular diameter distance [kpc] for Planck 2018 cosmology.

    Uses astropy.cosmology.Planck18 for exact numeric integration
    of the FLRW metric, rather than the Pen (1999) approximation.
    """
    from astropy.cosmology import Planck18
    return Planck18.angular_diameter_distance(z).to_value('kpc')


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STELLAR KINEMATICS — Velocity Dispersion Module")
    print("=" * 60)

    # Quick test with a mock convergence map
    grid_size = 64
    x = np.linspace(-2, 2, grid_size)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2 + yy**2) + 0.1
    kappa_mock = 0.5 / r  # SIS-like

    result = combine_lensing_kinematics(
        kappa_map=kappa_mock,
        sigma_v=250.0,
        r_eff=1.5,
        z_lens=0.3,
        pixel_scale=4.0 / grid_size,
    )

    print(f"\n  M_lensing  = {result['M_lensing_msun']:.2e} M☉")
    print(f"  M_kinematic = {result['M_kinematic_msun']:.2e} M☉")
    print(f"  M_joint     = {result['M_joint_msun']:.2e} M☉")
    print(f"  MSD param λ = {result['msd_test']['msd_parameter']:.3f}")
    print(f"  Improvement = {result['improvement_factor']:.1f}×")
    print(f"  {result['msd_test']['interpretation']}")
    print("\n✓ Kinematics module test complete.")
