"""
Test multi-plane gravitational lensing implementation.
"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from lens_models.multi_plane import MultiPlaneLens, LensPlane
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from lens_models.mass_profiles import NFWProfile
from lens_models.lens_system import LensSystem

def test_multiplane_lensing():
    """Test multi-plane lensing system"""
    
    print("=" * 70)
    print("MULTI-PLANE LENSING TEST")
    print("=" * 70)
    print()
    
    # Create cosmology
    print("1. Setting up cosmology (Planck 2018)")
    print("-" * 70)
    cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
    print(f"[OK] H0 = {cosmo.H0}")
    print(f"[OK] Omega_m = {cosmo.Om0}")
    print(f"[OK] Omega_Lambda = {1 - cosmo.Om0}")
    print()
    
    # Create multi-plane lens
    print("2. Creating multi-plane lens system")
    print("-" * 70)
    source_z = 2.0
    lens = MultiPlaneLens(source_redshift=source_z, cosmology=cosmo)
    print(f"[OK] Source redshift: z = {source_z}")
    print(f"[OK] Source distance: Ds = {lens.Ds:.1f} Mpc")
    print()
    
    # Add lens planes
    print("3. Adding lens planes")
    print("-" * 70)
    
    # Foreground perturber at z=0.3
    z1 = 0.3
    M1 = 1e13  # Solar masses
    lens_sys1 = LensSystem(z_lens=z1, z_source=source_z, H0=67.4, Om0=0.315)
    nfw1 = NFWProfile(M_vir=M1, concentration=5, lens_system=lens_sys1)
    lens.add_plane(redshift=z1, profile=nfw1, center=(5.0, 0.0))
    print(f"[OK] Plane 1: z={z1}, M={M1:.1e} Msun, NFW c=5, center=(5.0, 0.0) arcsec")
    
    # Main cluster at z=0.5
    z2 = 0.5
    M2 = 5e14  # Solar masses
    lens_sys2 = LensSystem(z_lens=z2, z_source=source_z, H0=67.4, Om0=0.315)
    nfw2 = NFWProfile(M_vir=M2, concentration=4, lens_system=lens_sys2)
    lens.add_plane(redshift=z2, profile=nfw2, center=(0.0, 0.0))
    print(f"[OK] Plane 2: z={z2}, M={M2:.1e} Msun, NFW c=4, center=(0.0, 0.0) arcsec")
    print()
    
    # Get summary
    print("4. System summary")
    print("-" * 70)
    summary = lens.summary()
    for plane_info in summary['planes']:
        print(f"Plane {plane_info['index']}:")
        print(f"  Redshift: z = {plane_info['redshift']}")
        print(f"  Center: {plane_info['center']}")
        print(f"  Distance: Dd = {plane_info['Dd']:.1f} Mpc")
        print(f"  Dd-source: Dds = {plane_info['Dds']:.1f} Mpc")
        print(f"  Weight: {plane_info['weight']:.4f}")
        print(f"  Profile: {plane_info['profile_type']}")
        print()
    
    # Test ray tracing
    print("5. Ray tracing test")
    print("-" * 70)
    
    # Test positions
    test_positions = np.array([
        [0.0, 0.0],    # Center
        [2.0, 0.0],    # Near main lens
        [5.0, 0.0],    # Near perturber
        [10.0, 10.0]   # Far field
    ])
    
    print("Testing ray tracing at different positions:")
    print()
    for i, theta in enumerate(test_positions):
        beta = lens.ray_trace(theta)
        alpha_eff = lens.effective_deflection(theta)
        
        print(f"Position {i+1}: θ = ({theta[0]:.1f}, {theta[1]:.1f}) arcsec")
        print(f"  Source position: β = ({beta[0]:.4f}, {beta[1]:.4f}) arcsec")
        print(f"  Deflection: α = ({alpha_eff[0]:.4f}, {alpha_eff[1]:.4f}) arcsec")
        print(f"  |Deflection|: {np.sqrt(alpha_eff[0]**2 + alpha_eff[1]**2):.4f} arcsec")
        print()
    
    # Test intermediate positions
    print("6. Intermediate plane positions")
    print("-" * 70)
    theta_test = np.array([2.0, 0.0])
    positions = lens.ray_trace(theta_test, return_intermediate=True)
    
    print(f"Ray tracing from θ = ({theta_test[0]:.1f}, {theta_test[1]:.1f}) arcsec")
    print()
    # Extract single point from each position
    pos0 = positions[0] if positions[0].ndim == 1 else positions[0][0]
    print(f"Image plane:   ({pos0[0]:.4f}, {pos0[1]:.4f}) arcsec")
    for i in range(len(lens.planes)):
        pos = positions[i+1] if positions[i+1].ndim == 1 else positions[i+1][0]
        print(f"After plane {i+1}:  ({pos[0]:.4f}, {pos[1]:.4f}) arcsec")
    print()
    
    # Test convergence map
    print("7. Convergence map")
    print("-" * 70)
    try:
        kappa = lens.convergence_map(image_size=128, fov=30.0)
        print(f"[OK] Generated convergence map: {kappa.shape}")
        print(f"  Min kappa: {np.min(kappa):.6f}")
        print(f"  Max kappa: {np.max(kappa):.6f}")
        print(f"  Mean kappa: {np.mean(kappa):.6f}")
        print()
    except Exception as e:
        print(f"[FAIL] Convergence map failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test magnification map
    print("8. Magnification map")
    print("-" * 70)
    try:
        mu = lens.magnification_map(image_size=128, fov=30.0)
        print(f"[OK] Generated magnification map: {mu.shape}")
        print(f"  Min mu: {np.min(mu):.4f}")
        print(f"  Max mu: {np.max(mu):.4f}")
        print(f"  Mean mu: {np.mean(mu):.4f}")
        
        # Find high magnification regions
        high_mag = np.sum(np.abs(mu) > 5)
        print(f"  Pixels with |mu| > 5: {high_mag} ({100*high_mag/mu.size:.2f}%)")
        print()
    except Exception as e:
        print(f"[FAIL] Magnification map failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test critical curves
    print("9. Critical curves")
    print("-" * 70)
    try:
        critical = lens.critical_curves(image_size=128, fov=30.0, threshold=10.0)
        n_critical = np.sum(critical)
        print(f"[OK] Found critical curve pixels: {n_critical}")
        print(f"  Fraction of FOV: {100*n_critical/critical.size:.2f}%")
        print()
    except Exception as e:
        print(f"[FAIL] Critical curve finding failed: {e}")
        print()
    
    print("=" * 70)
    print("[SUCCESS] MULTI-PLANE LENSING TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_multiplane_lensing()
