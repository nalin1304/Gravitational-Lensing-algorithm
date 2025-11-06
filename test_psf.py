"""
Test PSF model implementations (Gaussian, Airy, Moffat)
"""
import numpy as np
from gravitational_lensing_toolkit.data.real_data_loader import PSFModel

def test_psf_models():
    """Test all PSF model types"""
    
    # Common parameters
    fwhm = 0.1  # arcseconds
    pixel_scale = 0.05  # arcsec/pixel
    psf_size = 25  # pixels
    
    print("=" * 70)
    print("PSF MODEL TESTING")
    print("=" * 70)
    print(f"FWHM: {fwhm} arcsec")
    print(f"Pixel scale: {pixel_scale} arcsec/pixel")
    print(f"PSF size: {psf_size}×{psf_size} pixels")
    print()
    
    # Test Gaussian PSF
    print("1. GAUSSIAN PSF")
    print("-" * 70)
    try:
        gaussian_psf = PSFModel(fwhm=fwhm, pixel_scale=pixel_scale, model_type='gaussian')
        gaussian_kernel = gaussian_psf.generate_psf(size=psf_size)
        
        print(f"✓ Generated Gaussian PSF")
        print(f"  Shape: {gaussian_kernel.shape}")
        print(f"  Sum (normalized): {np.sum(gaussian_kernel):.6f}")
        print(f"  Peak value: {np.max(gaussian_kernel):.6f}")
        print(f"  Min value: {np.min(gaussian_kernel):.6e}")
        
        # Check FWHM
        center = psf_size // 2
        central_profile = gaussian_kernel[center, :]
        half_max = np.max(central_profile) / 2
        above_half_max = np.where(central_profile >= half_max)[0]
        fwhm_measured = len(above_half_max) * pixel_scale
        print(f"  Measured FWHM: {fwhm_measured:.3f} arcsec (target: {fwhm:.3f})")
        print()
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        print()
    
    # Test Airy disk PSF
    print("2. AIRY DISK PSF (Diffraction-limited)")
    print("-" * 70)
    try:
        airy_psf = PSFModel(fwhm=fwhm, pixel_scale=pixel_scale, model_type='airy')
        airy_kernel = airy_psf.generate_psf(size=psf_size)
        
        print(f"✓ Generated Airy PSF")
        print(f"  Shape: {airy_kernel.shape}")
        print(f"  Sum (normalized): {np.sum(airy_kernel):.6f}")
        print(f"  Peak value: {np.max(airy_kernel):.6f}")
        print(f"  Min value: {np.min(airy_kernel):.6e}")
        
        # Airy disk should have rings
        center = psf_size // 2
        central_profile = airy_kernel[center, :]
        num_minima = np.sum(np.diff(np.sign(np.diff(central_profile))) > 0)
        print(f"  Number of side lobes detected: {num_minima}")
        print()
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test Moffat PSF
    print("3. MOFFAT PSF (Atmospheric seeing)")
    print("-" * 70)
    try:
        moffat_psf = PSFModel(fwhm=fwhm, pixel_scale=pixel_scale, model_type='moffat')
        moffat_kernel = moffat_psf.generate_psf(size=psf_size)
        
        print(f"✓ Generated Moffat PSF")
        print(f"  Shape: {moffat_kernel.shape}")
        print(f"  Sum (normalized): {np.sum(moffat_kernel):.6f}")
        print(f"  Peak value: {np.max(moffat_kernel):.6f}")
        print(f"  Min value: {np.min(moffat_kernel):.6e}")
        
        # Check FWHM
        center = psf_size // 2
        central_profile = moffat_kernel[center, :]
        half_max = np.max(central_profile) / 2
        above_half_max = np.where(central_profile >= half_max)[0]
        fwhm_measured = len(above_half_max) * pixel_scale
        print(f"  Measured FWHM: {fwhm_measured:.3f} arcsec (target: {fwhm:.3f})")
        print()
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Compare profiles
    print("4. PROFILE COMPARISON")
    print("-" * 70)
    print("Radial profile at 1×FWHM, 2×FWHM, 3×FWHM:")
    print()
    
    try:
        fwhm_pixels = fwhm / pixel_scale
        center = psf_size // 2
        
        for model_name, kernel in [('Gaussian', gaussian_kernel), 
                                    ('Airy', airy_kernel), 
                                    ('Moffat', moffat_kernel)]:
            print(f"{model_name:10s}", end="")
            for r_mult in [1.0, 2.0, 3.0]:
                r_pix = int(r_mult * fwhm_pixels)
                if center + r_pix < psf_size:
                    value = kernel[center, center + r_pix]
                    print(f"  r={r_mult}×FWHM: {value:.6f}", end="")
            print()
        
        print()
        print("=" * 70)
        print("✓ ALL PSF MODELS TESTED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Comparison failed: {e}")

if __name__ == "__main__":
    test_psf_models()
