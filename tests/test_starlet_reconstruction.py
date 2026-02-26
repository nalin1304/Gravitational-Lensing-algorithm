import pytest
import numpy as np
from src.ml.starlet_reconstruction import StarletTransform

def test_starlet_transform_identity():
    """Verify that Starlet Transform reconstruction perfectly recovers the original image."""
    np.random.seed(42)
    # 32x32 random image
    image = np.random.randn(32, 32)
    
    starlet = StarletTransform(num_scales=3)
    
    # Decompose
    coeffs = starlet.transform(image)
    assert coeffs.shape == (4, 32, 32) # (scales+1, H, W)
    
    # Reconstruct
    rec_image = starlet.inverse_transform(coeffs)
    
    # Identity constraint (undecimated wavelets should preserve flux exactly)
    np.testing.assert_allclose(rec_image, image, atol=1e-10)

def test_apply_sparsity_threshold():
    """Verify that thresholding successfully zeroes out small coefficients."""
    starlet = StarletTransform(num_scales=2)
    
    # Fake coefficients (2 scales + 1 coarse)
    coeffs = np.ones((3, 10, 10))
    coeffs[0] = 0.5 # High freq scale 1
    coeffs[1] = 2.0 # Mid freq scale 2
    coeffs[2] = 5.0 # Coarse scale
    
    # Thresholds
    thresholds = np.array([1.0, 1.0])
    
    thresholded = starlet.apply_sparsity_threshold(coeffs, thresholds)
    
    # Scale 0 should be stripped entirely to 0
    np.testing.assert_allclose(thresholded[0], 0.0)
    # Scale 1 should be shrunk by 1.0
    np.testing.assert_allclose(thresholded[1], 1.0)
    # Scale 2 (coarse) should be untouched
    np.testing.assert_allclose(thresholded[2], 5.0)

def test_solve_sparse_source_fista():
    """Verify that the FISTA solver successfully reconstructs a target under identity lensing."""
    starlet = StarletTransform(num_scales=2)
    
    # Target image we want to reconstruct
    original = np.zeros((20, 20))
    original[10, 10] = 100.0 # Point source
    
    # Noisier observation
    np.random.seed(0)
    obs = original + np.random.normal(0, 1.0, (20, 20))
    
    # Mock specific identity lensing operators
    def forward_operator(x): return x
    def adjoint_operator(res): return res
    
    thresholds = np.array([5.0, 5.0])
    
    rec_source = starlet.solve_sparse_source_fista(
        observed_image=obs,
        forward_operator=forward_operator,
        adjoint_operator=adjoint_operator,
        thresholds=thresholds,
        max_iter=50,
        learning_rate=0.5
    )
    
    # L1 norm should suppress the background noise effectively
    residual_noise_std = np.std(rec_source[0:5, 0:5])
    assert residual_noise_std < 1.0 # Should suppress noise cleanly below raw 1.0 std
    assert np.max(rec_source) > 50.0 # Point source should survive
