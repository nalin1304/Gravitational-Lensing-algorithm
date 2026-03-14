import pytest
import numpy as np
from src.ml.multiband_sed import SEDMorphologyJointLikelihood

def test_multiband_joint_likelihood():
    """Verify that multiple bands with the exact same morphology yield expected log-likelihood scalings."""
    num_bands = 3
    joint = SEDMorphologyJointLikelihood(num_bands=num_bands)
    
    # 10x10 shared morphology (just a blob in the center)
    shared_morphology = np.zeros((10, 10))
    shared_morphology[4:6, 4:6] = 1.0
    
    # Let's say we have 3 bands: F475W, F814W, F160W
    sed_amplitudes = np.array([0.5, 2.0, 10.0])
    
    # Dummy mock lens operators (just identity mapping for this test)
    lens_operators = [
        lambda x: x * 1.0,
        lambda x: x * 1.0,
        lambda x: x * 1.0,
    ]
    
    # True images without noise
    images = [
        shared_morphology * sed_amplitudes[0],
        shared_morphology * sed_amplitudes[1],
        shared_morphology * sed_amplitudes[2]
    ]
    
    # Small noise for testing
    noises = [np.ones((10, 10)) * 0.1 for _ in range(num_bands)]
    
    # The Log-Likelihood of perfect data should just be the constant normalization term
    ll = joint.compute_joint_log_likelihood(
        images=images,
        noises=noises,
        lens_operators=lens_operators,
        shared_morphology=shared_morphology,
        sed_amplitudes=sed_amplitudes
    )
    
    # 3 bands * 100 pixels = 300 pixels
    # LL = - 0.5 * chi^2 - 0.5 * sum(log(2pi sigma^2))
    # chi^2 is 0 here
    expected_normalization = -0.5 * 300 * np.log(2.0 * np.pi * 0.1**2)
    np.testing.assert_allclose(ll, expected_normalization, rtol=1e-5)
    

def test_optimize_sed_amplitudes_linear():
    """Verify that the analytical linear ML estimator recovers the exact SED amplitudes from imaging."""
    num_bands = 2
    joint = SEDMorphologyJointLikelihood(num_bands=num_bands)
    
    shared_morphology = np.zeros((5, 5))
    shared_morphology[2, 2] = 5.0
    
    true_amplitudes = np.array([3.0, 7.5])
    
    lens_operators = [
        lambda x: x,
        lambda x: x
    ]
    
    images = [
        shared_morphology * true_amplitudes[0],
        shared_morphology * true_amplitudes[1]
    ]
    
    noises = [np.ones((5, 5)) * 1.0, np.ones((5, 5)) * 2.0]
    
    recovered_amplitudes = joint.optimize_sed_amplitudes_linear(
        images=images,
        noises=noises,
        lens_operators=lens_operators,
        shared_morphology=shared_morphology
    )
    
    np.testing.assert_allclose(recovered_amplitudes, true_amplitudes, rtol=1e-5)

