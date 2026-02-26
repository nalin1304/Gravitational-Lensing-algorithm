import numpy as np
from typing import List, Callable, Dict

class SEDMorphologyJointLikelihood:
    """
    Joint likelihood module for multi-band gravitational lensing data.
    Enforces a strict astrophysical constraint: 
    The source galaxy has a single, shared spatial morphology across all bands,
    and the difference entirely governed by a set of flux parameters dictated
    by the Spectral Energy Distribution (SED).
    
    This breaks the severe degeneracies in multi-band lens modeling.
    """
    
    def __init__(self, num_bands: int):
        """
        :param num_bands: The number of imaging bands being jointly modeled (e.g., 3 for F475W, F814W, F160W).
        """
        self.num_bands = num_bands
        
    def compute_joint_log_likelihood(
        self,
        images: List[np.ndarray],
        noises: List[np.ndarray],
        lens_operators: List[Callable[[np.ndarray], np.ndarray]],
        shared_morphology: np.ndarray,
        sed_amplitudes: np.ndarray
    ) -> float:
        """
        Computes the log-likelihood over all bands simultaneously.
        
        :param images: List of multi-band target observations.
        :param noises: List of noise maps (sigma) per band.
        :param lens_operators: List of forward models (lens equation + PSF + rendering) specifically tuned for each band's resolution/PSF.
        :param shared_morphology: The single inherent unlensed source morphology (e.g., Starlet reconstructed array or Sersic evaluation).
        :param sed_amplitudes: Array length `num_bands`. The relative flux scaling for each band derived from SED parameters.
        :return: Total joint Log-Likelihood.
        """
        if len(images) != self.num_bands or len(lens_operators) != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} bands, got mismatch in inputs.")
            
        total_log_likelihood = 0.0
        
        for i in range(self.num_bands):
            # 1. Scale the shared, normalized morphology by the physical SED amplitude for this band
            source_band_i = shared_morphology * sed_amplitudes[i]
            
            # 2. Forward model the source through the lens equation (which is specific to this band's PSF)
            model_image_i = lens_operators[i](source_band_i)
            
            # 3. Compute Gaussian log-likelihood for this band
            chi2 = np.sum(((images[i] - model_image_i) / noises[i])**2)
            
            # Log likelihood component
            ll_i = -0.5 * chi2 - 0.5 * np.sum(np.log(2.0 * np.pi * noises[i]**2))
            
            total_log_likelihood += ll_i
            
        return total_log_likelihood

    def optimize_sed_amplitudes_linear(
        self,
        images: List[np.ndarray],
        noises: List[np.ndarray],
        lens_operators: List[Callable[[np.ndarray], np.ndarray]],
        shared_morphology: np.ndarray
    ) -> np.ndarray:
        """
        Given a fixed shared morphology, computes the exact linear Maximum Likelihood (ML) 
        estimators for the SED amplitudes in each band independent of each other 
        (since bands don't mix linearly with each other in the image plane).
        
        :return: Array of optimal SED amplitudes [A_0, A_1, ..., A_N].
        """
        optimal_amplitudes = np.zeros(self.num_bands)
        
        for i in range(self.num_bands):
            # Base response of the unit morphology in the data plane
            base_model_i = lens_operators[i](shared_morphology)
            
            # Using variance-weighted Ordinary Least Squares for A_i
            # Model = A_i * base_model_i
            # chi2 = sum [ (img - A_i * base_model_i)^2 / noise^2 ]
            # dchi2/dA_i = -2 * sum [ base_model_i * (img - A_i * base_model_i) / noise^2 ] = 0
            # A_i = sum(img * base_model_i / noise^2) / sum(base_model_i^2 / noise^2)
            
            variance = noises[i] ** 2
            
            numerator = np.sum((images[i] * base_model_i) / variance)
            denominator = np.sum((base_model_i ** 2) / variance)
            
            if denominator > 0:
                optimal_amplitudes[i] = numerator / denominator
            else:
                optimal_amplitudes[i] = 0.0
                
            # Restrict to strictly positive astrophysical fluxes
            if optimal_amplitudes[i] < 0:
                optimal_amplitudes[i] = 0.0
                
        return optimal_amplitudes
