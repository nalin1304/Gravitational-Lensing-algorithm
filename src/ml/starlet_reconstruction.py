import numpy as np
from scipy.signal import convolve2d

class StarletTransform:
    """
    Implements the Isotropic Undecimated Wavelet Transform (Starlet).
    Used for multi-scale, sparsity-enforcing morphological reconstruction of lensed sources.
    Uses the B-spline of degree 3 (a trous algorithm) for separation.
    """
    
    def __init__(self, num_scales: int = 4):
        """
        Initializes the Starlet Transform.
        :param num_scales: Number of wavelet scales to extract.
        """
        self.num_scales = num_scales
        # 1D B-spline filter: [1, 4, 6, 4, 1] / 16
        h1d = np.array([1, 4, 6, 4, 1]) / 16.0
        # 2D B-spline filter
        self.h2d = np.outer(h1d, h1d)
        
    def _expand_filter(self, scale: int) -> np.ndarray:
        """
        Dilates the generating filter `h2d` by inserting 2^scale - 1 zeros between coefficients (a trous).
        """
        if scale == 0:
            return self.h2d
            
        step = 2**scale
        size = 4 * step + 1
        h_dilated = np.zeros((size, size))
        
        # Insert coefficients
        for i in range(5):
            for j in range(5):
                h_dilated[i*step, j*step] = self.h2d[i, j]
                
        return h_dilated
        
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Decomposes the image into `num_scales` wavelet scales + 1 coarse scale.
        
        :param image: 2D numpy array containing the flux image.
        :return: 3D numpy array of shape (num_scales + 1, height, width).
        """
        h, w = image.shape
        coeffs = np.zeros((self.num_scales + 1, h, w))
        
        c_j = image.copy()
        
        for j in range(self.num_scales):
            # Compute smoothed version
            h_j = self._expand_filter(j)
            
            # Convolve (using 'symm' boundary conditions via 'symm' padding emulation)
            # scipy's convolve2d with mode='same' and boundary='symm' handles it well
            c_j_plus_1 = convolve2d(c_j, h_j, mode='same', boundary='symm')
            
            # Wavelet coefficient is the high-frequency detail difference
            w_j = c_j - c_j_plus_1
            coeffs[j] = w_j
            
            # Prepare for next scale
            c_j = c_j_plus_1
            
        # The remaining low-frequency background
        coeffs[-1] = c_j
        
        return coeffs
        
    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstructs the image from its Starlet coefficients.
        Since it's an undecimated transform, reconstruction is simple addition.
        
        :param coeffs: 3D array of wavelet coefficients + coarse background.
        :return: 2D reconstructed image.
        """
        return np.sum(coeffs, axis=0)
        
    def apply_sparsity_threshold(self, coeffs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Applies soft-thresholding to the Starlet coefficients (typically used in FISTA for L1 norm).
        
        :param coeffs: 3D array of coefficients.
        :param thresholds: 1D array of sigma thresholds per scale (length num_scales). Coarse scale is usually untouched.
        :return: Thresholded 3D coefficients.
        """
        thresholded = np.zeros_like(coeffs)
        
        # Soft threshold highest-frequency wavelet scales
        for j in range(self.num_scales):
            tau = thresholds[j]
            # Soft thresholding: sign(w) * max(|w| - tau, 0)
            thresholded[j] = np.sign(coeffs[j]) * np.maximum(np.abs(coeffs[j]) - tau, 0)
            
        # Leave coarse scale untouched (no sparsity prior on the broad background)
        thresholded[-1] = coeffs[-1]
        
        return thresholded
        
    def solve_sparse_source_fista(
        self, 
        observed_image: np.ndarray, 
        forward_operator: callable, 
        adjoint_operator: callable, 
        thresholds: np.ndarray, 
        max_iter: int = 150, 
        learning_rate: float = 1.0
    ) -> np.ndarray:
        """
        Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) to solve the L1-regularized inverse problem:
        w_opt = argmin_w 0.5 * ||y - H S(w)||_2^2 + lambda ||w||_1
        
        where:
        - w are the starlet coefficients
        - S is the inverse starlet transform (w -> source image)
        - H is the forward lensing operator (source image -> lensed image)
        - y is the observed image
        
        :param observed_image: The target lensed image.
        :param forward_operator: Function mapping source_image -> lensed_image.
        :param adjoint_operator: Function mapping lensed_image -> source_image (gradient).
        :param thresholds: Array of L1 regularization strengths per scale.
        :param max_iter: Maximum number of iterations.
        :param learning_rate: Gradient descent step size.
        :return: Optimized source image reconstructed from thresholded wavelets.
        """
        h, w = observed_image.shape
        
        # Initialize coefficients
        x = np.zeros((self.num_scales + 1, h, w))
        y = np.copy(x)
        t = 1.0
        
        for i in range(max_iter):
            # 1. Forward pass (evaluate current source at data plane)
            current_source = self.inverse_transform(y)
            pred_image = forward_operator(current_source)
            
            # 2. Gradient of data fidelity term: H^T (H x - y)
            residual = pred_image - observed_image
            grad_source = adjoint_operator(residual)
            
            # Project gradient back to wavelet space
            grad_coeffs = self.transform(grad_source)
            
            # 3. Gradient descent step
            x_next = y - learning_rate * grad_coeffs
            
            # 4. Proximal operator (Soft Thresholding)
            x_next = self.apply_sparsity_threshold(x_next, thresholds * learning_rate)
            
            # 5. FISTA Momentum update
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            
            x = x_next
            t = t_next
            
        return self.inverse_transform(x)
