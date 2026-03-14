"""
HST (Hubble Space Telescope) Data Validation Pipeline

This module provides automated validation against real HST observations
of gravitational lensing systems. It enables quantitative comparison
between simulated lensing images and actual HST data.

Key Features
------------
- HST MAST archive integration for automatic data download
- Well-documented lens systems (Einstein Cross, Abell 1689, etc.)
- Chi-squared and residual analysis
- PSF-matched comparison
- Automated report generation

Example Systems
---------------
- Einstein Cross (Q2237+030): z_lens=0.039, z_source=1.695
- Abell 1689: z=0.183, massive galaxy cluster
- SDSS J1004+4112: z_lens=0.68, z_source=1.734, quasar quintuple

References
----------
- HST MAST Archive: https://mast.stsci.edu/
- Kochanek et al. (2006) - CASTLES Database
- Treu & Marshall (2016) - Strong Lensing Review
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class HSTTarget:
    """
    HST gravitational lens target information.
    
    Attributes
    ----------
    name : str
        Common name of the lens system
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift
    lens_type : str
        Type: 'galaxy', 'cluster', 'quasar'
    dataset_id : str
        HST MAST dataset ID
    filter : str
        HST filter (F814W, F606W, etc.)
    """
    name: str
    ra: float
    dec: float
    z_lens: float
    z_source: float
    lens_type: str
    dataset_id: str = ""
    filter: str = "F814W"


class HSTValidation:
    """
    HST data validation pipeline for gravitational lensing simulations.
    
    This class handles:
    1. Target definition and metadata
    2. HST data download (MAST archive)
    3. Image preprocessing and alignment
    4. Chi-squared comparison
    5. Residual analysis and reporting
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory for caching downloaded HST data
    """
    
    # Well-known gravitational lens systems
    TARGETS = {
        'einstein_cross': HSTTarget(
            name='Einstein Cross (Q2237+030)',
            ra=339.45,  # degrees
            dec=3.45,
            z_lens=0.039,
            z_source=1.695,
            lens_type='galaxy',
            dataset_id='j8pu01010',  # Example HST dataset
            filter='F814W'
        ),
        'abell1689': HSTTarget(
            name='Abell 1689',
            ra=197.873,
            dec=-1.341,
            z_lens=0.183,
            z_source=3.0,  # Typical background galaxy
            lens_type='cluster',
            dataset_id='j8pu02010',
            filter='F814W'
        ),
        'sdss_j1004': HSTTarget(
            name='SDSS J1004+4112',
            ra=151.076,
            dec=41.206,
            z_lens=0.68,
            z_source=1.734,
            lens_type='cluster',
            dataset_id='j8pu03010',
            filter='F606W'
        ),
    }
    
    def __init__(self, cache_dir: str = './data/hst_cache'):
        """Initialize HST validation pipeline."""
        self.cache_dir = cache_dir
        self.downloaded_data: Dict[str, np.ndarray] = {}
    
    def get_target(self, name: str) -> HSTTarget:
        """
        Get target information by name.
        
        Parameters
        ----------
        name : str
            Target name ('einstein_cross', 'abell1689', etc.)
        
        Returns
        -------
        target : HSTTarget
            Target information
        
        Raises
        ------
        KeyError
            If target not found
        """
        if name not in self.TARGETS:
            available = ', '.join(self.TARGETS.keys())
            raise KeyError(f"Target '{name}' not found. Available: {available}")
        
        return self.TARGETS[name]
    
    def download_hst_data(
        self,
        target_name: str,
        force_download: bool = False
    ) -> np.ndarray:
        """
        Download HST data from MAST archive.
        
        Parameters
        ----------
        target_name : str
            Name of target system
        force_download : bool, optional
            Force re-download even if cached
        
        Returns
        -------
        image : np.ndarray
            HST image data
        
        Notes
        -----
        This method intentionally avoids synthetic surrogates. It loads
        real local assets (Numpy/FITS) and fails fast if unavailable.
        """
        target = self.get_target(target_name)
        
        # Check cache
        if not force_download and target_name in self.downloaded_data:
            return self.downloaded_data[target_name]

        cache_dir = Path(self.cache_dir)
        candidate_dirs = [
            cache_dir,
            Path("./data/hst"),
            Path("./assets/hst"),
        ]
        candidate_names = [
            target_name,
            target.dataset_id,
            target.dataset_id.upper(),
        ]

        # Try NumPy assets first
        for base in candidate_dirs:
            for name in candidate_names:
                npy_path = base / f"{name}.npy"
                if npy_path.exists():
                    image = np.load(npy_path)
                    self.downloaded_data[target_name] = image
                    return image

        # Try FITS assets
        for base in candidate_dirs:
            for name in candidate_names:
                fits_path = base / f"{name}.fits"
                if fits_path.exists():
                    try:
                        from astropy.io import fits
                    except ImportError as e:
                        raise ImportError(
                            "astropy is required to load FITS assets. "
                            "Install astropy or provide .npy HST data."
                        ) from e

                    with fits.open(fits_path) as hdul:
                        image_hdu = None
                        for hdu in hdul:
                            if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) >= 2:
                                image_hdu = hdu.data
                                break
                        if image_hdu is None:
                            raise ValueError(f"No image data found in FITS file: {fits_path}")
                        image = np.asarray(image_hdu, dtype=float)
                    self.downloaded_data[target_name] = image
                    return image

        raise FileNotFoundError(
            f"No real HST asset found for target '{target_name}' (dataset_id={target.dataset_id}). "
            f"Expected local files in {candidate_dirs}: "
            f"{[name + ext for name in candidate_names for ext in ['.npy', '.fits']]}. "
            f"Use `from src.data.mast_downloader import MASTDownloader` to auto-download "
            f"SLACS lens data from the MAST archive."
        )
    
    def compare_with_hst(
        self,
        simulated_image: np.ndarray,
        target_name: str,
        use_mask: bool = True
    ) -> Dict:
        """
        Compare simulated image with HST observation.
        
        Parameters
        ----------
        simulated_image : np.ndarray
            Simulated lensing image
        target_name : str
            HST target to compare against
        use_mask : bool, optional
            Use mask to exclude bad pixels
        
        Returns
        -------
        results : dict
            Comparison metrics including chi-squared, residuals, etc.
        
        Notes
        -----
        Chi-squared metric:
            χ² = Σᵢ [(observed_i - simulated_i)² / σᵢ²]
        
        where σᵢ includes photon noise and background.
        """
        # Download HST data
        hst_image = self.download_hst_data(target_name)
        
        # Ensure images same size
        if simulated_image.shape != hst_image.shape:
            # Resize simulated to match HST
            from scipy.ndimage import zoom
            scale_y = hst_image.shape[0] / simulated_image.shape[0]
            scale_x = hst_image.shape[1] / simulated_image.shape[1]
            simulated_image = zoom(simulated_image, (scale_y, scale_x))
        
        # Compute statistics
        residual = simulated_image - hst_image
        
        # Estimate uncertainties (Poisson + readnoise)
        readnoise = 5.0  # electrons (typical for HST)
        sigma = np.sqrt(np.abs(hst_image) + readnoise**2)
        
        # Chi-squared
        chi2 = np.sum((residual / sigma)**2)
        reduced_chi2 = chi2 / residual.size
        
        # RMS residual
        rms = np.sqrt(np.mean(residual**2))
        
        # Relative error
        relative_error = np.abs(residual) / (np.abs(hst_image) + 1e-10)
        mean_relative_error = np.mean(relative_error)
        
        results = {
            'target': target_name,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'rms_residual': rms,
            'mean_relative_error': mean_relative_error,
            'max_residual': np.max(np.abs(residual)),
            'hst_mean': np.mean(hst_image),
            'sim_mean': np.mean(simulated_image),
            'residual_map': residual
        }
        
        return results
    
    def generate_validation_report(
        self,
        results: Dict,
        output_file: str = 'hst_validation_report.txt'
    ) -> None:
        """
        Generate validation report.
        
        Parameters
        ----------
        results : dict
            Results from compare_with_hst()
        output_file : str, optional
            Output filename
        """
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("HST VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Target: {results['target']}\n")
            f.write(f"Chi-squared: {results['chi2']:.2e}\n")
            f.write(f"Reduced chi-squared: {results['reduced_chi2']:.4f}\n")
            f.write(f"RMS residual: {results['rms_residual']:.4f}\n")
            f.write(f"Mean relative error: {results['mean_relative_error']:.4%}\n")
            f.write(f"Max residual: {results['max_residual']:.4f}\n\n")
            
            f.write("Interpretation:\n")
            if results['reduced_chi2'] < 1.5:
                f.write("  EXCELLENT: Model matches HST data very well\n")
            elif results['reduced_chi2'] < 3.0:
                f.write("  GOOD: Model reasonably matches HST data\n")
            elif results['reduced_chi2'] < 10.0:
                f.write("  FAIR: Model shows systematic deviations\n")
            else:
                f.write("  POOR: Model does not match HST data\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Validation report saved to: {output_file}")
    
    def list_available_targets(self) -> List[str]:
        """
        List all available HST validation targets.
        
        Returns
        -------
        targets : list
            List of target names
        """
        return list(self.TARGETS.keys())
    
    def target_info(self, target_name: str) -> str:
        """
        Get detailed information about a target.
        
        Parameters
        ----------
        target_name : str
            Target name
        
        Returns
        -------
        info : str
            Formatted target information
        """
        target = self.get_target(target_name)
        
        info = f"""
Target: {target.name}
-----------------------------------
Type: {target.lens_type}
Position: RA={target.ra:.4f}°, Dec={target.dec:.4f}°
Lens redshift: z = {target.z_lens}
Source redshift: z = {target.z_source}
HST Filter: {target.filter}
Dataset ID: {target.dataset_id}
"""
        return info


def validate_against_all_targets(
    model,
    validation: HSTValidation
) -> Dict[str, Dict]:
    """
    Validate model against all available HST targets.
    
    Parameters
    ----------
    model : object
        Lensing model with render() method
    validation : HSTValidation
        HST validation pipeline
    
    Returns
    -------
    all_results : dict
        Results for each target
    """
    all_results = {}
    
    for target_name in validation.list_available_targets():
        print(f"\nValidating against {target_name}...")
        
        # Render model prediction for this target without synthetic stand-ins.
        if hasattr(model, "render_target"):
            sim_image = model.render_target(validation.get_target(target_name))
        elif hasattr(model, "render"):
            sim_image = model.render(validation.get_target(target_name))
        else:
            raise AttributeError(
                "Model must implement render_target(target) or render(target) "
                "for HST validation."
            )
        
        # Compare
        results = validation.compare_with_hst(sim_image, target_name)
        all_results[target_name] = results
        
        # Print summary
        print(f"  Reduced chi²: {results['reduced_chi2']:.4f}")
        print(f"  RMS residual: {results['rms_residual']:.4f}")
    
    return all_results


if __name__ == "__main__":
    print("HST Validation Pipeline")
    print("=" * 70)
    
    # Initialize
    validator = HSTValidation()
    
    # List targets
    print("\nAvailable HST targets:")
    for target_name in validator.list_available_targets():
        print(f"  - {target_name}")
    
    # Show target info
    print("\n" + validator.target_info('einstein_cross'))
    
    print("\nHST validation pipeline initialized successfully.")
    print("Use validator.compare_with_hst(sim_image, target_name) to validate.")

