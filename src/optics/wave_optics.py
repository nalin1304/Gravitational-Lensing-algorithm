"""
Wave Optics Engine for Gravitational Lensing

This module implements physical optics calculations including diffraction
and interference effects in gravitational lensing, extending beyond the
geometric optics approximation.

Wave optics is important when:
- Wavelength comparable to Schwarzschild radius
- Interference fringes between multiple images
- Chromatic effects in lensing
"""

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    import numpy as np
    jnp = np

import numpy as np
from typing import Tuple, Dict, Optional
# matplotlib is imported lazily (inside plotting methods) to avoid headless-env warnings.
from astropy import constants as const
from astropy import units as u
from scipy.ndimage import label, gaussian_filter


class WaveOpticsEngine:
    """
    Calculate diffraction and interference effects using physical optics.
    
    This class computes the wave optical amplification factor accounting for
    the finite wavelength of light, which can produce interference patterns
    and differs from geometric optics predictions.
    
    The key equation is (Nakamura & Deguchi 1999, Eq. 4.2):
    F(ω) = (ω / 2πi) ∫ d²θ exp[iω τ(θ,β)]

    where τ(θ,β) is the Fermat potential (time delay surface) and
    ω = 2πc/λ is the radiation angular frequency [rad/s].
    
    Parameters
    ----------
    None (stateless calculator)
    
    Examples
    --------
    >>> from lens_models import LensSystem, PointMassProfile
    >>> from optics import WaveOpticsEngine
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> lens = PointMassProfile(1e12, lens_sys)
    >>> engine = WaveOpticsEngine()
    >>> result = engine.compute_amplification_factor(
    ...     lens, source_position=(0.5, 0.0), wavelength=500.0
    ... )
    >>> print(f"Wave optics shows interference fringes")
    """
    
    def __init__(self):
        """Initialize the wave optics engine."""
        pass
    
    def compute_amplification_factor(
        self,
        lens_model,
        source_position: Tuple[float, float] = (0.5, 0.0),
        wavelength: float = 500.0,
        grid_size: int = 512,
        grid_extent: float = 3.0,
        return_geometric: bool = True
    ) -> Dict:
        """
        Calculate wave optical amplification including diffraction/interference.

        Implements the diffraction integral for gravitational lensing:
            F(ω) = (ω / 2πi) ∫ d²θ  exp[iω τ(θ)]
        where ω = 2πf·(1+z_L)·(D_d D_s / D_ds) is the dimensionless frequency.

        Ref: Nakamura & Deguchi (1999, Prog. Theor. Phys. Suppl. 133, 137), Eq. 4.2
        Ref: Schneider et al. (1992) "Gravitational Lenses", §4.5 (wave optics)
        Ref: Takahashi & Nakamura (2003, ApJ, 595, 1039), Eq. 3–5

        Algorithm:
        1. Compute Fermat potential on lens plane grid:
           Φ(θ) = 0.5|θ − β|² − ψ(θ)    — Eq. 4.14 in Schneider (1992)
           where ψ is the lensing potential from lens_model

        2. Calculate wave phase ωτ(θ,β):
           wave_phase = (2πc/λ) × Δt(θ)  — dimensionless, equals ω×τ(θ,β)

        3. Evaluate N&D (1999) Eq. 4.2 scalar integral:
           F(ω) = (ω/2πi) ∫ d²θ exp[iωτ(θ,β)]
           where ω = 2πc/λ is the radiation angular frequency.

        4. Compute scalar magnification: |F(ω)|²
        
        Parameters
        ----------
        lens_model : MassProfile
            The lens model (must have lensing_potential method)
        source_position : tuple of float, optional
            Source position (β_x, β_y) in arcseconds (default: (0.5, 0.0))
        wavelength : float, optional
            Observation wavelength in nanometers (default: 500 nm for optical)
        grid_size : int, optional
            Grid size for computation (default: 512, recommend power of 2 for FFT)
        grid_extent : float, optional
            Physical extent of grid in arcseconds (default: 3.0)
        return_geometric : bool, optional
            Whether to compute geometric optics for comparison (default: True)
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'F_omega': complex amplification factor F(ω) [scalar]
            - 'magnification_wave': |F(ω)|² scalar magnification
            - 'wave_phase': 2D array of phase ωτ(θ,β) [radians]
            - 'grid_x': 1D array of x-coordinates in arcsec
            - 'grid_y': 1D array of y-coordinates in arcsec
            - 'wavelength': wavelength used in nm
            - 'fermat_potential': 2D array of Φ(θ) in arcsec²
            - 'geometric_comparison': dict with geometric optics result (if requested)
            
        Notes
        -----
        The geometric scale factor converts the dimensionless Fermat potential
        to physical time delay using cosmological distances. The wave phase
        accumulates as light travels along different paths.
        
        For a lens at z_l with source at z_s, the time delay is:
        Δt = (1 + z_l) × (D_l × D_s / D_ls) / c × Φ(θ)
        
        The wave phase is then: φ = 2π × Δt / (λ / c) = (2πc/λ) × Δt
        """
        beta_x, beta_y = source_position
        
        # Step 1: Create computational grid on image plane (θ space)
        x = jnp.linspace(-grid_extent, grid_extent, grid_size)
        y = jnp.linspace(-grid_extent, grid_extent, grid_size)
        xx, yy = jnp.meshgrid(x, y)
        
        # Step 2: Compute lensing potential ψ(θ)
        # Flatten for vectorized computation
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        
        psi_flat = lens_model.lensing_potential(x_flat, y_flat)
        psi = psi_flat.reshape(xx.shape)
        
        # Step 3: Compute Fermat potential
        # Φ(θ) = (1/2)|θ - β|² - ψ(θ)
        # This is the arrival time surface (dimensionless, in arcsec²)
        theta_minus_beta_x = xx - beta_x
        theta_minus_beta_y = yy - beta_y
        geometric_term = 0.5 * (theta_minus_beta_x**2 + theta_minus_beta_y**2)
        
        fermat_potential = geometric_term - psi
        
        # Step 4: Convert to physical time delay and then to wave phase
        # Get cosmological distances from lens system (already have units)
        D_l = lens_model.lens_system.angular_diameter_distance_lens()  # Quantity in Mpc
        D_s = lens_model.lens_system.angular_diameter_distance_source()  # Quantity in Mpc
        D_ls = lens_model.lens_system.angular_diameter_distance_lens_source()  # Quantity in Mpc
        z_l = lens_model.lens_system.z_l
        
        # Geometric factor: (1 + z_l) × D_l × D_s / D_ls / c
        # This converts Φ [arcsec²] to time delay [seconds]
        # First convert arcsec² to radians²
        arcsec_to_rad = (1.0 / 206265.0)  # 1 arcsec = 1/206265 radians
        
        # Distance factor in meters (distances already have units from astropy)
        D_l_m = D_l.to(u.m).value
        D_s_m = D_s.to(u.m).value
        D_ls_m = D_ls.to(u.m).value
        
        # Geometric factor [m/rad²]
        geometric_factor = (1.0 + z_l) * D_l_m * D_s_m / D_ls_m
        
        # Time delay in seconds
        c_light = const.c.value  # m/s
        time_delay = geometric_factor * fermat_potential * (arcsec_to_rad**2) / c_light
        
        # Convert wavelength to meters
        wavelength_m = wavelength * 1e-9  # nm to m
        
        # Wave phase φ = 2π × Δt × (c/λ) = 2π × Δt / T where T = λ/c is period
        # φ = 2π × c × Δt / λ
        wave_phase = 2.0 * jnp.pi * c_light * time_delay / wavelength_m
        
        # Step 5 → 6: N&D (1999) Eq. 4.2: F(ω) = (ω/2πi) ∫ d²θ exp[iωτ(θ,β)]
        # wave_phase = ωτ(θ,β) is already computed above.
        dtheta_rad = (2.0 * grid_extent / grid_size) * arcsec_to_rad  # rad/pixel
        d2theta = dtheta_rad**2  # solid angle per pixel [rad²]
        # Dimensionless effective frequency ω_eff = 2πf × T_0 where
        # T_0 = (1+z_l)·D_l·D_s/(c·D_ls) is the geometric time-delay scale.
        # This converts the EM angular frequency to the lens-plane frequency
        # that appears in the N&D (1999) / T&N (2003) Eq. 3 prefactor.
        T_0 = geometric_factor / c_light  # geometric time-delay scale [seconds]
        omega_rad = 2.0 * jnp.pi * c_light / wavelength_m  # EM angular frequency [rad/s]
        omega_eff = omega_rad * T_0  # dimensionless effective frequency
        integrand = jnp.exp(1j * wave_phase)  # per-pixel complex phase field
        F_omega = (omega_eff / (2.0 * jnp.pi * 1j)) * jnp.sum(integrand) * d2theta  # scalar complex
        magnification_wave = float(jnp.abs(F_omega)**2)  # scalar |F(ω)|²

        result = {
            'F_omega': complex(F_omega),
            'magnification_wave': magnification_wave,
            'wave_phase': np.array(wave_phase),
            'grid_x': np.array(x),
            'grid_y': np.array(y),
            'wavelength': wavelength,
            'fermat_potential': np.array(fermat_potential),
            'grid_extent': grid_extent,
        }
        
        # Step 8: Optionally compute geometric optics for comparison
        if return_geometric:
            from .ray_tracing import ray_trace
            
            geo_result = ray_trace(
                source_position,
                lens_model,
                grid_extent=grid_extent,
                grid_resolution=grid_size,
                threshold=0.05,
                return_maps=True
            )
            
            result['geometric_comparison'] = {
                'image_positions': geo_result['image_positions'],
                'magnifications': geo_result['magnifications'],
                'convergence_map': geo_result['convergence_map']
            }
        
        return result
    
    def detect_fringes(
        self,
        amplitude_map: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray
    ) -> Dict:
        """
        Detect and characterize interference fringes in amplitude map.
        
        Parameters
        ----------
        amplitude_map : np.ndarray
            2D intensity map from wave optics calculation
        grid_x : np.ndarray
            x-coordinates in arcsec
        grid_y : np.ndarray
            y-coordinates in arcsec
            
        Returns
        -------
        fringe_info : dict
            Dictionary containing:
            - 'fringe_spacing': average spacing in arcsec
            - 'n_fringes': number of distinct fringes detected
            - 'fringe_contrast': (I_max - I_min) / (I_max + I_min)
        """
        # Compute radial profile
        dx = grid_x[1] - grid_x[0]
        center_idx = len(grid_x) // 2
        y_center = amplitude_map.shape[0] // 2
        
        # Extract radial profile along x-axis through center
        radial_profile = amplitude_map[y_center, :]
        r = grid_x
        
        # Find peaks in radial profile
        # Smooth slightly to avoid noise
        from scipy.signal import find_peaks
        smoothed = gaussian_filter(radial_profile, sigma=2.0)
        peaks, properties = find_peaks(smoothed, prominence=0.1*jnp.max(smoothed))
        
        if len(peaks) > 1:
            # Compute average spacing between peaks
            peak_positions = r[peaks]
            spacings = np.diff(peak_positions)
            avg_spacing = jnp.mean(jnp.abs(spacings))
        else:
            avg_spacing = 0.0
        
        # Compute fringe contrast
        I_max = jnp.max(amplitude_map)
        I_min = jnp.min(amplitude_map)
        contrast = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0.0
        
        return {
            'fringe_spacing': avg_spacing,
            'n_fringes': len(peaks),
            'fringe_contrast': contrast
        }
    
    def compare_with_geometric(
        self,
        wave_result: Dict,
        fractional_threshold: float = 0.01
    ) -> Dict:
        """
        Compare wave optics result with geometric optics.
        
        Parameters
        ----------
        wave_result : dict
            Result dictionary from compute_amplification_factor
        fractional_threshold : float, optional
            Threshold for significant difference (default: 0.01 = 1%)
            
        Returns
        -------
        comparison : dict
            Dictionary containing:
            - 'fractional_difference_map': 2D array of |wave - geo|/geo
            - 'max_difference': maximum fractional difference
            - 'mean_difference': mean fractional difference
            - 'significant_pixels': fraction of pixels with difference > threshold
        """
        if 'geometric_comparison' not in wave_result:
            raise ValueError("Wave result must include geometric comparison")
        
        # Derive 2D interference field from wave phase for spatial comparison
        wave_phase_2d = np.asarray(wave_result['wave_phase'])
        amplitude_map = (1.0 + np.cos(wave_phase_2d)) * 0.5  # normalized to [0, 1]
        convergence_map = wave_result['geometric_comparison']['convergence_map']
        
        # Normalize both maps for comparison
        amp_norm = amplitude_map / jnp.sum(amplitude_map)
        conv_norm = convergence_map / jnp.sum(convergence_map)
        
        # Compute fractional difference
        # Avoid division by zero
        epsilon = 1e-10
        frac_diff = jnp.abs(amp_norm - conv_norm) / (conv_norm + epsilon)
        
        # Mask regions where both are very small (not meaningful)
        mask = (amp_norm > 0.01 * jnp.max(amp_norm)) | (conv_norm > 0.01 * jnp.max(conv_norm))
        frac_diff_masked = frac_diff * mask
        
        max_diff = jnp.max(frac_diff_masked)
        mean_diff = jnp.mean(frac_diff_masked[mask]) if jnp.sum(mask) > 0 else 0.0
        significant = jnp.sum(frac_diff_masked > fractional_threshold) / jnp.sum(mask) if jnp.sum(mask) > 0 else 0.0
        
        return {
            'fractional_difference_map': frac_diff_masked,
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'significant_pixels': significant
        }
    
    def plot_interference_pattern(
        self,
        wave_result: Dict,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Create publication-quality figure of wave optics results.
        
        Parameters
        ----------
        wave_result : dict
            Result from compute_amplification_factor
        figsize : tuple, optional
            Figure size in inches (default: (12, 10))
        save_path : str, optional
            Path to save figure (default: None, display only)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='#1a1a1a')
        fig.suptitle(
            f'Wave Optics: λ = {wave_result["wavelength"]:.0f} nm',
            fontsize=16, color='white', y=0.98
        )
        
        wave_phase_2d = np.asarray(wave_result['wave_phase'])
        interference_map = (1.0 + np.cos(wave_phase_2d)) * 0.5  # [0,1] normalized
        fermat_potential = wave_result['fermat_potential']
        extent = wave_result['grid_extent']
        extent_plot = [-extent, extent, -extent, extent]
        
        # 1. Interference map: (1+cos(ωτ))/2 — constructive/destructive regions
        ax1 = axes[0, 0]
        im1 = ax1.imshow(
            interference_map,
            extent=extent_plot,
            origin='lower',
            cmap='RdBu_r',
            aspect='auto',
            vmin=0, vmax=1
        )
        ax1.set_xlabel('θ_x (arcsec)', color='white')
        ax1.set_ylabel('θ_y (arcsec)', color='white')
        ax1.set_title('Interference Pattern (1+cos(ωτ))/2', color='white', fontsize=12)
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#0a0a0a')
        plt.colorbar(im1, ax=ax1, label='Normalized Field')
        
        # 2. Wave phase ωτ(θ,β) map (mod 2π for display)
        ax2 = axes[0, 1]
        im2 = ax2.imshow(
            np.mod(wave_phase_2d, 2.0 * np.pi),
            extent=extent_plot,
            origin='lower',
            cmap='twilight',
            aspect='auto',
            vmin=0,
            vmax=2.0 * np.pi
        )
        ax2.set_xlabel('θ_x (arcsec)', color='white')
        ax2.set_ylabel('θ_y (arcsec)', color='white')
        ax2.set_title('Wave Phase ωτ(θ,β) mod 2π', color='white', fontsize=12)
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#0a0a0a')
        plt.colorbar(im2, ax=ax2, label='Phase mod 2π (rad)')
        
        # 3. Fermat potential
        ax3 = axes[1, 0]
        im3 = ax3.imshow(
            fermat_potential,
            extent=extent_plot,
            origin='lower',
            cmap='viridis',
            aspect='auto'
        )
        ax3.set_xlabel('θ_x (arcsec)', color='white')
        ax3.set_ylabel('θ_y (arcsec)', color='white')
        ax3.set_title('Fermat Potential Φ(θ)', color='white', fontsize=12)
        ax3.tick_params(colors='white')
        ax3.set_facecolor('#0a0a0a')
        plt.colorbar(im3, ax=ax3, label='Φ (arcsec²)')
        
        # 4. Radial profile showing fringes
        ax4 = axes[1, 1]
        y_center = interference_map.shape[0] // 2
        radial_profile = interference_map[y_center, :]
        x_coords = wave_result['grid_x']
        
        ax4.plot(x_coords, radial_profile, color='#00ff41', linewidth=2)
        ax4.set_xlabel('θ_x (arcsec)', color='white')
        ax4.set_ylabel('Intensity', color='white')
        ax4.set_title('Radial Intensity Profile', color='white', fontsize=12)
        ax4.tick_params(colors='white')
        ax4.set_facecolor('#0a0a0a')
        ax4.grid(True, alpha=0.2, color='white')
        
        # Detect and annotate fringes
        fringe_info = self.detect_fringes(
            interference_map, x_coords, wave_result['grid_y']
        )
        
        textstr = (
            f"Fringes detected: {fringe_info['n_fringes']}\n"
            f"Avg spacing: {fringe_info['fringe_spacing']:.3f} arcsec\n"
            f"Contrast: {fringe_info['fringe_contrast']:.3f}"
        )
        ax4.text(
            0.05, 0.95, textstr,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
            color='white'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, facecolor='#1a1a1a')
            print(f"Saved wave optics plot to {save_path}")
        
        return fig


def plot_wave_vs_geometric(
    lens_model,
    source_position: Tuple[float, float],
    wavelength: float = 500.0,
    grid_size: int = 512,
    grid_extent: float = 3.0,
    save_path: Optional[str] = None
):
    """
    Create side-by-side comparison of wave vs geometric optics.
    
    Parameters
    ----------
    lens_model : MassProfile
        The lens model
    source_position : tuple
        Source position (x, y) in arcsec
    wavelength : float, optional
        Wavelength in nm (default: 500)
    grid_size : int, optional
        Grid size (default: 512)
    grid_extent : float, optional
        Grid extent in arcsec (default: 3.0)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The comparison figure
    """
    import matplotlib.pyplot as plt
    # Compute wave optics
    engine = WaveOpticsEngine()
    wave_result = engine.compute_amplification_factor(
        lens_model,
        source_position=source_position,
        wavelength=wavelength,
        grid_size=grid_size,
        grid_extent=grid_extent,
        return_geometric=True
    )
    
    # Get comparison
    comparison = engine.compare_with_geometric(wave_result)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#1a1a1a')
    fig.suptitle(
        f'Wave Optics vs Geometric Optics (λ = {wavelength:.0f} nm)',
        fontsize=16, color='white', y=0.98
    )
    
    extent_plot = [-grid_extent, grid_extent, -grid_extent, grid_extent]
    
    # 1. Geometric optics (convergence map)
    ax1 = axes[0, 0]
    geo_map = wave_result['geometric_comparison']['convergence_map']
    im1 = ax1.imshow(
        geo_map,
        extent=extent_plot,
        origin='lower',
        cmap='hot',
        aspect='auto'
    )
    ax1.set_xlabel('θ_x (arcsec)', color='white')
    ax1.set_ylabel('θ_y (arcsec)', color='white')
    ax1.set_title('Geometric Optics (Ray Tracing)', color='white', fontsize=12)
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#0a0a0a')
    plt.colorbar(im1, ax=ax1, label='Convergence κ')
    
    # Mark image positions
    img_pos = wave_result['geometric_comparison']['image_positions']
    if len(img_pos) > 0:
        ax1.plot(img_pos[:, 0], img_pos[:, 1], 'c*', markersize=15, 
                markeredgecolor='white', markeredgewidth=1.5, label='Images')
    ax1.plot(source_position[0], source_position[1], 'r*', markersize=15,
            markeredgecolor='white', markeredgewidth=1.5, label='Source')
    ax1.legend(loc='upper right', fontsize=8)
    
    # 2. Wave optics: interference pattern from N&D (1999) phase
    ax2 = axes[0, 1]
    wave_map = (1.0 + np.cos(np.asarray(wave_result['wave_phase']))) * 0.5
    im2 = ax2.imshow(
        wave_map,
        extent=extent_plot,
        origin='lower',
        cmap='RdBu_r',
        aspect='auto',
        vmin=0, vmax=1
    )
    ax2.set_xlabel('θ_x (arcsec)', color='white')
    ax2.set_ylabel('θ_y (arcsec)', color='white')
    ax2.set_title('Wave Optics (N&D 1999 Phase)', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#0a0a0a')
    plt.colorbar(im2, ax=ax2, label='(1+cos(ωτ))/2')
    
    # 3. Fractional difference map
    ax3 = axes[1, 0]
    diff_map = comparison['fractional_difference_map']
    im3 = ax3.imshow(
        diff_map,
        extent=extent_plot,
        origin='lower',
        cmap='RdYlBu_r',
        aspect='auto',
        vmin=0,
        vmax=min(1.0, np.percentile(diff_map, 99))
    )
    ax3.set_xlabel('θ_x (arcsec)', color='white')
    ax3.set_ylabel('θ_y (arcsec)', color='white')
    ax3.set_title('Fractional Difference |Wave - Geo|/Geo', color='white', fontsize=12)
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#0a0a0a')
    plt.colorbar(im3, ax=ax3, label='Fractional Diff')
    
    # Highlight regions with >1% difference
    significant_mask = diff_map > 0.01
    if jnp.sum(significant_mask) > 0:
        ax3.contour(
            diff_map, levels=[0.01], colors='lime', linewidths=2,
            extent=extent_plot, origin='lower'
        )
        ax3.text(
            0.05, 0.95, '>1% difference\n(green contour)',
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', color='lime',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )
    
    # 4. Statistics and summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    COMPARISON STATISTICS
    {'='*40}
    
    Wavelength: {wavelength:.1f} nm
    Grid size: {grid_size} × {grid_size}
    Grid extent: ±{grid_extent:.2f} arcsec
    
    Geometric Optics:
      Images found: {len(img_pos)}
      Total |μ|: {jnp.sum(jnp.abs(wave_result['geometric_comparison']['magnifications'])):.3f}
    
    Wave Optics:
      Max difference: {comparison['max_difference']:.3f}
      Mean difference: {comparison['mean_difference']:.3f}
      Pixels >1% diff: {comparison['significant_pixels']*100:.1f}%
    
    Fringe Detection:
    """
    
    fringe_info = engine.detect_fringes(
        wave_map, wave_result['grid_x'], wave_result['grid_y']
    )
    stats_text += f"""  N fringes: {fringe_info['n_fringes']}
      Avg spacing: {fringe_info['fringe_spacing']:.3f} arcsec
      Contrast: {fringe_info['fringe_contrast']:.3f}
    """
    
    ax4.text(
        0.1, 0.9, stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment='top',
        family='monospace',
        color='white',
        bbox=dict(boxstyle='round', facecolor='#0a0a0a', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='#1a1a1a')
        print(f"Saved comparison plot to {save_path}")
    
    return fig

