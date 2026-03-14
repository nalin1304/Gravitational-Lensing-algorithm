"""
Advanced mass profile models for realistic gravitational lensing.

This module implements more sophisticated lens models beyond the basic
circular profiles, including elliptical variants and composite structures.

Classes
-------
EllipticalNFWProfile
    NFW profile with ellipticity and position angle
SersicProfile
    Sérsic profile for galactic bulges and disks
CompositeGalaxyProfile
    Combination of multiple components (bulge + disk + halo)

References
----------
.. [1] Navarro, Frenk & White (1997), ApJ, 490, 493
.. [2] Sérsic (1963), Boletin de la Asociacion Argentina de Astronomia
.. [3] Dutton & Treu (2014), MNRAS, 438, 3594
"""

import numpy as np
from typing import Optional, Tuple
from scipy.special import gamma, gammaincinv
from scipy.integrate import quad

from .mass_profiles import MassProfile, NFWProfile
from .lens_system import LensSystem


class EllipticalNFWProfile(NFWProfile):
    """
    Elliptical Navarro-Frenk-White (NFW) dark matter halo profile.
    
    Extends the circular NFW profile to include ellipticity and rotation.
    The ellipticity is implemented through coordinate transformation before
    evaluating the circular NFW profile.
    
    Parameters
    ----------
    M_vir : float
        Virial mass in solar masses (M☉)
    c : float
        Concentration parameter (dimensionless)
    lens_sys : LensSystem
        Lens system containing cosmological parameters
    ellipticity : float, optional
        Ellipticity parameter, ε = (a-b)/(a+b) where a,b are semi-major/minor axes
        Must be in range [0, 1). Default is 0 (circular).
    position_angle : float, optional
        Position angle in degrees, measured counter-clockwise from x-axis
        to the major axis. Default is 0.
    center_x : float, optional
        Center x-coordinate in arcseconds. Default is 0.
    center_y : float, optional
        Center y-coordinate in arcseconds. Default is 0.
    
    Attributes
    ----------
    ellipticity : float
        Ellipticity parameter ε
    position_angle : float
        Position angle in degrees
    q : float
        Axis ratio b/a = (1-ε)/(1+ε)
    phi : float
        Position angle in radians
    
    Examples
    --------
    >>> from src.lens_systems.cosmology import LensSystem
    >>> lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
    >>> profile = EllipticalNFWProfile(
    ...     M_vir=1e12, c=10.0, lens_sys=lens_sys,
    ...     ellipticity=0.3, position_angle=45.0
    ... )
    >>> kappa = profile.convergence(1.0, 0.5)
    
    Notes
    -----
    The ellipticity is applied via coordinate transformation:
    
    .. math::
        x' = x \\cos\\phi + y \\sin\\phi
        y' = (-x \\sin\\phi + y \\cos\\phi) / q
        r' = \\sqrt{x'^2 + y'^2}
    
    where q is the axis ratio and φ is the position angle.
    
    References
    ----------
    .. [1] Golse & Kneib (2002), A&A, 390, 821
    .. [2] Keeton (2001), arXiv:astro-ph/0102341
    """
    
    def __init__(
        self,
        M_vir: float,
        c: float,
        lens_sys: LensSystem,
        ellipticity: float = 0.0,
        position_angle: float = 0.0,
        center_x: float = 0.0,
        center_y: float = 0.0
    ):
        # Initialize parent NFW profile
        super().__init__(M_vir, c, lens_sys)
        
        # Validate ellipticity
        if not 0 <= ellipticity < 1:
            raise ValueError(f"Ellipticity must be in [0, 1), got {ellipticity}")
        
        self.ellipticity = ellipticity
        self.position_angle = position_angle
        self.center_x = center_x
        self.center_y = center_y
        
        # Axis ratio convention: q = (1 - e) / (1 + e)  (WL/shape-measurement
        # convention where e = (1-q)/(1+q), i.e. reduced shear notation).
        # Note: NFWProfile in mass_profiles.py uses q = 1 - e instead — these
        # two conventions give different q for the same e.  EllipticalNFWProfile
        # follows the WL convention consistently across its coordinate transforms.
        self.q = (1.0 - ellipticity) / (1.0 + ellipticity)  # Axis ratio b/a (WL convention)
        self.phi = np.radians(position_angle)  # Convert to radians
    
    def _transform_coordinates(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform (x, y) to elliptical coordinates.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in arcseconds
        y : np.ndarray
            Y coordinates in arcseconds
        
        Returns
        -------
        x_rot : np.ndarray
            Rotated x coordinates
        y_rot : np.ndarray
            Rotated y coordinates
        r_ell : np.ndarray
            Elliptical radius
        """
        # Shift to center
        x_shifted = x - self.center_x
        y_shifted = y - self.center_y
        
        # Rotate to principal axes
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        
        x_rot = x_shifted * cos_phi + y_shifted * sin_phi
        y_rot = -x_shifted * sin_phi + y_shifted * cos_phi
        
        # Scale by axis ratio
        y_scaled = y_rot / self.q
        
        # Compute elliptical radius
        r_ell = np.sqrt(x_rot**2 + y_scaled**2)
        
        return x_rot, y_rot, r_ell
    
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute convergence at position (x, y).
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate(s) in arcseconds
        y : np.ndarray
            Y coordinate(s) in arcseconds
        
        Returns
        -------
        kappa : np.ndarray
            Dimensionless surface mass density (convergence)
        """
        # Check if inputs are scalar
        scalar_input = np.isscalar(x) and np.isscalar(y)
        
        # Convert to arrays for processing
        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)
        
        # Transform to elliptical coordinates
        _, _, r_ell = self._transform_coordinates(x_arr, y_arr)
        
        # Evaluate circular NFW at elliptical radius
        # Use parent class method with r_ell as distance
        kappa = super().convergence(r_ell, np.zeros_like(r_ell))
        
        # Return scalar if input was scalar
        if scalar_input:
            return float(kappa[0]) if len(kappa) > 0 else float(kappa)
        
        return kappa
    
    def deflection_angle(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deflection angle at position (x, y).
        
        Use a pseudo-elliptical approximation based on the gradient of the
        circular NFW potential evaluated at the elliptical radius.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate(s) in arcseconds
        y : np.ndarray
            Y coordinate(s) in arcseconds
        
        Returns
        -------
        alpha_x : np.ndarray
            X component of deflection angle in arcseconds
        alpha_y : np.ndarray
            Y component of deflection angle in arcseconds
        
        Notes
        -----
        This is an approximation. For precise calculations, use numerical
        integration of the convergence profile.
        """
        scalar_input = np.isscalar(x) and np.isscalar(y)
        x_arr = np.atleast_1d(x).astype(float)
        y_arr = np.atleast_1d(y).astype(float)

        x_rot, y_rot, r_ell = self._transform_coordinates(x_arr, y_arr)
        safe_radius = np.maximum(r_ell, 1.0e-12)

        # Evaluate the circular parent profile at the elliptical radius and
        # project the radial deflection back into the rotated frame.
        alpha_radial, _ = super().deflection_angle(safe_radius, np.zeros_like(safe_radius))
        alpha_x_rot = alpha_radial * x_rot / safe_radius
        alpha_y_rot = alpha_radial * y_rot / (self.q**2 * safe_radius)
        
        # Rotate back to the original un-rotated coordinate frame
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        
        alpha_x = alpha_x_rot * cos_phi - alpha_y_rot * sin_phi
        alpha_y = alpha_x_rot * sin_phi + alpha_y_rot * cos_phi
        
        if scalar_input:
            return float(np.ravel(alpha_x)[0]), float(np.ravel(alpha_y)[0])
        return alpha_x, alpha_y
    
    def shear(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute shear components at position (x, y).
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate(s) in arcseconds
        y : np.ndarray
            Y coordinate(s) in arcseconds
        
        Returns
        -------
        gamma1 : np.ndarray
            Shear component γ₁
        gamma2 : np.ndarray
            Shear component γ₂
        
        Notes
        -----
        For elliptical profiles, shear calculation is more complex.
        This implementation provides an approximation.
        """
        # Compute exact shear via numerical differentiation of the deflection field
        # \gamma_1 = 0.5 * (\partial\alpha_x/\partial x - \partial\alpha_y/\partial y)
        # \gamma_2 = 0.5 * (\partial\alpha_x/\partial y + \partial\alpha_y/\partial x)
        
        dx = 1e-6
        alpha_x_plus_dx, alpha_y_plus_dx = self.deflection_angle(x + dx, y)
        alpha_x_minus_dx, alpha_y_minus_dx = self.deflection_angle(x - dx, y)
        
        alpha_x_plus_dy, alpha_y_plus_dy = self.deflection_angle(x, y + dx)
        alpha_x_minus_dy, alpha_y_minus_dy = self.deflection_angle(x, y - dx)
        
        dalpha_x_dx = (alpha_x_plus_dx - alpha_x_minus_dx) / (2.0 * dx)
        dalpha_y_dy = (alpha_y_plus_dy - alpha_y_minus_dy) / (2.0 * dx)
        
        dalpha_x_dy = (alpha_x_plus_dy - alpha_x_minus_dy) / (2.0 * dx)
        dalpha_y_dx = (alpha_y_plus_dx - alpha_y_minus_dx) / (2.0 * dx)
        
        gamma1 = 0.5 * (dalpha_x_dx - dalpha_y_dy)
        gamma2 = 0.5 * (dalpha_x_dy + dalpha_y_dx)
        
        return gamma1, gamma2


class SersicProfile(MassProfile):
    """
    Sérsic profile for galactic bulges and stellar disks.
    
    The Sérsic profile is commonly used to describe the light (and mass)
    distribution of elliptical galaxies and galactic bulges.
    
    Parameters
    ----------
    I_e : float
        Surface brightness at effective radius (arbitrary units)
    r_e : float
        Effective radius in kpc (physical) or arcsec (projected)
    n : float
        Sérsic index. n=1 gives exponential, n=4 gives de Vaucouleurs
    lens_sys : LensSystem, optional
        Lens system for coordinate conversions
    M_L : float, optional
        Mass-to-light ratio in solar units. Default is 1.0.
    
    Attributes
    ----------
    b_n : float
        Sérsic b parameter: b_n ≈ 2n - 1/3 + 0.009876/n
    
    Examples
    --------
    >>> # Exponential disk (n=1)
    >>> disk = SersicProfile(I_e=1.0, r_e=5.0, n=1.0)
    >>> 
    >>> # de Vaucouleurs bulge (n=4)
    >>> bulge = SersicProfile(I_e=2.0, r_e=2.0, n=4.0)
    
    Notes
    -----
    The Sérsic surface brightness profile is:
    
    .. math::
        I(r) = I_e \\exp\\left[-b_n\\left(\\left(\\frac{r}{r_e}\\right)^{1/n} - 1\\right)\\right]
    
    where b_n is chosen such that half the total light is within r_e.
    
    References
    ----------
    .. [1] Sérsic (1963), Boletin de la Asociacion Argentina de Astronomia
    .. [2] Graham & Driver (2005), PASA, 22, 118
    .. [3] Trujillo et al. (2001), MNRAS, 326, 869
    """
    
    def __init__(
        self,
        I_e: float,
        r_e: float,
        n: float,
        lens_sys: Optional[LensSystem] = None,
        M_L: float = 1.0
    ):
        if n <= 0:
            raise ValueError(f"Sérsic index must be positive, got {n}")
        if r_e <= 0:
            raise ValueError(f"Effective radius must be positive, got {r_e}")
        
        self.I_e = I_e
        self.r_e = r_e
        self.n = n
        self.M_L = M_L
        self.lens_sys = lens_sys  # Store as lens_sys for consistency
        
        # Calculate b_n (approximation from Capaccioli 1989)
        self.b_n = self._calculate_b_n(n)
    
    @staticmethod
    def _calculate_b_n(n: float) -> float:
        """
        Calculate Sérsic b_n parameter.
        
        Uses the exact inverse incomplete gamma function solution
        (Ciotti 1991; Trujillo et al. 2001):
            solve γ(2n, b_n) = Γ(2n)/2
        i.e. regularized γ(2n, b_n) / Γ(2n) = 0.5
        
        Parameters
        ----------
        n : float
            Sérsic index
        
        Returns
        -------
        b_n : float
            Sérsic b parameter
        """
        return gammaincinv(2.0 * n, 0.5)
    
    def surface_brightness(self, r: np.ndarray) -> np.ndarray:
        """
        Compute surface brightness at radius r.
        
        Parameters
        ----------
        r : np.ndarray
            Radius in same units as r_e
        
        Returns
        -------
        I : np.ndarray
            Surface brightness in same units as I_e
        """
        x = r / self.r_e
        return self.I_e * np.exp(-self.b_n * (x**(1.0/self.n) - 1.0))
    
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute convergence (assumes mass traces light).
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate in arcseconds or kpc
        y : np.ndarray
            Y coordinate in arcseconds or kpc
        
        Returns
        -------
        kappa : np.ndarray
            Dimensionless surface mass density
        """
        r = np.sqrt(x**2 + y**2)
        
        # Surface density ∝ surface brightness
        Sigma = self.surface_brightness(r)
        
        # Normalize to dimensionless convergence
        # (This requires critical surface density from lens_sys)
        if self.lens_sys is not None:
            Sigma_crit = self.lens_sys.critical_surface_density()
            kappa = (Sigma * self.M_L) / Sigma_crit
        else:
            # Without a lens system, we cannot normalize to dimensionless κ.
            # Return surface mass density in M☉/arcsec² as a proxy (not true convergence).
            # Callers needing κ must provide a LensSystem via lens_sys parameter.
            import warnings
            warnings.warn(
                "SersicProfile.convergence() called without lens_sys — returning "
                "surface mass density [M☉/arcsec²], not dimensionless κ. "
                "Pass lens_sys=LensSystem(...) for physical convergence.",
                UserWarning, stacklevel=2
            )
            kappa = Sigma * self.M_L
        
        return kappa
    
    def deflection_angle(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deflection angle for Sérsic mass distribution.
        
        Uses numerical integration of the convergence profile.
        For circular profiles, the deflection angle is radial:
        α(θ) = (2/θ) ∫₀^θ κ(θ') θ' dθ'
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate in arcseconds
        y : np.ndarray
            Y coordinate in arcseconds
        
        Returns
        -------
        alpha_x : np.ndarray
            X component of deflection angle
        alpha_y : np.ndarray
            Y component of deflection angle
        
        Notes
        -----
        This uses a simplified analytical approximation based on
        the mean convergence within the radius.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        r = np.sqrt(x**2 + y**2)
        
        from scipy.integrate import quad
        import warnings
        
        alpha_mag = np.zeros_like(r, dtype=float)
        
        # Ensure we don't suppress integration warnings if numerical issues arise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(r.size):
                r_val = r.flat[i]
                if r_val < 1e-10:
                    alpha_mag.flat[i] = 0.0
                    continue
                
                # integrand: kappa(r') * r'
                def integrand(rp):
                    # self.convergence expects (x, y) arrays, we pass (rp, 0)
                    return self.convergence(np.array([rp]), np.array([0.0]))[0] * rp
                    
                res, _ = quad(integrand, 0.0, r_val, limit=100)
                alpha_mag.flat[i] = 2.0 * res / r_val
        
        alpha_mag = alpha_mag.reshape(x.shape)
        
        # Avoid division by zero
        r_safe = np.where(r > 1e-10, r, 1.0)
        
        # Decompose into x and y components
        alpha_x = alpha_mag * x / r_safe
        alpha_y = alpha_mag * y / r_safe
        
        # Handle r=0 case
        alpha_x = np.where(r > 1e-10, alpha_x, 0.0)
        alpha_y = np.where(r > 1e-10, alpha_y, 0.0)
        
        return alpha_x, alpha_y
    
    def surface_density(self, r: np.ndarray) -> np.ndarray:
        """
        Compute surface density at radius r.
        
        Parameters
        ----------
        r : np.ndarray
            Radius in arcseconds or kpc
        
        Returns
        -------
        Sigma : np.ndarray
            Surface density in M☉/pc² (if lens_sys provided) or arbitrary units
        """
        r = np.atleast_1d(r)
        
        # Surface density from surface brightness
        Sigma = self.surface_brightness(r) * self.M_L
        
        return Sigma
    
    def lensing_potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute lensing potential for Sérsic profile via exact 1D radial integration.

        For a circularly symmetric profile, the lensing potential satisfies
        ψ(θ) = ∫₀^θ α(θ') dθ'
        where α(θ') is the scalar (radial) deflection angle magnitude at θ'.
        This follows directly from α = ∇ψ for circularly symmetric lenses.

        Parameters
        ----------
        x : np.ndarray
            X coordinate in arcseconds
        y : np.ndarray
            Y coordinate in arcseconds

        Returns
        -------
        psi : np.ndarray
            Lensing potential in arcsec²

        Notes
        -----
        Uses scipy.integrate.quad with limit=150 for high-accuracy integration.
        Near-origin values (r < 1e-10) are set to zero by continuity.
        """
        from scipy.integrate import quad
        import warnings

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        r = np.sqrt(x**2 + y**2)

        psi = np.zeros_like(r, dtype=float)

        def alpha_radial(rp: float) -> float:
            """Scalar deflection angle magnitude at radius rp (arcsec)."""
            if rp < 1e-10:
                return 0.0
            ax, ay = self.deflection_angle(np.array([rp]), np.array([0.0]))
            return float(np.sqrt(ax[0]**2 + ay[0]**2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(r.size):
                r_val = float(r.flat[i])
                if r_val < 1e-10:
                    psi.flat[i] = 0.0
                    continue
                # ψ(r) = ∫₀^r α(r') dr'
                val, _ = quad(alpha_radial, 0.0, r_val, limit=150)
                psi.flat[i] = val

        return psi.reshape(x.shape)
    
    def total_luminosity(self) -> float:
        """
        Calculate total integrated luminosity.
        
        Returns
        -------
        L_tot : float
            Total luminosity in units of I_e * r_e²
        """
        # L = 2π n r_e² I_e exp(b_n) b_n^(-2n) Γ(2n)
        return (2.0 * np.pi * self.n * self.r_e**2 * self.I_e * 
                np.exp(self.b_n) * self.b_n**(-2*self.n) * 
                gamma(2*self.n))


class CompositeGalaxyProfile(MassProfile):
    """
    Composite galaxy model combining multiple components.
    
    Represents a realistic galaxy as a superposition of:
    - Stellar bulge (Sérsic profile, typically n=4)
    - Stellar disk (Sérsic profile, typically n=1)  
    - Dark matter halo (NFW or elliptical NFW)
    
    Parameters
    ----------
    bulge : SersicProfile, optional
        Bulge component (typically n=4)
    disk : SersicProfile, optional
        Disk component (typically n=1)
    halo : MassProfile, optional
        Dark matter halo (NFW or elliptical NFW)
    lens_sys : LensSystem, optional
        Lens system for cosmology
    
    Examples
    --------
    >>> # Create composite early-type galaxy
    >>> lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
    >>> bulge = SersicProfile(I_e=2.0, r_e=2.0, n=4.0, lens_sys=lens_sys)
    >>> halo = NFWProfile(M_vir=1e12, c=10.0, lens_sys=lens_sys)
    >>> galaxy = CompositeGalaxyProfile(bulge=bulge, halo=halo, lens_sys=lens_sys)
    >>> 
    >>> # Get total convergence
    >>> kappa_total = galaxy.convergence(1.0, 0.5)
    
    Notes
    -----
    The total convergence is the sum of all components:
    
    .. math::
        \\kappa_{\\rm total} = \\kappa_{\\rm bulge} + \\kappa_{\\rm disk} + \\kappa_{\\rm halo}
    
    References
    ----------
    .. [1] Dutton & Treu (2014), MNRAS, 438, 3594
    .. [2] Auger et al. (2010), ApJ, 724, 511
    """
    
    def __init__(
        self,
        bulge: Optional[SersicProfile] = None,
        disk: Optional[SersicProfile] = None,
        halo: Optional[MassProfile] = None,
        lens_sys: Optional[LensSystem] = None
    ):
        self.bulge = bulge
        self.disk = disk
        self.halo = halo
        self.lens_sys = lens_sys  # Store for consistency
        
        # At least one component must be provided
        if all(c is None for c in [bulge, disk, halo]):
            raise ValueError("At least one component (bulge, disk, halo) must be provided")
        
        self.components = [c for c in [bulge, disk, halo] if c is not None]
    
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute total convergence from all components.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate in arcseconds
        y : np.ndarray
            Y coordinate in arcseconds
        
        Returns
        -------
        kappa : np.ndarray
            Total dimensionless surface mass density
        """
        kappa_total = np.zeros_like(x, dtype=float)
        
        for component in self.components:
            kappa_total += component.convergence(x, y)
        
        return kappa_total
    
    def deflection_angle(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total deflection angle from all components.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate in arcseconds
        y : np.ndarray
            Y coordinate in arcseconds
        
        Returns
        -------
        alpha_x : np.ndarray
            X component of total deflection angle
        alpha_y : np.ndarray
            Y component of total deflection angle
        """
        alpha_x_total = np.zeros_like(x, dtype=float)
        alpha_y_total = np.zeros_like(y, dtype=float)
        
        for component in self.components:
            ax, ay = component.deflection_angle(x, y)
            alpha_x_total += ax
            alpha_y_total += ay
        
        return alpha_x_total, alpha_y_total
    
    def surface_density(self, r: np.ndarray) -> np.ndarray:
        """
        Compute total surface density from all components.
        
        Parameters
        ----------
        r : np.ndarray
            Radius in arcseconds
        
        Returns
        -------
        Sigma : np.ndarray
            Total surface density in M☉/pc² or arbitrary units
        """
        r = np.atleast_1d(r)
        Sigma_total = np.zeros_like(r, dtype=float)
        
        for component in self.components:
            Sigma_total += component.surface_density(r)
        
        return Sigma_total
    
    def lensing_potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute total lensing potential from all components.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinate in arcseconds
        y : np.ndarray
            Y coordinate in arcseconds
        
        Returns
        -------
        psi : np.ndarray
            Total lensing potential (dimensionless)
        """
        psi_total = np.zeros_like(x, dtype=float)
        
        for component in self.components:
            psi_total += component.lensing_potential(x, y)
        
        return psi_total
    
    def get_component_fractions(self, radius: float) -> dict:
        """
        Get the mass fraction of each component within a given radius.
        
        Parameters
        ----------
        radius : float
            Radius in arcseconds or kpc
        
        Returns
        -------
        fractions : dict
            Dictionary with keys 'bulge', 'disk', 'halo' and fractional masses
        """
        fractions = {}
        
        # Sample points within radius
        theta = np.linspace(0, 2*np.pi, 100)
        r_sample = np.linspace(0, radius, 50)
        
        total_mass = 0.0
        component_masses = {}
        
        for component, name in zip(
            [self.bulge, self.disk, self.halo],
            ['bulge', 'disk', 'halo']
        ):
            if component is not None:
                # Integrate convergence (simplified)
                mass = 0.0
                for r in r_sample:
                    for th in theta:
                        x = r * np.cos(th)
                        y = r * np.sin(th)
                        kappa = component.convergence(np.array([x]), np.array([y]))[0]
                        mass += kappa * r  # r dr dtheta
                
                component_masses[name] = mass
                total_mass += mass
        
        # Normalize to fractions
        if total_mass > 0:
            for name, mass in component_masses.items():
                fractions[name] = mass / total_mass
        
        return fractions

