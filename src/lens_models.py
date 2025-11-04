"""
Gravitational Lensing Models
Implements NFW profile and elliptical NFW profile for lensing calculations
"""

import numpy as np
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u  # This gives us access to u.km, u.Mpc, etc.
from astropy import constants as const  # This gives us access to physical constants

# Define commonly used units and constants
kpc = u.kpc  # type: ignore
Mpc = u.Mpc  # type: ignore
km = u.km  # type: ignore
s = u.s  # type: ignore
kg = u.kg  # type: ignore
m = u.m  # type: ignore
G = const.G  # type: ignore

class LensSystem:
    def __init__(self, z_lens=0.5, z_source=1.5, H0=70*u.km/u.s/u.Mpc, Om0=0.3):  # type: ignore
        """
        Initialize lens system with redshifts and cosmology
        
        Args:
            z_lens (float): Lens redshift
            z_source (float): Source redshift
            H0 (astropy.units.Quantity): Hubble constant (default: 70 km/s/Mpc)
            Om0 (float): Matter density parameter
        """
        self.z_lens = z_lens
        self.z_source = z_source
        
        # Create cosmology
        if isinstance(H0, u.Quantity):
            H0_val = H0.to(u.km/u.s/u.Mpc).value  # type: ignore
        else:
            H0_val = float(H0)
        # Use positional arguments to avoid static type checker warnings about keyword names
        self.cosmo = FlatLambdaCDM(H0_val, Om0)  # type: ignore
        
        # Calculate distances in kpc
        # angular_diameter_distance returns Mpc by default, convert to kpc
        self.D_l = self.cosmo.angular_diameter_distance(z_lens).to(u.kpc).value  # type: ignore
        self.D_s = self.cosmo.angular_diameter_distance(z_source).to(u.kpc).value  # type: ignore
        # For D_ls, calculate the angular diameter distance between lens and source redshifts
        self.D_ls = (self.cosmo.angular_diameter_distance_z1z2(z_lens, z_source) * u.Mpc).to(u.kpc).value  # type: ignore
        
        # Critical surface density
        self.Sigma_crit = 1.0 / (4.0 * np.pi * 6.67430e-11 * self.D_l * self.D_ls / self.D_s)  # kg/m^2

class NFWProfile:
    """NFW (Navarro-Frenk-White) density profile for dark matter halos"""
    
    def __init__(self, M_vir, concentration, lens_system):
        """
        Initialize NFW profile
        
        Args:
            M_vir (float): Virial mass in solar masses
            concentration (float): Concentration parameter
            lens_system (LensSystem): Lens system object
        """
        self.M_vir = M_vir * 1.989e30  # Convert to kg
        self.c = concentration
        self.lens_system = lens_system
        
        # Calculate derived parameters
        self.r_vir = (3 * self.M_vir / (4 * np.pi * 200 * self.lens_system.cosmo.critical_density(
            self.lens_system.z_lens).to(u.kg/u.m**3).value))**(1/3)  # type: ignore
        self.r_s = self.r_vir / self.c
        
        # Characteristic density
        f = np.log(1 + self.c) - self.c / (1 + self.c)
        self.rho_s = self.M_vir / (4 * np.pi * self.r_s**3 * f)
    
    def convergence(self, x, y):
        """
        Calculate convergence κ(x,y)
        
        Args:
            x (array): x coordinates in arcsec
            y (array): y coordinates in arcsec
            
        Returns:
            array: Convergence map
        """
        # Convert angles to physical distances
        x_kpc = x * self.lens_system.D_l * np.pi / (180 * 3600)
        y_kpc = y * self.lens_system.D_l * np.pi / (180 * 3600)
        r = np.sqrt(x_kpc**2 + y_kpc**2)
        
        x = r / self.r_s
        
        # NFW convergence formula
        convergence = np.zeros_like(x)
        
        # x < 1
        mask = x < 1
        if np.any(mask):
            convergence[mask] = (2 * self.rho_s * self.r_s / self.lens_system.Sigma_crit *
                               (1 - 1/np.sqrt(1-x[mask]**2) * np.arctanh(np.sqrt((1-x[mask])/(1+x[mask])))))
        
        # x > 1
        mask = x > 1
        if np.any(mask):
            convergence[mask] = (2 * self.rho_s * self.r_s / self.lens_system.Sigma_crit *
                               (1 - 1/np.sqrt(x[mask]**2-1) * np.arctan(np.sqrt((x[mask]-1)/(x[mask]+1)))))
        
        # x = 1
        mask = x == 1
        if np.any(mask):
            convergence[mask] = 2 * self.rho_s * self.r_s / (3 * self.lens_system.Sigma_crit)
        
        return convergence

class EllipticalNFWProfile(NFWProfile):
    """Elliptical NFW profile for dark matter halos"""
    
    def __init__(self, M_vir, concentration, ellipticity, theta, lens_system):
        """
        Initialize elliptical NFW profile
        
        Args:
            M_vir (float): Virial mass in solar masses
            concentration (float): Concentration parameter
            ellipticity (float): Ellipticity parameter (0 ≤ e < 1)
            theta (float): Position angle in degrees
            lens_system (LensSystem): Lens system object
        """
        super().__init__(M_vir, concentration, lens_system)
        self.e = ellipticity
        self.theta = np.radians(theta)
        
        # Pre-compute transformation matrices
        self.cos_theta = np.cos(self.theta)
        self.sin_theta = np.sin(self.theta)
        self.q = 1 - self.e  # Axis ratio
    
    def convergence(self, x, y):
        """
        Calculate convergence κ(x,y) for elliptical NFW
        
        Args:
            x (array): x coordinates in arcsec
            y (array): y coordinates in arcsec
            
        Returns:
            array: Convergence map
        """
        # Rotate and scale coordinates
        x_rot = self.cos_theta * x + self.sin_theta * y
        y_rot = -self.sin_theta * x + self.cos_theta * y
        
        # Apply ellipticity transformation
        x_ell = x_rot
        y_ell = y_rot / self.q
        
        # Calculate elliptical radius
        r_ell = np.sqrt(x_ell**2 + y_ell**2)
        
        # Use parent class method with transformed coordinates
        return super().convergence(r_ell, np.zeros_like(r_ell))