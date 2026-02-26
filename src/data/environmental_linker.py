import numpy as np
import pandas as pd
from typing import Union
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G, c, M_sun
import astropy.units as u

class EnvironmentalLinker:
    """
    Ingests catalogs of surrounding galaxies to compute the line-of-sight 
    external convergence (kappa_ext) and external shear (gamma_ext) 
    acting on the primary lens system.
    """
    
    def __init__(self, main_lens_ra: float, main_lens_dec: float, main_lens_z: float, source_z: float):
        """
        :param main_lens_ra: Right Ascension of the primary lens (degrees)
        :param main_lens_dec: Declination of the primary lens (degrees)
        :param main_lens_z: Redshift of the primary lens
        :param source_z: Redshift of the background source
        """
        self.primary_ra = main_lens_ra
        self.primary_dec = main_lens_dec
        self.primary_z = main_lens_z
        self.source_z = source_z
        
        # Cosmo distances for primary lensing
        self.D_s = cosmo.angular_diameter_distance(source_z).to(u.Mpc)
        
    def _angular_separation(self, ra1, dec1, ra2, dec2):
        """Returns separation in arcseconds using small angle approximation."""
        dra = (ra1 - ra2) * np.cos(np.radians(dec1))
        ddec = dec1 - dec2
        return np.sqrt(dra**2 + ddec**2) * 3600.0

    def _estimate_einstein_radius(self, mass_msun: float, lens_z: float) -> float:
        """
        Estimates the Einstein radius in arcseconds for a point mass.
        theta_E = sqrt(4 G M / c^2 * D_ls / (D_l * D_s))
        """
        if lens_z >= self.source_z:
            return 0.0 # Background galaxies don't lens the source
            
        D_l = cosmo.angular_diameter_distance(lens_z).to(u.m)
        D_s = cosmo.angular_diameter_distance(self.source_z).to(u.m)
        D_ls = cosmo.angular_diameter_distance_z1z2(lens_z, self.source_z).to(u.m)
        
        mass_kg = mass_msun * M_sun.value
        
        theta_E_rad = np.sqrt(
            (4.0 * G.value * mass_kg / c.value**2) * (D_ls.value / (D_l.value * D_s.value))
        )
        return np.degrees(theta_E_rad) * 3600.0
        
    def compute_kappa_ext(self, catalog_path: str, format: str = 'csv') -> float:
        """
        Reads a catalog and computes the total external convergence at the primary lens position.
        Assumes Singular Isothermal Sphere (SIS) mass profiles for simplicity of environmental sum.
        kappa_SIS(theta) = theta_E / (2 * theta)
        
        :param catalog_path: Path to the FITS or CSV catalog.
        :param format: 'csv' or 'fits'
        :return: Total kappa_ext
        """
        if format == 'csv':
            df = pd.read_csv(catalog_path)
            ra = df['ra'].values
            dec = df['dec'].values
            z = df['redshift'].values
            mass = df['mass_msun'].values
        elif format == 'fits':
            with fits.open(catalog_path) as hdul:
                data = hdul[1].data
                ra = data['ra']
                dec = data['dec']
                z = data['redshift']
                mass = data['mass_msun']
        else:
            raise ValueError("Format must be 'csv' or 'fits'")
            
        kappa_ext_total = 0.0
        
        for i in range(len(ra)):
            # Distance from primary lens in arcsec
            sep_arcsec = self._angular_separation(self.primary_ra, self.primary_dec, ra[i], dec[i])
            
            # Avoid the primary lens itself (separation ~ 0)
            if sep_arcsec < 1.0:
                continue
                
            # Estimate Einstein radius
            theta_e = self._estimate_einstein_radius(mass[i], z[i])
            
            if theta_e > 0:
                # SIS approximation for convergence: kappa = theta_E / (2 * theta)
                kappa_i = theta_e / (2.0 * sep_arcsec)
                kappa_ext_total += kappa_i
                
        return kappa_ext_total
