"""
Gravitational Lens Models Module

This module provides classes for modeling gravitational lens systems,
including cosmological calculations and various mass profiles.
"""

from .lens_system import LensSystem
from .mass_profiles import (
    MassProfile, 
    PointMassProfile, 
    NFWProfile,
    WarmDarkMatterProfile,
    SIDMProfile,
    DarkMatterFactory
)
from .advanced_profiles import (
    EllipticalNFWProfile,
    SersicProfile,
    CompositeGalaxyProfile
)
from .critical_curves import (
    lens_jacobian,
    magnification_map,
    convergence_shear,
    find_critical_curves,
    find_caustics,
    tangential_and_radial_critical_curves,
    solve_lens_equation,
    full_lensing_analysis,
)

__all__ = [
    'LensSystem', 
    'MassProfile', 
    'PointMassProfile', 
    'NFWProfile',
    'WarmDarkMatterProfile',
    'SIDMProfile',
    'DarkMatterFactory',
    'EllipticalNFWProfile',
    'SersicProfile',
    'CompositeGalaxyProfile',
    'lens_jacobian',
    'magnification_map',
    'convergence_shear',
    'find_critical_curves',
    'find_caustics',
    'tangential_and_radial_critical_curves',
    'solve_lens_equation',
    'full_lensing_analysis',
]
