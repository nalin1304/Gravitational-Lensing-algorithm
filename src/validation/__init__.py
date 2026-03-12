"""
Scientific Validation Framework

Comprehensive validation tools for gravitational lensing predictions.
"""

from .scientific_validator import (
    ValidationLevel,
    ValidationResult,
    ScientificValidator,
    quick_validate,
    rigorous_validate
)
from .observational_diagnostics import (
    IMAGE_SPACE_THRESHOLDS,
    ObservationalFitResult,
    build_hst_psf_kernel,
    fit_lensed_host_observation,
    subtract_smooth_foreground,
)

__all__ = [
    'ValidationLevel',
    'ValidationResult',
    'ScientificValidator',
    'quick_validate',
    'rigorous_validate',
    'IMAGE_SPACE_THRESHOLDS',
    'ObservationalFitResult',
    'build_hst_psf_kernel',
    'fit_lensed_host_observation',
    'subtract_smooth_foreground',
]
