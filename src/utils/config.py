"""
Configuration Utilities for Gravitational Lensing (Phase 33)

Replaces legacy loose YAML configuration dicts with a strictly-typed
Pydantic "Caskade-style" validation schema pipeline. This guarantees
reproducible, type-safe environments tracking exact physical parameters.
"""

import yaml  # type: ignore[import-untyped]
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Literal, Union

class LensModelConfig(BaseModel):
    """Configuration mapping for Mass Models."""
    model_type: Literal['NFW', 'PointMass', 'SIE', 'EPL'] = Field(..., description="Type of macroscopic lens profile")
    mass_solar: float = Field(1e12, description="Mass in Solar Masses")
    concentration: Optional[float] = Field(None, description="Halo concentration for NFW profiles")
    ellipticity: Optional[float] = Field(None, description="Ellipticity for SIE/EPL models")
    z_l: float = Field(0.5, description="Lens redshift (cosmological)")

class SourceConfig(BaseModel):
    """Configuration tracking empirical Source parameters."""
    position: Tuple[float, float] = Field((0.0, 0.0), description="Source position (x, y) in arcsec")
    radius_arcsec: float = Field(0.1, description="Source emission characteristic radius")
    z_s: float = Field(2.0, description="Source redshift (cosmological)")
    wavelength_nm: Optional[float] = Field(None, description="Active observation wavelength for wave-optics")

class SimulationConfig(BaseModel):
    """Numerical solver constants for integration frameworks."""
    grid_extent: float = Field(3.0, description="Field of view extent bounding box in arcsec")
    grid_resolution: int = Field(512, description="Target pixel resolution size")
    ray_tracing_threshold: float = Field(0.01, description="Root finding limit for geometric boundaries")

class CaskadePipelineConfig(BaseModel):
    """
    Root strictly-typed Pydantic configuration structure.
    Used to deserialize experimental environments from demos/*.yaml scripts.
    """
    lens: LensModelConfig
    source: SourceConfig
    simulation: SimulationConfig
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'CaskadePipelineConfig':
        """Load configuration enforcing strict Pydantic type validation."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration payload: {path}")
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls(**data)
