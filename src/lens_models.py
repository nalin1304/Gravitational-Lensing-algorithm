"""Legacy compatibility imports for historical `src/lens_models.py` usage.

Canonical implementations live under the package directory:
`src/lens_models/`.

This module exists only to keep older imports working while avoiding duplicate
implementations that can drift scientifically.
"""

from __future__ import annotations

from src.lens_models import (
    DarkMatterFactory,
    EllipticalNFWProfile,
    LensSystem,
    MassProfile,
    NFWProfile,
    PointMassProfile,
    SIDMProfile,
    SersicProfile,
    WarmDarkMatterProfile,
)

__all__ = [
    "DarkMatterFactory",
    "EllipticalNFWProfile",
    "LensSystem",
    "MassProfile",
    "NFWProfile",
    "PointMassProfile",
    "SIDMProfile",
    "SersicProfile",
    "WarmDarkMatterProfile",
]

