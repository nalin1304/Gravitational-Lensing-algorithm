"""Compatibility wrappers for shared app styling utilities.

This module preserves legacy imports (`from app.styles import ...`) while delegating
all styling logic to `app.utils.ui`.
"""

from __future__ import annotations

try:
    from app.utils.ui import CUSTOM_CSS, inject_custom_css, render_card, render_header
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils.ui import CUSTOM_CSS, inject_custom_css, render_card, render_header

__all__ = [
    "CUSTOM_CSS",
    "inject_custom_css",
    "render_card",
    "render_header",
]
