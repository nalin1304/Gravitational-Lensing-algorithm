"""Compatibility layer for legacy imports of app scientific helper functions.

Historically, Phase 10 helper functions lived in `app/utils.py`.
They are now organized under `app/core/web_utils.py` to separate:

1. App orchestration/UI code (`app/pages`, `app/utils/ui.py`)
2. Testable scientific utility logic (`app/core/web_utils.py`)

This module re-exports the same public API to avoid breaking existing imports.
"""

from __future__ import annotations

try:
    from app.core.web_utils import (
        compute_classification_entropy,
        format_parameter_value,
        generate_synthetic_convergence,
        load_pretrained_model,
        plot_classification_probs,
        plot_comparison,
        plot_convergence_map,
        plot_uncertainty_bars,
        prepare_model_input,
    )
except ImportError:  # pragma: no cover - fallback when run with app/ on PYTHONPATH
    from core.web_utils import (  # type: ignore
        compute_classification_entropy,
        format_parameter_value,
        generate_synthetic_convergence,
        load_pretrained_model,
        plot_classification_probs,
        plot_comparison,
        plot_convergence_map,
        plot_uncertainty_bars,
        prepare_model_input,
    )

__all__ = [
    "compute_classification_entropy",
    "format_parameter_value",
    "generate_synthetic_convergence",
    "load_pretrained_model",
    "plot_classification_probs",
    "plot_comparison",
    "plot_convergence_map",
    "plot_uncertainty_bars",
    "prepare_model_input",
]
