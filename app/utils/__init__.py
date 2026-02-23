"""
app.utils package initializer
Makes the `app.utils` directory a proper Python package so imports like
`from app.utils.session_state import init_session_state` work in all environments.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .session_state import *  # noqa: F401,F403

# Backward compatibility bridge:
# Historically, application utility functions were implemented in `app/utils.py`.
# With this package present, `import app.utils` resolves here, so we explicitly
# expose the legacy functions to keep API/app/benchmark imports stable.
_legacy_utils_path = Path(__file__).resolve().parent.parent / "utils.py"
_legacy_spec = spec_from_file_location("app._legacy_utils_module", _legacy_utils_path)
_legacy_module = None
if _legacy_spec and _legacy_spec.loader:
    _legacy_module = module_from_spec(_legacy_spec)
    _legacy_spec.loader.exec_module(_legacy_module)

if _legacy_module is not None:
    generate_synthetic_convergence = getattr(_legacy_module, "generate_synthetic_convergence")
    load_pretrained_model = getattr(_legacy_module, "load_pretrained_model")
    prepare_model_input = getattr(_legacy_module, "prepare_model_input")
    compute_classification_entropy = getattr(_legacy_module, "compute_classification_entropy")
    format_parameter_value = getattr(_legacy_module, "format_parameter_value")
else:
    # Keep import-time errors explicit if the legacy implementation cannot load.
    def _missing(*_args, **_kwargs):
        raise ImportError(f"Legacy utility module could not be loaded: {_legacy_utils_path}")

    generate_synthetic_convergence = _missing
    load_pretrained_model = _missing
    prepare_model_input = _missing
    compute_classification_entropy = _missing
    format_parameter_value = _missing

__all__ = [
    'init_session_state',
    'get_state',
    'set_state',
    'clear_state',
    'reset_computation_results',
    'get_lens_parameters',
    'get_grid_parameters',
    'parameter_changed',
    'update_from_dict',
    'generate_synthetic_convergence',
    'load_pretrained_model',
    'prepare_model_input',
    'compute_classification_entropy',
    'format_parameter_value',
]
