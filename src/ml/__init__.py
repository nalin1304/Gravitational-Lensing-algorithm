"""
Machine Learning Module for Gravitational Lensing

This module provides physics-informed neural networks for:
- Lens parameter inference via variational physics optimization
- Dark matter model classification
- Training and evaluation utilities

Hardware-Agnostic Backend: Detects JAX and reports unavailable otherwise.
All sub-modules are guarded against missing dependencies.

"""

from typing import Any

# --------------------------------------------------------------------------
# Backend detection
# --------------------------------------------------------------------------
BACKEND: str = "unavailable"
"""Active compute backend: ``"jax"`` or ``"unavailable"``."""

try:
    import jax  # noqa: F401
    BACKEND = "jax"
except ImportError:
    pass


def check_backend() -> str:
    """Print a diagnostic summary of the active compute backend.

    Returns the backend name for programmatic use.
    """
    lines = [
        f"LensPINN Backend: {BACKEND}",
        f"  JAX available  : {BACKEND == 'jax'}",
    ]
    if BACKEND == "jax":
        import jax
        lines.append(f"  JAX version    : {jax.__version__}")
        lines.append(f"  Default device : {jax.default_backend()}")
    print("\n".join(lines))
    return BACKEND


# --------------------------------------------------------------------------
# Physics-Informed Neural Network (requires JAX/Equinox)
# --------------------------------------------------------------------------
PhysicsInformedNN: Any

try:
    from . import pinn as _pinn
    PhysicsInformedNN = _pinn.PhysicsInformedNN
    physics_informed_loss = _pinn.physics_informed_loss
except ImportError:
    # JAX and Equinox are optional dependencies.
    PhysicsInformedNN = None

    def physics_informed_loss(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("JAX/Equinox dependencies are required for physics_informed_loss.")

# --------------------------------------------------------------------------
# Dataset generation (NumPy-only — always available)
# --------------------------------------------------------------------------
try:
    from .generate_dataset import (
        generate_training_data,
        generate_convergence_map_vectorized,
        generate_convergence_map
    )
except ImportError:
    generate_training_data = None
    generate_convergence_map_vectorized = None
    generate_convergence_map = None

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
try:
    from .evaluate import evaluate_model, compute_metrics
except ImportError:
    evaluate_model = None
    compute_metrics = None

# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
try:
    from .augmentation import (
        RandomRotation, RandomFlip, RandomBrightness, RandomNoise,
        Compose, ToTensor, Normalize, get_training_transforms
    )
except ImportError:
    RandomRotation = RandomFlip = RandomBrightness = RandomNoise = None
    Compose = ToTensor = Normalize = None
    get_training_transforms = None

# --------------------------------------------------------------------------
# TensorBoard logging (optional)
# --------------------------------------------------------------------------
try:
    from .tensorboard_logger import PINNLogger
except Exception:
    PINNLogger = None  # type: ignore

# --------------------------------------------------------------------------
# Performance utilities
# --------------------------------------------------------------------------
try:
    from .performance import (
        get_backend, set_backend, GPU_AVAILABLE,
        PerformanceMonitor, timer,
        benchmark_convergence_map, compare_cpu_gpu_performance,
        cached_convergence, clear_cache
    )
except ImportError:
    get_backend = set_backend = None
    GPU_AVAILABLE = False
    PerformanceMonitor = timer = None
    benchmark_convergence_map = compare_cpu_gpu_performance = None
    cached_convergence = clear_cache = None

# --------------------------------------------------------------------------
# Transfer Learning
# --------------------------------------------------------------------------
try:
    from .transfer_learning import (
        TransferConfig,
        DomainAdaptationNetwork,
        MMDLoss,
        CORALLoss,
        BayesianUncertaintyEstimator,
        TransferLearningTrainer,
        create_synthetic_to_real_pipeline,
        compute_domain_discrepancy
    )
except ImportError:
    TransferConfig = DomainAdaptationNetwork = None
    MMDLoss = CORALLoss = None
    BayesianUncertaintyEstimator = TransferLearningTrainer = None
    create_synthetic_to_real_pipeline = compute_domain_discrepancy = None

__all__ = [
    # Backend
    'BACKEND',
    'check_backend',
    # PINN
    'PhysicsInformedNN',
    'physics_informed_loss',
    # Dataset
    'generate_training_data',
    'generate_convergence_map_vectorized',
    'generate_convergence_map',
    # Evaluation
    'evaluate_model',
    'compute_metrics',
    # Augmentation
    'RandomRotation',
    'RandomFlip',
    'RandomBrightness',
    'RandomNoise',
    'Compose',
    'ToTensor',
    'Normalize',
    'get_training_transforms',
    # Logging
    'PINNLogger',
    # Performance
    'get_backend',
    'set_backend',
    'GPU_AVAILABLE',
    'PerformanceMonitor',
    'timer',
    'benchmark_convergence_map',
    'compare_cpu_gpu_performance',
    'cached_convergence',
    'clear_cache',
    # Transfer Learning
    'TransferConfig',
    'DomainAdaptationNetwork',
    'MMDLoss',
    'CORALLoss',
    'BayesianUncertaintyEstimator',
    'TransferLearningTrainer',
    'create_synthetic_to_real_pipeline',
    'compute_domain_discrepancy',
]

