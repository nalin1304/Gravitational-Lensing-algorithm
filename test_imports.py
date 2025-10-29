import sys
sys.path.insert(0, '.')

print("Testing imports...")

try:
    from src.lens_models import LensSystem, NFWProfile, EllipticalNFWProfile
    print("✅ lens_models OK")
except Exception as e:
    print(f"❌ lens_models FAILED: {e}")

try:
    from src.ml.pinn import PhysicsInformedNN
    print("✅ pinn OK")
except Exception as e:
    print(f"❌ pinn FAILED: {e}")

try:
    from src.ml.generate_dataset import generate_convergence_map_vectorized
    print("✅ generate_dataset OK")
except Exception as e:
    print(f"❌ generate_dataset FAILED: {e}")

try:
    from src.ml.transfer_learning import BayesianUncertaintyEstimator, DomainAdaptationNetwork, TransferConfig
    print("✅ transfer_learning OK")
except Exception as e:
    print(f"❌ transfer_learning FAILED: {e}")

try:
    from src.data.real_data_loader import FITSDataLoader, preprocess_real_data, ObservationMetadata, ASTROPY_AVAILABLE
    print("✅ real_data_loader OK")
except Exception as e:
    print(f"❌ real_data_loader FAILED: {e}")

try:
    from src.validation import ScientificValidator, ValidationLevel, quick_validate, rigorous_validate
    print("✅ validation OK")
except Exception as e:
    print(f"❌ validation FAILED: {e}")

try:
    from src.ml.uncertainty import BayesianPINN, UncertaintyCalibrator, visualize_uncertainty, print_uncertainty_summary
    print("✅ uncertainty OK")
except Exception as e:
    print(f"❌ uncertainty FAILED: {e}")

print("\nAll tests complete!")
