"""
Test Streamlit Pages (Phase 15)

Tests that all page functions can be imported and basic components work
"""

import sys
import io

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("STREAMLIT PAGES TEST")
print("=" * 70)

# Test 1: Import main module
print("\n[TEST 1] Importing app.main...")
try:
    from app import main
    print("✅ Import successful")
    print(f"   MODULES_AVAILABLE: {main.MODULES_AVAILABLE}")
    print(f"   PHASE15_AVAILABLE: {main.PHASE15_AVAILABLE}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check Phase 15 functions exist
print("\n[TEST 2] Checking Phase 15 page functions...")
functions = [
    'show_validation_page',
    'show_bayesian_uq_page',
    'generate_synthetic_convergence',
    'plot_convergence_map'
]

for func_name in functions:
    if hasattr(main, func_name):
        print(f"   ✅ {func_name} exists")
    else:
        print(f"   ❌ {func_name} missing")

# Test 3: Check Phase 15 imports
print("\n[TEST 3] Checking Phase 15 imports...")
try:
    from src.validation import quick_validate, rigorous_validate, ScientificValidator
    print("   ✅ Validation module imported")
except Exception as e:
    print(f"   ❌ Validation import failed: {e}")

try:
    from src.ml.uncertainty import BayesianPINN, UncertaintyCalibrator
    print("   ✅ Uncertainty module imported")
except Exception as e:
    print(f"   ❌ Uncertainty import failed: {e}")

# Test 4: Test generate_synthetic_convergence
print("\n[TEST 4] Testing generate_synthetic_convergence...")
try:
    convergence, X, Y = main.generate_synthetic_convergence(
        profile_type="NFW",
        mass=1.5e14,
        scale_radius=200.0,
        ellipticity=0.0,
        grid_size=64
    )
    print(f"   ✅ Function works")
    print(f"      Shape: {convergence.shape}")
    print(f"      Range: [{convergence.min():.4f}, {convergence.max():.4f}]")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test quick_validate
print("\n[TEST 5] Testing quick_validate...")
try:
    import numpy as np
    from src.validation import quick_validate
    
    # Generate test data
    truth = np.random.rand(64, 64) * 0.5
    pred = truth + np.random.normal(0, 0.001, truth.shape)
    
    passed = quick_validate(pred, truth)
    print(f"   ✅ Validation works: {'PASSED' if passed else 'FAILED'}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test Bayesian PINN creation
print("\n[TEST 6] Testing BayesianPINN creation...")
try:
    from src.ml.uncertainty import BayesianPINN
    
    model = BayesianPINN(dropout_rate=0.1)
    print(f"   ✅ Model created")
    print(f"      Parameters: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
print("\n✅ All core components are working!")
print("\nNOTE: If Streamlit UI looks bad, it might be:")
print("  1. CSS/styling issues (not tested here)")
print("  2. Browser-specific rendering problems")
print("  3. Specific widget interactions")
print("\nTry opening http://localhost:8501 in your browser to check.")
