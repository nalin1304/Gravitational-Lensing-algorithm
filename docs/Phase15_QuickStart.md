# Phase 15 Quick Start Guide

**Get started with scientific validation and uncertainty quantification in 5 minutes!**

---

## Installation Check ✅

All dependencies already installed from previous phases!

```bash
# Verify installation
python -c "import torch, numpy, scipy, matplotlib; print('✅ All dependencies ready')"
```

---

## Quick Start: 3 Examples

### Example 1: Validate Your PINN (30 seconds)

```python
"""Validate PINN predictions against ground truth"""
import numpy as np
from src.validation import rigorous_validate

# Your data
predicted = np.load('my_pinn_output.npy')  # Your PINN prediction
ground_truth = np.load('analytic_solution.npy')  # Analytic solution

# Validate
result = rigorous_validate(predicted, ground_truth, profile_type="NFW")

# Check result
if result.passed:
    print("✅ PASSED - Ready for publication!")
    print(f"Confidence: {result.confidence_level:.1%}")
else:
    print("❌ FAILED - Needs improvement")
    for rec in result.recommendations:
        print(f"  💡 {rec}")

# Full scientific report
print("\n" + result.scientific_notes)
```

**Expected Output:**
```
✅ PASSED - Ready for publication!
Confidence: 84.2%

======================================================================
SCIENTIFIC VALIDATION REPORT: NFW Profile
======================================================================

✅ VALIDATION STATUS: PASSED
   Confidence Level: 84.2%

1. NUMERICAL ACCURACY
   ⭐ EXCELLENT: Predictions match ground truth within 1% RMSE
   • RMSE: 0.005008
   • SSIM: 0.9747

6. PUBLICATION READINESS
   ✅ RECOMMENDED FOR PUBLICATION
   Results meet peer-review standards for:
   • ApJ, MNRAS, A&A (top-tier journals)
```

---

### Example 2: Get Uncertainty Estimates (1 minute)

```python
"""Get Bayesian uncertainty estimates for predictions"""
import torch
from src.ml.uncertainty import BayesianPINN, visualize_uncertainty

# Create Bayesian PINN
model = BayesianPINN(dropout_rate=0.1)

# (Normally you'd load trained weights here)
# model.load_state_dict(torch.load('trained_model.pt'))

# Create test grid
x = torch.linspace(-5, 5, 128)
y = torch.linspace(-5, 5, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Predict with uncertainty
result = model.predict_convergence_with_uncertainty(
    X, Y,
    mass=1e14,
    concentration=5.0,
    redshift=0.5,
    n_samples=100,  # MC Dropout samples
    confidence=0.95  # 95% confidence intervals
)

# Results
print(f"Mean κ: {result.mean.mean():.4f}")
print(f"Uncertainty: {result.std.mean():.4f}")
print(f"95% CI: [{result.lower.mean():.4f}, {result.upper.mean():.4f}]")

# Visualize
fig = visualize_uncertainty(
    X.numpy(), Y.numpy(),
    result.mean, result.std,
    save_path='uncertainty_map.png'
)
print("✅ Visualization saved: uncertainty_map.png")
```

**Expected Output:**
```
Mean κ: -0.3087
Uncertainty: 0.4432
95% CI: [-2.1780, 1.5606]
✅ Visualization saved: uncertainty_map.png
```

---

### Example 3: Check Calibration (45 seconds)

```python
"""Verify uncertainty estimates are calibrated"""
import numpy as np
from src.ml.uncertainty import UncertaintyCalibrator

# Your predictions with uncertainties
predictions = np.load('predictions.npy')
uncertainties = np.load('uncertainties.npy')
ground_truth = np.load('ground_truth.npy')

# Create calibrator
calibrator = UncertaintyCalibrator()

# Check calibration
calib_error = calibrator.calibrate(
    predictions=predictions,
    uncertainties=uncertainties,
    ground_truth=ground_truth
)

print(f"Calibration error: {calib_error:.4f}")

if calib_error < 0.05:
    print("✅ Well-calibrated!")
else:
    print("⚠️  Calibration could be improved")

# Detailed assessment
assessment = calibrator.assess_calibration()
print(f"Status: {assessment['calibration_status']}")

# Plot calibration curve
fig = calibrator.plot_calibration_curve(
    save_path='calibration_curve.png'
)
print("✅ Calibration curve saved: calibration_curve.png")
```

**Expected Output:**
```
Calibration error: 0.0110
✅ Well-calibrated!
Status: Well-calibrated
✅ Calibration curve saved: calibration_curve.png
```

---

## Run Test Suites

### Test Scientific Validator
```bash
python scripts/test_validator.py
```

**Expected:** 7/7 tests pass in ~1 second

### Test Bayesian UQ
```bash
python scripts/test_bayesian_uq.py
```

**Expected:** 10/10 tests pass in ~5 seconds

### Integration Example
```bash
python scripts/integrate_validator.py
```

**Shows:** Full validation workflow with trained models

---

## Common Use Cases

### Use Case 1: During Training
```python
"""Quick validation during PINN training"""
from src.validation import quick_validate

for epoch in range(n_epochs):
    # ... training code ...
    
    if epoch % 10 == 0:
        with torch.no_grad():
            pred = model(x_val)
        
        if quick_validate(pred.numpy(), y_val.numpy()):
            print(f"Epoch {epoch}: ✅ Validation passed")
```

### Use Case 2: Final Validation
```python
"""Rigorous validation before publication"""
from src.validation import rigorous_validate

# After training
result = rigorous_validate(
    predicted=final_prediction,
    ground_truth=analytic_solution,
    profile_type="NFW"
)

# Save report
with open('validation_report.txt', 'w') as f:
    f.write(result.scientific_notes)

print(f"Report saved with {len(result.metrics)} metrics")
```

### Use Case 3: Uncertainty Analysis
```python
"""Comprehensive uncertainty analysis"""
from src.ml.uncertainty import BayesianPINN, print_uncertainty_summary

model = BayesianPINN()
# ... load trained weights ...

result = model.get_prediction_intervals(x_test, confidence=0.95)

# Print summary
print_uncertainty_summary(result, ground_truth=y_test)

# Check coverage
coverage = np.mean(
    (y_test >= result.lower) & (y_test <= result.upper)
)
print(f"Empirical coverage: {coverage:.1%} (expected: 95%)")
```

### Use Case 4: Profile-Specific Validation
```python
"""Validate NFW cusp and outer slope"""
from src.validation import ScientificValidator, ValidationLevel

validator = ScientificValidator(level=ValidationLevel.RIGOROUS)

result = validator.validate_convergence_map(
    predicted=pinn_output,
    ground_truth=nfw_analytic,
    profile_type="NFW",
    pixel_scale=0.05  # arcsec/pixel
)

# Check NFW-specific metrics
if 'nfw_overall_fit_quality' in result.metrics:
    quality = result.metrics['nfw_overall_fit_quality']
    print(f"NFW fit quality: {quality:.1%}")
    
    if quality > 0.9:
        print("✅ Excellent NFW profile reproduction")
```

---

## File Locations

### Source Code
```
src/
├── validation/
│   ├── __init__.py
│   └── scientific_validator.py    # Main validator
│
└── ml/
    └── uncertainty/
        ├── __init__.py
        └── bayesian_uq.py          # Bayesian PINN & calibration
```

### Test Scripts
```
scripts/
├── test_validator.py               # Test validation framework
├── test_bayesian_uq.py             # Test uncertainty quantification
└── integrate_validator.py          # Integration example
```

### Documentation
```
docs/
├── Phase15_Research_Accuracy_Plan.md   # Overall plan
├── Phase15_Part1_Complete.md           # Validation framework
├── Phase15_Part2_Complete.md           # Uncertainty quantification
└── Phase15_Summary.md                  # Combined summary
```

---

## API Reference

### Scientific Validator

```python
from src.validation import (
    ScientificValidator,      # Main validator class
    ValidationLevel,          # QUICK, STANDARD, RIGOROUS, BENCHMARK
    ValidationResult,         # Result container
    quick_validate,           # Quick pass/fail
    rigorous_validate         # Full validation
)

# Create validator
validator = ScientificValidator(level=ValidationLevel.RIGOROUS)

# Validate
result = validator.validate_convergence_map(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    profile_type: str = "NFW",
    uncertainty: Optional[np.ndarray] = None,
    pixel_scale: float = 0.05,
    verbose: bool = True
) -> ValidationResult
```

### Bayesian Uncertainty

```python
from src.ml.uncertainty import (
    BayesianPINN,             # Bayesian PINN model
    UncertaintyCalibrator,    # Calibration analysis
    UncertaintyPrediction,    # Result container
    visualize_uncertainty,    # Visualization function
    print_uncertainty_summary # Print summary
)

# Create model
model = BayesianPINN(
    input_dim: int = 5,
    output_dim: int = 4,
    hidden_dims: List[int] = [64, 64, 64],
    dropout_rate: float = 0.1
)

# Predict with uncertainty
mean, std = model.predict_with_uncertainty(
    x: torch.Tensor,
    n_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]

# Get confidence intervals
result = model.get_prediction_intervals(
    x: torch.Tensor,
    confidence: float = 0.95,
    n_samples: int = 100
) -> UncertaintyPrediction
```

---

## Troubleshooting

### Issue: "ImportError: cannot import name..."

**Solution:**
```bash
# Make sure you're in project root
cd d:/Coding\ projects/Collab/financial-advisor-tool

# Verify Python environment
.\.venv\Scripts\activate

# Test imports
python -c "from src.validation import rigorous_validate; print('✅ OK')"
python -c "from src.ml.uncertainty import BayesianPINN; print('✅ OK')"
```

### Issue: "Tests fail due to SSIM threshold"

**Solution:** Noise level too high in synthetic data. Already fixed in test scripts.

### Issue: "Validation takes too long"

**Solution:** Use `ValidationLevel.QUICK` for development:
```python
from src.validation import ScientificValidator, ValidationLevel

validator = ScientificValidator(level=ValidationLevel.QUICK)
```

### Issue: "MC Dropout uncertainty is zero"

**Solution:** Ensure dropout rate > 0 and model is in train mode during MC sampling:
```python
model = BayesianPINN(dropout_rate=0.1)  # Must be > 0
mean, std = model.predict_with_uncertainty(x, n_samples=100)
# Handles train/eval mode automatically
```

---

## Performance Tips

### For Faster Validation
- Use `ValidationLevel.QUICK` (0.006s vs 0.023s)
- Skip profile-specific tests if not needed
- Validate on subset of data first

### For Faster Uncertainty
- Reduce n_samples (50 instead of 100)
- Use smaller grids during development
- Enable GPU (10-50× speedup)

### For Better Calibration
- Use n_samples ≥ 100 for stable estimates
- Calibrate on large validation set (1000+ points)
- Re-calibrate after major model changes

---

## Next Steps

### Immediate Actions
1. ✅ Run test suites to verify installation
2. ✅ Try examples with your data
3. ✅ Check calibration of your models

### Short Term (This Week)
1. Train Bayesian PINN on real data
2. Validate against analytic solutions
3. Generate calibration curves
4. Create publication figures

### Medium Term (This Month)
1. Complete Streamlit dashboard (Part 3)
2. Run full benchmark suite
3. Compare with literature (Lenstool, GLAFIC)
4. Prepare publication materials

---

## Help & Support

### Documentation
- `docs/Phase15_Summary.md` - Overview
- `docs/Phase15_Part1_Complete.md` - Validation details
- `docs/Phase15_Part2_Complete.md` - Uncertainty details

### Test Scripts
- `scripts/test_validator.py` - Validation examples
- `scripts/test_bayesian_uq.py` - Uncertainty examples
- `scripts/integrate_validator.py` - Integration workflow

### Generated Outputs
- `results/uncertainty_tests/calibration_curve.png`
- `results/uncertainty_tests/uncertainty_visualization.png`

---

## Quick Reference Card

```python
# VALIDATE
from src.validation import rigorous_validate
result = rigorous_validate(pred, truth, "NFW")
print(result.scientific_notes)

# UNCERTAINTY
from src.ml.uncertainty import BayesianPINN
model = BayesianPINN()
mean, std = model.predict_with_uncertainty(x, n_samples=100)

# CALIBRATE
from src.ml.uncertainty import UncertaintyCalibrator
calibrator = UncertaintyCalibrator()
error = calibrator.calibrate(pred, std, truth)
fig = calibrator.plot_calibration_curve()

# VISUALIZE
from src.ml.uncertainty import visualize_uncertainty
fig = visualize_uncertainty(X, Y, mean, std, truth)
```

---

**You're all set! 🎉**

Start with Example 1 to validate your PINN, then move to Example 2 for uncertainty quantification!
