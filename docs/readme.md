# Gravitational Lensing Simulation Framework

## A Multi-Modal Framework for Probing Cosmological Frontiers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a high-fidelity gravitational lensing simulation framework for testing fundamental cosmological theories. The framework integrates wave optics, alternative dark matter models, and time-delay cosmography to probe the nature of dark matter and measure cosmic expansion.

## 🌟 Features

### Phase 1: Geometric Ray Tracing ✅
- **Cosmological Framework**: Full ΛCDM cosmology with configurable parameters
- **Mass Profiles**: 
  - Point Mass (analytical solution for validation)
  - NFW Profile (realistic dark matter halos with correct deflection angles)
- **Ray Tracing Engine**: Find multiple lensed images via ray shooting algorithm
- **Magnification Calculations**: Jacobian-based magnification for each image
- **Time Delay Calculations**: Fermat potential and time delays between images
- **Comprehensive Visualization**: Convergence maps, deflection fields, radial profiles

### Phase 2: Wave Optics ✅ **NEW!**
- **Physical Optics**: Beyond geometric approximation with full wave treatment
- **Fermat Potential**: Accurate time delay surface computation
- **Interference Patterns**: Diffraction and interference from wave nature of light
- **Fringe Detection**: Automatic detection and characterization of interference fringes
- **Chromatic Effects**: Wavelength-dependent lensing (400-600 nm demonstrated)
- **Wave-Geometric Comparison**: Quantify differences between wave and ray optics
- **FFT Propagation**: Efficient 2D FFT for wave propagation to observer plane

### Future Phases
- **Phase 3**: Time-delay cosmography (H₀ measurement)
- **Phase 4**: Alternative dark matter (WDM, SIDM models)
- **Phase 5**: Physics-Informed Neural Networks
- **Phase 6**: Real data validation (HST, JWST)

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```powershell
git clone https://github.com/yourusername/financial-advisor-tool.git
cd financial-advisor-tool
```

2. **Create virtual environment** (optional but recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

### Verify Installation

Run the test suite to ensure everything is working:

```powershell
python -m pytest tests/ -v
```

**Expected output**: `91 passed` - All tests should pass ✓
- Phase 1 (Geometric): 63 tests
- Phase 2 (Wave Optics): 28 tests

### Basic Usage - Geometric Optics

```python
from src.lens_models import LensSystem, PointMassProfile
from src.optics import ray_trace
from src.utils import plot_lens_system

# Setup cosmology
lens_sys = LensSystem(z_lens=0.5, z_source=1.5)

# Create lens (10^12 solar masses)
lens = PointMassProfile(mass=1e12, lens_system=lens_sys)

# Find lensed images
source_pos = (0.5, 0.0)  # arcsec
results = ray_trace(source_pos, lens)

# Visualize
plot_lens_system(lens, source_pos, 
                results['image_positions'],
                results['magnifications'],
                results['convergence_map'],
                results['grid_x'],
                results['grid_y'])
```

### Wave Optics Usage (Phase 2)

```python
from src.lens_models import LensSystem, PointMassProfile
from src.optics import WaveOpticsEngine, plot_wave_vs_geometric

# Setup lens system
lens_sys = LensSystem(z_lens=0.5, z_source=1.5)
lens = PointMassProfile(mass=1e12, lens_system=lens_sys)

# Compute wave optics at optical wavelength
engine = WaveOpticsEngine()
wave_result = engine.compute_amplification_factor(
    lens,
    source_position=(0.5, 0.0),
    wavelength=500.0,  # nm
    grid_size=512
)

# Analyze interference fringes
fringe_info = engine.detect_fringes(
    wave_result['amplitude_map'],
    wave_result['grid_x'],
    wave_result['grid_y']
)
print(f"Fringes detected: {fringe_info['n_fringes']}")

# Compare wave vs geometric optics
fig = plot_wave_vs_geometric(lens, (0.5, 0.0), wavelength=500.0)
```

### Run Demo Notebooks

```powershell
# Geometric optics (Phase 1)
jupyter lab notebooks/phase1_demo.ipynb

# Wave optics (Phase 2) - NEW!
jupyter lab notebooks/phase2_wave_demo.ipynb
```

### Quick Demo

```powershell
# Run wave optics demonstration
python demo_wave_optics.py
```

This will generate visualizations in `results/` showing:
- Interference patterns with detected fringes
- Wave vs geometric optics comparison
- Numerical analysis of differences

## 📂 Project Structure

```
financial-advisor-tool/
├── src/
│   ├── lens_models/
│   │   ├── __init__.py
│   │   ├── lens_system.py      # Cosmological calculations
│   │   └── mass_profiles.py    # Point Mass, NFW profiles
│   ├── optics/
│   │   ├── __init__.py
│   │   └── ray_tracing.py      # Ray shooting algorithm
│   └── utils/
│       ├── __init__.py
│       └── visualization.py    # Plotting functions
├── tests/
│   ├── test_lens_system.py
│   ├── test_mass_profiles.py
│   └── test_ray_tracing.py
├── notebooks/
│   └── phase1_demo.ipynb       # Interactive demonstration
├── data/                        # Datasets (future)
├── docs/                        # Documentation
├── results/                     # Output figures
├── requirements.txt
└── README.md
```

## 🧪 Running Tests

Run the full test suite:
```powershell
pytest tests/ -v
```

Run specific test file:
```powershell
pytest tests/test_lens_system.py -v
```

Check test coverage:
```powershell
pytest --cov=src tests/
```

## 📊 Example Results

### Point Mass Lens
For a 10¹² M☉ lens at z=0.5 lensing a source at z=1.5:
- **Einstein radius**: θ_E ≈ 1.0 arcsec
- **Number of images**: 2-4 (depending on alignment)
- **Total magnification**: > 1 (flux conservation)

### NFW Dark Matter Halo
- **Smooth convergence profile**: No central singularity
- **Scale radius**: r_s = r_vir / c
- **Extended mass distribution**: Realistic galaxy halo

## 🔬 Scientific Validation

### Physical Sanity Checks ✓
- ✅ Einstein radius matches theoretical formula (1% accuracy)
- ✅ Image positions near Einstein radius
- ✅ Magnifications satisfy flux conservation
- ✅ Critical surface density ~10⁹ M☉/pc²

### Numerical Accuracy ✓
- ✅ Image positions stable to 0.01 arcsec
- ✅ Magnifications accurate to 1%
- ✅ No crashes at singularities (r=0)

### Code Quality ✓
- ✅ Full NumPy-style docstrings
- ✅ Type hints on all functions
- ✅ 100% test pass rate
- ✅ Edge cases handled gracefully

## 📖 Documentation

### Key Classes

#### `LensSystem`
Handles cosmological calculations for gravitational lensing.

```python
lens_sys = LensSystem(z_lens=0.5, z_source=1.5, H0=70.0, Om0=0.3)
sigma_cr = lens_sys.critical_surface_density()  # Msun/pc²
theta_E = lens_sys.einstein_radius_scale(1e12)  # arcsec
```

#### `PointMassProfile`
Simple point mass lens (exact analytical solution).

```python
lens = PointMassProfile(mass=1e12, lens_system=lens_sys)
alpha_x, alpha_y = lens.deflection_angle(x, y)  # arcsec
kappa = lens.convergence(x, y)  # dimensionless
```

#### `NFWProfile`
Navarro-Frenk-White dark matter halo profile.

```python
halo = NFWProfile(M_vir=1e12, concentration=5, lens_system=lens_sys)
sigma = halo.surface_density(r)  # Msun/pc²
```

#### `ray_trace()`
Main ray shooting algorithm to find lensed images.

```python
results = ray_trace(
    source_position=(0.5, 0.0),
    lens_model=lens,
    grid_extent=3.0,
    grid_resolution=300,
    threshold=0.05
)
```

## 🎯 Research Goals

### Core Question
Can an integrated simulation combining wave optics with alternative dark matter theories resolve ambiguities in lensing data and provide more accurate H₀ measurements?

### Hypothesis
Wave optics effects introduce systematic biases in time-delay measurements that depend on the underlying dark matter model. A Physics-Informed Neural Network can distinguish between CDM, WDM, and SIDM with >90% accuracy.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{lensing_framework_2025,
  author = {Your Name},
  title = {Gravitational Lensing Simulation Framework},
  year = {2025},
  url = {https://github.com/yourusername/financial-advisor-tool}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Astropy**: For cosmological calculations
- **NumPy/SciPy**: For numerical computing
- **Matplotlib**: For visualization
- **Wright & Brainerd (2000)**: NFW deflection formulas

## 📧 Contact

For questions or collaborations:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🗺️ Roadmap

- [x] Phase 1: Geometric ray tracing
- [ ] Phase 2: Physical optics module
- [ ] Phase 3: Time-delay cosmography
- [ ] Phase 4: Alternative dark matter models
- [ ] Phase 5: Physics-Informed Neural Networks
- [ ] Phase 6: Real data validation

---

**Status**: Phase 1 Complete ✅ | **Last Updated**: October 2025
