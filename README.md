# 🌌 Gravitational Lensing Analysis Toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Advanced gravitational lensing simulation and analysis toolkit with Physics-Informed Neural Networks, General Relativity integration, and multi-plane cosmological modeling.**

![Project Banner](docs/images/banner.png)

## 🎯 Overview

This toolkit provides a comprehensive platform for simulating, analyzing, and validating gravitational lensing phenomena. Built for the **Intel International Science and Engineering Fair (ISEF)**, it combines cutting-edge machine learning with rigorous physics modeling.

### ✨ Key Features

- **🤖 Physics-Informed Neural Networks (PINNs)**: Deep learning models constrained by gravitational lensing equations
- **⚡ General Relativity Integration**: Full geodesic integration using Schwarzschild metric
- **🌌 Multi-Plane Lensing**: Cosmologically accurate modeling of multiple lens planes
- **📊 Real Data Support**: Load and analyze HST, JWST, and SDSS observations
- **🎯 Bayesian Uncertainty Quantification**: Rigorous uncertainty estimation with calibration
- **🔬 Scientific Validation**: Automated validation against known lensing systems
- **🔭 Substructure Detection**: Dark matter sub-halo identification algorithms
- **📈 Interactive Web Interface**: Professional Streamlit dashboard with 12 analysis modes

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM recommended

### Installation

```powershell
# Clone the repository
git clone https://github.com/yourusername/gravitational-lensing-algorithm.git
cd gravitational-lensing-algorithm

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app/main.py
```

The app will open at **http://localhost:8501**

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [🚀 QUICKSTART.md](QUICKSTART.md) | Get started in 5 minutes |
| [📖 COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md) | Detailed installation and troubleshooting |
| [🎓 MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md) | Train custom PINN models (500+ lines) |
| [🔬 REAL_DATA_SOURCES.md](REAL_DATA_SOURCES.md) | Access HST/JWST/SDSS data |
| [🏆 ISEF_QUICK_REFERENCE.md](ISEF_QUICK_REFERENCE.md) | Demo script for presentations |
| [⚙️ CONFIG_SETUP.md](CONFIG_SETUP.md) | Configuration and deployment |

## 🎨 Features Overview

### 1. Synthetic Data Generation
Generate convergence maps from NFW profiles with:
- Customizable mass, concentration, ellipticity
- Multiple dark matter models (CDM, WDM, SIDM)
- Realistic noise simulation

### 2. Real Data Analysis
- **FITS file support**: Load HST/JWST observations
- **PSF modeling**: Gaussian, Airy, Moffat PSFs
- **Preprocessing pipeline**: Normalization, background subtraction
- **WCS coordinate handling**: Astropy integration

### 3. Model Inference
- **Pre-trained PINNs**: Instant parameter estimation
- **Custom architectures**: Conv2D + Dense layers
- **GPU acceleration**: CUDA support
- **Batch processing**: Analyze multiple images

### 4. Uncertainty Quantification
- **Monte Carlo Dropout**: Sample uncertainty distributions
- **Bayesian calibration**: Temperature scaling
- **Confidence intervals**: 95% credible regions
- **Visualization**: Uncertainty heatmaps

### 5. Scientific Validation
- **Known systems**: Einstein Cross, Twin Quasar, etc.
- **Automated metrics**: Relative errors, correlations
- **Ground truth comparison**: Validate against literature
- **Research-grade accuracy**: Publication-ready results

### 6. Multi-Plane Lensing
- **Cosmological distances**: FlatLambdaCDM
- **Multiple lens planes**: Cumulative deflection
- **Redshift evolution**: z = 0.1 to 4.0
- **3D ray tracing**: Full light path simulation

### 7. GR vs Simplified Comparison
- **Schwarzschild geodesics**: Numerical integration
- **Born approximation**: Standard thin-lens
- **Error analysis**: Quantify approximation validity
- **Impact parameter study**: Strong vs weak lensing regimes

### 8. Substructure Detection
- **Sub-halo generation**: Realistic mass functions
- **Perturbation analysis**: Identify anomalies
- **Statistical tests**: Chi-squared, KS tests
- **Mass reconstruction**: Infer substructure properties

## 🧪 Testing

```powershell
# Run all tests
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_lens_system.py -v
python -m pytest tests/test_ml.py -v
python -m pytest tests/test_mass_profiles.py -v

# Check imports
python test_imports.py
```

**Current Test Status**: ✅ 61/61 tests passing (100%)

## 📊 Project Structure

```
gravitational-lensing-algorithm/
├── app/                    # Streamlit web interface
│   ├── main.py            # Main application (3,142 lines)
│   └── styles.py          # Custom CSS styling
├── src/                   # Core library
│   ├── lens_models/       # Mass profiles, lens systems
│   ├── ml/                # PINN, training, uncertainty
│   ├── optics/            # Ray tracing, geodesics
│   ├── data/              # FITS loading, preprocessing
│   ├── validation/        # Scientific validators
│   └── dark_matter/       # Substructure detection
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter tutorials
├── docs/                  # Comprehensive documentation
├── scripts/               # Utility scripts
├── benchmarks/            # Performance profiling
└── requirements.txt       # Python dependencies
```

## 🔬 Scientific Background

### Physics

This toolkit implements gravitational lensing based on:
- **Einstein's General Relativity**: Full geodesic equations
- **Lens equation**: θ = β + α(θ)
- **Convergence**: κ = Σ / Σ_crit
- **Deflection angle**: α = (4GM/c²) × (D_LS / D_L × D_S)

### Machine Learning

Our Physics-Informed Neural Networks:
- **Architecture**: Conv2D → Dense → Dual heads (regression + classification)
- **Loss function**: MSE + physics constraints + classification cross-entropy
- **Training data**: 50,000+ synthetic convergence maps
- **Uncertainty**: Monte Carlo dropout + Bayesian calibration

### Validation

Tested against:
- **Einstein Cross (Q2237+0305)**: z_lens=0.04, z_source=1.695
- **Twin Quasar (Q0957+561)**: First discovered gravitational lens
- **SDSS J1004+4112**: Five-image quasar lens system
- **Literature values**: Sub-5% error on Einstein radii

## 🏆 ISEF Presentation

This project was developed for Intel ISEF 2025. For judges and presenters:

1. **Launch Demo**: `streamlit run app/main.py`
2. **Follow**: [ISEF_QUICK_REFERENCE.md](ISEF_QUICK_REFERENCE.md)
3. **Show**: Live synthetic generation → Inference → Validation
4. **Highlight**: GR geodesics, multi-plane lensing, uncertainty quantification

**Key Talking Points**:
- Combines ML with physics constraints (not pure black-box)
- Full GR implementation (not just Born approximation)
- Research-grade accuracy on known systems
- Production-ready with 61 passing tests

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```powershell
# Install dev dependencies
pip install -r requirements.txt pytest black flake8

# Run tests
pytest tests/ -v

# Format code
black src/ app/ tests/

# Lint
flake8 src/ app/ tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **Astropy Community**: FITS file handling
- **PyTorch Team**: Deep learning framework
- **Streamlit**: Interactive web framework
- **ISEF**: Motivation and platform

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/gravitational-lensing-algorithm/issues)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for gravitational lensing research and ISEF 2025**
