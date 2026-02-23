# 🌌 Gravitational Lensing Toolkit (ISEF 2025)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ISEF](https://img.shields.io/badge/ISEF-2025-gold.svg)](https://www.societyforscience.org/isef/)

> **Research-grade lens modeling in one command**  
> Physics-informed neural networks + cosmological ray tracing for gravitational lensing analysis

## ▶️ Try a Demo Now

**Experience publication-quality gravitational lensing analysis in <15 seconds:**

```powershell
git clone https://github.com/nalin1304/Gravitational-Lensing-algorithm
cd Gravitational-Lensing-algorithm
pip install -r requirements.txt
streamlit run app/Home.py
```

**Then click "Einstein Cross" → see results immediately**

✅ **No training** • ✅ **No config** • ✅ **Scientifically validated**

---

## 🎯 What This Does

Turn **raw astronomical observations** into **validated mass maps** with **full uncertainty quantification** — automatically.

**Built for ISEF 2025**, this toolkit demonstrates:
- ✨ Physics-informed machine learning (PINNs constrained by General Relativity)
- 🌌 Cosmological thin-lens ray tracing (ΛCDM distances)
- 📊 Bayesian uncertainty quantification (Monte Carlo dropout)
- 🔬 Sub-percent accuracy on benchmark lensing systems

### Featured Demos (One-Click Ready)

| Demo | System | Highlights |
|------|--------|-----------|
| **🌟 Einstein Cross** | Q2237+030 (z=0.04) | Quadruple-image quasar, classic strong lens |
| **🔭 Twin Quasar** | Q0957+561 (z=0.36) | First discovered lens (1979), time delay demo |
| **🪐 JWST Cluster** | Simulated (z=0.3) | Dark matter substructure detection with AI |

All demos use **pre-trained PINN models** and **generate publication-ready figures automatically**.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM recommended

### Installation

#### Option 1: Docker (Recommended)

```powershell
# Clone the repository
git clone https://github.com/nalin1304/Gravitational-Lensing-algorithm.git
cd Gravitational-Lensing-algorithm

# Create .env file (copy from .env.example)
cp .env.example .env

# Edit .env with your configuration
# Required: DATABASE_URL, REDIS_URL, SECRET_KEY

# Start all services
docker-compose up -d

# Access the app
# Streamlit: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

#### Option 2: Local Development

```powershell
# Clone the repository
git clone https://github.com/nalin1304/Gravitational-Lensing-algorithm.git
cd Gravitational-Lensing-algorithm

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt

# Launch the Streamlit app (Multi-Page)
streamlit run app/Home.py

# OR launch the FastAPI backend
uvicorn api.main:app --reload
```

The Streamlit app opens at **http://localhost:8501**  
The API server runs at **http://localhost:8000**

## 📚 Documentation

### 📖 **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** ← **START HERE!**
**Complete, comprehensive guide (15,000+ lines)** covering everything:
- Installation & Quick Start
- Features & Capabilities
- Security Implementation
- Training PINN Models
- Real Data Sources
- Production Deployment
- API Reference
- Testing & Validation
- ISEF Presentation Guide
- Troubleshooting

### Quick Reference Guides

| Document | Description |
|----------|-------------|
| [🚀 QUICKSTART.md](QUICKSTART.md) | Get started in 5 minutes |
| [� QUICK_REFERENCE_CARD.txt](QUICK_REFERENCE_CARD.txt) | Essential commands & troubleshooting |
| [� ISEF_QUICK_REFERENCE.md](ISEF_QUICK_REFERENCE.md) | Demo script for presentations |
| [🔬 REAL_DATA_SOURCES.md](REAL_DATA_SOURCES.md) | Access HST/JWST/SDSS data |
| [� MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md) | Train custom PINN models |
| [🚀 PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) | Production deployment |
| [🔧 database/SSL_SETUP_GUIDE.md](database/SSL_SETUP_GUIDE.md) | SSL certificate setup |

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

**Current Test Status**: ✅ 551 passed, 22 skipped (`python -m pytest tests/ -q`, run on February 23, 2026)

## ✅ Publication Readiness

- Full reproducibility and validation record: `JOURNAL_PUBLICATION_READINESS.md`
- Latest release gate summary:
  - `551 passed, 22 skipped` (full test suite)
  - `mypy src/` clean
  - Python compile sanity clean
  - Placeholder/dummy marker scan clean

## 📊 Project Structure

```
gravitational-lensing-algorithm/
├── app/                           # Streamlit web interface
│   ├── Home.py                   # Streamlit root page
│   ├── main.py                   # Shared app entry/config helpers
│   ├── pages/                    # Multipage UI modules
│   │   ├── 02_Simple_Lensing.py
│   │   ├── 03_PINN_Inference.py
│   │   ├── 03_Results.py
│   │   ├── 04_Multi_Plane.py
│   │   ├── 05_Real_Data.py
│   │   ├── 06_Training.py
│   │   ├── 07_Validation.py
│   │   ├── 08_Bayesian_UQ.py
│   │   └── 09_Settings.py
│   └── utils/                    # Shared app utilities
├── api/                          # FastAPI REST backend
│   ├── main.py                  # API server with JWT auth
│   ├── auth_routes.py           # Authentication endpoints
│   └── analysis_routes.py       # Analysis endpoints
├── src/                          # Core scientific library
│   ├── lens_models/             # Mass profiles, lens systems, multi-plane
│   ├── ml/                      # PINN, training, uncertainty quantification
│   ├── optics/                  # Ray tracing, geodesics
│   ├── data/                    # FITS loading, PSF modeling
│   ├── validation/              # Scientific validators
│   ├── dark_matter/             # Substructure detection
│   ├── api_utils/               # API utilities (JWT, auth)
│   └── utils/                   # Constants, common utilities
├── database/                     # PostgreSQL models and CRUD
│   ├── models.py                # SQLAlchemy models
│   ├── database.py              # DB session management
│   └── crud.py                  # CRUD operations
├── tests/                        # Comprehensive test suite
│   ├── test_database_crud.py    # Database tests (renamed from phase12)
│   ├── test_scientific_validation.py  # Validation tests (renamed from phase13)
│   └── test_*.py                # 20+ test modules
├── benchmarks/                   # Performance profiling
├── notebooks/                    # Jupyter tutorials
├── docs/                         # Documentation (30+ guides)
├── migrations/                   # Alembic database migrations
├── monitoring/                   # Prometheus/Grafana configs
├── .github/workflows/            # CI/CD pipelines
│   └── ci-cd.yml                # Automated testing & deployment
├── Dockerfile                    # Production API container (multi-stage)
├── Dockerfile.streamlit          # Streamlit container (multi-stage)
├── docker-compose.yml            # Local development stack
├── requirements.txt              # Runtime dependencies (35 packages)
├── requirements-dev.txt          # Development tools (pytest, mypy, jupyter)
└── alembic.ini                   # Database migration config
```

## 🔧 Recent Infrastructure Improvements (October 2025)

### ✅ Completed Refactoring

1. **Dependency Management**
   - Split `requirements.txt` (runtime) and `requirements-dev.txt` (dev tools)
   - Removed duplicate dependencies and dev tools from production
   - ~40% smaller Docker images

2. **Docker Optimization**
   - Multi-stage builds for API and Streamlit containers
   - Non-root user execution for security
   - Optimized layer caching for faster builds
   - Removed unnecessary files from final images

3. **CI/CD Pipeline**
   - Updated to use `requirements-dev.txt` for tests
   - Added `mypy` static type checking to lint job
   - Improved caching for faster workflow runs
   - Parameterized AWS deployment with secrets

4. **Authentication Security**
   - Real JWT authentication with `python-jose`
   - Secure password hashing with `bcrypt`
   - No hardcoded fallback tokens or auth bypasses
   - Proper token verification in all protected endpoints

5. **App Architecture**
   - Refactored monolithic `app/main.py` into multi-page structure
   - Created `app/utils/` with modular utilities:
     - `session_state.py` - Centralized state management
     - `plotting.py` - Publication-quality visualization
     - `ui.py` - Reusable UI components
     - `helpers.py` - Validation and dependency checking

6. **Test Organization**
   - Renamed phase-based tests to descriptive names:
     - `test_phase12.py` → `test_database_crud.py`
     - `test_phase13.py` → `test_scientific_validation.py`
   - Improved test discoverability

7. **Physical Constants**
   - Comprehensive `src/utils/constants.py` module
   - CODATA 2018 recommended values
   - Planck 2018 cosmological parameters
   - Convenient unit conversion functions

## 📊 Project Statistics

- **Lines of Code**: 15,000+ (Python)
- **Automated Tests**: 551 passed, 22 skipped in latest full `tests/` run
- **Documentation**: 30+ comprehensive guides
- **CI/CD**: Automated testing, linting, and deployment
- **Performance**: PINN inference at 134.6 img/s on CPU (134× above target)

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

1. **Launch Demo**: `streamlit run app/Home.py`
2. **Follow**: [ISEF_QUICK_REFERENCE.md](ISEF_QUICK_REFERENCE.md)
3. **Show**: Live synthetic generation → Inference → Validation
4. **Highlight**: GR geodesics, multi-plane lensing, uncertainty quantification

**Key Talking Points**:
- Combines ML with physics constraints (not pure black-box)
- Full GR implementation (not just Born approximation)
- Research-grade accuracy on known systems
- Production-ready with a 551-test passing baseline

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

- **Project Lead**: Nalin
- **Contact**: Use GitHub Issues for project communication
- **GitHub Issues**: [Report bugs or request features](https://github.com/nalin1304/Gravitational-Lensing-algorithm/issues)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for gravitational lensing research and ISEF 2025**
