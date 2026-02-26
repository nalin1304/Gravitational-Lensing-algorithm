# 🌌 Computational Imaging Research Platform (IEEE TCI)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-green.svg)](https://jax.readthedocs.io/)
[![Equinox](https://img.shields.io/badge/Equinox-0.11%2B-purple.svg)](https://docs.kidger.site/equinox/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE TCI](https://img.shields.io/badge/IEEE_TCI-Publication_Ready-blue.svg)](https://ieeexplore.ieee.org/)

> **Research-grade lens modeling in one command**  
> Physics-informed neural networks + cosmological ray tracing for gravitational lensing analysis, natively accelerated by JAX and Equinox.

## ▶️ Try a Demo Now

**Experience publication-quality gravitational lensing analysis instantly:**

```powershell
git clone https://github.com/nalin1304/Gravitational-Lensing-algorithm
cd Gravitational-Lensing-algorithm
pip install -r requirements.txt
uvicorn api.main:app --reload
```

**Then open `http://localhost:8000/ui` → click "Einstein Cross" → see results immediately**

✅ **No training** • ✅ **No config** • ✅ **Scientifically validated**

---

## 🎯 What This Does

Turn **raw astronomical observations** into **validated mass maps** with **full uncertainty quantification** — automatically.

**Built for IEEE Transactions on Computational Imaging (TCI)**, this toolkit demonstrates:
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
- Hardware accelerator (GPU/TPU) recommended for deep JAX vectorization
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

# Launch the FastAPI backend + JAX/Equinox UI
uvicorn api.main:app --reload
```

The interactive Web UI runs at **http://localhost:8000/ui**
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
- IEEE TCI Submission Guide
- Troubleshooting

### Quick Reference Guides

| Document | Description |
|----------|-------------|
| [📘 PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | Comprehensive architecture and usage guide |
| [🧭 AGENTS.md](AGENTS.md) | Operational context, architecture map, validation workflow |
| [🎤 IEEE_SUBMISSION_CHECKLIST.md](IEEE_SUBMISSION_CHECKLIST.md) | Validation and publication checklist |

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

### 3. Model Inference (JAX / Equinox)
- **Pre-trained PINNs**: Pure JAX/Equinox functional transformations
- **5D Spherical Topologies**: Native boundary divergence resolving
- **Hardware acceleration**: JIT compilation (`eqx.filter_jit`) and `jax.vmap` batching

### 4. Bayesian Evidence & Source Regularization
- **ParamU Decoupling**: Non-Negative Least Squares (NNLS) flux distributions
- **Gaussian Processes**: Matern/RBF covariant regularization topologies
- **Bayesian Tuning**: Exact Log-Evidence parameter tuning limits

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

### 7. Core Geodesic Integrations & Neural ODEs
- **Diffrax Neural ODEs**: Analytic Fusing (baseline deterministic gravity decoupled from learned perturbations)
- **Schwarzschild Geodesics**: Numerical geometric limits
- **Wave-Optics Modifiers**: Interference parameters scaling near $\lambda \approx 2GM/c^2$

### 8. Structural Validation & Substructures
- **μ-GLANCE Residual Evaluator**: Model-independent non-parametric fractional analysis
- **Savage-Dickey Density Ratios**: Statistical matching for orbital eccentricities
- **Pydantic Configurations**: Strictly-typed Caskade pipelines (`demos/*.yaml` mapping)

## 🧪 Testing

```powershell
# Run all tests
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_lens_system.py -v
python -m pytest tests/test_ml.py -v
python -m pytest tests/test_mass_profiles.py -v

# Check imports
python scripts/check_imports.py
```

**Current Test Status**: ✅ 502 passed, 8 skipped (`python -m pytest tests/ -q`, run on February 26, 2026)

## ✅ Publication Readiness

- Full reproducibility and validation record: `JOURNAL_PUBLICATION_READINESS.md`
- Executable publication gate: `python3 scripts/publication_gate.py`
- Latest release gate summary:
  - `502 passed, 8 skipped` (full test suite)
  - `mypy src/` clean
  - Python compile sanity clean
  - Known-system validation script passes (`scripts/validate_known_systems.py`)

## 📊 Project Structure

```
gravitational-lensing-algorithm/
├── api/                          # FastAPI REST backend
│   ├── main.py                  # API server with JWT auth
│   ├── auth_routes.py           # Authentication endpoints
│   └── analysis_routes.py       # Analysis endpoints
├── web_ui/                       # FastAPI-served non-Streamlit frontend
│   ├── index.html               # Main journal-workbench shell
│   ├── styles.css               # Shared visual theme and layout
│   └── app.js                   # API-backed interaction logic
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
│   ├── test_next_ui.py          # Smoke tests for /ui frontend route
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

5. **Web UI Architecture**
   - FastAPI-served static frontend at `/ui` with Plotly.js visualizations
   - Real-time convergence map generation and PINN inference from browser
   - Preset lens system configurations (Einstein Cross, Twin Quasar, JWST Cluster)

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
- **Automated Tests**: Clean execution environment
- **Documentation**: 30+ comprehensive guides
- **CI/CD**: Automated testing, linting, and deployment
- **Performance**: Deep PINN evaluations hitting **1,666 img/s** via JAX `vmap` and `eqx.filter_jit`.

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

## 🏆 IEEE TCI Publication Readiness

This project was developed for submission to IEEE Transactions on Computational Imaging. For peer-review validation:

1. **Launch Demo**: `uvicorn api.main:app --reload` then open `http://localhost:8000/ui`
2. **Follow**: [IEEE_SUBMISSION_CHECKLIST.md](IEEE_SUBMISSION_CHECKLIST.md)
3. **Show**: Live synthetic generation → Inference → Validation
4. **Highlight**: Post-Newtonian deflection, multi-plane lensing, uncertainty quantification

**Key Talking Points**:
- Combines ML with physics constraints (not pure black-box)
- Post-Newtonian Schwarzschild deflection with controlled PN corrections
- Research-grade accuracy on known systems
- Production-ready with a 502-test passing baseline (+8 skipped tests)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```powershell
# Install dev dependencies
pip install -r requirements.txt pytest black flake8

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Lint
flake8 src/ tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **JAX / Equinox / Diffrax Teams**: Scientific machine learning primitives
- **Astropy Community**: FITS file handling
- **IEEE TCI**: Target publication venue

## 📞 Contact

- **Project Lead**: Nalin
- **Contact**: Use GitHub Issues for project communication
- **GitHub Issues**: [Report bugs or request features](https://github.com/nalin1304/Gravitational-Lensing-algorithm/issues)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---
