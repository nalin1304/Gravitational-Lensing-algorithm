# Computational Imaging Research Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-green.svg)](https://jax.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-647%20passed-brightgreen.svg)](#testing)

> Physics-constrained gravitational lensing toolkit with analytic validation, real HST/SLACS data diagnostics, differentiable inference, and a browser-based research workbench.

## Quick Start

```bash
git clone https://github.com/nalin1304/Gravitational-Lensing-algorithm.git
cd Gravitational-Lensing-algorithm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open **http://localhost:8000/ui** for the web workbench, **http://localhost:8000/docs** for the OpenAPI explorer.

## What This Does

A reproducible computational-imaging platform combining thin-lens and multi-plane gravitational lensing physics, JAX/Equinox and PyTorch ML components, real HST/SLACS data ingestion, and a web UI that exposes validation artifacts directly.

### Scientific Core

| Module | Description | Key References |
|--------|-------------|----------------|
| **Lens Models** | NFW, SIS, power-law, Sérsic profiles; multi-plane ray tracing | NFW (1997); Wright & Brainerd (2000); Schneider (1992) |
| **Critical Curves** | Caustics, magnification maps, image classification | Schneider (1992) §5.3; Birrer & Amara (2018) |
| **Wave Optics** | Diffraction integral F(ω) for interference/magnification | Nakamura & Deguchi (1999); Takahashi & Nakamura (2003) |
| **Time Delays** | Fermat potential cosmography | Refsdal (1964); Schneider (1992) |
| **PINN** | Physics-informed neural nets with ∇²ψ = 2κ constraint | Schneider (1992) Eq. 3.11 |
| **NUTS-HMC** | Differentiable lensing simulator + No-U-Turn Sampler | Hoffman & Gelman (2014) |
| **PI-SBI** | Physics-informed simulation-based inference (EM + GW) | Novel contribution |
| **Kinematics** | Jeans equation, mass-sheet degeneracy testing | Treu & Koopmans (2004) |
| **Nested Sampling** | Bayesian model evidence (NFW vs SIS) | Skilling (2004) |
| **Source Models** | GP-regularized pixelized source reconstruction | — |

### Validation Pipeline

- **Known systems**: Einstein Cross (Q2237+030), Twin Quasar (Q0957+561) with raw Einstein-radius errors < 2.3%
- **SLACS diagnostics**: PSF-convolved lensed Sérsic fits on real HST/ACS data (5/5 systems pass)
- **Uncertainty calibration**: MC-dropout on synthetic NFW analogs (ECE ≈ 0.062, coverage@90 ≈ 0.936)
- **Reproducibility gate**: `python3 scripts/publication_gate.py --quick`

### Web Interface

NASA-inspired dark theme with 9 pages: Dashboard, Workbench, Validation, Analyses, Stage IV Survey, Inference, Lensing Analysis, API Explorer, Account. All pages consume live API data — no hardcoded fallbacks.

## Project Structure

```
├── api/                    FastAPI backend (55 endpoints)
│   ├── main.py             Core routes
│   ├── auth_routes.py      JWT authentication
│   └── analysis_routes.py  Analysis CRUD
├── src/                    Scientific library
│   ├── lens_models/        Mass profiles, lens system, multi-plane
│   ├── ml/                 PINN, neural ODE, nested sampling, PI-SBI
│   ├── optics/             Ray tracing, wave optics, ePSF
│   ├── inference/          Differentiable simulator, NUTS-HMC
│   ├── validation/         HST targets, kinematics, diagnostics
│   ├── data/               MAST downloader, pixel covariance
│   ├── time_delay/         Fermat potential cosmography
│   └── utils/              Constants, blinding
├── web_ui/                 Static frontend (served at /ui)
├── paper/                  IEEE TCI manuscript + BibTeX
├── scripts/                Benchmark and reproducibility scripts
├── tests/                  647 tests
├── demos/                  YAML preset configurations
└── notebooks/              Colab quickstart
```

## Testing

```bash
# Full suite (647 tests)
uv run python -m pytest tests/ -q

# Publication gate
python3 scripts/publication_gate.py --quick

# One-command reproducibility (6 benchmarks)
bash scripts/reproduce.sh
```

**Current status**: 647 passed, 1 skipped

## Docker

```bash
cp .env.example .env   # Configure DB_PASSWORD, SECRET_KEY
docker-compose up -d
# Web UI: http://localhost:8000/ui
```

## Documentation

| Document | Purpose |
|----------|---------|
| **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** | Comprehensive architecture and usage guide |
| **[AGENTS.md](AGENTS.md)** | Operational context, file map, validation workflow |
| **[IEEE_SUBMISSION_CHECKLIST.md](IEEE_SUBMISSION_CHECKLIST.md)** | Publication readiness checklist |
| **[paper/main.tex](paper/main.tex)** | IEEE TCI manuscript |

## Key Design Principles

1. **No silent fallbacks** — missing checkpoints return 503, missing data raises errors
2. **Explicit provenance** — every formula cites its source paper and equation number
3. **Functional randomness** — all stochastic components accept `seed: int` parameters
4. **Guard all imports** — optional deps (JAX, PyTorch) use try/except, never crash at import
5. **Checkpoint-gated inference** — no heuristic stand-ins when models are unavailable

## License

MIT — see [LICENSE](LICENSE)

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata.
