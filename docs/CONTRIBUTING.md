# Contributing to Gravitational Lensing Tool

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Scientific Validation](#scientific-validation)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/financial-advisor-tool.git
   cd financial-advisor-tool
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/financial-advisor-tool.git
   ```

## Development Setup

### Prerequisites

- Python 3.9+ (3.10 recommended)
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Unix/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 isort mypy
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Run linting
black --check src/ tests/
flake8 src/ tests/
```

## How to Contribute

### Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:
- Clear description
- Reproducible example
- Environment details
- Error messages

### Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if applicable)
- Scientific references (if applicable)

### Contributing Code

1. **Find or create an issue** describing what you'll work on
2. **Create a branch** from `develop`:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following coding standards
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add elliptical NFW profile"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Open a Pull Request** using the PR template

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **Black** for code formatting (line length: 100)
- Use **isort** for import sorting
- Use **type hints** for function signatures

```python
from typing import Tuple, Optional
import numpy as np

def compute_convergence(
    x: np.ndarray, 
    y: np.ndarray, 
    M_vir: float,
    c: float
) -> np.ndarray:
    """
    Compute convergence map for NFW profile.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates in arcseconds
    y : np.ndarray
        Y coordinates in arcseconds
    M_vir : float
        Virial mass in solar masses
    c : float
        Concentration parameter
    
    Returns
    -------
    convergence : np.ndarray
        Dimensionless surface mass density
    """
    # Implementation here
    pass
```

### Docstring Format

Use **NumPy-style docstrings**:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Short description (one line).
    
    Longer description with more details about what the function does,
    its purpose, and any important notes.
    
    Parameters
    ----------
    param1 : type1
        Description of param1
    param2 : type2
        Description of param2
    
    Returns
    -------
    return_name : return_type
        Description of return value
    
    Raises
    ------
    ValueError
        When input is invalid
    
    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0
    
    Notes
    -----
    Any additional scientific context or references.
    
    References
    ----------
    .. [1] Author et al. (Year), Journal, Volume, Pages
    """
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `NFWProfile`, `PhysicsInformedNN`)
- **Functions/methods**: `snake_case` (e.g., `compute_convergence`, `train_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SPEED_OF_LIGHT`, `G_NEWTON`)
- **Private methods**: `_leading_underscore` (e.g., `_compute_internal`)

### Physical Units

Always specify units in docstrings and use consistent conventions:

- **Distances**: kpc, Mpc (physical), arcseconds (angular)
- **Masses**: Solar masses (M☉)
- **Time**: Gyr (cosmological), days (time delays)
- **Cosmology**: H₀ in km/s/Mpc, dimensionless Ω

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- One test file per module

```python
# tests/test_new_profile.py

import pytest
import numpy as np
from src.lens_models.advanced_profiles import EllipticalNFWProfile

class TestEllipticalNFWProfile:
    """Test suite for elliptical NFW profile."""
    
    def test_initialization(self):
        """Test profile can be initialized with valid parameters."""
        profile = EllipticalNFWProfile(
            M_vir=1e12, c=10.0, lens_sys=None,
            ellipticity=0.3, position_angle=45.0
        )
        assert profile.ellipticity == 0.3
        assert profile.position_angle == 45.0
    
    def test_reduces_to_circular_nfw(self):
        """Test that ellipticity=0 gives circular NFW."""
        # Implementation
        pass
    
    def test_convergence_symmetry(self):
        """Test convergence respects elliptical symmetry."""
        # Implementation
        pass
    
    @pytest.mark.parametrize("ellipticity", [0.0, 0.2, 0.5, 0.8])
    def test_various_ellipticities(self, ellipticity):
        """Test profile works for different ellipticities."""
        # Implementation
        pass
```

### Test Coverage

- Aim for **>80% code coverage**
- All new functions must have tests
- Test edge cases and error conditions

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Performance Tests

For performance-critical code, add benchmarks:

```python
# tests/test_performance.py

def test_convergence_map_speed(benchmark):
    """Benchmark convergence map generation."""
    profile = NFWProfile(M_vir=1e12, c=10.0, lens_sys=None)
    
    def compute():
        return profile.convergence_map(grid_size=128)
    
    result = benchmark(compute)
    assert result.shape == (128, 128)
```

## Documentation

### Updating Documentation

- Update docstrings for all modified functions
- Update `readme.md` if adding new features
- Create/update tutorial notebooks for significant features
- Add examples to `docs/examples/`

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme numpydoc

# Build docs
cd docs/
make html

# View docs
open _build/html/index.html
```

## Pull Request Process

1. **Update CHANGELOG.md** with your changes
2. **Ensure all tests pass** locally
3. **Ensure code is formatted**:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```
4. **Update version numbers** if applicable (semantic versioning)
5. **Open PR** using the template
6. **Address review comments**
7. **Squash commits** before merge (optional)

### PR Review Criteria

Your PR will be reviewed for:
- ✅ Code quality and style
- ✅ Test coverage
- ✅ Documentation completeness
- ✅ Scientific correctness
- ✅ Performance impact
- ✅ Backward compatibility

## Scientific Validation

For changes to physical models or algorithms:

### 1. Analytical Validation

Compare against known analytical solutions:

```python
def test_nfw_analytical_mass():
    """Test NFW mass matches analytical formula."""
    profile = NFWProfile(M_vir=1e12, c=10.0, lens_sys=None)
    
    # Compute numerical mass within r_200
    numerical_mass = profile.mass_within_radius(profile.r_200)
    
    # Should equal M_vir by definition
    np.testing.assert_allclose(numerical_mass, 1e12, rtol=0.01)
```

### 2. Cross-Tool Validation

Benchmark against established tools:

- **Lenstronomy**: https://github.com/lenstronomy/lenstronomy
- **PyAutoLens**: https://github.com/Jammy2211/PyAutoLens
- **LensTool**: https://projets.lam.fr/projects/lenstool

```python
def test_compare_lenstronomy():
    """Compare convergence with Lenstronomy."""
    from lenstronomy.LensModel.Profiles.nfw import NFW as LenstroNFW
    
    # Setup matching profiles
    our_profile = NFWProfile(...)
    lenstro_profile = LenstroNFW()
    
    # Compare convergence values
    # ...
```

### 3. Physical Reasonableness

- Check units are correct
- Verify values are in physical range
- Test conservation laws (e.g., mass conservation)
- Validate against observations

## Git Commit Messages

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(lens_models): add elliptical NFW profile

Implement elliptical NFW profile with position angle parameter.
Includes coordinate transformation and validation tests.

Closes #42
```

```
fix(optics): correct phase calculation in wave optics

Fixed sign error in Fermat potential calculation that was causing
incorrect interference patterns for extended sources.

Fixes #87
```

## Questions?

If you have questions not covered here:
- Open a [Discussion](https://github.com/OWNER/REPO/discussions)
- Check existing [Issues](https://github.com/OWNER/REPO/issues)
- Contact maintainers

## Thank You!

Your contributions help advance gravitational lensing research and make this tool better for the entire community. Every contribution, no matter how small, is valued and appreciated! 🌟
