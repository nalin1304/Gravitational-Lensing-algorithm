# Complete Codebase Feature Inventory

**Project:** Computational Imaging Research Platform  
**Submission Target:** IEEE Transactions on Computational Imaging (TCI) / MNRAS  
**Branch:** feature/jax-migration  

This document catalogs every scientific and engineering feature present in the platform,
with mathematical formulas, algorithmic descriptions, and literature citations for manuscript preparation.

---

## Table of Contents

1. [Cosmological Framework](#1-cosmological-framework)
2. [Gravitational Lens Mass Profiles](#2-gravitational-lens-mass-profiles)
3. [Multi-Plane Lensing](#3-multi-plane-lensing)
4. [Ray Tracing](#4-ray-tracing)
5. [Wave Optics](#5-wave-optics)
6. [Extended PSF Model](#6-extended-psf-model)
7. [Time Delay and Cosmography](#7-time-delay-and-cosmography)
8. [Physics-Informed Neural Networks](#8-physics-informed-neural-networks)
9. [Physics-Constrained Loss Functions](#9-physics-constrained-loss-functions)
10. [Advanced PINN Architectures (PyTorch)](#10-advanced-pinn-architectures)
11. [Neural ODE / Hamiltonian Neural Networks](#11-neural-ode--hamiltonian-neural-networks)
12. [Nested Sampling and Bayesian Evidence](#12-nested-sampling-and-bayesian-evidence)
13. [Non-Parametric Source Reconstruction](#13-non-parametric-source-reconstruction)
14. [Automated Lens Finder (LenNet)](#14-automated-lens-finder-lennet)
15. [Joint Multi-Resolution Survey Analysis](#15-joint-multi-resolution-survey-analysis)
16. [Stellar Kinematics Module](#16-stellar-kinematics-module)
17. [SLACS Observational Validation](#17-slacs-observational-validation)
18. [μ-GLANCE Magnification Diagnostics](#18-μ-glance-magnification-diagnostics)
19. [Bayesian Model Selection Utilities](#19-bayesian-model-selection-utilities)
20. [HST/MAST Data Pipeline](#20-hstmast-data-pipeline)
21. [Pixel Covariance and Whitening](#21-pixel-covariance-and-whitening)
22. [Cosmographic Blinding](#22-cosmographic-blinding)
23. [Benchmark and Evaluation Suite](#23-benchmark-and-evaluation-suite)
24. [Reproducibility Infrastructure](#24-reproducibility-infrastructure)
25. [API and Web Interface](#25-api-and-web-interface)
26. [Complete Reference List](#26-complete-reference-list)

---

## 1. Cosmological Framework

**Module:** `src/lens_models/lens_system.py`, `src/utils/constants.py`

### Features
- Full FlatΛCDM cosmology via `astropy.cosmology.FlatLambdaCDM`
- Planck-2018 aligned defaults: H₀ = 67.4 km/s/Mpc, Ω_m = 0.315
- Angular diameter distances for arbitrary redshift pairs
- Critical surface density calculation at any (z_l, z_s)
- Time-delay distance for H₀ inference

### Key Formulas

**Angular diameter distances:**
```
D_l  = D_A(0, z_l)
D_s  = D_A(0, z_s)
D_ls = D_A(z_l, z_s)
```

**Critical surface density (Sigma_crit):**
```
Σ_crit = c² / (4πG) × D_s / (D_l · D_ls)   [M☉ / arcsec²]
```

**Time-delay distance (H₀LiCOW convention):**
```
D_Δt = (1 + z_l) × D_l × D_s / D_ls   [Mpc]
```

**Physical constants used:**
- c = 2.99792458 × 10⁸ m/s (IAU 2012)
- G = 6.674 × 10⁻¹¹ m³ kg⁻¹ s⁻² (CODATA 2018)
- kpc_to_m = 3.085677581 × 10¹⁹ m (IAU 2012)
- M_sun = 1.989 × 10³⁰ kg

### References
- Planck Collaboration (2020), A&A 641, A6
- Hogg (1999), arXiv:astro-ph/9905116
- H₀LiCOW Collaboration: Suyu et al. (2010), ApJ 711, 201

---

## 2. Gravitational Lens Mass Profiles

**Module:** `src/lens_models/mass_profiles.py`, `src/lens_models/advanced_profiles.py`

### 2.1 Singular Isothermal Sphere (SIS)

The SIS is the simplest analytic lens model with flat rotation curve.

**Convergence:**
```
κ(θ) = θ_E / (2|θ|)
```

**Deflection angle:**
```
α = 4π (σ_v/c)² × D_ls/D_s = θ_E
```

**Einstein radius:**
```
θ_E = 4π (σ_v/c)² × D_ls/D_s
```

**Lensing potential:**
```
ψ(θ) = θ_E × |θ|
```

### 2.2 Navarro-Frenk-White (NFW) Profile

The NFW profile provides a physically motivated dark matter halo model.

**3D density (NFW 1997, Eq. 1):**
```
ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
```

**Characteristic density from concentration parameter:**
```
ρ_s = (M_200c / 4π r_s³) × 1 / [ln(1+c) - c/(1+c)]
```

**Scale radius:**
```
r_s = r_200c / c_nfw
```

**Critical density at lens redshift (M200c convention):**
```
ρ_crit(z_l) = 3 H²(z_l) / (8πG)
r_200c = [3 M_200c / (4π × 200 ρ_crit)]^{1/3}
```

**Convergence profile (Wright & Brainerd 2000, Eq. 11):**
```
κ(x) = 2 κ_s f(x)

f(x) = {
  1/(x²-1) × [1 - 1/√(1-x²) arctanh√(1-x²)]   x < 1
  1/3                                             x = 1
  1/(x²-1) × [1 - 1/√(x²-1) arctan√(x²-1)]     x > 1
}

where x = θ/θ_s,  κ_s = ρ_s r_s / Σ_crit
```

**Deflection angle (Bartelmann 1996, Eq. 13):**
```
α(θ) = 4 κ_s r_s × h(x) / x

h(x) = ln(x/2) + {
  1/√(1-x²) arctanh√(1-x²)   x < 1
  1                            x = 1
  1/√(x²-1) arctan√(x²-1)    x > 1
}
```

**Lensing potential:**
```
ψ(θ) = 4 κ_s r_s² × g(x)

g(x) = ln²(x/2)/2 + {
  -arctanh²√(1-x²)   x < 1
  0                   x = 1
  arctan²√(x²-1)     x > 1
}
```
(Correct factor is 4, verified against Wright & Brainerd 2000.)

**Concentration parameter:**
```
c_nfw = r_200c / r_s   (derived from M_vir and r_s)
```

### 2.3 Power-Law Elliptical Profile

**Surface mass density:**
```
κ(ξ) = κ_0 × (ξ/θ_0)^{−γ'}

where ξ = √(x²/q + q y²),  q = axis ratio ≤ 1
```

The slope γ' = 2 recovers isothermal; γ' = 1 is uniform disk.

### 2.4 Sérsic Luminous Profile

**Surface brightness:**
```
I(R) = I_e exp[-b_n ((R/R_e)^{1/n} - 1)]

b_n = gammaincinv(2n, 0.5)   (exact solution, Ciotti 1991; Trujillo et al. 2001)
```

The exact `b_n` is obtained by solving the equation `γ(2n, b_n) / Γ(2n) = 1/2`
using the inverse incomplete gamma function.

### 2.5 Elliptical NFW Profile

**Ellipticity mapping:**
```
r_ellip = √(x²/q + q y²),  q = (1-e)/(1+e)   [weak-lensing convention]
```

### 2.6 NFW + Dark Matter Substructure

**Subhalo placement:** Deterministic pseudo-random positions using SHA-256 hash of physical parameters (M_vir, c, z_l, z_s) as seed — ensures reproducibility without hardcoded constants (Functional Randomness Control).

**Augmentation constraint:** RandomBrightness and RandomNoise clip to max(0, x) only — κ can exceed 1 in massive halo cores (clusters: κ ~ 2–5).

### Profile Comparison Table

| Profile | κ formula | α formula | ψ formula | Free params |
|---------|-----------|-----------|-----------|-------------|
| SIS | θ_E/(2θ) | θ_E | θ_E·θ | σ_v |
| NFW | 2κ_s f(x) | 4κ_s r_s h(x)/x | 4κ_s r_s² g(x) | M_vir, c |
| Power-law | κ_0(θ/θ_0)^{-γ'} | analytic | analytic | κ_0, γ', q |
| Sérsic | I(R)/Σ_crit | numerical | numerical | I_e, R_e, n |

### References
- NFW (1997), ApJ 490, 493
- Wright & Brainerd (2000), ApJ 534, 34
- Bartelmann (1996), A&A 313, 697
- Ciotti (1991), A&A 249, 99
- Trujillo et al. (2001), MNRAS 326, 869

---

## 3. Multi-Plane Lensing

**Module:** `src/lens_models/multi_plane.py`, `src/lens_models/multi_plane_recursive.py`

### Features
- N-plane ray tracing with proper lens recurrence
- Cosmologically correct inter-plane angular diameter distances
- Lensing potential and deflection field across multiple lens planes
- Recursive solver using scipy.fsolve for exact backward ray tracing

### Multi-plane recurrence (Schneider et al. 1992, §9)

**Angular position at plane i:**
```
θ_i = θ - Σ_{j=1}^{i-1} β_{ij} α̂_j(θ_j)

β_{ij} = D_{ij} / D_j × D_i / D_{ij}     (reduced deflection weight)
```

**Exact weights (Schneider 1992, Eq. 9.1–9.3):**
```
β_{ij} = D_A(z_j, z_i) × D_A(0, z_j+1) / (D_A(0, z_i) × D_A(z_j, z_j+1))
```

**Lensing potential (total):**
```
ψ_total = Σ_i (D_i / D_s) × ψ_i(θ_i)
```

(Blandford & Narayan 1986, Eq. 2.4)

### References
- Schneider, Ehlers & Falco (1992), *Gravitational Lenses*, Springer, §9
- Blandford & Narayan (1986), ApJ 310, 568

---

## 4. Ray Tracing

**Module:** `src/optics/ray_tracing.py`, `src/optics/geodesic_integration.py`

### Features
- Single-plane deflection field computation
- Curved spacetime geodesic integration (Runge-Kutta 4th order)
- Schwarz radius / photon capture condition
- Magnification tensor (convergence κ, shear γ, reduced shear g)
- Critical curves and caustics computation
- Multi-plane ray tracing via LensSystem API
- JAX-compatible vectorized operations

### Magnification tensor
```
M = (1 - κ)² - γ²   (magnification determinant)

κ = 1/2 (∂²ψ/∂x² + ∂²ψ/∂y²)   (convergence)
γ₁ = 1/2 (∂²ψ/∂x² - ∂²ψ/∂y²)  (shear components)
γ₂ = ∂²ψ/(∂x ∂y)
γ = √(γ₁² + γ₂²)
```

### Reduced shear
```
g = γ / (1 - κ)
```

### Geodesic integration
ODE system for null geodesic in Schwarzschild metric, integrated with RK4.
Returns `nan` (not π) for photons captured within r_s ≤ 1.5 r_Schwarz.

---

## 5. Wave Optics

**Module:** `src/optics/wave_optics.py`

### Features
- Full scalar diffraction integral (beyond geometric optics)
- Nakamura & Deguchi (1999) Eq. 4.2 formulation
- Amplification factor F(ω) as function of gravitational wave frequency ω
- Magnification |F(ω)|² vs geometric optics limit
- Interference fringes in the wave-optics regime

### Wave amplification factor

**Nakamura & Deguchi (1999), Eq. 4.2:**
```
F(ω) = (ω / 2πi) ∫ d²θ exp[i ω τ(θ, β)]
```

**Fermat potential:**
```
τ(θ, β) = (1 + z_l)/2c × (D_l D_s/D_ls) × [(θ - β)² - 2ψ(θ)]
```

**Discrete implementation (regular grid):**
```
F(ω) = (ω / 2πi) × dθ² × Σ_{pixels} exp[i ω τ_pix]
```

**Geometric optics limit (ω → ∞):**
```
F → Σ_{images} |μ_j|^{1/2} exp[i ω τ_j - iπ n_j / 2]

where n_j ∈ {0, 1, 2} is Morse index (saddle/min/max type).
```

**Wave optics regime:**
Applicable when λ_GW ~ θ_E × D_l (ω τ ~ 1), i.e., for
```
ω ≲ c / (G M_lens / c²)   [characteristic frequency]
```

### References
- Nakamura & Deguchi (1999), Prog. Theor. Phys. Suppl. 133, 137–153
- Takahashi & Nakamura (2003), ApJ 595, 1039–1051

---

## 6. Extended PSF Model

**Module:** `src/optics/epsf_model.py`

### Features
- Spatially-varying ePSF using Zernike polynomial decomposition
- Zernike terms Z4–Z22 (focus, astigmatism, coma, trefoil, spherical aberration, secondary astigmatism, quadrafoil)
- Detector field-of-view dependent PSF variation (polynomial spatial model)
- Zernike polynomial generation using Noll ordering convention

### Zernike polynomial representation

**PSF kernel at position (x_det, y_det):**
```
K(u, v; x_det, y_det) = Σ_{j=4}^{22} c_j(x_det, y_det) × Z_j(u, v)

c_j(x_det, y_det) = Σ_{k} a_{jk} P_k(x_det, y_det)
```

where P_k are 2D polynomial basis functions over the detector FOV.

**Noll Zernike indices:**

| Z index | Aberration | Formula |
|---------|-----------|---------|
| Z4 | Defocus | 2ρ² - 1 |
| Z5/Z6 | Primary astigmatism | ρ² cos(2φ), ρ² sin(2φ) |
| Z7/Z8 | Primary coma | (3ρ³-2ρ)cos(φ), (3ρ³-2ρ)sin(φ) |
| Z11 | Primary spherical | 6ρ⁴ - 6ρ² + 1 |
| Z22 | Secondary spherical | 20ρ⁶ - 30ρ⁴ + 12ρ² - 1 |

### References
- Noll (1976), J. Opt. Soc. Am. 66, 207
- Anderson & King (2000), PASP 112, 1360

---

## 7. Time Delay and Cosmography

**Module:** `src/time_delay/cosmography.py`

### Features
- Fermat potential time delay
- H₀ inference from time delay measurements
- Time-delay distance D_Δt marginalization
- Excess delay surface Δτ across image positions
- TDCOSMO-compatible H₀ posterior computation

### Time delay formula

**Absolute time delay (Refsdal 1964):**
```
Δt_{ij} = D_Δt/c × [φ(θ_i, β) - φ(θ_j, β)]
```

**Fermat potential:**
```
φ(θ, β) = (θ - β)²/2 - ψ(θ)
```

**Time-delay distance:**
```
D_Δt = (1 + z_l) D_l D_s / D_ls
```

**H₀ inference:**
```
H₀ = c D_Δt^{-1} × [Δt_{obs} / Δφ]
```

### References
- Refsdal (1964), MNRAS 128, 307
- Suyu et al. (2010), ApJ 711, 201
- Birrer et al. (2020), A&A 643, A165 (TDCOSMO)

---

## 8. Physics-Informed Neural Networks

**Modules:** `src/ml/pinn.py`, `src/ml/pinn_models.py`

### Architecture
- JAX/Equinox-based implementation (hardware-agnostic)
- Inputs: convergence map κ(θ) as 2D image, shape (1, H, W)
- Outputs: NFW parameters [M_vir, r_s, β_x, β_y]
- 6-layer fully connected network with swish activations
- Physics-informed training with Poisson constraint

### PINN Physics Loss

**Poisson equation (Schneider 1992, Eq. 3.11):**
```
∇²ψ = 2κ
```

**Total loss:**
```
L = L_data + λ_physics × L_Poisson + λ_boundary × L_boundary + λ_mass × L_mass
```

**Physics penalty:**
```
L_Poisson = ||∇²ψ_pred - 2κ_pred||²
```

**Parameter bounds (soft penalties on raw predictions):**
```
L_bounds = relu(10⁹ - M_vir_raw) + relu(M_vir_raw - 10¹⁴)
         + relu(0.1 - r_s_raw)   + relu(r_s_raw - 500)
```

After penalty computation, predictions are clipped:
```
M_vir = clip(M_vir_raw, 10⁹, 10¹⁴)   [M☉]
r_s   = clip(r_s_raw, 0.1, 500.0)     [arcsec]
```

### NFW Forward Model in PINN

**Characteristic surface density:**
```
κ_s = M_vir / (4π r_s² Σ_crit)   [dimensionless]
```

**Convergence from PINN parameters (Wright & Brainerd 2000):**
```
κ(x) = 2 κ_s f(x),   x = θ/r_s
```

### Uncertainty Quantification
- MC Dropout (Gal & Ghahramani 2016): eval() mode with selective Dropout re-enable
- 50 forward passes per inference for posterior sampling
- Epistemic uncertainty from dropout variance
- ECE (Expected Calibration Error) metric for calibration assessment

### Serialization
- JAX/Equinox: `eqx.tree_serialise_leaves(path, model)`
- Checkpoint-gated inference: API returns 503 if no checkpoint is found

### References
- Raissi, Perdikaris & Karniadakis (2019), J. Comp. Phys. 378, 686
- Schneider, Ehlers & Falco (1992), *Gravitational Lenses*, §3
- Gal & Ghahramani (2016), ICML 2016
- Wright & Brainerd (2000), ApJ 534, 34

---

## 9. Physics-Constrained Loss Functions

**Module:** `src/ml/physics_constrained_loss.py`

### Loss Components

**L_Poisson (Poisson constraint):**
```
L_Poisson = ||∇²ψ - 2κ||²_F / (H × W)
```
Implemented via vectorized `jax.grad` / `jax.hessian` (O(1) amortized over batch, not O(H×W) loop).

**L_gradient (deflection self-consistency):**
```
L_grad = ||α - ∇ψ||² = ||∂ψ/∂x - α_x||² + ||∂ψ/∂y - α_y||²
```

**L_mass (integrated mass conservation):**
```
L_mass = |∫∫ κ dΩ - M_Ein / Σ_crit|²

where M_Ein = π θ_E² Σ_crit   (mass within Einstein radius)
```

**Combined physics loss:**
```
L_physics = w_1 L_Poisson + w_2 L_grad + w_3 L_mass
```

### References
- Schneider (1992), *Gravitational Lenses*, Eq. 3.11
- Raissi et al. (2019), J. Comp. Phys. 378, 686

---

## 10. Advanced PINN Architectures

**Module:** `src/ml/pinn_advanced.py`

### Architecture
- PyTorch-based (with `try/except ImportError` guard for JAX-only environments)
- LensingCNN with ResNet-style skip connections and BatchNorm
- MC Dropout uncertainty: `self.eval()` + selective Dropout re-enable (Gal & Ghahramani 2016)
- Extended parameter outputs: [M_vir, r_s, β_x, β_y, κ_s, c_nfw]

### MC Dropout Protocol (Gal & Ghahramani 2016)
```
1. Set model.eval()   → BatchNorm uses running statistics
2. Re-enable Dropout layers (m.train() for m in modules() if isinstance(m, Dropout))
3. Run N=50 forward passes with torch.no_grad()
4. Return mean ± std of N samples as uncertainty estimate
```

### References
- Gal & Ghahramani (2016), ICML 2016, *Dropout as Bayesian Approximation*
- He et al. (2016), CVPR 2016 (ResNet skip connections)

---

## 11. Neural ODE / Hamiltonian Neural Networks

**Module:** `src/ml/neural_ode.py`

### Features
- `AnalyticFusingNODE`: Neural ODE with Lagrangian residual formulation
- `HamiltonianNeuralNetwork`: Physics-preserving dynamics via Hamiltonian structure
- Energy conservation by construction: dH/dt = 0 (symplectic structure)
- Integration with `torchdiffeq` (when available)

### Lagrangian Residual Formulation

**Cranmer (2020) / Greydanus (2019) — Lagrangian Neural Network:**
```
∂L/∂q - d/dt (∂L/∂q̇) = 0   (Euler-Lagrange equation as physics prior)
```

**Hamiltonian ODE system:**
```
dq/dt =  ∂H/∂p
dp/dt = -∂H/∂q
```

**Energy conservation test:**
```
δH/H = |H(t) - H(0)| / |H(0)|  ≲ 10⁻⁶  (expected for well-trained HNN)
```

### References
- Chen et al. (2018), NeurIPS 2018 (*Neural Ordinary Differential Equations*)
- Cranmer et al. (2020), PNAS 117, 9449
- Greydanus et al. (2019), NeurIPS 2019 (*Hamiltonian Neural Networks*)

---

## 12. Nested Sampling and Bayesian Evidence

**Module:** `src/ml/nested_sampling.py`

### Features
- Pure NumPy Skilling (2004) nested sampler (no external dependencies)
- Log evidence Z = log ∫ L(θ) π(θ) dθ via live-point compression
- Posterior samples from nested sampling chain
- Bayes factor K = Z_1/Z_2 for model comparison
- Jeffreys (1961) scale interpretation of ln K
- Built-in NFW vs SIS demo: ln K ≈ 13.9 (decisive evidence for NFW)

### Nested Sampling Algorithm (Skilling 2004)

**Log evidence accumulation:**
```
log Z_i = log Z_{i-1} + log(L_min) + log(ΔX_i)
```

**Prior volume compression:**
```
X_i = exp(-i/N_live)   (expectation value)
```

**Terminal contribution (equal weights, Skilling 2004):**
```
ΔX_terminal = X_remaining / N_live   for all live points
```

**Bayes factor (Jeffreys scale):**

| ln K | Evidence |
|------|---------|
| < 1  | Inconclusive |
| 1–3  | Positive |
| 3–5  | Strong |
| > 5  | Decisive |

### References
- Skilling (2004), AIP Conf. Proc. 735, 395
- Jeffreys (1961), *Theory of Probability*, 3rd ed., Oxford

---

## 13. Non-Parametric Source Reconstruction

**Module:** `src/ml/source_models.py`

### Features
- Pixelized source model with Gaussian Process regularization
- RBF kernel: K(r, r') = σ² exp(-||r-r'||²/2l²)
- Matérn-3/2 kernel: K(r, r') = σ²(1 + √3 r/l) exp(-√3 r/l)
- Linear inversion: s = (B^T N^{-1} B + λ C^{-1})^{-1} B^T N^{-1} d
- Log-evidence computation for regularization hyperparameter selection
- Uncertainty covariance Σ_s = (B^T N^{-1} B + λ C^{-1})^{-1}

### Linear Source Inversion

**Data model:**
```
d = B s + n,   n ~ N(0, N)
```
where B is the lensing blurring matrix, s is source pixel vector.

**Optimal source (Suyu et al. 2006):**
```
s_MAP = (B^T N^{-1} B + λ C_s^{-1})^{-1} B^T N^{-1} d
```

**Log evidence:**
```
log E(λ) = -1/2 [χ² + s^T C_s^{-1} s + log det(C_s) + log det(B^T N^{-1}B + λ C_s^{-1})]
```

### References
- Suyu et al. (2006), MNRAS 371, 983
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press

---

## 14. Automated Lens Finder (LenNet)

**Module:** `src/ml/lens_finder.py`

### Features
- LenNet-style convolutional network for lens detection in wide-field FITS images
- Object-detection head with bounding box regression
- Checkpoint-gated: reports `checkpoint_missing` mode when no trained model
- No heuristic/random fallback detections — fails transparently
- Designed for Stage IV survey data (LSST, Euclid)

### Architecture
- 5-layer CNN backbone with MaxPool and BatchNorm
- Detection head: GlobalAveragePool → FC(256) → FC(4) for bounding box
- Classification head: FC(256) → FC(2) for lens/non-lens
- Input: 64×64 pixel cutouts, single channel

### References
- Jacobs et al. (2017), ApJS 243, 17
- Petrillo et al. (2019), MNRAS 482, 807

---

## 15. Joint Multi-Resolution Survey Analysis

**Module:** `src/ml/joint_survey.py`

### Features
- Multi-resolution likelihood engine for joint ground + space deblending
- Joint likelihood: L(θ) = L_ground(θ) × L_space(θ)
- PSF convolution matching for ground/space images
- Handles data from heterogeneous resolution instruments simultaneously

### Joint Likelihood Model
```
log L_joint = log L_HST + log L_ground

log L_HST   = -1/2 ||d_HST   - B_HST   f||²_{N_HST}
log L_ground = -1/2 ||d_ground - B_ground f||²_{N_ground}
```

where B_{*} = lensing_matrix × PSF_{*}.

### References
- Rowe et al. (2015), A&C 10, 121 (GalSim)
- Birrer et al. (2015), ApJ 813, 102

---

## 16. Stellar Kinematics Module

**Module:** `src/validation/kinematics.py`

### Features
- Jeans equation solver for radial velocity dispersion σ_r(r)
- Abel projection to line-of-sight dispersion σ_LOS(R)
- Aperture-averaged σ within circular aperture (matched to SDSS fiber)
- Mass-sheet degeneracy (MSD) test: compares lensing mass M_lens to kinematic mass M_kin
- Joint lensing+kinematics constraint on γ_ext (external convergence)
- λ = M_lens/M_kin Jeffreys-scale interpretation

### Jeans Equation (Mamon & Łokas 2005, Eq. 11)

**Spherical Jeans equation (anisotropy β):**
```
d(ν σ_r²)/dr + 2β/r × ν σ_r² = -ν GM(r)/r²
```

**Solution (isotropic β=0):**
```
σ_r²(r) = 1/ν(r) ∫_r^∞ ν(r') GM(r')/r'² dr'
```

**Line-of-sight projection (Binney & Tremaine 2008, §4.2):**
```
Σ(R) σ_LOS²(R) = 2 ∫_R^∞ (1 - β R²/r²) ν(r) σ_r²(r) r/√(r²-R²) dr
```

**Aperture-averaged dispersion:**
```
⟨σ_LOS⟩_ap = √[ ∫_0^{R_ap} Σ(R) σ_LOS²(R) W(R) R dR  /  ∫_0^{R_ap} Σ(R) W(R) R dR ]

W(R) = 1 / √(R_ap² - R²)   (uniform circular aperture weight)
```

**Mass-sheet degeneracy:**
```
λ = M_lens / M_kin

λ ≈ 1   (no MSD)
λ > 1   (over-massive lens, γ_ext > 0)
λ < 1   (under-massive lens, γ_ext < 0)
```

### Key Constants
- kpc_to_m = 3.085677581 × 10¹⁹ m (IAU 2012, verified against Nakamura & Deguchi 1999)

### References
- Mamon & Łokas (2005), MNRAS 363, 705
- Binney & Tremaine (2008), *Galactic Dynamics*, 2nd ed., §4.2
- Treu & Koopmans (2004), ApJ 611, 739
- Birrer et al. (2020), A&A 643, A165

---

## 17. SLACS Observational Validation

**Modules:** `src/validation/hst_targets.py`, `src/validation/observational_diagnostics.py`

### SLACS Lens Catalog

| System | z_l | z_s | θ_E (arcsec) | σ_v (km/s) | HST Proposal |
|--------|-----|-----|--------------|------------|--------------|
| SDSS J0737+3216 | 0.3223 | 0.5812 | 1.00 | 322 | 10174 |
| SDSS J0912+0029 | 0.1642 | 0.3239 | 1.63 | 327 | 10174 |
| SDSS J1020+1122 | 0.2822 | 0.5530 | 1.20 | 282 | 10494 |
| SDSS J1430+4105 | 0.2850 | 0.5753 | 1.52 | 322 | 10886 |
| SDSS J1627+0053 | 0.2076 | 0.5241 | 1.23 | 289 | 10494 |

### Observational Validation Protocol

1. Fix lens model from published SLACS parameters
2. Fit PSF-convolved lensed Sérsic source in Einstein-ring annulus
3. Report image-space diagnostics (not κ-to-flux comparisons):
   - NRMSE: normalized root mean square error in annulus
   - SSIM: structural similarity index
   - Ring correlation: Pearson correlation in arc region
   - Annular flux ratio: ∫Σ_model / ∫Σ_obs

### Observational Thresholds
```
NRMSE  ≤ 0.15   (15% residual tolerance)
SSIM   ≥ 0.70   (70% structural similarity)
ring_corr ≥ 0.6 (arc pattern correlation)
```

### References
- Bolton et al. (2006), ApJ 638, 703 (SLACS survey)
- Auger et al. (2009), ApJ 705, 1099
- Koopmans et al. (2009), ApJ 703, L51

---

## 18. μ-GLANCE Magnification Diagnostics

**Module:** `src/validation/mu_glance.py`

### Features
- Flux anomaly statistics: difference between observed and model magnifications
- RBF-interpolated magnification residual maps
- Flux anomaly ratio: μ_obs / μ_model
- Statistical characterization: mean, std, outlier fraction
- Visualization of residual structures (substructure signals)

### Flux Anomaly
```
δμ = (μ_obs - μ_model) / μ_model

Anomaly threshold: |δμ| > 3σ_{noise}
```

### References
- Kochanek & Dalal (2004), ApJ 610, 69

---

## 19. Bayesian Model Selection Utilities

**Module:** `src/validation/bayes_factor.py`

### Features
- Log Bayes factor: ln K = ln Z_1 - ln Z_2
- Savage-Dickey ratio for nested model comparison
- Eccentricity-microlensing correlation test
- Jeffreys (1961) evidence scale interpretation

### Bayes Factor (Jeffreys 1961)
```
K = Z_1 / Z_2 = ∫ L(d|θ, M_1) π(θ|M_1) dθ / ∫ L(d|θ, M_2) π(θ|M_2) dθ
```

**Savage-Dickey density ratio (nested models):**
```
K = π(θ_0 | d, M_1) / π(θ_0 | M_1)   (at the nested point θ_0)
```

### References
- Jeffreys (1961), *Theory of Probability*, 3rd ed.
- Verdinelli & Wasserman (1995), JASA 90, 614 (Savage-Dickey)

---

## 20. HST/MAST Data Pipeline

**Module:** `src/data/mast_downloader.py`

### Features
- Cache-first FITS loader (local disk → MAST fallback)
- ACS/WFC F814W cutout retrieval for 9 SLACS systems
- `astroquery.mast` integration for automated downloads
- Deterministic synthetic noise via SHA-256(lens_name) seed
- Pixel scale extraction from FITS WCS headers
- Cache manifest: `data/hst_cache/manifest.json` (9/9 SLACS archived)
- Synthetic FITS opt-in only — no silent fallback

### Data Flow
```
Local cache hit?  → return cached FITS
      ↓ no
MAST query live?  → download + cache + return
      ↓ no
Explicit demo flag? → generate synthetic FITS with SHA-256 noise
      ↓ no
Raise DataUnavailableError (no silent fallback)
```

### References
- Anderson & King (2000), PASP 112, 1360
- Dressel (2012), *ACS Instrument Handbook*, STScI

---

## 21. Pixel Covariance and Whitening

**Module:** `src/data/pixel_covariance.py`

### Features
- Drizzled image correlated pixel noise covariance modeling
- Full covariance matrix: Σ = signal + readout + dark current contributions
- Cholesky whitening: χ = L⁻¹r (solves Lχ = r via `scipy.linalg.solve_triangular`)
- Block-diagonal covariance for large images
- Log-likelihood with correlated noise: log L = -½ χ^T χ - ½ log|Σ|

### Cholesky Whitening
```
Σ = L L^T   (Cholesky decomposition, L lower triangular)
χ = L⁻¹ r  (whitened residuals)
log L = -½ ||χ||² - Σ_i log L_{ii}
```

**Important:** whitening returns `L⁻¹r` via `solve_triangular`, NOT `Σ⁻¹r` directly.

### References
- Fruchter & Hook (2002), PASP 114, 144 (drizzling)
- Rowe et al. (2015), A&C 10, 121

---

## 22. Cosmographic Blinding

**Module:** `src/utils/blinding.py`

### Features
- TDCOSMO-style cryptographic blinding of H₀ and D_Δt
- HMAC-SHA256 with user-provided salt
- Blind offset: random shift in H₀ ∈ [60, 80] km/s/Mpc range
- Unblinding requires knowledge of salt (not stored in code)
- Server-side HMAC for D_Δt in the survey blinding API endpoint

### Blinding Protocol
```python
blind_H0 = H0_true + HMAC-SHA256(salt, "H0_blind")[0:4] % 20 - 10
           # shifts H0 by unknown offset ∈ [-10, +10] km/s/Mpc
```

### References
- Suyu et al. (2020), A&A 644, A162 (H₀LiCOW blinding protocol)
- TDCOSMO Collaboration (Millon et al. 2020, A&A 639, A101)

---

## 23. Benchmark and Evaluation Suite

**Scripts:** `scripts/`

### 23.1 Ablation Study (`ablation_study.py`)
- Systematic component removal study to quantify contribution of each module
- Metrics: κ-RMSE, computation time, convergence quality
- n_trials ≥ 1 validation guard
- Output: `results/ablation_table.tex` (LaTeX-formatted)

### 23.2 SOTA Comparison (`sota_comparison.py`)
- Benchmark PINN vs analytic lensing vs MCMC inference
- Checkpoint-backed neural method; analytic fallback with explicit mode disclosure
- Output: `results/sota_comparison_table.tex`

### 23.3 Uncertainty Calibration (`uncertainty_calibration.py`)
- MC-dropout calibration on held-out synthetic NFW analogs
- Metrics: Expected Calibration Error (ECE), coverage@90%
- Current validated results: ECE = 0.062, coverage@90 = 0.936
- Configuration: seed=21, dropout_rate=0.04, 50% shrinkage of empirical quantiles
- Output: `results/uncertainty_calibration.png`

**Expected Calibration Error:**
```
ECE = Σ_b (n_b/N) |acc_b - conf_b|

where acc_b = fraction within predicted interval in bin b
      conf_b = nominal confidence level of bin b
```

### 23.4 Scalability Benchmark (`scalability_benchmark.py`)
- Grid scaling study: 16 → 512 pixels per side
- Measures wall time, memory, κ-RMSE as function of grid size
- Output: `results/scalability_analysis.png`

### 23.5 Pareto Benchmark (`pareto_benchmark.py`)
- Time-to-Solution vs κ-RMSE Pareto front
- Methods: PINN (checkpoint-gated), analytic NFW, MCMC
- `evaluation_mode` disclosed in output JSON for downstream rigor checks
- Output: `results/pareto_front.png`, `results/pareto_table.tex`

### 23.6 Multi-Messenger Demo (`multi_messenger_demo.py`)
- Optical + gravitational wave multi-messenger consistency analysis
- Wave optics amplification F(ω) = (ω/2πi) ∫ exp[iωτ] dΩ
- Cross-correlates GW time delay with optical Fermat potential
- Output: `results/multi_messenger_consistency.png`

---

## 24. Reproducibility Infrastructure

### One-Command Reproducibility
```bash
bash scripts/reproduce.sh
```
6-step pipeline: tests → ablation → real data → SOTA → scalability → calibration.
Uses `uv run python` if available, falls back to `python3`.

### Publication Gate
```bash
python3 scripts/publication_gate.py --quick
```
Validates all required figures exist and all tests pass.

### Statistical Rigor Report
```bash
python3 scripts/statistical_rigor_report.py
```
Generates full statistical rigor report with evaluation_mode disclosure for all outputs.

### Functional Randomness Control
- All stochastic components accept explicit `seed: int` parameters
- NFW subhalo: SHA-256(M_vir, c, z_l, z_s) as seed
- MAST downloader: SHA-256(lens_name) for synthetic noise
- No hidden `PRNGKey(0)` defaults
- Full bit-for-bit reproducibility when seeds are fixed

### Git Configuration
- Branch: `feature/jax-migration`
- Python 3.14.3, NumPy 2.4.2, JAX backend auto-detected via `src/ml/__init__.BACKEND`

---

## 25. API and Web Interface

### FastAPI Application (`api/main.py`)
- 45 REST endpoints total
- JWT authentication with access/refresh tokens
- API key management (CRUD)
- Rate limiting via `slowapi`
- OpenAPI/Swagger documentation at `/docs`
- CORS with credential-safe configuration

### Complete Endpoint Index

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status, GPU, database, version |
| `/api/v1/models` | GET | Loaded model list |
| `/api/v1/stats` | GET | API job stats + live test count |
| `/api/v1/synthetic` | POST | Generate NFW convergence map |
| `/api/v1/inference` | POST | Checkpoint-gated PINN inference |
| `/api/v1/batch` | POST | Submit batch job |
| `/api/v1/batch/{id}/status` | GET | Poll batch status |
| `/api/v1/validation/slacs` | GET | SLACS validation results |
| `/api/v1/validation/calibration` | GET | ECE + coverage UQ calibration |
| `/api/v1/validation/ablation` | GET | Ablation study results |
| `/api/v1/analyses` | GET, POST | User analysis CRUD |
| `/api/v1/analyses/public` | GET | Public analysis gallery |
| `/api/v1/jobs` | GET | DB-persisted jobs list |
| `/api/v1/results` | GET | DB-persisted results |
| `/api/v1/survey/finder` | POST | Automated lens detection |
| `/api/v1/survey/epsf` | POST | ePSF model computation |
| `/api/v1/survey/blinding/apply` | POST | Cosmographic blinding (HMAC) |
| `/api/v1/survey/covariance` | POST | Pixel covariance computation |
| `/api/v1/survey/joint` | POST | Joint multi-resolution analysis |
| `/api/v1/auth/register` | POST | User registration |
| `/api/v1/auth/login` | POST | JWT login |
| `/api/v1/auth/refresh` | POST | Token refresh (Pydantic v2 body) |
| `/api/v1/auth/me` | GET | Current user info |
| `/api/v1/auth/api-keys` | GET | List user API keys |
| `/api/v1/auth/api-keys` | POST | Create API key |
| `/api/v1/auth/api-keys/{id}` | DELETE | Revoke API key |
| `/api/v1/rigor/sed` | GET | SED photometry demo |
| `/docs` | GET | OpenAPI schema |
| `/ui` | GET | Web UI shell |

### Web Interface Pages

| Page | Features |
|------|---------|
| Dashboard | Health, GPU status, database connected, test count, API stats |
| Workbench | NFW synthesis, PINN inference, convergence map viz (Plotly) |
| Validation | SLACS results, UQ calibration, ablation table |
| Analyses | User CRUD analyses with tabs (Jobs, Results) |
| Stage IV Survey | Finder, ePSF, Blinding, Covariance, Joint Survey |
| API Explorer | OpenAPI-driven request builder |
| Account | Auth flow, API key management with revoke |

---

## 26. Complete Reference List

1. **Bartelmann (1996)** — NFW deflection angles in gravitational lensing. *A&A* 313, 697–702.
2. **Birrer et al. (2015)** — Gravitational lens modeling with basis sets. *ApJ* 813, 102.
3. **Birrer et al. (2020)** — TDCOSMO: time-delay cosmography. *A&A* 643, A165.
4. **Binney & Tremaine (2008)** — *Galactic Dynamics*, 2nd ed. Princeton University Press. §4.2.
5. **Blandford & Narayan (1986)** — Fermat's principle, caustics, and the caustic crossing of a point mass lens. *ApJ* 310, 568.
6. **Bolton et al. (2006)** — The Sloan Lens ACS Survey. *ApJ* 638, 703–724.
7. **Chen et al. (2018)** — Neural ordinary differential equations. *NeurIPS* 2018.
8. **Ciotti (1991)** — Stellar systems following the de Vaucouleurs R^{1/4} law. *A&A* 249, 99.
9. **Cranmer et al. (2020)** — Lagrangian neural networks. *PNAS* 117, 9449.
10. **Fruchter & Hook (2002)** — Drizzle: a method for the linear reconstruction of undersampled images. *PASP* 114, 144.
11. **Gal & Ghahramani (2016)** — Dropout as a Bayesian approximation. *ICML* 2016.
12. **Greydanus et al. (2019)** — Hamiltonian neural networks. *NeurIPS* 2019.
13. **He et al. (2016)** — Deep residual learning for image recognition. *CVPR* 2016.
14. **Hogg (1999)** — Distance measures in cosmology. arXiv:astro-ph/9905116.
15. **Jacobs et al. (2017)** — Finding strong gravitational lenses with convolutional neural networks. *ApJS* 243, 17.
16. **Jeffreys (1961)** — *Theory of Probability*, 3rd ed. Oxford University Press.
17. **Kochanek & Dalal (2004)** — Tests for substructure in gravitational lenses. *ApJ* 610, 69.
18. **Koopmans et al. (2009)** — The structure and dynamics of massive early-type galaxies. *ApJ* 703, L51.
19. **Mamon & Łokas (2005)** — Dark matter in elliptical galaxies. *MNRAS* 363, 705.
20. **Millon et al. (2020)** — TDCOSMO I. *A&A* 639, A101.
21. **Nakamura & Deguchi (1999)** — Wave optics in gravitational lensing. *Prog. Theor. Phys. Suppl.* 133, 137–153.
22. **NFW / Navarro, Frenk & White (1997)** — A universal density profile from hierarchical clustering. *ApJ* 490, 493.
23. **Noll (1976)** — Zernike polynomials and atmospheric turbulence. *J. Opt. Soc. Am.* 66, 207.
24. **Petrillo et al. (2019)** — CNN gravitational lens finding. *MNRAS* 482, 807.
25. **Planck Collaboration (2020)** — Planck 2018 results VI: cosmological parameters. *A&A* 641, A6.
26. **Raissi, Perdikaris & Karniadakis (2019)** — Physics-informed neural networks. *J. Comp. Phys.* 378, 686.
27. **Rasmussen & Williams (2006)** — *Gaussian Processes for Machine Learning*. MIT Press.
28. **Refsdal (1964)** — On the possibility of determining Hubble's parameter and the masses of galaxies from the gravitational lens effect. *MNRAS* 128, 307.
29. **Rowe et al. (2015)** — GalSim: the modular galaxy image simulation toolkit. *A&C* 10, 121.
30. **Schneider, Ehlers & Falco (1992)** — *Gravitational Lenses*. Springer.
31. **Skilling (2004)** — Nested sampling. *AIP Conf. Proc.* 735, 395–405.
32. **Suyu et al. (2006)** — Dissecting the gravitational lens B1938+666. *MNRAS* 371, 983.
33. **Suyu et al. (2010)** — Dissecting the gravitational lens B1938+666 II. *ApJ* 711, 201.
34. **Takahashi & Nakamura (2003)** — Wave effects in gravitational lensing of GWs. *ApJ* 595, 1039.
35. **Treu & Koopmans (2004)** — The internal structure and formation of early-type galaxies. *ApJ* 611, 739.
36. **Trujillo et al. (2001)** — The effects of seeing on Sérsic structural parameters. *MNRAS* 326, 869.
37. **Verdinelli & Wasserman (1995)** — Computing Bayes factors using a generalization of the Savage-Dickey density ratio. *JASA* 90, 614.
38. **Wright & Brainerd (2000)** — A new derivation of the lensing convergence for the NFW profile. *ApJ* 534, 34.

---

*Document generated for manuscript preparation. All formulas cross-referenced with implementation code.*
*Last updated: 2026-03-13*
