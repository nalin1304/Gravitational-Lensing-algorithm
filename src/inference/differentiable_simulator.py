"""
Differentiable Strong-Lens Simulator — PyTorch Native.

Every operation from mass-profile parameters to convergence κ, deflection α,
and lensed images is composed of differentiable primitives so that
``torch.autograd.grad`` can back-propagate through the full forward model.

This is the foundation for gradient-based posterior sampling (NUTS-HMC)
and Fisher-matrix forecasting.

Physics
-------
NFW convergence (Wright & Brainerd 2000, Eq. 11-13):
    κ(x) = 2 ρ_s r_s / Σ_crit × f(x)
where x = r / r_s and f(x) is the piecewise function:
    f(x<1) = (1 - arccosh(1/x)/√(1-x²)) / (x²-1)
    f(x=1) = 1/3
    f(x>1) = (1 - arccos(1/x)/√(x²-1)) / (x²-1)

SIS deflection:
    α(θ) = θ_E × θ / |θ|

References
----------
- Navarro, Frenk & White (1997), ApJ 490, 493
- Wright & Brainerd (2000), astro-ph/0001341
- Bartelmann (1996), A&A 313, 697
- Galan et al. (2022), A&A 668 — Herculens differentiable lens modeling
"""

import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


# ---------------------------------------------------------------------------
# Physical constants (SI, converted once)
# ---------------------------------------------------------------------------
_c_m_s = const.c.to(u.m / u.s).value
_G_SI = const.G.to(u.m**3 / (u.kg * u.s**2)).value
_Mpc_m = u.Mpc.to(u.m)
_Msun_kg = const.M_sun.to(u.kg).value
_arcsec_rad = (1.0 * u.arcsec).to(u.rad).value


# ---------------------------------------------------------------------------
# Differentiable NFW profile
# ---------------------------------------------------------------------------
class DifferentiableNFW(nn.Module):
    """Fully differentiable NFW convergence and deflection in PyTorch.

    All computations use smooth approximations so gradients exist everywhere,
    including at x = 1 where the standard piecewise NFW function is continuous
    but has a cusp.  We use the Bartelmann (1996) / Wright & Brainerd (2000)
    analytic forms with a small softening ε to avoid branch-point issues.

    Parameters
    ----------
    log10_M_vir : float
        log₁₀ of virial mass in solar masses.
    concentration : float
        NFW concentration parameter c = r_vir / r_s.
    z_lens, z_source : float
        Lens and source redshifts.
    H0 : float
        Hubble constant (km/s/Mpc), default 67.4 (Planck 2018).
    Om0 : float
        Matter density parameter, default 0.315 (Planck 2018).
    """

    def __init__(
        self,
        log10_M_vir: float = 14.0,
        concentration: float = 5.0,
        z_lens: float = 0.3,
        z_source: float = 1.5,
        H0: float = 67.4,
        Om0: float = 0.315,
    ):
        super().__init__()
        # Learnable / samplable parameters
        self.log10_M_vir = nn.Parameter(torch.tensor(log10_M_vir, dtype=torch.float64))
        self.concentration = nn.Parameter(torch.tensor(concentration, dtype=torch.float64))

        # Fixed cosmological context
        self.z_lens = z_lens
        self.z_source = z_source

        # Pre-compute angular diameter distances (not differentiable w.r.t. cosmology)
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        D_l = cosmo.angular_diameter_distance(z_lens).to(u.m).value
        D_s = cosmo.angular_diameter_distance(z_source).to(u.m).value
        D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.m).value
        rho_crit_z = cosmo.critical_density(z_lens).to(u.kg / u.m**3).value

        self.register_buffer("D_l", torch.tensor(D_l, dtype=torch.float64))
        self.register_buffer("D_s", torch.tensor(D_s, dtype=torch.float64))
        self.register_buffer("D_ls", torch.tensor(D_ls, dtype=torch.float64))
        self.register_buffer("rho_crit_z", torch.tensor(rho_crit_z, dtype=torch.float64))

        # Critical surface density  Σ_crit = c²/(4πG) × D_s / (D_l × D_ls)
        Sigma_crit = (_c_m_s**2 / (4.0 * math.pi * _G_SI)) * D_s / (D_l * D_ls)
        self.register_buffer("Sigma_crit", torch.tensor(Sigma_crit, dtype=torch.float64))

    def _nfw_derived(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute r_s, rho_s, kappa_s from current parameters."""
        M_vir = 10.0 ** self.log10_M_vir * _Msun_kg  # kg
        c = torch.clamp(self.concentration, min=1.0, max=30.0)

        # Virial radius from M_vir = (4/3)π r_vir³ × 200 ρ_crit
        r_vir = (3.0 * M_vir / (4.0 * math.pi * 200.0 * self.rho_crit_z)) ** (1.0 / 3.0)
        r_s = r_vir / c  # scale radius (m)

        # Characteristic over-density
        delta_c = (200.0 / 3.0) * c**3 / (torch.log1p(c) - c / (1.0 + c))
        rho_s = delta_c * self.rho_crit_z  # kg/m³

        # Convergence normalisation
        # κ_s = ρ_s × r_s / Σ_crit  (dimensionless)
        kappa_s = rho_s * r_s / self.Sigma_crit

        return r_s, rho_s, kappa_s

    @staticmethod
    def _f_nfw(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Smooth NFW convergence kernel f(x) — Wright & Brainerd (2000).

        Piecewise:
            x < 1: f = [1/(x²-1)] × [1 - arccosh(1/x)/√(1-x²)]
            x = 1: f = 1/3
            x > 1: f = [1/(x²-1)] × [1 - arccos(1/x)/√(x²-1)]

        Uses torch.where for differentiability through both branches.
        A small softening ensures numerical stability at x ≈ 1.
        """
        x = torch.clamp(x, min=1e-8)

        # --- Branch x < 1 ---
        # arccosh(1/x) = ln(1/x + sqrt(1/x² - 1))  for x < 1
        one_minus_x2 = torch.clamp(1.0 - x * x, min=eps * eps)
        sqrt_1mx2 = torch.sqrt(one_minus_x2)
        inv_x = torch.clamp(1.0 / x, max=1e8)
        acosh_val = torch.log(inv_x + torch.sqrt(torch.clamp(inv_x * inv_x - 1.0, min=eps * eps)))
        f_lt1 = (1.0 - acosh_val / sqrt_1mx2) / (x * x - 1.0 - eps)

        # --- Branch x > 1 ---
        x2_minus_1 = torch.clamp(x * x - 1.0, min=eps * eps)
        sqrt_x2m1 = torch.sqrt(x2_minus_1)
        acos_val = torch.acos(torch.clamp(1.0 / x, min=-1.0 + 1e-7, max=1.0 - 1e-7))
        f_gt1 = (1.0 - acos_val / sqrt_x2m1) / (x * x - 1.0 + eps)

        # --- Blend at x ≈ 1 ---
        f_at_one = torch.tensor(1.0 / 3.0, dtype=x.dtype, device=x.device)

        # Use smooth blending near x=1 to avoid cusp
        sigma = 0.05
        w = torch.exp(-0.5 * ((x - 1.0) / sigma) ** 2)
        f_piecewise = torch.where(x < 1.0, f_lt1, f_gt1)
        f = (1.0 - w) * f_piecewise + w * f_at_one

        return f

    def convergence(self, theta_x: torch.Tensor, theta_y: torch.Tensor) -> torch.Tensor:
        """Compute convergence κ(θ) on a grid of angular positions (arcsec).

        Parameters
        ----------
        theta_x, theta_y : Tensor
            Angular coordinates in arcseconds.

        Returns
        -------
        Tensor
            Convergence map κ(θ).
        """
        r_s, _, kappa_s = self._nfw_derived()

        # Angular scale radius (radians → arcsec)
        theta_s = (r_s / self.D_l) / _arcsec_rad  # arcsec

        r_arcsec = torch.sqrt(theta_x**2 + theta_y**2 + 1e-12)
        x = r_arcsec / theta_s

        return 2.0 * kappa_s * self._f_nfw(x)

    def deflection(self, theta_x: torch.Tensor, theta_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute deflection angles α_x, α_y (arcsec).

        Uses the NFW deflection integral h(x) — Bartelmann (1996) Eq. 13:
            α(θ) = 4 κ_s r_s/D_l × h(x) × θ/|θ|
        where h(x) = ln(x/2) + arccosh(1/x)/sqrt(1-x²)  [x<1]
                    = ln(x/2) + 1                          [x=1]
                    = ln(x/2) + arccos(1/x)/sqrt(x²-1)    [x>1]
        """
        r_s, _, kappa_s = self._nfw_derived()
        theta_s = (r_s / self.D_l) / _arcsec_rad

        r_arcsec = torch.sqrt(theta_x**2 + theta_y**2 + 1e-12)
        x = r_arcsec / theta_s
        x = torch.clamp(x, min=1e-8)

        eps = 1e-8
        log_half_x = torch.log(x / 2.0 + eps)

        # --- Branch x < 1: h = ln(x/2) + arccosh(1/x)/sqrt(1-x²) ---
        one_minus_x2 = torch.clamp(1.0 - x * x, min=eps)
        sqrt_1mx2 = torch.sqrt(one_minus_x2)
        inv_x = torch.clamp(1.0 / x, max=1e8)
        acosh_val = torch.log(inv_x + torch.sqrt(torch.clamp(inv_x * inv_x - 1.0, min=eps)))
        h_lt1 = log_half_x + acosh_val / sqrt_1mx2

        # --- Branch x > 1: h = ln(x/2) + arccos(1/x)/sqrt(x²-1) ---
        x2_minus_1 = torch.clamp(x * x - 1.0, min=eps)
        sqrt_x2m1 = torch.sqrt(x2_minus_1)
        acos_val = torch.acos(torch.clamp(1.0 / x, min=-1.0 + 1e-7, max=1.0 - 1e-7))
        h_gt1 = log_half_x + acos_val / sqrt_x2m1

        # --- Blend at x ≈ 1: h → ln(1/2) + 1 ---
        h_at_one = math.log(0.5) + 1.0
        sigma = 0.05
        w = torch.exp(-0.5 * ((x - 1.0) / sigma) ** 2)
        h_piecewise = torch.where(x < 1.0, h_lt1, h_gt1)
        h = (1.0 - w) * h_piecewise + w * h_at_one

        # Deflection magnitude (arcsec)
        alpha_mag = 4.0 * kappa_s * theta_s * h / x

        # Project onto x, y
        cos_phi = theta_x / r_arcsec
        sin_phi = theta_y / r_arcsec

        return alpha_mag * cos_phi, alpha_mag * sin_phi

    def forward(self, theta_x: torch.Tensor, theta_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass: convergence + deflection."""
        kappa = self.convergence(theta_x, theta_y)
        alpha_x, alpha_y = self.deflection(theta_x, theta_y)
        return {"convergence": kappa, "alpha_x": alpha_x, "alpha_y": alpha_y}


# ---------------------------------------------------------------------------
# Differentiable SIS profile
# ---------------------------------------------------------------------------
class DifferentiableSIS(nn.Module):
    """Differentiable Singular Isothermal Sphere.

    κ(θ) = θ_E / (2|θ|),  α(θ) = θ_E × θ̂
    """

    def __init__(self, theta_E: float = 1.0):
        super().__init__()
        self.theta_E = nn.Parameter(torch.tensor(theta_E, dtype=torch.float64))

    def convergence(self, theta_x: torch.Tensor, theta_y: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(theta_x**2 + theta_y**2 + 1e-12)
        return torch.abs(self.theta_E) / (2.0 * r)

    def deflection(self, theta_x: torch.Tensor, theta_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = torch.sqrt(theta_x**2 + theta_y**2 + 1e-12)
        alpha_x = torch.abs(self.theta_E) * theta_x / r
        alpha_y = torch.abs(self.theta_E) * theta_y / r
        return alpha_x, alpha_y

    def forward(self, theta_x: torch.Tensor, theta_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        kappa = self.convergence(theta_x, theta_y)
        alpha_x, alpha_y = self.deflection(theta_x, theta_y)
        return {"convergence": kappa, "alpha_x": alpha_x, "alpha_y": alpha_y}


# ---------------------------------------------------------------------------
# Full differentiable lens simulator
# ---------------------------------------------------------------------------
class DifferentiableLensSimulator(nn.Module):
    """End-to-end differentiable strong-lens simulator.

    Composes: parameters → mass profile → convergence / deflection → ray-traced
    source-plane mapping → lensed image (with optional PSF convolution).

    This enables:
    - Exact gradient computation for NUTS-HMC posterior sampling
    - Fisher information matrix via autograd Hessian
    - End-to-end training with physics in the loop

    Parameters
    ----------
    profile : nn.Module
        A differentiable mass profile (DifferentiableNFW or DifferentiableSIS).
    grid_size : int
        Number of pixels per side.
    extent_arcsec : float
        Half-width of the field of view in arcseconds.
    source_type : str
        Source model: 'gaussian' or 'sersic'.
    """

    def __init__(
        self,
        profile: nn.Module,
        grid_size: int = 64,
        extent_arcsec: float = 3.0,
        source_type: str = "gaussian",
    ):
        super().__init__()
        self.profile = profile
        self.grid_size = grid_size
        self.extent_arcsec = extent_arcsec
        self.source_type = source_type

        # Build coordinate grid (fixed, not learnable)
        lin = torch.linspace(-extent_arcsec, extent_arcsec, grid_size, dtype=torch.float64)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def _source_brightness(
        self,
        beta_x: torch.Tensor,
        beta_y: torch.Tensor,
        src_x: float = 0.1,
        src_y: float = 0.05,
        src_sigma: float = 0.3,
        src_amplitude: float = 1.0,
    ) -> torch.Tensor:
        """Evaluate source surface brightness at source-plane positions."""
        dx = beta_x - src_x
        dy = beta_y - src_y
        r2 = dx**2 + dy**2
        if self.source_type == "sersic":
            # Sérsic n=1 (exponential)
            r = torch.sqrt(r2 + 1e-12)
            b_1 = 1.678  # b_n for n=1
            return src_amplitude * torch.exp(-b_1 * (r / src_sigma - 1.0))
        else:
            return src_amplitude * torch.exp(-0.5 * r2 / src_sigma**2)

    def simulate(
        self,
        source_params: Optional[Dict[str, float]] = None,
        noise_std: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run the full differentiable forward model.

        Parameters
        ----------
        source_params : dict, optional
            Source position and shape: {src_x, src_y, src_sigma, src_amplitude}.
        noise_std : float
            Gaussian noise standard deviation (0 = noiseless).

        Returns
        -------
        dict with keys:
            convergence, alpha_x, alpha_y, lensed_image, source_image
        """
        sp = source_params or {}
        src_x = sp.get("src_x", 0.1)
        src_y = sp.get("src_y", 0.05)
        src_sigma = sp.get("src_sigma", 0.3)
        src_amplitude = sp.get("src_amplitude", 1.0)

        # Forward through mass profile
        out = self.profile(self.grid_x, self.grid_y)
        kappa = out["convergence"]
        alpha_x = out["alpha_x"]
        alpha_y = out["alpha_y"]

        # Ray-trace to source plane:  β = θ - α
        beta_x = self.grid_x - alpha_x
        beta_y = self.grid_y - alpha_y

        # Evaluate source at ray-traced positions
        lensed = self._source_brightness(beta_x, beta_y, src_x, src_y, src_sigma, src_amplitude)

        # Unlensed source (for comparison)
        unlensed = self._source_brightness(self.grid_x, self.grid_y, src_x, src_y, src_sigma, src_amplitude)

        # Optional noise
        if noise_std > 0:
            lensed = lensed + noise_std * torch.randn_like(lensed)

        return {
            "convergence": kappa,
            "alpha_x": alpha_x,
            "alpha_y": alpha_y,
            "lensed_image": lensed,
            "source_image": unlensed,
        }

    def forward(self, source_params: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Alias for simulate()."""
        return self.simulate(source_params)

    def log_likelihood(
        self,
        observed: torch.Tensor,
        noise_std: float = 0.01,
        source_params: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Gaussian log-likelihood: log p(data | θ).

        Parameters
        ----------
        observed : Tensor
            Observed lensed image (grid_size × grid_size).
        noise_std : float
            Known noise level per pixel.
        source_params : dict, optional
            Source parameters.

        Returns
        -------
        Tensor (scalar)
            Log-likelihood value (differentiable w.r.t. profile parameters).
        """
        sim = self.simulate(source_params, noise_std=0.0)
        residual = observed - sim["lensed_image"]
        n_pix = residual.numel()
        chi2 = torch.sum(residual**2) / (noise_std**2)
        log_norm = -0.5 * n_pix * math.log(2.0 * math.pi * noise_std**2)
        return log_norm - 0.5 * chi2
