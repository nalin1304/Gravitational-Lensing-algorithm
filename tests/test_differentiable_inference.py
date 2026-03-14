"""
Tests for the Differentiable Inference Engine (NUTS-HMC + Fisher Information).

Tests cover:
1. Differentiable NFW forward model
2. Differentiable SIS forward model
3. Full differentiable lens simulator
4. Gradient computation through the full pipeline
5. NUTS-HMC posterior sampling
6. Fisher information matrix
7. Amortized refinement pipeline
"""

import pytest
import numpy as np
import torch

from src.inference.differentiable_simulator import (
    DifferentiableNFW,
    DifferentiableSIS,
    DifferentiableLensSimulator,
)
from src.inference.nuts_hmc import (
    NUTSSampler,
    FisherInformation,
    AmortizedRefinement,
    LensingLogPosterior,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def nfw_profile():
    return DifferentiableNFW(
        log10_M_vir=14.0, concentration=5.0,
        z_lens=0.3, z_source=1.5,
    )


@pytest.fixture
def sis_profile():
    return DifferentiableSIS(theta_E=1.2)


@pytest.fixture
def nfw_simulator(nfw_profile):
    return DifferentiableLensSimulator(
        nfw_profile, grid_size=32, extent_arcsec=3.0,
    )


@pytest.fixture
def sis_simulator(sis_profile):
    return DifferentiableLensSimulator(
        sis_profile, grid_size=32, extent_arcsec=3.0,
    )


# ---------------------------------------------------------------------------
# Test Differentiable NFW
# ---------------------------------------------------------------------------
class TestDifferentiableNFW:
    def test_convergence_shape(self, nfw_profile):
        theta = torch.linspace(-3, 3, 32, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        kappa = nfw_profile.convergence(tx, ty)
        assert kappa.shape == (32, 32)

    def test_convergence_positive(self, nfw_profile):
        theta = torch.linspace(-3, 3, 32, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        kappa = nfw_profile.convergence(tx, ty)
        assert torch.all(kappa >= 0)

    def test_convergence_radial_decrease(self, nfw_profile):
        """NFW convergence should decrease with radius at large r."""
        r_inner = torch.tensor([0.5], dtype=torch.float64)
        r_outer = torch.tensor([10.0], dtype=torch.float64)
        zero = torch.tensor([0.0], dtype=torch.float64)
        k_inner = nfw_profile.convergence(r_inner, zero)
        k_outer = nfw_profile.convergence(r_outer, zero)
        assert k_inner > k_outer

    def test_deflection_shape(self, nfw_profile):
        theta = torch.linspace(-3, 3, 32, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        ax, ay = nfw_profile.deflection(tx, ty)
        assert ax.shape == (32, 32)
        assert ay.shape == (32, 32)

    def test_gradient_exists(self, nfw_profile):
        """Verify gradients flow through NFW convergence computation."""
        theta = torch.linspace(-3, 3, 16, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        kappa = nfw_profile.convergence(tx, ty)
        loss = kappa.sum()
        loss.backward()
        assert nfw_profile.log10_M_vir.grad is not None
        assert nfw_profile.concentration.grad is not None
        assert not torch.isnan(nfw_profile.log10_M_vir.grad)

    def test_forward_dict_keys(self, nfw_profile):
        theta = torch.linspace(-3, 3, 16, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        out = nfw_profile(tx, ty)
        assert "convergence" in out
        assert "alpha_x" in out
        assert "alpha_y" in out

    def test_mass_sensitivity(self):
        """Higher mass → higher convergence."""
        low = DifferentiableNFW(log10_M_vir=13.0, concentration=5.0, z_lens=0.3, z_source=1.5)
        high = DifferentiableNFW(log10_M_vir=15.0, concentration=5.0, z_lens=0.3, z_source=1.5)
        theta = torch.linspace(-3, 3, 16, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        k_low = low.convergence(tx, ty).sum()
        k_high = high.convergence(tx, ty).sum()
        assert k_high > k_low


# ---------------------------------------------------------------------------
# Test Differentiable SIS
# ---------------------------------------------------------------------------
class TestDifferentiableSIS:
    def test_convergence_shape(self, sis_profile):
        theta = torch.linspace(-3, 3, 32, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        kappa = sis_profile.convergence(tx, ty)
        assert kappa.shape == (32, 32)

    def test_deflection_constant_magnitude(self, sis_profile):
        """SIS deflection magnitude is constant = θ_E."""
        theta = torch.linspace(-3, 3, 32, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        ax, ay = sis_profile.deflection(tx, ty)
        mag = torch.sqrt(ax**2 + ay**2)
        # Should be approximately constant everywhere (= theta_E)
        expected = torch.abs(sis_profile.theta_E)
        assert torch.allclose(mag, expected, atol=0.01)

    def test_gradient_exists(self, sis_profile):
        theta = torch.linspace(-3, 3, 16, dtype=torch.float64)
        tx, ty = torch.meshgrid(theta, theta, indexing="ij")
        kappa = sis_profile.convergence(tx, ty)
        kappa.sum().backward()
        assert sis_profile.theta_E.grad is not None


# ---------------------------------------------------------------------------
# Test Full Simulator
# ---------------------------------------------------------------------------
class TestDifferentiableLensSimulator:
    def test_simulate_output_keys(self, nfw_simulator):
        out = nfw_simulator.simulate()
        for key in ("convergence", "alpha_x", "alpha_y", "lensed_image", "source_image"):
            assert key in out, f"Missing key: {key}"

    def test_simulate_shapes(self, nfw_simulator):
        out = nfw_simulator.simulate()
        assert out["convergence"].shape == (32, 32)
        assert out["lensed_image"].shape == (32, 32)

    def test_lensed_image_positive(self, nfw_simulator):
        out = nfw_simulator.simulate()
        assert torch.all(out["lensed_image"] >= -0.01)  # Gaussian source, near-zero edges

    def test_gradient_through_simulator(self, nfw_simulator):
        """Full backprop test: d(lensed_image) / d(log10_M_vir) exists."""
        out = nfw_simulator.simulate()
        loss = out["lensed_image"].sum()
        loss.backward()
        profile = nfw_simulator.profile
        assert profile.log10_M_vir.grad is not None
        assert not torch.isnan(profile.log10_M_vir.grad)

    def test_log_likelihood_scalar(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        ll = nfw_simulator.log_likelihood(observed, noise_std=0.01)
        assert ll.dim() == 0  # scalar

    def test_log_likelihood_gradient(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        ll = nfw_simulator.log_likelihood(observed, noise_std=0.01)
        ll.backward()
        assert nfw_simulator.profile.log10_M_vir.grad is not None

    def test_sersic_source(self, nfw_profile):
        sim = DifferentiableLensSimulator(
            nfw_profile, grid_size=32, extent_arcsec=3.0, source_type="sersic",
        )
        out = sim.simulate()
        assert out["lensed_image"].shape == (32, 32)

    def test_noise_injection(self, nfw_simulator):
        out_clean = nfw_simulator.simulate(noise_std=0.0)
        out_noisy = nfw_simulator.simulate(noise_std=0.1)
        diff = (out_noisy["lensed_image"] - out_clean["lensed_image"]).abs().mean()
        assert diff > 0.01  # Noise should be visible


# ---------------------------------------------------------------------------
# Test NUTS Sampler
# ---------------------------------------------------------------------------
class TestNUTSSampler:
    def test_nuts_runs(self, nfw_simulator):
        """NUTS produces samples without crashing."""
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()

        log_post = LensingLogPosterior(
            nfw_simulator, observed, noise_std=0.01,
            priors={"log10_M_vir": (14.0, 1.0), "concentration": (5.0, 2.0)},
        )

        sampler = NUTSSampler(log_post, step_size=0.001, max_tree_depth=3, seed=42)
        result = sampler.sample(n_samples=10, warmup=5, progress=False)

        assert "log10_M_vir" in result
        assert "concentration" in result
        assert len(result["log10_M_vir"]) == 10
        assert result["wall_time_s"] > 0

    def test_nuts_posterior_near_truth(self, nfw_simulator):
        """Posterior mean should be near the true parameters."""
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()

        log_post = LensingLogPosterior(
            nfw_simulator, observed, noise_std=0.01,
            priors={"log10_M_vir": (14.0, 0.5), "concentration": (5.0, 1.0)},
        )

        sampler = NUTSSampler(log_post, step_size=0.001, max_tree_depth=4, seed=42)
        result = sampler.sample(n_samples=20, warmup=10, progress=False)

        mean_M = np.mean(result["log10_M_vir"])
        mean_c = np.mean(result["concentration"])
        assert abs(mean_M - 14.0) < 2.0, f"log10_M_vir mean {mean_M} too far from 14.0"
        assert abs(mean_c - 5.0) < 5.0, f"concentration mean {mean_c} too far from 5.0"

    def test_nuts_sis(self, sis_simulator):
        """NUTS works with SIS profile too."""
        out = sis_simulator.simulate()
        observed = out["lensed_image"].detach()

        log_post = LensingLogPosterior(
            sis_simulator, observed, noise_std=0.01,
            priors={"theta_E": (1.2, 0.3)},
        )

        sampler = NUTSSampler(log_post, step_size=0.001, max_tree_depth=3, seed=42)
        result = sampler.sample(n_samples=10, warmup=5, progress=False)
        assert "theta_E" in result
        assert len(result["theta_E"]) == 10

    def test_nuts_metadata(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        log_post = LensingLogPosterior(nfw_simulator, observed, noise_std=0.01)
        sampler = NUTSSampler(log_post, step_size=0.001, max_tree_depth=3, seed=42)
        result = sampler.sample(n_samples=5, warmup=3, progress=False)

        assert result["method"] == "NUTS-HMC"
        assert result["n_samples"] == 5
        assert result["warmup"] == 3
        assert 0 <= result["accept_rate"] <= 1


# ---------------------------------------------------------------------------
# Test Fisher Information
# ---------------------------------------------------------------------------
class TestFisherInformation:
    def test_fisher_shape(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        log_post = LensingLogPosterior(nfw_simulator, observed, noise_std=0.01)

        fisher = FisherInformation(log_post)
        result = fisher.compute()

        assert result["fisher_matrix"].shape == (2, 2)
        assert len(result["marginal_errors"]) == 2
        assert result["covariance"].shape == (2, 2)
        assert result["correlation"].shape == (2, 2)

    def test_fisher_positive_definite(self, nfw_simulator):
        """Fisher matrix should be positive semi-definite at the truth."""
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        log_post = LensingLogPosterior(nfw_simulator, observed, noise_std=0.01)

        fisher = FisherInformation(log_post)
        result = fisher.compute()

        eigvals = np.linalg.eigvalsh(result["fisher_matrix"])
        assert np.all(eigvals >= -1e-6), f"Fisher not PSD: eigenvalues = {eigvals}"

    def test_fisher_marginal_errors_positive(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        log_post = LensingLogPosterior(nfw_simulator, observed, noise_std=0.01)

        fisher = FisherInformation(log_post)
        result = fisher.compute()

        for name, err in zip(result["parameter_names"], result["marginal_errors"]):
            assert err > 0 or np.isnan(err), f"Non-positive error for {name}: {err}"


# ---------------------------------------------------------------------------
# Test Amortized Refinement
# ---------------------------------------------------------------------------
class TestAmortizedRefinement:
    def test_refinement_runs(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()

        refiner = AmortizedRefinement(
            nfw_simulator, observed, noise_std=0.01,
            priors={"log10_M_vir": (14.0, 1.0), "concentration": (5.0, 2.0)},
        )

        result = refiner.refine(n_samples=5, warmup=3, seed=42, max_tree_depth=3)
        assert result["method"] == "amortized_refined_NUTS"
        assert len(result["log10_M_vir"]) == 5

    def test_initialization_from_amortized(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()

        refiner = AmortizedRefinement(
            nfw_simulator, observed, noise_std=0.01,
            priors={"log10_M_vir": (14.0, 1.0), "concentration": (5.0, 2.0)},
        )

        fake_samples = np.array([[13.8, 4.5], [14.1, 5.2], [14.0, 5.0]])
        refiner.initialize_from_amortized(fake_samples, ["log10_M_vir", "concentration"])

        # Check parameters were set to median
        profile = nfw_simulator.profile
        assert abs(float(profile.log10_M_vir) - 14.0) < 0.2
        assert abs(float(profile.concentration) - 5.0) < 0.3

    def test_full_pipeline(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()

        refiner = AmortizedRefinement(
            nfw_simulator, observed, noise_std=0.01,
            priors={"log10_M_vir": (14.0, 1.0), "concentration": (5.0, 2.0)},
        )

        fake_samples = np.array([[14.0, 5.0]] * 10)
        result = refiner.full_pipeline(
            amortized_samples=fake_samples,
            param_names=["log10_M_vir", "concentration"],
            n_samples=5, warmup=3, seed=42,
        )

        assert "fisher" in result
        assert result["total_wall_time_s"] > 0
        assert result["fisher"]["fisher_matrix"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Test Log Posterior
# ---------------------------------------------------------------------------
class TestLensingLogPosterior:
    def test_log_posterior_scalar(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        log_post = LensingLogPosterior(nfw_simulator, observed, noise_std=0.01)
        lp = log_post()
        assert lp.dim() == 0

    def test_log_posterior_with_priors(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        log_post = LensingLogPosterior(
            nfw_simulator, observed, noise_std=0.01,
            priors={"log10_M_vir": (14.0, 1.0)},
        )
        lp = log_post()
        assert lp.dim() == 0
        assert not torch.isnan(lp)

    def test_log_posterior_gradient(self, nfw_simulator):
        out = nfw_simulator.simulate()
        observed = out["lensed_image"].detach()
        log_post = LensingLogPosterior(nfw_simulator, observed, noise_std=0.01)
        lp = log_post()
        lp.backward()
        assert nfw_simulator.profile.log10_M_vir.grad is not None

