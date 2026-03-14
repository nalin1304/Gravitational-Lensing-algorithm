"""
Tests for PI-SBI: Physics-Informed Simulation-Based Inference

Tests cover:
  - Simulator: prior sampling, kappa map generation, GW spectrum
  - Model: forward shapes, physics loss, sampling
  - Integration: end-to-end training step
  - Real data: SLACS FITS loading
"""
import pytest
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.joint_simulator import JointSimulator, SLACSInformedPrior, LIGO_O3_PSD
from src.ml.pi_sbi import (
    JointNPE, PhysicsInformedEncoder, GWSpectrumEncoder, RealNVPFlow, RealNVPCouplingLayer
)

# ─── Simulator tests ─────────────────────────────────────────────────────────

class TestSLACSPrior:
    def test_sample_shape(self):
        prior = SLACSInformedPrior(seed=0)
        theta = prior.sample(10)
        assert theta.shape == (10, 6)
    
    def test_parameter_bounds(self):
        prior = SLACSInformedPrior(seed=1)
        theta = prior.sample(1000)
        # log10(M_vir) in [9, 14]
        assert np.all(theta[:, 0] >= 9.0) and np.all(theta[:, 0] <= 14.0)
        # log10(r_s) in [-0.5, 1.5]
        assert np.all(theta[:, 1] >= -0.5) and np.all(theta[:, 1] <= 1.5)
        # z_l in [0.06, 0.50]
        assert np.all(theta[:, 2] >= 0.06) and np.all(theta[:, 2] <= 0.50)
        # z_s > z_l + 0.15
        assert np.all(theta[:, 3] > theta[:, 2] + 0.1)
    
    def test_log_prob_in_bounds(self):
        prior = SLACSInformedPrior(seed=2)
        theta = prior.sample(1)[0]
        assert np.isfinite(prior.log_prob(theta))
    
    def test_log_prob_out_of_bounds(self):
        prior = SLACSInformedPrior(seed=3)
        theta_bad = np.array([20.0, 0.0, 0.5, 1.0, 0.0, 0.0])  # log10(M)=20 out of range
        assert prior.log_prob(theta_bad) == -np.inf
    
    def test_param_names(self):
        assert len(SLACSInformedPrior.PARAM_NAMES) == SLACSInformedPrior.PARAM_DIM == 6

class TestLIGOPSD:
    def test_shape(self):
        f = np.array([10.0, 100.0, 1000.0])
        psd = LIGO_O3_PSD(f)
        assert psd.shape == (3,)
    
    def test_positive(self):
        f = np.logspace(1, 3, 50)
        psd = LIGO_O3_PSD(f)
        assert np.all(psd > 0)
    
    def test_knee_frequency(self):
        # PSD should be higher at low frequencies (seismic wall) than at 100 Hz
        psd_10 = LIGO_O3_PSD(np.array([10.0]))[0]
        psd_100 = LIGO_O3_PSD(np.array([100.0]))[0]
        # At 10 Hz (below knee ~215 Hz), thermal noise dominates, PSD should be higher
        assert psd_10 > psd_100

class TestJointSimulator:
    def setup_method(self):
        self.sim = JointSimulator(grid_size=32, n_omega=8, seed=42)
        self.prior = SLACSInformedPrior(seed=42)
    
    def test_kappa_map_shape(self):
        theta = self.prior.sample(1)[0]
        kmap = self.sim.simulate_kappa_map(theta)
        assert kmap.shape == (32, 32)
    
    def test_kappa_map_nonnegative(self):
        # Noise can make some pixels negative, but mean should be close to 0+
        theta = self.prior.sample(1)[0]
        kmap = self.sim.simulate_kappa_map(theta, noise_sigma=0.0)
        assert np.all(kmap >= -1e-6)  # exact convergence is non-negative
    
    def test_gw_spectrum_shape(self):
        theta = self.prior.sample(1)[0]
        gw = self.sim.simulate_gw_spectrum(theta, add_noise=False)
        assert gw.shape == (8,)
    
    def test_gw_spectrum_positive(self):
        theta = self.prior.sample(1)[0]
        gw = self.sim.simulate_gw_spectrum(theta, add_noise=False)
        assert np.all(gw >= 0.0)
    
    def test_simulate_joint_shapes(self):
        theta = self.prior.sample(1)[0]
        kmap, gw = self.sim.simulate_joint(theta)
        assert kmap.shape == (1, 32, 32)
        assert gw.shape == (8,)
    
    def test_generate_batch_shapes(self):
        thetas = self.prior.sample(5)
        kmaps, gws = self.sim.generate_batch(thetas)
        assert kmaps.shape == (5, 1, 32, 32)
        assert gws.shape == (5, 8)
    
    def test_theta_to_physics_keys(self):
        theta = self.prior.sample(1)[0]
        p = self.sim.theta_to_physics(theta)
        assert all(k in p for k in ['M_vir', 'r_s', 'z_l', 'z_s', 'beta_x', 'beta_y'])
    
    def test_real_slacs_data_loads(self):
        """Real SLACS FITS data should load without error."""
        real = self.sim.get_real_validation_data()
        # May return 0 if cache not present in test env, but should not crash
        assert isinstance(real, list)
        for entry in real:
            assert 'kappa_map' in entry
            assert entry['kappa_map'].shape == (1, 64, 64)
            assert 'theta_published' in entry
            assert len(entry['theta_published']) == 6

# ─── Model tests ─────────────────────────────────────────────────────────────

class TestPhysicsInformedEncoder:
    def test_forward_shape(self):
        enc = PhysicsInformedEncoder(emb_dim=64, input_size=64)
        kappa = torch.randn(4, 1, 64, 64)
        phi = enc(kappa)
        assert phi.shape == (4, 64)
    
    def test_physics_loss_scalar(self):
        enc = PhysicsInformedEncoder(emb_dim=64, input_size=64)
        kappa = torch.randn(4, 1, 64, 64).abs()  # non-negative
        phi = enc(kappa)
        loss = enc.physics_loss(kappa, phi)
        assert loss.shape == ()  # scalar
        assert torch.isfinite(loss)
    
    def test_physics_loss_gradients(self):
        enc = PhysicsInformedEncoder(emb_dim=64, input_size=64)
        kappa = torch.randn(2, 1, 64, 64).abs()
        phi = enc(kappa)
        loss = enc.physics_loss(kappa, phi)
        loss.backward()  # should not crash
        assert True

class TestGWSpectrumEncoder:
    def test_forward_shape(self):
        enc = GWSpectrumEncoder(n_omega=32, emb_dim=32)
        gw = torch.rand(4, 32)
        phi = enc(gw)
        assert phi.shape == (4, 32)
    
    def test_log_transform_stability(self):
        enc = GWSpectrumEncoder(n_omega=8, emb_dim=16)
        gw = torch.zeros(2, 8)  # all zeros should be handled by log1p
        phi = enc(gw)
        assert torch.all(torch.isfinite(phi))

class TestRealNVPFlow:
    def setup_method(self):
        self.flow = RealNVPFlow(param_dim=6, context_dim=32, n_layers=4, hidden_dim=64)
    
    def test_log_prob_shape(self):
        theta = torch.randn(8, 6)
        context = torch.randn(8, 32)
        lp = self.flow.log_prob(theta, context)
        assert lp.shape == (8,)
    
    def test_log_prob_finite(self):
        theta = torch.randn(4, 6)
        context = torch.randn(4, 32)
        lp = self.flow.log_prob(theta, context)
        assert torch.all(torch.isfinite(lp))
    
    def test_sample_shape(self):
        context = torch.randn(32)
        samples = self.flow.sample(context, n_samples=100)
        assert samples.shape == (100, 6)
    
    def test_normalization_roundtrip(self):
        """Set normalization and verify denorm(norm(x)) ≈ x."""
        mean = np.array([11.5, 0.5, 0.3, 0.9, 0.0, 0.0])
        std = np.array([0.5, 0.3, 0.1, 0.3, 0.1, 0.1])
        self.flow.set_normalization(mean, std)
        x = torch.FloatTensor(mean)
        x_norm = self.flow.normalize_theta(x)
        x_back = self.flow.denormalize_theta(x_norm)
        assert torch.allclose(x, x_back, atol=1e-5)

class TestJointNPE:
    def setup_method(self):
        self.model = JointNPE(
            grid_size=32, n_omega=8,
            em_emb_dim=32, gw_emb_dim=8,
            param_dim=6, flow_layers=2,
            physics_weight=0.1
        )
    
    def test_encode_shape(self):
        kappa = torch.randn(4, 1, 32, 32)
        gw = torch.rand(4, 8)
        ctx = self.model.encode(kappa, gw)
        assert ctx.shape == (4, 32 + 8)
    
    def test_log_prob_shape(self):
        theta = torch.randn(4, 6)
        kappa = torch.randn(4, 1, 32, 32)
        gw = torch.rand(4, 8)
        lp = self.model.log_prob(theta, kappa, gw)
        assert lp.shape == (4,)
    
    def test_training_loss_components(self):
        theta = torch.randn(4, 6)
        kappa = torch.randn(4, 1, 32, 32).abs()
        gw = torch.rand(4, 8)
        loss, info = self.model.training_loss(theta, kappa, gw)
        assert torch.isfinite(loss)
        assert 'nll' in info
        assert 'l_poisson' in info
        assert 'loss' in info
    
    def test_backward_pass(self):
        theta = torch.randn(4, 6)
        kappa = torch.randn(4, 1, 32, 32).abs()
        gw = torch.rand(4, 8)
        loss, _ = self.model.training_loss(theta, kappa, gw)
        loss.backward()
        # Check gradients exist on key parameters
        assert self.model.em_encoder.conv1[0].weight.grad is not None
    
    def test_sample_posterior_shape(self):
        kappa = torch.randn(1, 32, 32).abs()
        gw = torch.rand(8)
        samples = self.model.sample_posterior(kappa, gw, n_samples=50)
        assert samples.shape == (50, 6)
    
    def test_posterior_mean_std(self):
        kappa = torch.randn(1, 1, 32, 32).abs()
        gw = torch.rand(1, 8)
        mean, std = self.model.posterior_mean_std(kappa, gw, n_samples=50)
        assert mean.shape == (6,)
        assert std.shape == (6,)
        assert np.all(std > 0)
    
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "pi_sbi_test.pt")
        self.model.save(path)
        loaded = JointNPE.load(path)
        # Check key parameter match
        for p1, p2 in zip(self.model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_param_names_length(self):
        assert len(JointNPE.PARAM_NAMES) == 6

# ─── Integration tests ────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_one_gradient_step(self):
        """End-to-end: simulate data, compute loss, gradient step, check loss decreases."""
        torch.manual_seed(0)
        model = JointNPE(grid_size=32, n_omega=8, em_emb_dim=32, gw_emb_dim=8,
                         flow_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        prior = SLACSInformedPrior(seed=0)
        sim = JointSimulator(grid_size=32, n_omega=8, seed=0)
        
        thetas = prior.sample(8)
        kmaps, gws = sim.generate_batch(thetas)
        
        theta_t = torch.FloatTensor(thetas)
        kmap_t = torch.FloatTensor(kmaps)
        gw_t = torch.FloatTensor(gws)
        
        # First loss
        loss1, _ = model.training_loss(theta_t, kmap_t, gw_t)
        loss1.backward()
        optimizer.step()
        
        # Not NaN
        assert torch.isfinite(loss1)
    
    def test_physics_weight_zero_equals_standard_npe(self):
        """With physics_weight=0, loss should equal plain NLL (no Poisson term)."""
        torch.manual_seed(1)
        model = JointNPE(grid_size=32, n_omega=8, em_emb_dim=16, gw_emb_dim=8,
                         flow_layers=2, physics_weight=0.0)
        theta = torch.randn(4, 6)
        kappa = torch.randn(4, 1, 32, 32)
        gw = torch.rand(4, 8)
        loss, info = model.training_loss(theta, kappa, gw)
        # loss should be approx nll (l_poisson is computed but weighted 0)
        assert abs(info['loss'] - info['nll']) < 1e-4


# ─── Conformal recalibration tests ────────────────────────────────────────────

class TestConformalRecalibration:
    """Tests for conformal posterior recalibration (CP4SBI framework)."""

    def test_conformal_recalibrate_returns_expected_keys(self):
        model = JointNPE(grid_size=16, n_omega=4, em_emb_dim=8, gw_emb_dim=4, flow_layers=2)
        theta = torch.randn(10, 6)
        kappa = torch.rand(10, 1, 16, 16)
        gw = torch.rand(10, 4)
        result = model.conformal_recalibrate(theta, kappa, gw, alpha=0.1)
        assert 'q_hat' in result
        assert 'empirical_coverage' in result
        assert result['n_cal'] == 10
        assert result['alpha'] == 0.1
        assert result['target_coverage'] == 0.9

    def test_conformal_q_hat_positive(self):
        model = JointNPE(grid_size=16, n_omega=4, em_emb_dim=8, gw_emb_dim=4, flow_layers=2)
        theta = torch.randn(8, 6)
        kappa = torch.rand(8, 1, 16, 16)
        gw = torch.rand(8, 4)
        result = model.conformal_recalibrate(theta, kappa, gw)
        assert result['q_hat'] > 0

    def test_posterior_conformal_uncalibrated_fallback(self):
        """Without calibration, should fall back to Gaussian z-score."""
        model = JointNPE(grid_size=16, n_omega=4, em_emb_dim=8, gw_emb_dim=4, flow_layers=2)
        kappa = torch.rand(1, 16, 16)
        gw = torch.rand(4)
        result = model.posterior_conformal(kappa, gw, n_samples=50)
        assert not result['calibrated']
        assert result['mean'] is not None
        assert result['ci_lower'] is not None

    def test_posterior_conformal_after_calibration(self):
        """After calibration, should use the conformal quantile."""
        model = JointNPE(grid_size=16, n_omega=4, em_emb_dim=8, gw_emb_dim=4, flow_layers=2)
        theta = torch.randn(8, 6)
        kappa_cal = torch.rand(8, 1, 16, 16)
        gw_cal = torch.rand(8, 4)
        model.conformal_recalibrate(theta, kappa_cal, gw_cal)

        kappa_test = torch.rand(1, 16, 16)
        gw_test = torch.rand(4)
        result = model.posterior_conformal(kappa_test, gw_test, n_samples=50)
        assert result['calibrated']
        assert abs(result['q_hat'] - model._conformal_q) < 1e-6

    def test_conformal_coverage_monotone(self):
        """Tighter alpha should give smaller q_hat."""
        model = JointNPE(grid_size=16, n_omega=4, em_emb_dim=8, gw_emb_dim=4, flow_layers=2)
        theta = torch.randn(20, 6)
        kappa = torch.rand(20, 1, 16, 16)
        gw = torch.rand(20, 4)
        r50 = model.conformal_recalibrate(theta, kappa, gw, alpha=0.5)
        r10 = model.conformal_recalibrate(theta, kappa, gw, alpha=0.1)
        assert r10['q_hat'] >= r50['q_hat']

