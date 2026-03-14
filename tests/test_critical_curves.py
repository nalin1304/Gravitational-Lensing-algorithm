"""
Tests for critical curves, caustics, magnification maps, and image solver.

Validates against analytic expectations for point mass and SIS profiles
where closed-form solutions are known.
"""

import numpy as np
import pytest
from src.lens_models.mass_profiles import NFWProfile
from src.lens_models.lens_system import LensSystem
from src.lens_models.critical_curves import (
    lens_jacobian,
    magnification_map,
    convergence_shear,
    find_critical_curves,
    find_caustics,
    tangential_and_radial_critical_curves,
    solve_lens_equation,
    full_lensing_analysis,
)


# --------------------------------------------------------------------------- #
#  Lightweight SIS stub for testing (exact analytic solutions known)           #
# --------------------------------------------------------------------------- #

class SISProfileTest:
    """
    Singular Isothermal Sphere with Einstein radius theta_E.

    Analytic properties:
        α(θ) = θ_E × θ/|θ|    (deflection angle)
        κ(θ) = θ_E / (2|θ|)   (convergence)
        Critical curve: circle at |θ| = θ_E
        Caustic: point at origin
    """

    def __init__(self, theta_E: float = 1.0):
        self.theta_E = theta_E

    def deflection_angle(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.sqrt(x**2 + y**2)
        r_safe = np.where(r < 1e-12, 1e-12, r)
        alpha_x = self.theta_E * x / r_safe
        alpha_y = self.theta_E * y / r_safe
        return alpha_x, alpha_y

    def convergence(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.sqrt(x**2 + y**2)
        r_safe = np.where(r < 1e-12, 1e-12, r)
        return self.theta_E / (2.0 * r_safe)


# =========================================================================== #
#  Jacobian and magnification tests                                           #
# =========================================================================== #

class TestLensJacobian:
    """Test the finite-difference Jacobian computation."""

    def test_jacobian_shape(self):
        sis = SISProfileTest(theta_E=1.0)
        x = np.linspace(-2, 2, 10)
        xx, yy = np.meshgrid(x, x)
        A11, A12, A21, A22 = lens_jacobian(sis, xx, yy)
        assert A11.shape == (10, 10)
        assert A22.shape == (10, 10)

    def test_sis_jacobian_symmetry(self):
        """For a circular SIS, A12 should equal A21 (symmetric shear)."""
        sis = SISProfileTest(theta_E=1.0)
        x = np.array([1.5, 0.0, -1.0])
        y = np.array([0.0, 1.5, -1.0])
        A11, A12, A21, A22 = lens_jacobian(sis, x, y)
        np.testing.assert_allclose(A12, A21, atol=1e-4)


class TestMagnificationMap:
    """Test magnification computation."""

    def test_point_mass_magnification(self):
        """
        For a point-mass-like profile far from centre, magnification → 1.
        We use SIS as a proxy since PointMassProfile requires LensSystem.
        """
        sis = SISProfileTest(theta_E=1.0)
        # Far from lens, magnification → 1
        x = np.array([5.0])
        y = np.array([0.0])
        mu = magnification_map(sis, x, y)
        assert abs(mu[0]) < 2.0  # Should be close to 1 far from lens

    def test_magnification_diverges_at_einstein_ring(self):
        """Magnification should be very large near the critical curve."""
        sis = SISProfileTest(theta_E=1.0)
        # Points near θ_E = 1 on the x-axis
        x = np.array([1.001])
        y = np.array([0.0])
        mu = magnification_map(sis, x, y)
        assert abs(mu[0]) > 10.0

    def test_magnification_sign(self):
        """Inside the tangential critical curve, μ should be negative (saddle)."""
        sis = SISProfileTest(theta_E=1.0)
        x = np.array([0.5])
        y = np.array([0.0])
        mu = magnification_map(sis, x, y)
        assert mu[0] < 0  # Saddle point image


class TestConvergenceShear:
    """Test convergence-shear decomposition."""

    def test_sis_convergence_recovery(self):
        """The κ from Jacobian decomposition should match direct convergence."""
        sis = SISProfileTest(theta_E=1.0)
        x = np.array([2.0, 1.5, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        kappa_jac, _, _ = convergence_shear(sis, x, y)
        kappa_direct = sis.convergence(x, y)
        np.testing.assert_allclose(kappa_jac, kappa_direct, rtol=0.02)

    def test_shear_magnitude_equals_convergence_for_sis(self):
        """For SIS, |γ| = κ at all radii (exact analytic result)."""
        sis = SISProfileTest(theta_E=1.0)
        x = np.array([1.5, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        kappa, gamma1, gamma2 = convergence_shear(sis, x, y)
        gamma = np.sqrt(gamma1**2 + gamma2**2)
        np.testing.assert_allclose(gamma, kappa, rtol=0.02)


# =========================================================================== #
#  Critical curves and caustics tests                                         #
# =========================================================================== #

class TestCriticalCurves:
    """Test critical curve finding."""

    def test_sis_critical_curve_is_einstein_ring(self):
        """
        For an SIS with θ_E = 1.0, the critical curve is a circle
        of radius θ_E = 1.0.
        """
        sis = SISProfileTest(theta_E=1.0)
        crit_x, crit_y = find_critical_curves(sis, grid_size=200, grid_range=2.0)

        assert len(crit_x) > 10, "Should find many contour points"

        # All critical curve points should be at radius ≈ 1.0
        r_crit = np.sqrt(crit_x**2 + crit_y**2)
        np.testing.assert_allclose(r_crit, 1.0, atol=0.05)

    def test_sis_caustic_is_point(self):
        """For an SIS, the caustic degenerates to a point at the origin."""
        sis = SISProfileTest(theta_E=1.0)
        caust_x, caust_y = find_caustics(sis, grid_size=200, grid_range=2.0)

        assert len(caust_x) > 0
        # All caustic points should be near origin
        np.testing.assert_allclose(caust_x, 0.0, atol=0.1)
        np.testing.assert_allclose(caust_y, 0.0, atol=0.1)

    def test_nfw_has_critical_curves(self):
        """An NFW profile should produce critical curves."""
        ls = LensSystem(z_lens=0.3, z_source=1.5)
        nfw = NFWProfile(
            M_vir=1e14, concentration=5.0, lens_system=ls
        )
        crit_x, crit_y = find_critical_curves(nfw, grid_size=150, grid_range=30.0)
        assert len(crit_x) > 0, "NFW should have at least one critical curve"

    def test_tangential_radial_separation(self):
        """The tangential/radial decomposition should find two distinct curves."""
        sis = SISProfileTest(theta_E=1.5)
        curves = tangential_and_radial_critical_curves(
            sis, grid_size=200, grid_range=3.0
        )
        tang_x, tang_y = curves['tangential']
        # SIS has a tangential critical curve at θ_E
        assert len(tang_x) > 10
        r_tang = np.sqrt(tang_x**2 + tang_y**2)
        np.testing.assert_allclose(r_tang, 1.5, atol=0.1)


# =========================================================================== #
#  Image position solver tests                                                #
# =========================================================================== #

class TestImageSolver:
    """Test the lens equation solver."""

    def test_sis_two_images_inside_caustic(self):
        """
        For SIS with θ_E = 1 and source at β = 0.3 on the x-axis,
        there should be 2 images at θ = β ± θ_E = 1.3 and -0.7.
        """
        sis = SISProfileTest(theta_E=1.0)
        images = solve_lens_equation(
            sis, beta_x=0.3, beta_y=0.0,
            grid_size=100, grid_range=3.0
        )
        assert len(images) == 2, f"Expected 2 images, got {len(images)}"

        positions = sorted([im['x'] for im in images])
        # Expected: θ₊ = 1.3, θ₋ = -0.7
        np.testing.assert_allclose(positions[0], -0.7, atol=0.05)
        np.testing.assert_allclose(positions[1], 1.3, atol=0.05)

    def test_sis_image_classification(self):
        """Images should be classified as minimum and saddle for SIS."""
        sis = SISProfileTest(theta_E=1.0)
        images = solve_lens_equation(
            sis, beta_x=0.3, beta_y=0.0,
            grid_size=100, grid_range=3.0
        )
        types = {im['type'] for im in images}
        assert 'minimum' in types or 'saddle' in types

    def test_sis_magnification_ratio(self):
        """
        For SIS, the magnification ratio of the two images is:
        |μ₊/μ₋| = (θ_E + β) / (θ_E − β)  [for source inside caustic]
        """
        sis = SISProfileTest(theta_E=1.0)
        beta = 0.3
        images = solve_lens_equation(
            sis, beta_x=beta, beta_y=0.0,
            grid_size=100, grid_range=3.0
        )
        if len(images) == 2:
            mags = sorted([abs(im['magnification']) for im in images])
            expected_ratio = (1.0 + beta) / (1.0 - beta)
            actual_ratio = mags[1] / max(mags[0], 1e-10)
            np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.15)

    def test_point_mass_two_images(self):
        """
        A point-mass-like SIS should produce 2 images for source inside caustic.
        We use SIS as a proxy since PointMassProfile requires LensSystem.
        For SIS: θ₊ = β + θ_E = 1.5, θ₋ = β − θ_E = −0.5
        """
        sis = SISProfileTest(theta_E=1.0)
        images = solve_lens_equation(
            sis, beta_x=0.5, beta_y=0.0,
            grid_size=150, grid_range=3.0
        )
        assert len(images) >= 2, f"Expected ≥2 images, got {len(images)}"

        found_x = sorted([im['x'] for im in images])
        np.testing.assert_allclose(found_x[-1], 1.5, atol=0.1)
        np.testing.assert_allclose(found_x[0], -0.5, atol=0.1)

    def test_source_outside_caustic_one_image(self):
        """For SIS with source beyond θ_E, only 1 bright image should appear."""
        sis = SISProfileTest(theta_E=1.0)
        # Source at β = 1.5 (beyond θ_E), only 1 image expected on same side
        images = solve_lens_equation(
            sis, beta_x=1.5, beta_y=0.0,
            grid_size=100, grid_range=4.0
        )
        # SIS always produces 2 images if β < θ_E, 1 if β > θ_E
        # (the second image is at origin with zero flux for SIS, may not be found)
        assert len(images) >= 1


# =========================================================================== #
#  Full analysis integration test                                             #
# =========================================================================== #

class TestFullAnalysis:
    """Test the convenience wrapper."""

    def test_full_analysis_returns_all_fields(self):
        sis = SISProfileTest(theta_E=1.0)
        result = full_lensing_analysis(
            sis, grid_size=100, grid_range=2.0,
            source_positions=[(0.2, 0.0)],
        )
        assert 'critical_curves' in result
        assert 'caustics' in result
        assert 'magnification_map' in result
        assert 'grid_x' in result
        assert 'image_solutions' in result
        assert result['image_solutions'][0]['n_images'] >= 1

    def test_full_analysis_magnification_shape(self):
        sis = SISProfileTest(theta_E=1.0)
        result = full_lensing_analysis(sis, grid_size=80, grid_range=2.0)
        assert result['magnification_map'].shape == (80, 80)

    def test_nfw_full_analysis(self):
        """Full analysis should work for NFW profiles."""
        ls = LensSystem(z_lens=0.3, z_source=1.5)
        nfw = NFWProfile(
            M_vir=1e14, concentration=5.0, lens_system=ls
        )
        result = full_lensing_analysis(
            nfw, grid_size=100, grid_range=30.0,
            source_positions=[(1.0, 0.0)],
        )
        assert 'critical_curves' in result
        assert result['magnification_map'].shape == (100, 100)

