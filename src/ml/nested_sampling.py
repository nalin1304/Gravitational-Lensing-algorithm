"""
Lightweight Nested Sampling for Bayesian Model Evidence

Implements a minimal nested sampling algorithm (no external dependency
beyond NumPy) for computing the Bayesian log-evidence ln Z and performing
model selection between competing lens profiles (e.g., NFW vs SIS vs
power-law).

Physics context
---------------
Model selection is critical for deciding whether an observation favors
a singular isothermal sphere (SIS), an NFW profile, or a more complex
mass distribution.  The Bayes factor K = Z_A / Z_B quantitatively
answers "which model is preferred by the data?"

Ref: Skilling (2004) "Nested Sampling" — Bayesian Inference and Maximum
     Entropy Methods in Science and Engineering, AIP Conf. 735, 395
Ref: Feroz et al. (2009) "MultiNest" — MNRAS 398, 1601
Ref: Jeffreys (1961) "Theory of Probability" — Jeffreys scale interpretation

Usage
-----
    from src.ml.nested_sampling import NestedSampler, compute_bayes_factor

    sampler = NestedSampler(n_dims=3, n_live=200)
    result = sampler.run(log_likelihood, prior_transform)
    print(f"ln Z = {result['log_evidence']:.2f}")
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import warnings


class NestedSampler:
    """Nested sampling engine for Bayesian model evidence.

    Parameters
    ----------
    n_dims : int
        Dimensionality of the parameter space.
    n_live : int
        Number of live points (higher → more accurate but slower).
        Recommended: 25 × n_dims for reliable evidence estimates.
    seed : int
        Random seed for reproducibility.

    Notes
    -----
    Algorithm summary (Ref: Skilling 2004, Algorithm 1):
        1. Sample n_live points from the prior
        2. Find the point with lowest likelihood L_min
        3. Record L_min and contract prior volume: X_i → X_i × exp(-1/n_live)
        4. Replace dead point with new sample from prior subject to L > L_min
        5. Repeat until convergence
        6. ln Z = log(Σ w_i × L_i) where w_i are prior volume weights
    """

    def __init__(self, n_dims: int, n_live: int = 200, seed: int = 42):
        self.n_dims = n_dims
        self.n_live = n_live
        self.rng = np.random.RandomState(seed)

    def run(
        self,
        log_likelihood: Callable[[np.ndarray], float],
        prior_transform: Callable[[np.ndarray], np.ndarray],
        max_iter: int = 5000,
        tol: float = 0.1,
        n_replace_attempts: int = 50,
    ) -> Dict:
        """Run nested sampling.

        Parameters
        ----------
        log_likelihood : callable
            log L(θ) — log-likelihood function mapping parameters → scalar.
        prior_transform : callable
            Maps unit hypercube [0,1]^d → physical parameter space.
            This encodes the prior distribution.
        max_iter : int
            Maximum number of nested sampling iterations.
        tol : float
            Convergence tolerance on the remaining evidence estimate.
        n_replace_attempts : int
            Number of attempts to find a replacement point above the
            likelihood threshold per iteration.

        Returns
        -------
        result : dict
            Keys:
            - 'log_evidence': float — ln Z
            - 'log_evidence_error': float — estimated uncertainty
            - 'information': float — H (information in nats)
            - 'posterior_samples': np.ndarray — weighted posterior samples
            - 'posterior_weights': np.ndarray — sample weights
            - 'n_iterations': int
            - 'n_likelihood_calls': int
        """
        n_calls = 0

        # Step 1: Sample live points from prior
        # Ref: Skilling (2004), §2 — "Start with N objects sampled from π"
        live_u = self.rng.uniform(0, 1, (self.n_live, self.n_dims))
        live_theta = np.array([prior_transform(u) for u in live_u])
        live_logl = np.array([log_likelihood(th) for th in live_theta])
        n_calls += self.n_live

        # Dead points storage
        dead_points = []
        dead_logl = []
        dead_logvol = []

        log_vol = 0.0  # ln(X), prior volume remaining
        log_evidence = -np.inf
        H = 0.0  # Information

        for iteration in range(max_iter):
            # Step 2: Find worst (lowest likelihood) live point
            worst_idx = np.argmin(live_logl)
            logl_min = live_logl[worst_idx]

            # Record dead point
            dead_points.append(live_theta[worst_idx].copy())
            dead_logl.append(logl_min)
            dead_logvol.append(log_vol)

            # Step 3: Update prior volume
            # Ref: Skilling (2004) — X_i ≈ exp(-i/N)
            log_vol_new = log_vol - 1.0 / self.n_live
            log_weight = np.logaddexp(log_vol, log_vol_new) - np.log(2.0)

            # Update evidence: ln Z = ln(Z_prev + L_i × w_i)
            log_evidence_new = np.logaddexp(log_evidence, logl_min + log_weight)

            # Update information H
            if log_evidence_new > -np.inf:
                H = (
                    np.exp(logl_min + log_weight - log_evidence_new) * logl_min
                    + np.exp(log_evidence - log_evidence_new) * (H + log_evidence)
                    - log_evidence_new
                )

            log_evidence = log_evidence_new
            log_vol = log_vol_new

            # Step 4: Replace dead point with new sample L > L_min
            replaced = False
            for _ in range(n_replace_attempts):
                # Simple prior sampling with likelihood constraint
                # (More sophisticated: ellipsoidal sampling, slice sampling)
                u_new = self.rng.uniform(0, 1, self.n_dims)
                theta_new = prior_transform(u_new)
                logl_new = log_likelihood(theta_new)
                n_calls += 1

                if logl_new > logl_min:
                    live_u[worst_idx] = u_new
                    live_theta[worst_idx] = theta_new
                    live_logl[worst_idx] = logl_new
                    replaced = True
                    break

            if not replaced:
                # Try harder: use MCMC within prior (random walk from existing point)
                for _ in range(n_replace_attempts):
                    donor_idx = self.rng.randint(self.n_live)
                    u_new = live_u[donor_idx] + self.rng.normal(0, 0.1, self.n_dims)
                    u_new = np.clip(u_new, 0, 1)
                    theta_new = prior_transform(u_new)
                    logl_new = log_likelihood(theta_new)
                    n_calls += 1

                    if logl_new > logl_min:
                        live_u[worst_idx] = u_new
                        live_theta[worst_idx] = theta_new
                        live_logl[worst_idx] = logl_new
                        replaced = True
                        break

            if not replaced:
                warnings.warn(
                    f"Nested sampling: could not replace dead point at "
                    f"iteration {iteration}, stopping early."
                )
                break

            # Step 5: Check convergence
            # Remaining evidence ≈ max(L_live) × X_remaining
            log_remaining = np.max(live_logl) + log_vol
            if log_remaining < log_evidence + np.log(tol):
                break

        # Add surviving live points using Skilling (2004) order-statistic volumes.
        # Sort ascending by likelihood so the lowest-L point gets the largest
        # remaining volume fraction.
        # Ref: Skilling (2004), §4 — terminal live-point volume fractions.
        sorted_idx = np.argsort(live_logl)
        n = self.n_live
        for rank, i in enumerate(sorted_idx):
            # Expected remaining volume fraction for rank-th point out of n live points
            if rank < n - 1:
                frac = 1.0 / (n - rank)
                log_dV = log_vol + np.log(frac) - np.log(n - rank + 1)
            else:
                log_dV = log_vol
            dead_points.append(live_theta[i].copy())
            dead_logl.append(live_logl[i])
            dead_logvol.append(log_dV)
            log_evidence = np.logaddexp(log_evidence, live_logl[i] + log_dV)

        # Build posterior samples
        dead_points = np.array(dead_points)
        dead_logl = np.array(dead_logl)
        dead_logvol = np.array(dead_logvol)

        # Posterior weights: w_i ∝ L_i × ΔX_i / Z
        log_weights = dead_logl + dead_logvol - log_evidence
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()

        # Evidence uncertainty: σ(ln Z) ≈ √(H/N)
        log_evidence_error = np.sqrt(max(H, 0.0) / self.n_live)

        return {
            'log_evidence': float(log_evidence),
            'log_evidence_error': float(log_evidence_error),
            'information': float(H),
            'posterior_samples': dead_points,
            'posterior_weights': weights,
            'n_iterations': iteration + 1,
            'n_likelihood_calls': n_calls,
        }


def compute_bayes_factor(ln_Z_1: float, ln_Z_2: float) -> Dict:
    """Compute Bayes factor and interpret on the Jeffreys scale.

    Parameters
    ----------
    ln_Z_1 : float
        Log-evidence for model 1 (numerator).
    ln_Z_2 : float
        Log-evidence for model 2 (denominator).

    Returns
    -------
    result : dict
        Keys: 'ln_bayes_factor', 'bayes_factor', 'interpretation',
              'preferred_model'

    Notes
    -----
    Jeffreys scale (Ref: Jeffreys 1961, Kass & Raftery 1995):

    | ln K       | Interpretation         |
    |------------|------------------------|
    | < 1        | Not worth mention      |
    | 1 – 2.5    | Substantial            |
    | 2.5 – 5    | Strong                 |
    | > 5        | Decisive               |
    """
    ln_K = ln_Z_1 - ln_Z_2

    abs_ln_K = abs(ln_K)
    if abs_ln_K < 1.0:
        interpretation = "Not worth more than a bare mention"
    elif abs_ln_K < 2.5:
        interpretation = "Substantial evidence"
    elif abs_ln_K < 5.0:
        interpretation = "Strong evidence"
    else:
        interpretation = "Decisive evidence"

    preferred = 1 if ln_K > 0 else 2

    return {
        'ln_bayes_factor': float(ln_K),
        'bayes_factor': float(np.exp(min(ln_K, 700))),  # Avoid overflow
        'interpretation': interpretation,
        'preferred_model': preferred,
    }


def model_selection_demo() -> Dict:
    """Demonstrate model selection between NFW and SIS profiles.

    Generates synthetic data from an NFW profile and compares evidence
    for NFW vs SIS fits to show that nested sampling correctly identifies
    the true generating model.

    Returns
    -------
    result : dict
        Contains evidence values and Bayes factor.
    """
    rng = np.random.RandomState(42)

    # Generate synthetic data from NFW
    n_data = 50
    r_data = np.sort(rng.uniform(0.1, 2.0, n_data))

    # NFW convergence profile (simplified 1D)
    # Ref: Wright & Brainerd (2000), Eq. 11
    rs = 0.5  # Scale radius
    kappa_s = 0.3  # Characteristic convergence
    x = r_data / rs
    kappa_nfw_true = np.where(
        x < 1,
        kappa_s * 2 / (x**2 - 1) * (1 - np.arccosh(1/x) / np.sqrt(1 - x**2)),
        np.where(
            x > 1,
            kappa_s * 2 / (x**2 - 1) * (1 - np.arccos(1/x) / np.sqrt(x**2 - 1)),
            kappa_s * 2.0 / 3.0
        )
    )
    sigma_noise = 0.02
    data = kappa_nfw_true + rng.normal(0, sigma_noise, n_data)

    # Model A: NFW (2 params: kappa_s, r_s)
    def log_likelihood_nfw(params):
        ks, r_s = params
        if ks <= 0 or r_s <= 0:
            return -1e10
        x = r_data / r_s
        kappa = np.where(
            x < 1 - 1e-6,
            ks * 2 / (x**2 - 1) * (1 - np.arccosh(1/np.clip(x, 1e-6, 1-1e-6)) / np.sqrt(1 - x**2)),
            np.where(
                x > 1 + 1e-6,
                ks * 2 / (x**2 - 1) * (1 - np.arccos(1/np.clip(x, 1+1e-6, 100)) / np.sqrt(x**2 - 1)),
                ks * 2.0 / 3.0
            )
        )
        return -0.5 * np.sum(((data - kappa) / sigma_noise) ** 2)

    def prior_nfw(u):
        return np.array([u[0] * 1.0, u[1] * 2.0 + 0.1])  # kappa_s in [0,1], r_s in [0.1,2.1]

    # Model B: SIS (1 param: theta_E)
    def log_likelihood_sis(params):
        theta_E = params[0]
        if theta_E <= 0:
            return -1e10
        kappa = theta_E / (2 * r_data)
        return -0.5 * np.sum(((data - kappa) / sigma_noise) ** 2)

    def prior_sis(u):
        return np.array([u[0] * 2.0 + 0.01])  # theta_E in [0.01, 2.01]

    print("  Running nested sampling for NFW model...")
    sampler_nfw = NestedSampler(n_dims=2, n_live=200, seed=42)
    result_nfw = sampler_nfw.run(log_likelihood_nfw, prior_nfw, max_iter=3000)

    print("  Running nested sampling for SIS model...")
    sampler_sis = NestedSampler(n_dims=1, n_live=200, seed=42)
    result_sis = sampler_sis.run(log_likelihood_sis, prior_sis, max_iter=3000)

    # Compare
    bf = compute_bayes_factor(result_nfw['log_evidence'], result_sis['log_evidence'])

    print(f"\n  NFW: ln Z = {result_nfw['log_evidence']:.2f} ± {result_nfw['log_evidence_error']:.2f}")
    print(f"  SIS: ln Z = {result_sis['log_evidence']:.2f} ± {result_sis['log_evidence_error']:.2f}")
    print(f"  Bayes factor: ln K = {bf['ln_bayes_factor']:.2f} → {bf['interpretation']}")
    print(f"  Preferred model: {'NFW' if bf['preferred_model'] == 1 else 'SIS'}")

    return {
        'nfw': result_nfw,
        'sis': result_sis,
        'bayes_factor': bf,
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NESTED SAMPLING — Bayesian Model Evidence Demo")
    print("=" * 60)
    result = model_selection_demo()
    print("\n✓ Nested sampling demo complete.")
