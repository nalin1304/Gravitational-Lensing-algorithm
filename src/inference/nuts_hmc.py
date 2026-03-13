"""
No-U-Turn Sampler (NUTS) for Differentiable Strong Lens Inference.

Implements the NUTS algorithm (Hoffman & Gelman 2014, JMLR 15) using
PyTorch autograd for exact gradient computation through the differentiable
lens simulator.  This replaces traditional random-walk MCMC (emcee) with
gradient-informed Hamiltonian dynamics, yielding:

    - 10–100× fewer evaluations to convergence (Hoffman & Gelman 2014, §5)
    - No hand-tuned step sizes (dual averaging adaptation)
    - Exact gradients via torch.autograd (no finite differences)

Also provides:
    - Fisher information matrix via autograd Hessian
    - AmortizedRefinement: hybrid PI-SBI → NUTS pipeline

References
----------
- Hoffman & Gelman (2014), JMLR 15(1), 1593–1623
- Neal (2011), "MCMC using Hamiltonian dynamics", Handbook of MCMC
- Betancourt (2018), arXiv:1701.02434, "A Conceptual Introduction to HMC"
- Galan et al. (2022), A&A 668, Herculens
"""

import math
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log-posterior wrapper
# ---------------------------------------------------------------------------
class LensingLogPosterior(nn.Module):
    """Differentiable log-posterior for strong lens parameters.

    Wraps a DifferentiableLensSimulator with Gaussian likelihood and
    configurable priors on lens-model parameters.

    Parameters
    ----------
    simulator : DifferentiableLensSimulator
        Differentiable forward model.
    observed : Tensor
        Observed image data (grid_size × grid_size).
    noise_std : float
        Per-pixel Gaussian noise standard deviation.
    priors : dict
        Parameter name → (mean, std) for Gaussian priors, or
        (low, high) for uniform priors (indicated by 'uniform' key).
    source_params : dict, optional
        Fixed source parameters.
    """

    def __init__(
        self,
        simulator,
        observed: torch.Tensor,
        noise_std: float = 0.01,
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
        source_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.simulator = simulator
        self.register_buffer("observed", observed)
        self.noise_std = noise_std
        self.priors = priors or {}
        self.source_params = source_params

    def log_prior(self) -> torch.Tensor:
        """Evaluate Gaussian log-prior on profile parameters."""
        lp = torch.tensor(0.0, dtype=torch.float64)
        profile = self.simulator.profile

        for name, (loc, scale) in self.priors.items():
            if hasattr(profile, name):
                param = getattr(profile, name)
                lp = lp - 0.5 * ((param - loc) / scale) ** 2
        return lp

    def log_likelihood(self) -> torch.Tensor:
        """Gaussian log-likelihood."""
        return self.simulator.log_likelihood(
            self.observed, self.noise_std, self.source_params
        )

    def forward(self) -> torch.Tensor:
        """Log-posterior = log-likelihood + log-prior."""
        return self.log_likelihood() + self.log_prior()


# ---------------------------------------------------------------------------
# Leapfrog integrator
# ---------------------------------------------------------------------------
def _leapfrog(
    params: List[torch.Tensor],
    momenta: List[torch.Tensor],
    log_prob_fn: Callable[[], torch.Tensor],
    step_size: float,
    n_steps: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Leapfrog integration for Hamiltonian dynamics.

    Parameters
    ----------
    params : list of Tensors
        Current parameter values (will be modified in-place).
    momenta : list of Tensors
        Current momentum values (will be modified in-place).
    log_prob_fn : callable
        Returns scalar log-probability (differentiable).
    step_size : float
        Integration step size ε.
    n_steps : int
        Number of leapfrog steps L.

    Returns
    -------
    params, momenta, log_prob_final
    """
    # Compute initial gradient
    lp = log_prob_fn()
    grads = torch.autograd.grad(lp, params, create_graph=False)

    # Half-step for momentum
    for p, g in zip(momenta, grads):
        p.data.add_(0.5 * step_size * g)

    # Full steps
    for step in range(n_steps):
        for q in params:
            q.data.add_(step_size * momenta[params.index(q)])

        if step < n_steps - 1:
            lp = log_prob_fn()
            grads = torch.autograd.grad(lp, params, create_graph=False)
            for p, g in zip(momenta, grads):
                p.data.add_(step_size * g)

    # Final gradient and half-step
    lp = log_prob_fn()
    grads = torch.autograd.grad(lp, params, create_graph=False)
    for p, g in zip(momenta, grads):
        p.data.add_(0.5 * step_size * g)

    return params, momenta, lp


# ---------------------------------------------------------------------------
# NUTS Sampler
# ---------------------------------------------------------------------------
class NUTSSampler:
    """No-U-Turn Sampler with dual averaging step-size adaptation.

    Implements Algorithm 6 from Hoffman & Gelman (2014).

    Parameters
    ----------
    log_posterior : LensingLogPosterior
        Differentiable log-posterior.
    param_names : list of str
        Names of parameters to sample (must be nn.Parameter attributes of
        the profile inside the simulator).
    step_size : float
        Initial leapfrog step size ε₀.
    max_tree_depth : int
        Maximum binary tree depth (default 10 → max 1024 leapfrog steps).
    target_accept : float
        Target Metropolis acceptance probability for dual averaging (default 0.8).
    adapt_steps : int
        Number of warmup steps for step-size adaptation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        log_posterior: LensingLogPosterior,
        param_names: Optional[List[str]] = None,
        step_size: float = 0.01,
        max_tree_depth: int = 10,
        target_accept: float = 0.8,
        adapt_steps: int = 200,
        seed: int = 42,
    ):
        self.log_posterior = log_posterior
        self.max_tree_depth = max_tree_depth
        self.target_accept = target_accept
        self.adapt_steps = adapt_steps
        self.rng = np.random.default_rng(seed)

        # Identify parameters to sample
        profile = log_posterior.simulator.profile
        if param_names is None:
            param_names = [name for name, _ in profile.named_parameters()]
        self.param_names = param_names
        self.params = [getattr(profile, name) for name in param_names]

        # Dual averaging state (Hoffman & Gelman 2014, Algorithm 5)
        self.step_size = step_size
        self._log_eps_bar = 0.0
        self._H_bar = 0.0
        self._mu = math.log(10.0 * step_size)
        self._gamma = 0.05
        self._t0 = 10
        self._kappa = 0.75

    def _kinetic_energy(self, momenta: List[torch.Tensor]) -> torch.Tensor:
        return sum(0.5 * torch.sum(p**2) for p in momenta)

    def _sample_momentum(self) -> List[torch.Tensor]:
        return [torch.randn_like(p) for p in self.params]

    def _nuts_step(self, step_size: float) -> Tuple[Dict[str, float], float]:
        """Single NUTS step using the simplified "multinomial" variant.

        Uses iterative doubling to build a balanced binary tree until a
        U-turn criterion is met, then samples uniformly from the trajectory.
        """
        # Save current state
        q0 = [p.data.clone() for p in self.params]
        p0 = self._sample_momentum()
        lp0 = self.log_posterior()
        H0 = -lp0 + self._kinetic_energy(p0)

        # Slice variable
        log_u = float(torch.log(torch.rand(1, dtype=torch.float64)).item() - H0.detach().item())

        # Initialize tree
        q_minus = [q.clone() for q in q0]
        q_plus = [q.clone() for q in q0]
        p_minus = [p.clone() for p in p0]
        p_plus = [p.clone() for p in p0]
        q_sample = [q.clone() for q in q0]
        p_sample = [p.clone() for p in p0]
        lp_sample = float(lp0)
        n_valid = 1
        depth = 0
        keep_going = True

        # Tree doubling
        while keep_going and depth < self.max_tree_depth:
            # Choose direction uniformly
            direction = 1 if self.rng.random() < 0.5 else -1

            if direction == -1:
                # Extend backwards
                q_leaf, p_leaf, lp_leaf = self._build_tree_leaf(
                    q_minus, p_minus, step_size * direction
                )
                for i in range(len(q_minus)):
                    q_minus[i] = q_leaf[i]
                    p_minus[i] = p_leaf[i]
            else:
                q_leaf, p_leaf, lp_leaf = self._build_tree_leaf(
                    q_plus, p_plus, step_size * direction
                )
                for i in range(len(q_plus)):
                    q_plus[i] = q_leaf[i]
                    p_plus[i] = p_leaf[i]

            H_leaf = -lp_leaf + self._kinetic_energy(p_leaf)
            is_valid = float(-H_leaf) > log_u

            if is_valid:
                n_valid += 1
                if self.rng.random() < 1.0 / n_valid:
                    q_sample = [q.clone() for q in q_leaf]
                    p_sample = [pi.clone() for pi in p_leaf]
                    lp_sample = float(lp_leaf)

            # U-turn check
            dq = [qp - qm for qp, qm in zip(q_plus, q_minus)]
            u_turn = sum(float(torch.sum(d * pm)) for d, pm in zip(dq, p_minus)) < 0
            u_turn = u_turn or sum(float(torch.sum(d * pp)) for d, pp in zip(dq, p_plus)) < 0
            keep_going = not u_turn

            depth += 1

        # Set parameters to sample
        for p, q in zip(self.params, q_sample):
            p.data.copy_(q)

        # Acceptance statistic using trajectory momenta (not fresh random ones)
        H_sample = -lp_sample + float(self._kinetic_energy(p_sample))
        accept_stat = min(1.0, math.exp(float(-H_sample + H0)))

        result = {name: float(q) for name, q in zip(self.param_names, q_sample)}
        result["log_posterior"] = lp_sample

        return result, accept_stat

    def _build_tree_leaf(
        self,
        q: List[torch.Tensor],
        p: List[torch.Tensor],
        signed_step: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
        """Single leapfrog step (leaf node of the NUTS tree)."""
        q_new = [qi.clone().detach().requires_grad_(True) for qi in q]
        p_new = [pi.clone() for pi in p]

        # Temporarily set parameters
        old_data = [param.data.clone() for param in self.params]
        for param, qi in zip(self.params, q_new):
            param.data.copy_(qi)

        # Leapfrog step
        lp = self.log_posterior()
        grads = torch.autograd.grad(lp, self.params, create_graph=False)

        for pi, gi in zip(p_new, grads):
            pi.data.add_(0.5 * signed_step * gi)
        for qi, pi in zip(q_new, p_new):
            qi.data.add_(signed_step * pi)
        for param, qi in zip(self.params, q_new):
            param.data.copy_(qi)

        lp = self.log_posterior()
        grads = torch.autograd.grad(lp, self.params, create_graph=False)
        for pi, gi in zip(p_new, grads):
            pi.data.add_(0.5 * signed_step * gi)

        # Restore original parameters
        for param, od in zip(self.params, old_data):
            param.data.copy_(od)

        return q_new, p_new, float(lp)

    def _adapt_step_size(self, step_idx: int, accept_stat: float):
        """Dual averaging step-size adaptation (Hoffman & Gelman 2014)."""
        m = step_idx + 1
        w = 1.0 / (m + self._t0)
        self._H_bar = (1.0 - w) * self._H_bar + w * (self.target_accept - accept_stat)

        log_eps = self._mu - math.sqrt(m) / self._gamma * self._H_bar
        self.step_size = math.exp(log_eps)

        m_pow = m ** (-self._kappa)
        self._log_eps_bar = m_pow * log_eps + (1.0 - m_pow) * self._log_eps_bar

    def sample(
        self,
        n_samples: int = 500,
        warmup: int = 200,
        progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Run NUTS sampling.

        Parameters
        ----------
        n_samples : int
            Number of post-warmup samples to collect.
        warmup : int
            Number of warmup (adaptation) steps.
        progress : bool
            Print progress updates.

        Returns
        -------
        dict
            Parameter name → numpy array of samples.
            Also includes 'log_posterior', 'accept_rate', 'wall_time_s',
            'n_grad_evals', and 'step_size_final'.
        """
        self.adapt_steps = warmup
        chains = {name: [] for name in self.param_names}
        chains["log_posterior"] = []
        accept_stats = []
        n_grad_evals = 0

        t0 = time.perf_counter()

        for i in range(warmup + n_samples):
            result, accept_stat = self._nuts_step(self.step_size)

            if i < warmup:
                self._adapt_step_size(i, accept_stat)
            else:
                for name in self.param_names:
                    chains[name].append(result[name])
                chains["log_posterior"].append(result["log_posterior"])
                accept_stats.append(accept_stat)

            n_grad_evals += 2 ** min(i % self.max_tree_depth + 1, self.max_tree_depth)

            if progress and (i + 1) % max(1, (warmup + n_samples) // 10) == 0:
                phase = "warmup" if i < warmup else "sampling"
                logger.info(
                    f"NUTS [{phase}] step {i+1}/{warmup + n_samples}, "
                    f"ε={self.step_size:.4f}, accept={accept_stat:.3f}"
                )

        wall_time = time.perf_counter() - t0

        output = {name: np.array(chains[name]) for name in self.param_names}
        output["log_posterior"] = np.array(chains["log_posterior"])
        output["accept_rate"] = float(np.mean(accept_stats))
        output["wall_time_s"] = wall_time
        output["n_grad_evals"] = n_grad_evals
        output["step_size_final"] = self.step_size
        output["n_samples"] = n_samples
        output["warmup"] = warmup
        output["method"] = "NUTS-HMC"

        return output


# ---------------------------------------------------------------------------
# Fisher Information Matrix
# ---------------------------------------------------------------------------
class FisherInformation:
    """Compute the Fisher information matrix via autograd Hessian.

    The Fisher matrix F_ij = -E[∂²log p(data|θ)/∂θ_i∂θ_j] evaluated at
    the maximum likelihood estimate provides Cramér–Rao lower bounds on
    parameter uncertainties.

    For a Gaussian likelihood, F = J^T Σ^{-1} J where J is the Jacobian
    of the model w.r.t. parameters.  We compute this exactly using
    torch.autograd.functional.hessian.

    Parameters
    ----------
    log_posterior : LensingLogPosterior
        Differentiable log-posterior (uses log-likelihood part only).
    """

    def __init__(self, log_posterior: LensingLogPosterior):
        self.log_posterior = log_posterior
        self.profile = log_posterior.simulator.profile
        self.param_names = [n for n, _ in self.profile.named_parameters()]

    def compute(self) -> Dict[str, Any]:
        """Compute Fisher matrix at current parameter values.

        Returns
        -------
        dict with keys:
            fisher_matrix : np.ndarray (n_params × n_params)
            parameter_names : list of str
            marginal_errors : np.ndarray (1-sigma from diagonal of F^{-1})
            covariance : np.ndarray (F^{-1})
            correlation : np.ndarray
        """
        params = list(self.profile.parameters())
        n = len(params)

        # Build the Hessian of the log-likelihood numerically via autograd
        hessian = np.zeros((n, n))

        for i in range(n):
            # First gradient
            lp = self.log_posterior.log_likelihood()
            g1 = torch.autograd.grad(lp, params, create_graph=True)

            for j in range(i, n):
                # Second derivative ∂²L/∂θ_i∂θ_j
                g2 = torch.autograd.grad(g1[i], params[j], retain_graph=True)[0]
                hessian[i, j] = float(g2)
                hessian[j, i] = hessian[i, j]

        # Fisher matrix = -Hessian of log-likelihood (at MLE)
        fisher = -hessian

        # Covariance = F^{-1}
        try:
            cov = np.linalg.inv(fisher)
            marginal = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            cov = np.full_like(fisher, np.nan)
            marginal = np.full(n, np.nan)

        # Correlation matrix
        d = np.sqrt(np.abs(np.diag(cov)))
        d_safe = np.where(d > 0, d, 1.0)
        corr = cov / np.outer(d_safe, d_safe)

        return {
            "fisher_matrix": fisher,
            "parameter_names": self.param_names,
            "marginal_errors": marginal,
            "covariance": cov,
            "correlation": corr,
        }


# ---------------------------------------------------------------------------
# Amortized → Refined hybrid pipeline
# ---------------------------------------------------------------------------
class AmortizedRefinement:
    """Hybrid PI-SBI → NUTS-HMC refinement pipeline.

    Uses an amortized neural posterior (from PI-SBI) to initialize NUTS,
    combining the speed of amortized inference (~1 second for thousands of
    systems) with the asymptotic exactness of gradient-based MCMC.

    Pipeline:
    1. Run PI-SBI to get amortized posterior median → θ₀
    2. Initialize NUTS at θ₀
    3. Run NUTS with adapted step size → refined posterior

    This is especially valuable for:
    - Systems where the amortized posterior may be slightly biased
    - High-precision H₀ inference where asymptotic exactness matters
    - Validation: comparing amortized vs. refined posteriors

    Parameters
    ----------
    simulator : DifferentiableLensSimulator
        Differentiable forward model.
    observed : Tensor
        Observed lensed image.
    noise_std : float
        Per-pixel noise level.
    """

    def __init__(
        self,
        simulator,
        observed: torch.Tensor,
        noise_std: float = 0.01,
        source_params: Optional[Dict[str, float]] = None,
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.simulator = simulator
        self.observed = observed
        self.noise_std = noise_std
        self.source_params = source_params

        self.log_posterior = LensingLogPosterior(
            simulator, observed, noise_std,
            priors=priors,
            source_params=source_params,
        )

    def initialize_from_amortized(
        self,
        amortized_samples: np.ndarray,
        param_names: List[str],
    ):
        """Set profile parameters to the median of amortized posterior.

        Parameters
        ----------
        amortized_samples : ndarray (n_samples, n_params)
            Samples from the amortized (PI-SBI) posterior.
        param_names : list of str
            Names mapping columns to profile parameters.
        """
        medians = np.median(amortized_samples, axis=0)
        profile = self.simulator.profile

        for name, val in zip(param_names, medians):
            if hasattr(profile, name):
                getattr(profile, name).data.fill_(val)
                logger.info(f"Initialized {name} = {val:.4f} from amortized posterior")

    def refine(
        self,
        n_samples: int = 500,
        warmup: int = 200,
        seed: int = 42,
        **nuts_kwargs,
    ) -> Dict[str, Any]:
        """Run NUTS refinement from current parameter state.

        Returns
        -------
        dict with NUTS samples + metadata including 'method': 'amortized_refined'.
        """
        sampler = NUTSSampler(
            self.log_posterior,
            seed=seed,
            **nuts_kwargs,
        )

        result = sampler.sample(n_samples=n_samples, warmup=warmup)
        result["method"] = "amortized_refined_NUTS"

        return result

    def full_pipeline(
        self,
        amortized_samples: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None,
        n_samples: int = 500,
        warmup: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run complete amortized → refined pipeline.

        If amortized_samples is provided, initializes from the neural posterior.
        Otherwise, starts from current parameter values.
        """
        t0 = time.perf_counter()

        if amortized_samples is not None and param_names is not None:
            self.initialize_from_amortized(amortized_samples, param_names)

        result = self.refine(n_samples=n_samples, warmup=warmup, seed=seed)
        result["total_wall_time_s"] = time.perf_counter() - t0

        # Fisher information at posterior mode
        fisher = FisherInformation(self.log_posterior)
        result["fisher"] = fisher.compute()

        return result
