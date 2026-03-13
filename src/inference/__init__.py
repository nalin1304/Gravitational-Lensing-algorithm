"""
Differentiable Inference Engine for Strong Gravitational Lensing.

This package provides:

1. **DifferentiableLensSimulator** — A PyTorch-native, fully differentiable
   strong-lens forward model (NFW, SIS, composite profiles).  Every operation
   from mass-profile parameters → convergence → deflection → lensed image is
   composed of differentiable primitives, enabling exact gradient computation
   via ``torch.autograd``.

2. **NUTSSampler** — The No-U-Turn Sampler (Hoffman & Gelman 2014, JMLR 15),
   a self-tuning Hamiltonian Monte Carlo variant that exploits the differentiable
   simulator to draw posterior samples over lens-model parameters with no
   hand-tuned step sizes or trajectory lengths.

3. **FisherInformation** — Automatic Fisher-matrix computation using
   second-order autograd (Hessian of the log-likelihood), providing
   Cramér–Rao lower bounds for parameter forecasts.

4. **AmortizedRefinement** — A hybrid pipeline that initialises NUTS from an
   amortised neural posterior (PI-SBI), then refines with gradient-based HMC.
   This combines the speed of amortised inference with the asymptotic
   exactness of MCMC.

References
----------
- Hoffman & Gelman (2014), JMLR 15, "The No-U-Turn Sampler"
- Galan et al. (2022), A&A 668, "Herculens: differentiable strong lensing"
- Zhou et al. (2024), arXiv:2405.12607, "Hamiltonian differentiable ray tracing"
- Campeau-Poirier et al. (2023), arXiv:2309.15071, "NRE for time delay cosmography"
"""

try:
    from .differentiable_simulator import (
        DifferentiableNFW,
        DifferentiableSIS,
        DifferentiableLensSimulator,
    )
    from .nuts_hmc import (
        NUTSSampler,
        FisherInformation,
        AmortizedRefinement,
        LensingLogPosterior,
    )
except ImportError:
    # PyTorch not available — set to None for graceful degradation
    DifferentiableNFW = None  # type: ignore[assignment,misc]
    DifferentiableSIS = None  # type: ignore[assignment,misc]
    DifferentiableLensSimulator = None  # type: ignore[assignment,misc]
    NUTSSampler = None  # type: ignore[assignment,misc]
    FisherInformation = None  # type: ignore[assignment,misc]
    AmortizedRefinement = None  # type: ignore[assignment,misc]
    LensingLogPosterior = None  # type: ignore[assignment,misc]
