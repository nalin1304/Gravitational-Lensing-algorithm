"""
Neural ODE models for gravitational dynamics trajectories.
"""

from __future__ import annotations

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    jax = None  # type: ignore
    jnp = None  # type: ignore

try:
    import equinox as eqx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    eqx = None  # type: ignore

try:
    import diffrax  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    diffrax = None  # type: ignore


HAS_NODE_DEPS = bool(jax is not None and eqx is not None and diffrax is not None and jnp is not None)


if HAS_NODE_DEPS:
    class NeuralVectorField(eqx.Module):
        """
        Continuous neural vector field dy/dt = f(t, y).
        """

        mlp: eqx.nn.MLP

        def __init__(self, state_dim: int, hidden_dim: int, depth: int, key: jax.random.PRNGKey):
            self.mlp = eqx.nn.MLP(
                in_size=state_dim + 1,
                out_size=state_dim,
                width_size=hidden_dim,
                depth=depth,
                activation=jax.nn.softplus,
                key=key,
            )

        def __call__(self, t: float, y: jax.Array, args) -> jax.Array:
            time_input = jnp.array([t])
            inputs = jnp.concatenate([time_input, y])
            return self.mlp(inputs)


    class DynamicsNODE(eqx.Module):
        """
        Standard neural ODE for dynamical trajectory modeling.
        """

        vector_field: NeuralVectorField

        def __init__(self, state_dim: int = 12, hidden_dim: int = 64, layers: int = 3, *, seed: int = 42):
            key = jax.random.PRNGKey(seed)
            self.vector_field = NeuralVectorField(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                depth=layers,
                key=key,
            )

        def __call__(self, y0: jax.Array, ts: jax.Array) -> jax.Array:
            term = diffrax.ODETerm(self.vector_field)
            solver = diffrax.Tsit5()
            saveat = diffrax.SaveAt(ts=ts)
            stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=ts[0],
                t1=ts[-1],
                dt0=None,
                y0=y0,
                stepsize_controller=stepsize_controller,
                saveat=saveat,
                max_steps=4096,
            )
            return solution.ys


    class AnalyticFusingNODE(DynamicsNODE):
        """
        Hybrid NODE with explicit Newtonian baseline and learned residual dynamics.
        """

        G: float
        m_mw: float
        m_lmc: float

        def __init__(self, state_dim: int = 12, hidden_dim: int = 64, layers: int = 3, *, seed: int = 42):
            super().__init__(state_dim=state_dim, hidden_dim=hidden_dim, layers=layers, seed=seed)
            self.G = 4.3009e-6
            self.m_mw = 1.0e12
            self.m_lmc = 1.0e11

        def analytical_gravity(self, t: float, y: jax.Array) -> jax.Array:
            del t
            r1 = y[0:3]
            v1 = y[3:6]
            r2 = y[6:9]
            v2 = y[9:12]

            displacement = r2 - r1
            distance = jnp.linalg.norm(displacement) + 1e-4

            force_mag = self.G * self.m_mw * self.m_lmc / (distance**2)
            direction = displacement / distance

            a1 = (force_mag / self.m_mw) * direction
            a2 = -(force_mag / self.m_lmc) * direction
            return jnp.concatenate([v1, a1, v2, a2])

        def __call__(self, y0: jax.Array, ts: jax.Array) -> jax.Array:
            def fused_vector_field(t, y, args):
                analytic_term = self.analytical_gravity(t, y)
                neural_term = self.vector_field(t, y, args)
                return analytic_term + neural_term

            term = diffrax.ODETerm(fused_vector_field)
            solver = diffrax.Tsit5()
            saveat = diffrax.SaveAt(ts=ts)
            stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=ts[0],
                t1=ts[-1],
                dt0=None,
                y0=y0,
                stepsize_controller=stepsize_controller,
                saveat=saveat,
                max_steps=4096,
            )
            return solution.ys


else:
    class NeuralVectorField:  # type: ignore[no-redef]
        """Dependency stub when JAX/Equinox/Diffrax is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Neural ODE requires optional dependencies: jax, equinox, and diffrax."
            )


    class DynamicsNODE:  # type: ignore[no-redef]
        """Dependency stub when JAX/Equinox/Diffrax is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "DynamicsNODE requires optional dependencies: jax, equinox, and diffrax."
            )


    class AnalyticFusingNODE(DynamicsNODE):  # type: ignore[no-redef]
        """Dependency stub when JAX/Equinox/Diffrax is unavailable."""
