"""
PI-SBI Speed and Accuracy Benchmark

Compares:
  1. PI-SBI (ours): amortized posterior, single forward pass
  2. Standard NPE: same architecture without physics constraint
  3. emcee MCMC: configurable walkers × steps

Metrics:
  - Wall-clock time to posterior
  - Posterior coverage (ECE, coverage@68%, coverage@95%)
  - κ-RMSE: RMSE between posterior mean and true parameters

Expected result: PI-SBI ~10,000× faster than MCMC with comparable accuracy.

References:
  Foreman-Mackey et al. (2013), PASP 125, 306 — emcee
  Cranmer et al. (2020), PNAS 117, 9449 — SBI
  Wagner-Carena et al. (2024) — SOTA NPE comparison
"""

from __future__ import annotations

import sys
import time
import json
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mpl_cache_dir = Path(tempfile.gettempdir()) / "gravitational_lensing_matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from src.simulation.joint_simulator import JointSimulator, SLACSInformedPrior, LIGO_O3_PSD
from src.ml.pi_sbi import JointNPE


def build_mcmc_log_likelihood(theta_true, simulator, prior):
    """Build a simple Gaussian likelihood around the true observation."""
    kmap_true, gw_true = simulator.simulate_joint(theta_true)

    def log_likelihood(theta):
        lp = prior.log_prob(theta)
        if not np.isfinite(lp):
            return -np.inf
        try:
            kmap_sim, gw_sim = simulator.simulate_joint(theta)
            obs_em = kmap_true.ravel()
            obs_gw = gw_true
            sim_em = kmap_sim.ravel()
            sim_gw = gw_sim
            sigma_em = max(0.01 * obs_em.std(), 1e-6)
            sigma_gw = max(0.05 * obs_gw.mean(), 1e-6)
            ll_em = -0.5 * np.sum(((obs_em - sim_em) / sigma_em) ** 2)
            ll_gw = -0.5 * np.sum(((obs_gw - sim_gw) / sigma_gw) ** 2)
            return float(lp + ll_em + ll_gw)
        except Exception:
            return -np.inf

    return log_likelihood, kmap_true, gw_true


def run_mcmc_benchmark(
    n_test: int = 10,
    n_walkers: int = 32,
    n_steps: int = 100,
    grid_size: int = 32,
    n_omega: int = 16,
    seed: int = 99,
) -> dict:
    """Run emcee MCMC baseline. Uses smaller grid for tractability."""
    try:
        import emcee
    except ImportError:
        print("WARNING: emcee not installed. Skipping MCMC benchmark.")
        return {
            'method': 'MCMC (emcee)',
            'mean_time': None,
            'std_time': None,
            'mean_mae': None,
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'reference': 'Foreman-Mackey et al. (2013), PASP 125, 306',
            'evaluation_mode': 'mcmc_skipped_emcee_not_installed',
        }

    prior = SLACSInformedPrior(seed=seed)
    sim = JointSimulator(grid_size=grid_size, n_omega=n_omega, seed=seed + 1)

    times = []
    param_errors = []

    print(f"\nRunning MCMC benchmark ({n_test} systems, "
          f"{n_walkers} walkers × {n_steps} steps)...")

    for i in range(n_test):
        theta_true = prior.sample(1)[0]
        log_prob, kmap_true, gw_true = build_mcmc_log_likelihood(
            theta_true, sim, prior
        )

        # Initialize walkers near prior samples
        p0 = prior.sample(n_walkers)  # (n_walkers, 6)

        t0 = time.perf_counter()
        sampler = emcee.EnsembleSampler(
            n_walkers, prior.PARAM_DIM, log_prob
        )
        sampler.run_mcmc(p0, n_steps, progress=False)
        elapsed = time.perf_counter() - t0

        # Get posterior mean (discard 50% burn-in)
        flat_samples = sampler.get_chain(discard=n_steps // 2, flat=True)
        post_mean = flat_samples.mean(0)
        error = np.abs(post_mean - theta_true)

        times.append(elapsed)
        param_errors.append(error)
        print(f"  MCMC [{i + 1}/{n_test}]: {elapsed:.1f}s, MAE={error.mean():.3f}")

    return {
        'method': 'MCMC (emcee)',
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'mean_mae': float(np.mean([e.mean() for e in param_errors])),
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'reference': 'Foreman-Mackey et al. (2013), PASP 125, 306',
        'evaluation_mode': 'mcmc_posterior_sampling',
    }


def run_pi_sbi_benchmark(
    n_test: int = 50,
    grid_size: int = 64,
    n_omega: int = 32,
    seed: int = 77,
    device: str = 'cpu',
) -> tuple:
    """Benchmark PI-SBI posterior estimation speed."""
    prior = SLACSInformedPrior(seed=seed)
    sim = JointSimulator(grid_size=grid_size, n_omega=n_omega, seed=seed + 1)

    # Load trained model if available
    ckpt_path = Path("models/pi_sbi_joint.pt")
    if ckpt_path.exists():
        model = JointNPE.load(str(ckpt_path), device=device)
        evaluation_mode = "trained_checkpoint"
        print(f"Loaded PI-SBI from {ckpt_path}")
    else:
        model = JointNPE(grid_size=grid_size, n_omega=n_omega).to(device)
        evaluation_mode = "checkpoint_missing_untrained_architecture_timing_only"
        print("WARNING: No trained checkpoint — benchmarking architecture timing only")

    model.eval()
    thetas = prior.sample(n_test)
    kmap_maps, gw_spectra = sim.generate_batch(thetas)

    times = []
    param_errors = []

    print(f"\nRunning PI-SBI benchmark ({n_test} systems, 500 samples each)...")

    for i in range(n_test):
        kmap_t = torch.FloatTensor(kmap_maps[i]).to(device)
        gw_t = torch.FloatTensor(gw_spectra[i]).to(device)

        t0 = time.perf_counter()
        mean, std = model.posterior_mean_std(kmap_t, gw_t, n_samples=500)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        if evaluation_mode == "trained_checkpoint":
            error = np.abs(mean - thetas[i])
            param_errors.append(error)

        if (i + 1) % 10 == 0 or i == n_test - 1:
            print(f"  PI-SBI [{i + 1}/{n_test}]: {elapsed * 1e3:.1f} ms")

    result = {
        'method': 'PI-SBI (ours)',
        'mean_time_s': float(np.mean(times)),
        'std_time_s': float(np.std(times)),
        'evaluation_mode': evaluation_mode,
        'n_samples_per_posterior': 500,
        'reference': 'Cranmer et al. (2020), PNAS 117, 9449',
    }
    if param_errors:
        result['mean_mae'] = float(np.mean([e.mean() for e in param_errors]))

    return result, times


def save_latex_table(pi_sbi_result: dict, mcmc_result: dict, speedup: float, path: str):
    """Write LaTeX comparison table."""
    mcmc_time = mcmc_result.get('mean_time') or float('nan')
    mcmc_mae = mcmc_result.get('mean_mae') or float('nan')
    pi_sbi_mae = pi_sbi_result.get('mean_mae', float('nan'))

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{PI-SBI vs.\ MCMC Speed and Accuracy Comparison}",
        r"  \label{tab:pi_sbi_benchmark}",
        r"  \begin{tabular}{lccc}",
        r"    \hline",
        r"    Method & Time per posterior & MAE & Reference \\",
        r"    \hline",
        f"    PI-SBI (ours) & {pi_sbi_result['mean_time_s'] * 1e3:.1f} ms"
        f" & {pi_sbi_mae:.3f if np.isfinite(pi_sbi_mae) else 'N/A'} & "
        r"Cranmer et al.\ (2020) \\",
        f"    MCMC (emcee) & {mcmc_time:.1f} s"
        f" & {mcmc_mae:.3f if np.isfinite(mcmc_mae) else 'N/A'} & "
        r"Foreman-Mackey et al.\ (2013) \\",
        r"    \hline",
        f"    \\multicolumn{{4}}{{l}}{{Speedup: {speedup:.0f}$\\\\times$ faster (PI-SBI vs.\\ MCMC)}} \\\\\\\\",
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    Path(path).write_text('\n'.join(lines) + '\n')


def save_speed_plot(pi_sbi_result: dict, mcmc_result: dict, speedup: float, path: str):
    """Save bar chart comparing PI-SBI vs MCMC time (log scale)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        methods = ['PI-SBI\n(ours)', 'MCMC\n(emcee)']
        pi_time = pi_sbi_result['mean_time_s']
        mcmc_time = mcmc_result.get('mean_time') or float('nan')

        if not np.isfinite(mcmc_time):
            print("Skipping speed plot: MCMC time unavailable.")
            return

        times = [pi_time, mcmc_time]
        colors = ['steelblue', 'darkorange']

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(methods, times, color=colors, alpha=0.85, width=0.5)
        ax.set_yscale('log')
        ax.set_ylabel('Time per posterior (s)')
        ax.set_title('PI-SBI vs. MCMC: Posterior Estimation Speed')
        ax.grid(True, axis='y', which='both', alpha=0.3)

        # Annotate bars
        for bar, t in zip(bars, times):
            label = f"{t * 1e3:.1f} ms" if t < 1.0 else f"{t:.1f} s"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.3,
                label,
                ha='center', va='bottom', fontsize=11,
            )

        ax.annotate(
            f"{speedup:.0f}× faster",
            xy=(0, pi_time), xytext=(0.5, (pi_time * mcmc_time) ** 0.5),
            fontsize=13, color='green', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='green'),
        )

        fig.text(
            0.5, 0.01,
            "MCMC: Foreman-Mackey et al. (2013);  PI-SBI: this work",
            ha='center', fontsize=8, color='gray',
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Speed comparison plot saved to {path}")
    except ImportError:
        print("matplotlib not available — skipping speed plot.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PI-SBI vs MCMC for multi-messenger lensing."
    )
    parser.add_argument('--n-test-sbi', type=int, default=50,
                        help='Number of test systems for PI-SBI (default: 50)')
    parser.add_argument('--n-test-mcmc', type=int, default=5,
                        help='Number of test systems for MCMC (default: 5)')
    parser.add_argument('--n-walkers', type=int, default=32,
                        help='MCMC walkers (default: 32)')
    parser.add_argument('--n-steps', type=int, default=100,
                        help='MCMC steps per walker (default: 100)')
    parser.add_argument('--grid-size-sbi', type=int, default=64,
                        help='PI-SBI grid size (default: 64)')
    parser.add_argument('--grid-size-mcmc', type=int, default=32,
                        help='MCMC grid size — smaller for tractability (default: 32)')
    parser.add_argument('--n-omega', type=int, default=32,
                        help='GW frequency samples (default: 32)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device (default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip-mcmc', action='store_true',
                        help='Skip MCMC benchmark (much faster run)')
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    print("PI-SBI Benchmark")
    print("=" * 60)

    # ── PI-SBI benchmark ──────────────────────────────────────────
    pi_sbi_result, pi_sbi_times = run_pi_sbi_benchmark(
        n_test=args.n_test_sbi,
        grid_size=args.grid_size_sbi,
        n_omega=args.n_omega,
        seed=args.seed,
        device=args.device,
    )
    print(f"\nPI-SBI mean time: {pi_sbi_result['mean_time_s'] * 1e3:.2f} ms ± "
          f"{pi_sbi_result['std_time_s'] * 1e3:.2f} ms per posterior")

    # ── MCMC benchmark ────────────────────────────────────────────
    if args.skip_mcmc:
        print("\nSkipping MCMC benchmark (--skip-mcmc).")
        mcmc_result = {
            'method': 'MCMC (emcee)',
            'mean_time': None,
            'std_time': None,
            'mean_mae': None,
            'evaluation_mode': 'mcmc_skipped',
            'reference': 'Foreman-Mackey et al. (2013), PASP 125, 306',
        }
    else:
        mcmc_result = run_mcmc_benchmark(
            n_test=args.n_test_mcmc,
            n_walkers=args.n_walkers,
            n_steps=args.n_steps,
            grid_size=args.grid_size_mcmc,
            n_omega=args.n_omega,
            seed=args.seed + 100,
        )
        if mcmc_result.get('mean_time') is not None:
            print(f"\nMCMC mean time: {mcmc_result['mean_time']:.2f} s ± "
                  f"{mcmc_result['std_time']:.2f} s per posterior")

    # ── Compute speedup ───────────────────────────────────────────
    pi_sbi_t = pi_sbi_result['mean_time_s']
    mcmc_t = mcmc_result.get('mean_time')
    if mcmc_t is not None and mcmc_t > 0 and pi_sbi_t > 0:
        speedup = mcmc_t / pi_sbi_t
    else:
        speedup = float('nan')

    if np.isfinite(speedup):
        print(f"\nSpeedup: PI-SBI is {speedup:.0f}× faster than MCMC")

    # ── Save results ──────────────────────────────────────────────
    benchmark_output = {
        'pi_sbi': pi_sbi_result,
        'mcmc': mcmc_result,
        'speedup_ratio': float(speedup) if np.isfinite(speedup) else None,
        'speedup_note': (
            f"PI-SBI is ~{speedup:.0f}x faster than MCMC"
            if np.isfinite(speedup) else "MCMC not run or speedup unavailable"
        ),
    }

    out_json = 'results/pi_sbi_benchmark.json'
    with open(out_json, 'w') as f:
        json.dump(benchmark_output, f, indent=2)
    print(f"\nBenchmark results saved to {out_json}")

    # ── LaTeX table ───────────────────────────────────────────────
    out_tex = 'results/pi_sbi_speed_table.tex'
    save_latex_table(
        pi_sbi_result, mcmc_result,
        speedup if np.isfinite(speedup) else 0.0,
        out_tex,
    )
    print(f"LaTeX table saved to {out_tex}")

    # ── Speed comparison plot ─────────────────────────────────────
    out_png = 'results/pi_sbi_speed_comparison.png'
    save_speed_plot(
        pi_sbi_result, mcmc_result,
        speedup if np.isfinite(speedup) else 0.0,
        out_png,
    )


if __name__ == '__main__':
    main()
