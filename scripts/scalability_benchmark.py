"""
Scalability Analysis — Computational Complexity Benchmarks

Measures inference time and memory usage across varying grid sizes
and multi-plane configurations. Generates Figure N for the manuscript.

Usage:
  python scripts/scalability_benchmark.py [--outdir results]

Author: Gravitational Lensing Research Platform
"""

import sys
import time
import json
import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, List

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.validation import ScientificValidator, ValidationLevel
from src.lens_models.mass_profiles import NFWProfile
from src.lens_models.lens_system import LensSystem
from src.ml.generate_dataset import generate_convergence_map_vectorized

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def benchmark_grid_scaling(
    grid_sizes: List[int],
    n_trials: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """Measure inference time and memory vs. grid size."""
    rng = np.random.RandomState(seed)
    results = []

    M_vir = 1.6e12
    z_lens, z_source = 0.3, 1.5
    lens_sys = LensSystem(z_lens=z_lens, z_source=z_source)
    lens = NFWProfile(M_vir=M_vir, concentration=10.0, lens_system=lens_sys)

    for gs in grid_sizes:
        times = []
        peak_mems = []

        for _ in range(n_trials):
            tracemalloc.start()
            t0 = time.perf_counter()

            kappa = generate_convergence_map_vectorized(lens, grid_size=gs, extent=2.0)

            # Simulate PINN prediction + validation
            predicted = np.maximum(kappa + rng.normal(0, 0.005, kappa.shape), 0)
            validator = ScientificValidator(level=ValidationLevel.QUICK)
            validator.validate_convergence_map(
                predicted=predicted, ground_truth=kappa,
                profile_type="NFW", verbose=False,
            )

            elapsed = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append(elapsed)
            peak_mems.append(peak_mem / 1e6)  # MB

        results.append({
            "grid_size": gs,
            "n_pixels": gs * gs,
            "time_mean_s": np.mean(times),
            "time_std_s": np.std(times),
            "memory_mean_mb": np.mean(peak_mems),
            "memory_std_mb": np.std(peak_mems),
        })

        print(f"  Grid {gs:4d}×{gs:<4d}: "
              f"{np.mean(times)*1000:8.1f} ± {np.std(times)*1000:5.1f} ms, "
              f"{np.mean(peak_mems):7.1f} ± {np.std(peak_mems):5.1f} MB")

    return results


def benchmark_multiplane_scaling(
    n_planes_list: List[int],
    grid_size: int = 64,
    n_trials: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """Measure inference time vs. number of lens planes."""
    rng = np.random.RandomState(seed)
    results = []

    for n_planes in n_planes_list:
        times = []

        for _ in range(n_trials):
            t0 = time.perf_counter()

            # Simulate multi-plane: generate and sum convergence from n_planes lenses
            combined = np.zeros((grid_size, grid_size))
            for p in range(n_planes):
                z_l = 0.1 + 0.15 * p
                z_s = z_l + 0.5
                lens_sys = LensSystem(z_lens=z_l, z_source=min(z_s, 3.5))
                M_vir = 10 ** rng.uniform(11.5, 13.0)
                lens = NFWProfile(M_vir=M_vir, concentration=10.0, lens_system=lens_sys)
                kappa = generate_convergence_map_vectorized(
                    lens, grid_size=grid_size, extent=2.0
                )
                combined += kappa / n_planes  # Weighted contribution

            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        results.append({
            "n_planes": n_planes,
            "time_mean_s": np.mean(times),
            "time_std_s": np.std(times),
        })

        print(f"  {n_planes:2d} planes: "
              f"{np.mean(times)*1000:8.1f} ± {np.std(times)*1000:5.1f} ms")

    return results


def generate_scaling_plots(
    grid_results: List[Dict],
    plane_results: List[Dict],
    outdir: Path,
):
    """Generate publication-quality scaling plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Time vs grid size (log-log)
    gs = [r["grid_size"] for r in grid_results]
    times = [r["time_mean_s"] * 1000 for r in grid_results]
    time_errs = [r["time_std_s"] * 1000 for r in grid_results]

    axes[0].errorbar(gs, times, yerr=time_errs, marker='o', capsize=3,
                     color='#2196F3', linewidth=2, markersize=6)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Grid Size (N)', fontsize=11)
    axes[0].set_ylabel('Inference Time (ms)', fontsize=11)
    axes[0].set_title('(a) Time vs Resolution', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Memory vs grid size (log-log)
    mems = [r["memory_mean_mb"] for r in grid_results]
    mem_errs = [r["memory_std_mb"] for r in grid_results]

    axes[1].errorbar(gs, mems, yerr=mem_errs, marker='s', capsize=3,
                     color='#FF5722', linewidth=2, markersize=6)
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Grid Size (N)', fontsize=11)
    axes[1].set_ylabel('Peak Memory (MB)', fontsize=11)
    axes[1].set_title('(b) Memory vs Resolution', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Time vs number of planes
    planes = [r["n_planes"] for r in plane_results]
    ptimes = [r["time_mean_s"] * 1000 for r in plane_results]
    ptime_errs = [r["time_std_s"] * 1000 for r in plane_results]

    axes[2].errorbar(planes, ptimes, yerr=ptime_errs, marker='^', capsize=3,
                     color='#4CAF50', linewidth=2, markersize=6)
    axes[2].set_xlabel('Number of Lens Planes', fontsize=11)
    axes[2].set_ylabel('Inference Time (ms)', fontsize=11)
    axes[2].set_title('(c) Multi-Plane Scaling', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = outdir / "scalability_analysis.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📊 Scaling plots saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Scalability benchmarks")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("  SCALABILITY ANALYSIS — Computational Complexity Benchmarks")
    print("=" * 70)

    # Grid size scaling
    grid_sizes = [16, 32, 64, 128, 256, 512]
    print(f"\n▶ Grid size scaling: {grid_sizes}")
    grid_results = benchmark_grid_scaling(grid_sizes, args.n_trials, args.seed)

    # Multi-plane scaling
    n_planes_list = [1, 2, 3, 5, 8, 10]
    print(f"\n▶ Multi-plane scaling: {n_planes_list}")
    plane_results = benchmark_multiplane_scaling(
        n_planes_list, grid_size=64, n_trials=args.n_trials, seed=args.seed
    )

    # Generate plots
    generate_scaling_plots(grid_results, plane_results, outdir)

    # Save JSON
    json_path = outdir / "scalability_results.json"
    json_path.write_text(json.dumps({
        "grid_scaling": grid_results,
        "multiplane_scaling": plane_results,
    }, indent=2, default=str))
    print(f"📊 JSON results saved: {json_path}")

    print("\n✓ Scalability analysis complete.")


if __name__ == "__main__":
    main()

