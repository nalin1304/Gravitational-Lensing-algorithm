"""
PI-SBI Training Script

Trains the Physics-Informed Simulation-Based Inference model on simulated
multi-messenger (EM + GW) gravitational lensing data.

Training uses a SLACS-calibrated prior (Bolton et al. 2006; Auger et al. 2009)
and validates on real HST SLACS observations.

Usage:
  python scripts/train_pi_sbi.py --n-sims 10000 --epochs 50 --grid-size 64

Output:
  models/pi_sbi_joint.pt            — trained checkpoint
  results/pi_sbi_training_curves.png — loss curves
  results/pi_sbi_slacs_posteriors.png — posterior on real SLACS data
  results/pi_sbi_training_summary.json

References:
  Cranmer et al. (2020), PNAS 117, 9449
  Bolton et al. (2006), ApJ 638, 703
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mpl_cache_dir = Path(tempfile.gettempdir()) / "gravitational_lensing_matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from src.simulation.joint_simulator import JointSimulator, SLACSInformedPrior
from src.ml.pi_sbi import JointNPE


def generate_training_data(n_sims: int, grid_size: int, n_omega: int, seed: int):
    """Generate n_sims training triplets (theta, kappa_map, gw_spectrum)."""
    prior = SLACSInformedPrior(seed=seed)
    sim = JointSimulator(grid_size=grid_size, n_omega=n_omega, seed=seed + 1)

    print(f"Generating {n_sims} training simulations...")
    thetas = prior.sample(n_sims)
    kappa_maps, gw_spectra = sim.generate_batch(thetas)

    return thetas, kappa_maps, gw_spectra


def validate_on_slacs(model: JointNPE, grid_size: int, n_omega: int, device: str):
    """Validate posterior on real SLACS FITS data."""
    sim = JointSimulator(grid_size=grid_size, n_omega=n_omega)
    real_data = sim.get_real_validation_data()

    if not real_data:
        print("No real SLACS data found for validation.")
        return

    print(f"\nValidating on {len(real_data)} real SLACS systems:")
    results = []

    for entry in real_data:
        kmap = torch.FloatTensor(entry['kappa_map']).to(device)
        # Compute GW spectrum from published parameters
        theta_pub = entry['theta_published']
        gw = sim.simulate_gw_spectrum(theta_pub, add_noise=False)
        gw_t = torch.FloatTensor(gw).unsqueeze(0).to(device)

        mean, std = model.posterior_mean_std(kmap, gw_t, n_samples=500)
        theta_pub_a = entry['theta_published']

        print(f"  {entry['name']}:")
        print(f"    Published: log10(M)={theta_pub_a[0]:.2f}, log10(rs)={theta_pub_a[1]:.2f}, "
              f"z_l={theta_pub_a[2]:.3f}, z_s={theta_pub_a[3]:.3f}")
        print(f"    Posterior: log10(M)={mean[0]:.2f}±{std[0]:.2f}, "
              f"log10(rs)={mean[1]:.2f}±{std[1]:.2f}")

        results.append({
            'name': entry['name'],
            'posterior_mean': mean.tolist(),
            'posterior_std': std.tolist(),
            'theta_published': theta_pub_a.tolist(),
        })

    with open('results/pi_sbi_slacs_validation.json', 'w') as f:
        json.dump({'n_systems': len(results), 'results': results,
                   'data_source': 'real_hst_slacs_fits'}, f, indent=2)
    print("  SLACS validation saved to results/pi_sbi_slacs_validation.json")

    # Save posterior plot for first available system
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if results:
            r = results[0]
            means = np.array(r['posterior_mean'])
            stds = np.array(r['posterior_std'])
            pub = np.array(r['theta_published'])
            param_names = JointNPE.PARAM_NAMES

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(param_names))
            ax.bar(x - 0.2, means, 0.35, label='Posterior mean', color='steelblue', alpha=0.8)
            ax.errorbar(x - 0.2, means, yerr=stds, fmt='none', color='black', capsize=4)
            ax.bar(x + 0.2, pub, 0.35, label='Published', color='darkorange', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(param_names, rotation=20)
            ax.set_title(f"PI-SBI Posterior vs Published — {r['name']}")
            ax.legend()
            ax.grid(True, axis='y', alpha=0.4)
            plt.tight_layout()
            plt.savefig('results/pi_sbi_slacs_posteriors.png', dpi=150)
            plt.close()
            print("  Posterior plot saved to results/pi_sbi_slacs_posteriors.png")
    except ImportError:
        pass


def train(args):
    device = args.device
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    t_start = time.perf_counter()

    # Generate training data
    thetas, kappa_maps, gw_spectra = generate_training_data(
        args.n_sims, args.grid_size, args.n_omega, args.seed
    )

    # Compute normalization from training data
    theta_mean = thetas.mean(0)
    theta_std = thetas.std(0)

    # Create DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(thetas),
        torch.FloatTensor(kappa_maps),
        torch.FloatTensor(gw_spectra),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Build model
    model = JointNPE(
        grid_size=args.grid_size,
        n_omega=args.n_omega,
        physics_weight=args.physics_weight,
    ).to(device)
    model.flow.set_normalization(theta_mean, theta_std)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {'nll': [], 'l_poisson': [], 'loss': []}

    for epoch in range(args.epochs):
        model.train()
        epoch_nll = epoch_phys = epoch_total = 0.0
        n_batches = 0

        for theta_b, kmap_b, gw_b in loader:
            theta_b = theta_b.to(device)
            kmap_b = kmap_b.to(device)
            gw_b = gw_b.to(device)

            optimizer.zero_grad()
            loss, info = model.training_loss(theta_b, kmap_b, gw_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_nll += info['nll']
            epoch_phys += info['l_poisson']
            epoch_total += info['loss']
            n_batches += 1

        scheduler.step()

        avg_nll = epoch_nll / n_batches
        avg_phys = epoch_phys / n_batches
        avg_total = epoch_total / n_batches

        history['nll'].append(avg_nll)
        history['l_poisson'].append(avg_phys)
        history['loss'].append(avg_total)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch + 1:3d}/{args.epochs} | "
                  f"NLL={avg_nll:.3f} | Poisson={avg_phys:.4f} | Total={avg_total:.3f}")

    # Save model
    model.save("models/pi_sbi_joint.pt")
    print("Model saved to models/pi_sbi_joint.pt")

    # Save training curves plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history['nll'], label='NLL', color='blue')
        ax1.plot(history['loss'], label='Total', color='black')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('PI-SBI Training Loss')
        ax2.plot(history['l_poisson'], color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Poisson Residual')
        ax2.grid(True)
        ax2.set_title('Physics Constraint Loss (‖∇²ψ − 2κ‖²)')
        plt.tight_layout()
        plt.savefig('results/pi_sbi_training_curves.png', dpi=150)
        plt.close()
        print("Training curves saved to results/pi_sbi_training_curves.png")
    except ImportError:
        pass

    # Validate on real SLACS data
    validate_on_slacs(model, args.grid_size, args.n_omega, device)

    elapsed = time.perf_counter() - t_start

    # Save summary JSON
    summary = {
        'n_sims': args.n_sims,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'grid_size': args.grid_size,
        'n_omega': args.n_omega,
        'lr': args.lr,
        'physics_weight': args.physics_weight,
        'seed': args.seed,
        'device': args.device,
        'wall_time_s': float(elapsed),
        'final_nll': float(history['nll'][-1]),
        'final_l_poisson': float(history['l_poisson'][-1]),
        'final_loss': float(history['loss'][-1]),
        'theta_mean': theta_mean.tolist(),
        'theta_std': theta_std.tolist(),
        'evaluation_mode': 'slacs_calibrated_prior_synthetic_training',
        'real_data': 'slacs_9_hst_fits_validation',
        'param_names': JointNPE.PARAM_NAMES,
    }
    with open('results/pi_sbi_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Training summary saved to results/pi_sbi_training_summary.json")

    return model, history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PI-SBI model on multi-messenger lensing simulations."
    )
    parser.add_argument('--n-sims', type=int, default=10000,
                        help='Number of training simulations (default: 10000)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Mini-batch size (default: 128)')
    parser.add_argument('--grid-size', type=int, default=64,
                        help='Convergence map grid size (default: 64)')
    parser.add_argument('--n-omega', type=int, default=32,
                        help='GW frequency samples (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Adam learning rate (default: 3e-4)')
    parser.add_argument('--physics-weight', type=float, default=0.1,
                        help='Weight on Poisson physics loss (default: 0.1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device: cpu or cuda (default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


def main():
    args = parse_args()
    print("PI-SBI Training")
    print("=" * 60)
    print(f"  n_sims={args.n_sims}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}")
    print(f"  grid_size={args.grid_size}, n_omega={args.n_omega}")
    print(f"  lr={args.lr}, physics_weight={args.physics_weight}")
    print(f"  device={args.device}, seed={args.seed}")
    print("=" * 60)
    train(args)


if __name__ == '__main__':
    main()

