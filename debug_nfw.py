"""Quick debug script for NFW deflection"""
import torch
import numpy as np
import sys

sys.path.insert(0, r"d:\Coding projects\Collab\Gravitational lensing algorithm")

from src.ml.pinn import compute_nfw_deflection

# Test parameters
M_vir = torch.tensor([[10.0]])  # 10^13 M_sun
r_s = torch.tensor([[20.0]])  # 20 kpc
theta_x = torch.tensor([[1.0, 2.0, 3.0]])  # arcsec
theta_y = torch.tensor([[0.5, 1.0, 1.5]])  # arcsec

print("=" * 60)
print("NFW Deflection Debug Test")
print("=" * 60)
print(f"\nInput Parameters:")
print(f"  M_vir: {M_vir[0, 0]:.2f} × 10^12 M_sun = {M_vir[0, 0] * 1e12:.2e} M_sun")
print(f"  r_s: {r_s[0, 0]:.2f} kpc")
print(f"  theta_x: {theta_x[0].tolist()} arcsec")
print(f"  theta_y: {theta_y[0].tolist()} arcsec")

# Compute deflection
alpha_x, alpha_y = compute_nfw_deflection(M_vir, r_s, theta_x, theta_y)

print(f"\nOutput:")
print(f"  alpha_x: {alpha_x[0].tolist()} arcsec")
print(f"  alpha_y: {alpha_y[0].tolist()} arcsec")

# Compute magnitude
alpha_mag = torch.sqrt(alpha_x**2 + alpha_y**2)
print(f"  |alpha|: {alpha_mag[0].tolist()} arcsec")

# Check for zeros
if alpha_mag.max() < 1e-10:
    print("\n[ERROR] Deflection is zero or near-zero!")
    print("   This indicates a bug in the calculation.")
else:
    print(f"\n[SUCCESS] Non-zero deflection computed")
    print(f"   Max deflection: {alpha_mag.max().item():.6f} arcsec")

# Test mass scaling
print("\n" + "=" * 60)
print("Testing Mass Scaling (α ∝ M)")
print("=" * 60)

M1 = torch.tensor([[1.0]])  # 10^12 M_sun
M2 = torch.tensor([[2.0]])  # 2×10^12 M_sun
theta_test = torch.tensor([[5.0]])
theta_y_test = torch.tensor([[0.0]])

alpha_x1, _ = compute_nfw_deflection(M1, r_s, theta_test, theta_y_test)
alpha_x2, _ = compute_nfw_deflection(M2, r_s, theta_test, theta_y_test)

print(f"\nM1 = {M1[0, 0]:.1f} × 10^12: alpha = {alpha_x1[0, 0]:.6f} arcsec")
print(f"M2 = {M2[0, 0]:.1f} × 10^12: alpha = {alpha_x2[0, 0]:.6f} arcsec")

if alpha_x1[0, 0] > 1e-10:
    ratio = alpha_x2[0, 0] / alpha_x1[0, 0]
    print(f"\nRatio (alpha2/alpha1): {ratio:.4f}")
    if abs(ratio - 2.0) < 0.02:
        print("[PASS] Deflection scales linearly with mass")
    else:
        print(f"[FAIL] Expected ratio = 2.0, got {ratio:.4f}")
else:
    print("[FAIL] Alpha is zero, cannot compute ratio")
