"""Core utility functions for Streamlit scientific workflows.

This module hosts testable computation and plotting helpers used by the app.
It is intentionally Streamlit-independent so it can be validated in unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.lens_models import EllipticalNFWProfile, LensSystem, NFWProfile
from src.ml.generate_dataset import generate_convergence_map_vectorized
from src.ml.pinn import PhysicsInformedNN

ARCSEC_EXTENT = 2.0
DEFAULT_LENS_REDSHIFT = 0.5
DEFAULT_SOURCE_REDSHIFT = 1.5
DEFAULT_CONCENTRATION = 10.0
DEFAULT_POSITION_ANGLE_DEG = 45.0


def generate_synthetic_convergence(
    profile_type: str,
    mass: float,
    scale_radius: float,
    ellipticity: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic convergence map with coordinate grids.

    Parameters
    ----------
    profile_type
        Mass profile name (`NFW` or `Elliptical NFW`).
    mass
        Virial mass in solar masses.
    scale_radius
        Scale radius in kpc (kept for API compatibility).
    ellipticity
        Ellipticity for elliptical profile variants.
    grid_size
        Output map side length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Convergence map and corresponding X/Y coordinate grids.
    """
    _ = scale_radius  # retained in public signature for existing callers

    lens_system = LensSystem(z_lens=DEFAULT_LENS_REDSHIFT, z_source=DEFAULT_SOURCE_REDSHIFT)

    if profile_type == "NFW":
        lens_profile = NFWProfile(
            M_vir=mass,
            concentration=DEFAULT_CONCENTRATION,
            lens_system=lens_system,
        )
    elif profile_type == "Elliptical NFW":
        lens_profile = EllipticalNFWProfile(
            M_vir=mass,
            c=DEFAULT_CONCENTRATION,
            lens_sys=lens_system,
            ellipticity=ellipticity,
            position_angle=DEFAULT_POSITION_ANGLE_DEG,
        )
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    convergence_map = generate_convergence_map_vectorized(
        lens_profile,
        grid_size=grid_size,
        extent=ARCSEC_EXTENT,
    )

    x_axis = np.linspace(-ARCSEC_EXTENT, ARCSEC_EXTENT, grid_size)
    y_axis = np.linspace(-ARCSEC_EXTENT, ARCSEC_EXTENT, grid_size)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    return convergence_map, x_grid, y_grid


def plot_convergence_map(
    convergence_map: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    title: str = "Convergence Map",
    cmap: str = "viridis",
) -> plt.Figure:
    """Create a contour-based convergence map visualization."""
    figure, axis = plt.subplots(figsize=(8, 7))
    filled = axis.contourf(x_grid, y_grid, convergence_map, levels=20, cmap=cmap)
    axis.contour(x_grid, y_grid, convergence_map, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    axis.set_xlabel("x (arcsec)", fontsize=12)
    axis.set_ylabel("y (arcsec)", fontsize=12)
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_aspect("equal")

    colorbar = figure.colorbar(filled, ax=axis)
    colorbar.set_label("κ (convergence)", fontsize=12)
    plt.tight_layout()
    return figure


def plot_uncertainty_bars(
    parameter_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> plt.Figure:
    """Create an uncertainty bar chart for inferred parameters."""
    figure, axis = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(parameter_names))

    normalized_means = means / (np.abs(means) + 1e-10)
    normalized_stds = stds / (np.abs(means) + 1e-10)

    axis.bar(
        x_positions,
        normalized_means,
        yerr=normalized_stds,
        capsize=5,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(parameter_names, fontsize=11)
    axis.set_ylabel("Normalized Value ± Uncertainty", fontsize=12)
    axis.set_title("Parameter Estimates with Uncertainty", fontsize=14, fontweight="bold")
    axis.grid(axis="y", alpha=0.3)
    axis.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return figure


def plot_classification_probs(
    class_names: list[str],
    probabilities: np.ndarray,
    entropy: float,
) -> plt.Figure:
    """Visualize classification probabilities with bar and pie panels."""
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = axis_left.bar(class_names, probabilities, color=colors, alpha=0.7, edgecolor="black")
    axis_left.set_ylabel("Probability", fontsize=12)
    axis_left.set_title("Dark Matter Classification", fontsize=13, fontweight="bold")
    axis_left.set_ylim([0, 1])
    axis_left.grid(axis="y", alpha=0.3)

    for bar, probability in zip(bars, probabilities):
        bar_height = bar.get_height()
        axis_left.text(bar.get_x() + bar.get_width() / 2.0, bar_height, f"{probability:.1%}", ha="center", va="bottom", fontsize=10)

    axis_right.pie(
        probabilities,
        labels=class_names,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    axis_right.set_title(f"Confidence (Entropy: {entropy:.3f})", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return figure


def plot_comparison(
    original_map: np.ndarray,
    processed_map: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> plt.Figure:
    """Create a side-by-side comparison figure for map transformations."""
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(14, 6))

    original_panel = axis_left.contourf(x_grid, y_grid, original_map, levels=20, cmap="viridis")
    axis_left.set_xlabel("x (arcsec)", fontsize=11)
    axis_left.set_ylabel("y (arcsec)", fontsize=11)
    axis_left.set_title("Original Image", fontsize=12, fontweight="bold")
    axis_left.set_aspect("equal")
    plt.colorbar(original_panel, ax=axis_left, fraction=0.046)

    processed_panel = axis_right.contourf(x_grid, y_grid, processed_map, levels=20, cmap="viridis")
    axis_right.set_xlabel("x (arcsec)", fontsize=11)
    axis_right.set_ylabel("y (arcsec)", fontsize=11)
    axis_right.set_title("Processed (64×64, Normalized)", fontsize=12, fontweight="bold")
    axis_right.set_aspect("equal")
    plt.colorbar(processed_panel, ax=axis_right, fraction=0.046)

    plt.tight_layout()
    return figure


def load_pretrained_model(model_path: Optional[str] = None) -> PhysicsInformedNN:
    """Load a pre-trained PINN model if a valid checkpoint path is provided."""
    model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
    if model_path and Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            model.eval()
        except Exception:
            # Keep a usable model object even when checkpoint loading fails.
            pass
    return model


def prepare_model_input(convergence_map: np.ndarray, target_size: int = 64) -> torch.Tensor:
    """Resize, normalize, and package a map into a model-ready tensor."""
    if convergence_map.shape[0] != target_size:
        from scipy.ndimage import zoom

        resize_scale = target_size / convergence_map.shape[0]
        convergence_map = zoom(convergence_map, resize_scale, order=1)

    normalized_map = (convergence_map - convergence_map.min()) / (convergence_map.max() - convergence_map.min() + 1e-10)
    input_tensor = torch.from_numpy(normalized_map).float()
    return input_tensor.unsqueeze(0).unsqueeze(0)


def compute_classification_entropy(probabilities: np.ndarray) -> float:
    """Compute predictive entropy for classification outputs."""
    return float(-np.sum(probabilities * np.log(probabilities + 1e-10)))


def format_parameter_value(parameter_name: str, value: float) -> str:
    """Format parameter values for compact dashboard presentation."""
    if "M_vir" in parameter_name:
        return f"{value:.2e}"
    if "H" in parameter_name:
        return f"{value:.2f}"
    return f"{value:.4f}"
