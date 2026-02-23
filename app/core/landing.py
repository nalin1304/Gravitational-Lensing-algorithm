"""Landing-page core helpers and reusable data for the Streamlit home page."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


DEMO_CARDS: list[dict[str, str]] = [
    {
        "demo_id": "einstein_cross",
        "title": "Einstein Cross Q2237+030",
        "subtitle": "Quad image morphology, compact source, low-z lens.",
        "tags": "SIS-like morphology | z_l=0.04 | z_s=1.695",
        "button": "Run Einstein Cross",
    },
    {
        "demo_id": "twin_quasar",
        "title": "Twin Quasar Q0957+561",
        "subtitle": "Classic time-delay lens for cosmography workflows.",
        "tags": "NFW halo | time-delay use-case | historical benchmark",
        "button": "Run Twin Quasar",
    },
    {
        "demo_id": "jwst_cluster_demo",
        "title": "JWST Cluster Arc Field",
        "subtitle": "Cluster-scale lensing with substructure sensitivity.",
        "tags": "High-mass lens | arc morphology | subhalo analysis",
        "button": "Run JWST Cluster",
    },
]


def load_peak_inference_speed(benchmark_results_path: Path) -> float | None:
    """Return peak images/sec from benchmark results, if available."""
    if not benchmark_results_path.exists():
        return None

    try:
        payload: dict[str, Any] = json.loads(benchmark_results_path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    runs = payload.get("results", [])
    if not isinstance(runs, list):
        return None

    speeds: list[float] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        value = run.get("images_per_sec")
        if isinstance(value, (int, float)):
            speeds.append(float(value))

    return max(speeds) if speeds else None


def collect_home_stats(
    demos_dir: Path,
    tests_dir: Path,
    pages_dir: Path,
    benchmark_results_path: Path,
) -> dict[str, str]:
    """Collect lightweight repository stats for landing-page panels."""
    demo_count = len(list(demos_dir.glob("*.yaml")))
    test_modules = len(list(tests_dir.glob("test_*.py")))
    page_count = len(list(pages_dir.glob("*.py")))
    peak_speed = load_peak_inference_speed(benchmark_results_path)

    return {
        "demo_count": str(demo_count),
        "test_modules": str(test_modules),
        "page_count": str(page_count),
        "peak_speed": f"{peak_speed:.1f} img/s" if peak_speed is not None else "N/A",
    }


def _gaussian_2d(x_grid: np.ndarray, y_grid: np.ndarray, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Return an isotropic 2D Gaussian on a grid."""
    return np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma * sigma))


@st.cache_data(show_spinner=False)
def generate_demo_preview_png(demo_id: str) -> bytes:
    """Render a small synthetic thumbnail image for each demo system."""
    plt.style.use("dark_background")

    num_pixels = 220
    x_axis = np.linspace(-2.4, 2.4, num_pixels)
    y_axis = np.linspace(-2.4, 2.4, num_pixels)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    base_map = 0.02 * np.exp(-(x_grid * x_grid + y_grid * y_grid) / 8.0)

    if demo_id == "einstein_cross":
        image_map = base_map.copy()
        for center_x, center_y in [(-0.85, -0.12), (0.82, 0.08), (0.12, 0.86), (-0.08, -0.88)]:
            image_map += 1.35 * _gaussian_2d(x_grid, y_grid, center_x, center_y, 0.16)
        ring_map = np.exp(-((np.sqrt(x_grid**2 + y_grid**2) - 1.0) ** 2) / 0.05)
        image_map += 0.55 * ring_map
        color_map = "magma"
        panel_title = "Einstein Cross"
    elif demo_id == "twin_quasar":
        image_map = base_map.copy()
        image_map += 1.6 * _gaussian_2d(x_grid, y_grid, -0.95, 0.12, 0.2)
        image_map += 1.4 * _gaussian_2d(x_grid, y_grid, 0.88, -0.08, 0.22)
        image_map += 0.45 * np.exp(-(x_grid * x_grid / 0.9 + y_grid * y_grid / 2.3))
        arc_map = np.exp(-((y_grid - 0.5 * np.sin(1.6 * x_grid)) ** 2) / 0.12) * np.exp(-(x_grid + 0.3) ** 2 / 4.0)
        image_map += 0.35 * arc_map
        color_map = "inferno"
        panel_title = "Twin Quasar"
    else:
        image_map = base_map.copy()
        image_map += 0.55 * np.exp(-(x_grid * x_grid + y_grid * y_grid) / 3.5)
        for center_x, center_y, amplitude in [
            (-1.0, 0.9, 0.7),
            (0.7, 1.1, 0.6),
            (1.2, -0.5, 0.65),
            (-0.6, -1.1, 0.5),
        ]:
            image_map += amplitude * _gaussian_2d(x_grid, y_grid, center_x, center_y, 0.25)
        arc_1 = np.exp(-((y_grid + 0.8 - 0.25 * x_grid) ** 2) / 0.07) * np.exp(-(x_grid - 0.2) ** 2 / 5.0)
        arc_2 = np.exp(-((y_grid - 0.9 + 0.18 * x_grid) ** 2) / 0.06) * np.exp(-(x_grid + 0.5) ** 2 / 6.0)
        image_map += 0.95 * arc_1 + 0.85 * arc_2
        color_map = "viridis"
        panel_title = "JWST Cluster"

    figure, axis = plt.subplots(figsize=(3.2, 3.2), dpi=130)
    figure.patch.set_facecolor("#0A0E1A")
    axis.imshow(image_map, cmap=color_map, origin="lower")
    axis.set_title(panel_title, fontsize=10, color="#EAF4FF", pad=6)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#00D4FF")
        spine.set_linewidth(0.9)

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, facecolor="#0A0E1A", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    return buffer.getvalue()
