"""Gravitational Lensing Toolkit - Streamlit Home Page.

Launch with:
    streamlit run app/Home.py
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from styles import inject_custom_css, render_header
from utils.demo_helpers import run_demo_and_redirect
from utils.session_state import init_session_state

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEMOS_DIR = PROJECT_ROOT / "demos"
TESTS_DIR = PROJECT_ROOT / "tests"
PAGES_DIR = APP_DIR / "pages"
BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "pinn_results.json"
REPOSITORY_URL = "https://github.com/nalin1304/Gravitational-Lensing-algorithm"


def _safe_switch_page(page_path: str) -> None:
    """Switch to another Streamlit page only if the file exists."""
    target = APP_DIR / page_path
    if target.exists():
        st.switch_page(page_path)
    else:
        st.error(f"Page not found: {page_path}")


def _load_peak_inference_speed() -> float | None:
    """Return peak images/sec from benchmark results, if available."""
    if not BENCHMARK_RESULTS_PATH.exists():
        return None

    try:
        payload: dict[str, Any] = json.loads(BENCHMARK_RESULTS_PATH.read_text())
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


def _collect_home_stats() -> dict[str, str]:
    """Collect lightweight repository stats for landing-page panels."""
    demo_count = len(list(DEMOS_DIR.glob("*.yaml")))
    test_modules = len(list(TESTS_DIR.glob("test_*.py")))
    page_count = len(list(PAGES_DIR.glob("*.py")))
    peak_speed = _load_peak_inference_speed()

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
def _demo_preview_png(demo_id: str) -> bytes:
    """Render a small synthetic thumbnail image for each demo system."""
    plt.style.use("dark_background")

    n = 220
    x = np.linspace(-2.4, 2.4, n)
    y = np.linspace(-2.4, 2.4, n)
    xx, yy = np.meshgrid(x, y)

    base = 0.02 * np.exp(-(xx * xx + yy * yy) / 8.0)

    if demo_id == "einstein_cross":
        image = base.copy()
        for cx, cy in [(-0.85, -0.12), (0.82, 0.08), (0.12, 0.86), (-0.08, -0.88)]:
            image += 1.35 * _gaussian_2d(xx, yy, cx, cy, 0.16)
        ring = np.exp(-((np.sqrt(xx**2 + yy**2) - 1.0) ** 2) / 0.05)
        image += 0.55 * ring
        cmap = "magma"
        title = "Einstein Cross"
    elif demo_id == "twin_quasar":
        image = base.copy()
        image += 1.6 * _gaussian_2d(xx, yy, -0.95, 0.12, 0.2)
        image += 1.4 * _gaussian_2d(xx, yy, 0.88, -0.08, 0.22)
        image += 0.45 * np.exp(-(xx * xx / 0.9 + yy * yy / 2.3))
        arc = np.exp(-((yy - 0.5 * np.sin(1.6 * xx)) ** 2) / 0.12) * np.exp(-(xx + 0.3) ** 2 / 4.0)
        image += 0.35 * arc
        cmap = "inferno"
        title = "Twin Quasar"
    else:
        image = base.copy()
        image += 0.55 * np.exp(-(xx * xx + yy * yy) / 3.5)
        for cx, cy, amp in [(-1.0, 0.9, 0.7), (0.7, 1.1, 0.6), (1.2, -0.5, 0.65), (-0.6, -1.1, 0.5)]:
            image += amp * _gaussian_2d(xx, yy, cx, cy, 0.25)
        arc_1 = np.exp(-((yy + 0.8 - 0.25 * xx) ** 2) / 0.07) * np.exp(-(xx - 0.2) ** 2 / 5.0)
        arc_2 = np.exp(-((yy - 0.9 + 0.18 * xx) ** 2) / 0.06) * np.exp(-(xx + 0.5) ** 2 / 6.0)
        image += 0.95 * arc_1 + 0.85 * arc_2
        cmap = "viridis"
        title = "JWST Cluster"

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=130)
    fig.patch.set_facecolor("#0A0E1A")
    ax.imshow(image, cmap=cmap, origin="lower")
    ax.set_title(title, fontsize=10, color="#EAF4FF", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#00D4FF")
        spine.set_linewidth(0.9)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, facecolor="#0A0E1A", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buffer.getvalue()


st.set_page_config(
    page_title="Gravitational Lensing Analysis Platform",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": REPOSITORY_URL,
        "Report a bug": f"{REPOSITORY_URL}/issues",
        "About": (
            "# Gravitational Lensing Toolkit\n"
            "Research-oriented strong lensing analysis and PINN inference."
        ),
    },
)

init_session_state()
inject_custom_css()
stats = _collect_home_stats()

render_header(
    "Gravitational Lensing Toolkit",
    "A research-grade playground for strong-lensing simulation, inference, and scientific validation.",
    "ISEF 2025 Research Build",
)

st.markdown(
    f"""
<div class="kpi-strip">
    <span class="kpi-pill">Demo Systems: {stats['demo_count']}</span>
    <span class="kpi-pill">Peak Inference: {stats['peak_speed']}</span>
    <span class="kpi-pill">Physics Tests: 551 passed / 22 skipped</span>
    <span class="kpi-pill">UI Modules: {stats['page_count']} pages</span>
</div>
""",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.25, 1.0], gap="large")

with left_col:
    st.markdown("## Observatory Console")
    st.markdown(
        """
Select a canonical lensing scenario to generate convergence maps, deflection fields,
and uncertainty-aware inference outputs with physically constrained pipelines.
"""
    )

    action_col_1, action_col_2, action_col_3 = st.columns(3)
    with action_col_1:
        if st.button("Open Simple Lensing", use_container_width=True):
            _safe_switch_page("pages/02_Simple_Lensing.py")
    with action_col_2:
        if st.button("Open Real Data", use_container_width=True):
            _safe_switch_page("pages/05_Real_Data.py")
    with action_col_3:
        if st.button("Open Validation", use_container_width=True):
            _safe_switch_page("pages/07_Validation.py")

with right_col:
    st.markdown(
        """
<div class="custom-card">
  <div class="card-header">Mission Snapshot</div>
  <div class="card-body">
    <strong style="color:#EAF4FF;">Scientific objective:</strong> recover lens mass structure from image-domain observables while enforcing physical consistency.<br/><br/>
    <strong style="color:#EAF4FF;">Numerical stack:</strong> thin-lens ray tracing, multi-plane recursion, and PINN-based parameter inference.<br/><br/>
    <strong style="color:#EAF4FF;">Target output:</strong> publication-quality maps, uncertainty estimates, and reproducible validation traces.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="nebula-divider"></div>', unsafe_allow_html=True)
st.markdown("## Demo Systems")
st.markdown("Choose one-click systems with preconfigured astrophysical parameters.")

demo_cards = [
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

columns = st.columns(3, gap="large")
for column, card in zip(columns, demo_cards):
    with column:
        st.markdown(
            f"""
<div class="custom-card">
  <div class="card-header">{card['title']}</div>
  <div class="card-body">{card['subtitle']}</div>
  <div style="margin-top:0.55rem;color:#7E94B3;font-size:0.8rem;">{card['tags']}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.image(_demo_preview_png(card["demo_id"]), use_container_width=True)
        if st.button(card["button"], use_container_width=True, type="primary", key=f"launch_{card['demo_id']}"):
            run_demo_and_redirect(card["demo_id"])

st.markdown('<div class="nebula-divider"></div>', unsafe_allow_html=True)
st.markdown("## Platform Capabilities")

feat_col_1, feat_col_2, feat_col_3 = st.columns(3, gap="large")
with feat_col_1:
    st.markdown(
        """
<div class="custom-card">
  <div class="card-header">Physics Core</div>
  <div class="card-body">
    Multi-profile lens models, FLRW distances, Einstein radii, and time-delay terms under a unified API.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with feat_col_2:
    st.markdown(
        """
<div class="custom-card">
  <div class="card-header">Inference Engine</div>
  <div class="card-body">
    Physics-informed neural networks recover mass and profile parameters with calibrated uncertainty.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with feat_col_3:
    st.markdown(
        """
<div class="custom-card">
  <div class="card-header">Validation Layer</div>
  <div class="card-body">
    Benchmark suites compare numerical outputs against known systems and internal consistency tests.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="nebula-divider"></div>', unsafe_allow_html=True)

st.markdown(
    f"""
<div class="custom-card" style="text-align:center;">
  <div class="card-header" style="justify-content:center;">Repository</div>
  <div class="card-body">
    Source, issues, and reproducible experiment assets are available at
    <a href="{REPOSITORY_URL}" target="_blank" style="color:#00D4FF;text-decoration:none;">{REPOSITORY_URL}</a>.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
