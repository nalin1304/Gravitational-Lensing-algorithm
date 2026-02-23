"""
Gravitational Lensing Toolkit - Streamlit Home Page.

Launch with:
    streamlit run app/Home.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    """Collect light-weight repository stats for the hero section."""
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
    "Gravitational Lensing Analysis Platform",
    "Physics-informed machine learning for strong gravitational lensing",
    "Research Demo Suite • Streamlit",
)

st.markdown(
    f"""
<div style="text-align: center; padding: 2rem 0; animation: fadeInScale 0.8s ease-out;">
    <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">🌌 Gravitational Lensing Toolkit</h2>
    <p style="font-size: 1.15rem; color: var(--text-secondary); max-width: 900px; margin: 0 auto; line-height: 1.6;">
        Run scientifically grounded demo systems and inspect convergence, deflection, and inference outputs in one place.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Demo Systems", stats["demo_count"])
with metric_col_2:
    st.metric("Peak Inference", stats["peak_speed"])
with metric_col_3:
    st.metric("Test Modules", stats["test_modules"])
with metric_col_4:
    st.metric("UI Pages", stats["page_count"])

st.markdown("---")
st.markdown(
    """
<div style="margin: 2rem 0;">
    <h2 style="text-align: center; margin-bottom: 1.5rem; font-size: 2rem;">🚀 Launch a Demo</h2>
    <p style="text-align: center; font-size: 1.1rem; color: var(--text-secondary); max-width: 700px; margin: 0 auto 2rem;">
        Start from curated lens systems with ready-to-run parameters.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

demo_cards = [
    {
        "demo_id": "einstein_cross",
        "title": "🌟 Einstein Cross",
        "subtitle": "Quadruple-image quasar • z=0.04 lens",
        "button": "Launch Einstein Cross",
    },
    {
        "demo_id": "twin_quasar",
        "title": "🔭 Twin Quasar",
        "subtitle": "Historic 1979 system • time-delay workflow",
        "button": "Launch Twin Quasar",
    },
    {
        "demo_id": "jwst_cluster_demo",
        "title": "🪐 JWST Cluster",
        "subtitle": "Cluster-scale lensing • substructure sensitivity",
        "button": "Launch JWST Cluster",
    },
]

demo_columns = st.columns(3)
for column, card in zip(demo_columns, demo_cards):
    with column:
        st.markdown(
            f"""
<div style="text-align: center; margin-bottom: 1rem;">
    <h3 style="font-size: 1.45rem; margin-bottom: 0.5rem;">{card["title"]}</h3>
    <p style="color: var(--text-muted); font-size: 0.9rem;">{card["subtitle"]}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button(
            card["button"],
            use_container_width=True,
            type="primary",
            key=f"launch_{card['demo_id']}",
        ):
            run_demo_and_redirect(card["demo_id"])

st.markdown("---")
with st.expander("🔬 Advanced Workflows", expanded=False):
    st.markdown(
        """
Use the playgrounds below for custom parameter sweeps and FITS ingestion.
""",
        unsafe_allow_html=True,
    )
    advanced_col_1, advanced_col_2 = st.columns(2)
    with advanced_col_1:
        if st.button("📊 Simple Lensing Playground", use_container_width=True):
            _safe_switch_page("pages/02_Simple_Lensing.py")
    with advanced_col_2:
        if st.button("📂 Real Data Analysis", use_container_width=True):
            _safe_switch_page("pages/05_Real_Data.py")

st.markdown("---")
st.markdown(
    """
<h2 style="text-align: center; margin-bottom: 1.5rem;">🧭 Explore Features</h2>
""",
    unsafe_allow_html=True,
)

feature_col_1, feature_col_2, feature_col_3 = st.columns(3)
with feature_col_1:
    st.markdown(
        """
<div class="custom-card">
    <div class="card-header">
        <span style="font-size: 2rem;">📸</span>
        <span>Observation View</span>
    </div>
    <div class="card-body">
        HST/JWST-style image inspection with reproducible parameter presets.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
with feature_col_2:
    st.markdown(
        """
<div class="custom-card">
    <div class="card-header">
        <span style="font-size: 2rem;">🗺️</span>
        <span>Mass Mapping</span>
    </div>
    <div class="card-body">
        Convergence and deflection fields generated from physically motivated lens models.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
with feature_col_3:
    st.markdown(
        """
<div class="custom-card">
    <div class="card-header">
        <span style="font-size: 2rem;">📊</span>
        <span>Uncertainty Outputs</span>
    </div>
    <div class="card-body">
        Bayesian-style uncertainty views for predicted parameters and class probabilities.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; padding: 2.5rem 2rem; margin-top: 2rem; background: var(--bg-glass); backdrop-filter: blur(18px); border-radius: 16px; border: 1px solid var(--border-color);">
    <h3 style="margin-bottom: 0.5rem;">Project Links</h3>
    <p style="color: var(--text-secondary); margin-bottom: 0.75rem;">
        Source code and issue tracking are maintained in the public repository.
    </p>
    <p style="font-size: 0.95rem; margin: 0;">
        <a href="https://github.com/nalin1304/Gravitational-Lensing-algorithm" target="_blank" style="color: var(--accent-cyan); text-decoration: none;">
            github.com/nalin1304/Gravitational-Lensing-algorithm
        </a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
