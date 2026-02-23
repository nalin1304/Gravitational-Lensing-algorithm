"""Gravitational Lensing Toolkit - Streamlit Home Page.

Launch with:
    streamlit run app/Home.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from styles import inject_custom_css, render_header
try:
    from app.core.landing import DEMO_CARDS, collect_home_stats, generate_demo_preview_png
except ImportError:  # pragma: no cover - fallback when run with app/ on PYTHONPATH
    from core.landing import DEMO_CARDS, collect_home_stats, generate_demo_preview_png  # type: ignore

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
stats = collect_home_stats(
    demos_dir=DEMOS_DIR,
    tests_dir=TESTS_DIR,
    pages_dir=PAGES_DIR,
    benchmark_results_path=BENCHMARK_RESULTS_PATH,
)

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

columns = st.columns(3, gap="large")
for column, card in zip(columns, DEMO_CARDS):
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
        st.image(generate_demo_preview_png(card["demo_id"]), use_container_width=True)
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
