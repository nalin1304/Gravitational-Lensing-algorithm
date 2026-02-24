"""
Simple Gravitational Lensing - synthetic convergence map generation.
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

DEFAULT_LENS_REDSHIFT = 0.5
DEFAULT_SOURCE_REDSHIFT = 2.0
DEFAULT_CONCENTRATION = 5.0
GRID_EXTENT_ARCSEC = 50.0

# Configure page first.
st.set_page_config(
    page_title="Simple Lensing - Gravitational Lensing Platform",
    page_icon="🎨",
    layout="wide",
)

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import UI utilities with robust fallback for different launch contexts.
try:
    from app.utils.helpers import estimate_computation_time, validate_positive_number
    from app.utils.ui import inject_custom_css, render_header, show_error, show_success
except ImportError:
    try:
        sys.path.append(str(PROJECT_ROOT / "app"))
        from utils.helpers import estimate_computation_time, validate_positive_number
        from utils.ui import inject_custom_css, render_header, show_error, show_success
    except ImportError:
        def render_header(title: str, subtitle: str, badge: str = "") -> None:
            st.title(f"🔭 {title}")
            if subtitle:
                st.markdown(f"*{subtitle}*")
            if badge:
                st.caption(badge)

        def inject_custom_css() -> None:
            return None

        def show_success(message: str) -> None:
            st.success(message)

        def show_error(message: str) -> None:
            st.error(message)

        def validate_positive_number(value: float, _: str = "Value") -> None:
            if value <= 0:
                raise ValueError("Value must be positive.")

        def estimate_computation_time(grid_size: int, __: int = 1000) -> str:
            return f"~{grid_size / 64:.1f} seconds"

inject_custom_css()

LENS_MODELS_AVAILABLE = False
IMPORT_ERROR: str | None = None

try:
    from src.lens_models.advanced_profiles import EllipticalNFWProfile
    from src.lens_models.lens_system import LensSystem
    from src.lens_models.mass_profiles import NFWProfile

    LENS_MODELS_AVAILABLE = True
except ImportError as exc:
    IMPORT_ERROR = str(exc)


@st.cache_data(show_spinner=False)
def generate_synthetic_convergence(
    profile_type: str,
    mass_msun: float,
    concentration: float,
    ellipticity: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic convergence map from NFW-family profiles."""
    lens_system = LensSystem(z_lens=DEFAULT_LENS_REDSHIFT, z_source=DEFAULT_SOURCE_REDSHIFT)

    if profile_type == "NFW":
        profile = NFWProfile(
            M_vir=mass_msun,
            concentration=concentration,
            lens_system=lens_system,
        )
    elif profile_type == "Elliptical NFW":
        profile = EllipticalNFWProfile(
            M_vir=mass_msun,
            c=concentration,
            lens_sys=lens_system,
            ellipticity=ellipticity,
        )
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    x_axis = np.linspace(-GRID_EXTENT_ARCSEC, GRID_EXTENT_ARCSEC, grid_size)
    y_axis = np.linspace(-GRID_EXTENT_ARCSEC, GRID_EXTENT_ARCSEC, grid_size)
    mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)

    convergence_map = profile.convergence(mesh_x.ravel(), mesh_y.ravel()).reshape(grid_size, grid_size)
    return convergence_map, mesh_x, mesh_y


def plot_convergence_map(
    convergence: np.ndarray,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    title: str = "Convergence Map",
    cmap: str = "viridis",
) -> plt.Figure:
    """Render convergence map figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(mesh_x, mesh_y, convergence, levels=20, cmap=cmap)
    ax.set_xlabel("x (arcsec)", fontsize=12)
    ax.set_ylabel("y (arcsec)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    colorbar = plt.colorbar(contour, ax=ax, label="Convergence κ")
    colorbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def _format_time_estimate(grid_size: int) -> str:
    """Convert helper estimate into a display-safe label."""
    estimate = estimate_computation_time(grid_size)
    if isinstance(estimate, (int, float)):
        return f"~{estimate:.1f} seconds"
    return str(estimate)


def main() -> None:
    """Main Streamlit entry point for simple lensing page."""
    render_header(
        "Simple Gravitational Lensing",
        "Generate synthetic convergence maps with physically motivated halo profiles",
    )

    if not LENS_MODELS_AVAILABLE:
        if IMPORT_ERROR:
            st.warning(f"⚠️ Import error detected: {IMPORT_ERROR}")
        st.error("❌ Required lens-model modules are unavailable.")
        with st.expander("📦 Setup Instructions", expanded=True):
            st.markdown(
                """
Install dependencies and verify imports:

```bash
pip install numpy scipy matplotlib astropy h5py torch scikit-learn
python -c "from src.lens_models.lens_system import LensSystem; print('Lens models available')"
```
"""
            )
        return

    st.markdown(
        """
Generate synthetic gravitational lensing convergence maps using NFW-based profiles.
Tune halo mass and concentration, then inspect map statistics and downloads.
"""
    )

    controls_col, viz_col = st.columns([1, 2])

    with controls_col:
        st.subheader("⚙️ Configuration")

        profile_type = st.selectbox(
            "Lens Profile",
            ["NFW", "Elliptical NFW"],
            help="Dark-matter profile used to generate κ(x, y).",
        )

        virial_mass_1e12 = st.slider(
            "Virial Mass (×10¹² M☉)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Virial mass M_vir in units of 10^12 solar masses.",
        )

        concentration = st.slider(
            "Concentration (c)",
            min_value=2.0,
            max_value=20.0,
            value=float(DEFAULT_CONCENTRATION),
            step=0.5,
            help="NFW concentration parameter c = r_vir / r_s.",
        )

        ellipticity = 0.0
        if profile_type == "Elliptical NFW":
            ellipticity = st.slider(
                "Ellipticity",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Projected ellipticity (0 is circular).",
            )

        grid_size = st.select_slider(
            "Grid Size",
            options=[32, 64, 128, 256],
            value=64,
            help="Resolution of the generated convergence map.",
        )

        cmap = st.selectbox(
            "Colormap",
            ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"],
            index=0,
        )

        st.markdown("---")
        st.info(f"⏱️ Estimated computation time: {_format_time_estimate(grid_size)}")
        generate_clicked = st.button("🚀 Generate Map", type="primary", width="stretch")

    with viz_col:
        st.subheader("📊 Visualization")

        if generate_clicked:
            with st.spinner(f"Generating {grid_size}×{grid_size} convergence map..."):
                try:
                    validate_positive_number(virial_mass_1e12, "Virial Mass")
                    validate_positive_number(concentration, "Concentration")
                    mass_msun = virial_mass_1e12 * 1e12

                    convergence_map, mesh_x, mesh_y = generate_synthetic_convergence(
                        profile_type=profile_type,
                        mass_msun=mass_msun,
                        concentration=concentration,
                        ellipticity=ellipticity,
                        grid_size=grid_size,
                    )

                    st.session_state["convergence_map"] = convergence_map
                    st.session_state["mesh_x"] = mesh_x
                    st.session_state["mesh_y"] = mesh_y
                    st.session_state["profile_type"] = profile_type
                    st.session_state["grid_size"] = grid_size
                    st.session_state["colormap"] = cmap
                    show_success("Map generated successfully.")
                except Exception as exc:
                    show_error(f"Error generating map: {exc}")
                    st.code(traceback.format_exc())
                    return

        if {"convergence_map", "mesh_x", "mesh_y"}.issubset(st.session_state):
            convergence_map = st.session_state["convergence_map"]
            mesh_x = st.session_state["mesh_x"]
            mesh_y = st.session_state["mesh_y"]
            selected_colormap = st.session_state.get("colormap", cmap)

            fig = plot_convergence_map(
                convergence=convergence_map,
                mesh_x=mesh_x,
                mesh_y=mesh_y,
                title=f"{st.session_state.get('profile_type', 'NFW')} Convergence Map",
                cmap=selected_colormap,
            )
            st.pyplot(fig)

            st.markdown("### 📈 Statistics")
            stat_col_1, stat_col_2, stat_col_3, stat_col_4 = st.columns(4)
            stat_col_1.metric("Max κ", f"{float(np.max(convergence_map)):.4f}")
            stat_col_2.metric("Mean κ", f"{float(np.mean(convergence_map)):.4f}")
            stat_col_3.metric("Min κ", f"{float(np.min(convergence_map)):.4f}")
            stat_col_4.metric("Std κ", f"{float(np.std(convergence_map)):.4f}")

            st.markdown("### 💾 Download")
            map_buffer = io.BytesIO()
            np.save(map_buffer, convergence_map)
            map_buffer.seek(0)

            figure_buffer = io.BytesIO()
            fig.savefig(figure_buffer, format="png", dpi=300, bbox_inches="tight")
            figure_buffer.seek(0)
            plt.close(fig)

            download_col_1, download_col_2 = st.columns(2)
            with download_col_1:
                st.download_button(
                    label="Download Map (.npy)",
                    data=map_buffer,
                    file_name=f"convergence_map_{st.session_state.get('grid_size', grid_size)}.npy",
                    mime="application/octet-stream",
                    width="stretch",
                )
            with download_col_2:
                st.download_button(
                    label="Download Figure (.png)",
                    data=figure_buffer,
                    file_name=f"convergence_map_{st.session_state.get('grid_size', grid_size)}.png",
                    mime="image/png",
                    width="stretch",
                )

    with st.expander("ℹ️ About Convergence Maps"):
        st.markdown(
            """
**Convergence (κ)** is the projected surface density normalized by the critical
surface density:

- **κ < 1**: weak lensing regime
- **κ = 1**: critical curve threshold
- **κ > 1**: strong lensing regime where multiple images can form

The NFW family captures dark-matter halo structure with tunable concentration and
ellipticity for more realistic projected mass distributions.
"""
        )


if __name__ == "__main__":
    main()
