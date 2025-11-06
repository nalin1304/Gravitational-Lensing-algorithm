"""
Simple Gravitational Lensing - Synthetic Convergence Map Generation

Generate synthetic gravitational lensing convergence maps using different mass profiles.
Adjust parameters in real-time and visualize results instantly.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import io
import sys
import traceback

# Configure page FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="Simple Lensing - Gravitational Lensing Platform",
    page_icon="🎨",
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error
    from app.utils.helpers import validate_positive_number, estimate_computation_time
except ImportError as e:
    try:
        sys.path.append(str(project_root / 'app'))
        from utils.ui import render_header, inject_custom_css, show_success, show_error
        from utils.helpers import validate_positive_number, estimate_computation_time
    except ImportError as e:
        # Create minimal versions of required functions if imports fail
        def render_header(title, subtitle, category=None):
            st.title(f"🔭 {title}")
            if subtitle:
                st.markdown(f"*{subtitle}*")
        
        def inject_custom_css():
            pass
        
        def show_success(msg):
            st.success(msg)
        
        def show_error(msg):
            st.error(msg)
        
        def validate_positive_number(n):
            return n > 0
        
        def estimate_computation_time(grid_size):
            return grid_size * 0.01

# Apply custom CSS
inject_custom_css()

# Import required lens-model modules
LENS_MODELS_AVAILABLE = False
DATASET_AVAILABLE = False
import_error = None
dataset_warning = None

try:
    from src.lens_models.lens_system import LensSystem  # type: ignore
    from src.lens_models.mass_profiles import NFWProfile  # type: ignore
    from src.lens_models.advanced_profiles import EllipticalNFWProfile  # type: ignore
    LENS_MODELS_AVAILABLE = True
except ImportError as e:
    LENS_MODELS_AVAILABLE = False
    import_error = str(e)

# Optional dataset/ML helper
try:
    from src.ml.generate_dataset import generate_convergence_map_vectorized  # type: ignore
    DATASET_AVAILABLE = True
except Exception as e:
    DATASET_AVAILABLE = False
    dataset_warning = str(e)

MODULES_AVAILABLE = LENS_MODELS_AVAILABLE


def generate_synthetic_convergence(profile_type, mass, scale_radius, ellipticity, grid_size):
    """Generate synthetic convergence map."""
    # Create lens system first
    lens = LensSystem(z_lens=0.5, z_source=2.0)  # type: ignore
    
    if profile_type == "NFW":
        profile = NFWProfile(  # type: ignore
            M_vir=mass,  # Virial mass
            concentration=5.0,  # Concentration parameter
            lens_system=lens
        )
    else:  # Elliptical NFW
        profile = EllipticalNFWProfile(  # type: ignore
            M_vir=mass,  # Virial mass
            c=5.0,  # Concentration parameter
            lens_sys=lens,
            ellipticity=ellipticity  # Ellipticity parameter
        )
    
    # Generate grid in arcseconds
    extent = 50  # arcsec
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate convergence directly using the profile
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Calculate convergence using the profile
    convergence_map = profile.convergence(x_flat, y_flat)
    convergence_map = convergence_map.reshape(grid_size, grid_size)
    
    return convergence_map, X, Y


def plot_convergence_map(convergence, X, Y, title="Convergence Map", cmap="viridis"):
    """Plot convergence map."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(X, Y, convergence, levels=20, cmap=cmap)
    ax.set_xlabel("x (arcsec)", fontsize=12)
    ax.set_ylabel("y (arcsec)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax, label='Convergence κ')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    return fig


def main():
    """Main page function."""
    # Render header
    render_header(
        "Simple Gravitational Lensing",
        "Generate synthetic convergence maps with different mass profiles"
    )
    
    # Debug info
    if not MODULES_AVAILABLE:
        if import_error:
            st.warning(f"⚠️ Import error detected: {import_error}")
    
    # Show module status
    if not MODULES_AVAILABLE:
        st.error("❌ Required modules not available. Cannot generate maps.")
        
        with st.expander("📦 Setup Instructions", expanded=True):
            st.markdown("""
            ### Install Required Dependencies
            
            The following packages should already be installed. If you see this message, 
            please check the error above.
            
            ```bash
            # Install core dependencies
            pip install numpy scipy matplotlib astropy h5py
            
            # Install machine learning dependencies
            pip install torch scikit-learn
            
            # Verify installation
            python -c "import astropy; print(f'✅ Astropy {astropy.__version__} installed')"
            ```
            
            ### Check Module Availability
            
            ```bash
            python -c "from src.lens_models.lens_system import LensSystem; print('✅ Lens models available')"
            ```
            """)
        return
    
    # Parameter controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        profile_type = st.selectbox(
            "Lens Profile",
            ["NFW", "Elliptical NFW"],
            help="Select the dark matter density profile"
        )
        
        mass = st.slider(
            "Virial Mass (×10¹² M☉)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Total mass of the dark matter halo"
        )
        
        scale_radius = st.slider(
            "Scale Radius (kpc)",
            min_value=50.0,
            max_value=500.0,
            value=200.0,
            step=10.0,
            help="Characteristic radius of the density profile"
        )
        
        ellipticity = 0.0
        if profile_type == "Elliptical NFW":
            ellipticity = st.slider(
                "Ellipticity",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Ellipticity of the lens (0 = circular)"
            )
        
        grid_size = st.select_slider(
            "Grid Size",
            options=[32, 64, 128, 256],
            value=64,
            help="Resolution of the convergence map"
        )
        
        cmap = st.selectbox(
            "Colormap",
            ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"],
            index=0
        )
        
        st.markdown("---")
        st.info(f"⏱️ Estimated computation time: {grid_size * 0.01:.1f}s")
        generate_btn = st.button("🚀 Generate Map", type="primary", use_container_width=True)
    
    st.markdown("""
    Generate synthetic gravitational lensing convergence maps using different dark matter 
    density profiles. Adjust parameters in real-time and visualize the resulting convergence 
    distribution.
    
    **Features:**
    - NFW and Elliptical NFW profiles
    - Real-time parameter adjustment
    - High-resolution rendering (up to 256×256)
    - Multiple colormap options
    - Downloadable results
    """)
    
    st.write("Debug: Starting parameter controls")  # Debug line
    
    try:
        # Parameter controls
        col1, col2 = st.columns([1, 2])
        st.write("Debug: Columns created")  # Debug line
        
        with col1:
            st.write("Debug: Entering column 1")  # Debug line
            st.subheader("⚙️ Configuration")
    except Exception as e:
        st.error(f"Error in parameter controls setup: {str(e)}")
        st.write("Debug: Parameter controls failed")
        
        profile_type = st.selectbox(
            "Lens Profile",
            ["NFW", "Elliptical NFW"],
            help="Select the dark matter density profile"
        )
        
        mass = st.slider(
            "Virial Mass (×10¹² M☉)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Total mass of the dark matter halo"
        )
        
        scale_radius = st.slider(
            "Scale Radius (kpc)",
            min_value=50.0,
            max_value=500.0,
            value=200.0,
            step=10.0,
            help="Characteristic radius of the density profile"
        )
        
        ellipticity = 0.0
        if profile_type == "Elliptical NFW":
            ellipticity = st.slider(
                "Ellipticity",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Ellipticity of the lens (0 = circular)"
            )
        
        grid_size = st.select_slider(
            "Grid Size",
            options=[32, 64, 128, 256],
            value=64,
            help="Resolution of the convergence map"
        )
        
        cmap = st.selectbox(
            "Colormap",
            ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"],
            index=0
        )
        
        st.markdown("---")
        
        # Computation time estimate
        est_time = estimate_computation_time(grid_size)
        st.info(f"⏱️ Estimated computation time: {est_time:.1f}s")
        
        generate_btn = st.button("🚀 Generate Map", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Visualization")
        
        if generate_btn:
            with st.spinner(f"Generating {grid_size}×{grid_size} convergence map..."):
                try:
                    mass_solar = mass * 1e12
                    convergence_map, X, Y = generate_synthetic_convergence(
                        profile_type, mass_solar, scale_radius, ellipticity, grid_size
                    )
                    
                    # Store in session state
                    st.session_state['convergence_map'] = convergence_map
                    st.session_state['X'] = X
                    st.session_state['Y'] = Y
                    st.session_state['profile_type'] = profile_type
                    st.session_state['grid_size'] = grid_size
                    
                    show_success("Map generated successfully!")
                except Exception as e:
                    # Show both a friendly message and the full traceback in the UI
                    show_error(f"Error generating map: {e}")
                    st.exception(e)
                    st.text(traceback.format_exc())
                    return
        
        # Display map (only if it exists in session state)
        if 'convergence_map' in st.session_state and 'X' in st.session_state and 'Y' in st.session_state:
            fig = plot_convergence_map(
                st.session_state['convergence_map'],
                st.session_state['X'],
                st.session_state['Y'],
                title=f"{st.session_state.get('profile_type', 'NFW')} Convergence Map",
                cmap=cmap
            )
            st.pyplot(fig)
            plt.close()
            
            # Statistics
            st.markdown("### 📈 Statistics")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            kappa = st.session_state.get('convergence_map', None)
            if kappa is not None and hasattr(kappa, 'max'):
                col_a.metric("Max κ", f"{kappa.max():.4f}")
                col_b.metric("Mean κ", f"{kappa.mean():.4f}")
                col_c.metric("Min κ", f"{kappa.min():.4f}")
                col_d.metric("Std κ", f"{kappa.std():.4f}")
            
            # Download section
            st.markdown("### 💾 Download")
            
            if kappa is not None:
                # Save to buffer
                buf = io.BytesIO()
                np.save(buf, st.session_state['convergence_map'])
                buf.seek(0)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="Download Map (.npy)",
                        data=buf,
                        file_name=f"convergence_map_{grid_size}x{grid_size}.npy",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Save figure
                    fig_buf = io.BytesIO()
                    fig.savefig(fig_buf, format='png', dpi=300, bbox_inches='tight')
                    fig_buf.seek(0)
                    
                    st.download_button(
                        label="Download Figure (.png)",
                        data=fig_buf,
                        file_name=f"convergence_map_{grid_size}x{grid_size}.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    # Additional information
    with st.expander("ℹ️ About Convergence Maps"):
        st.markdown("""
        **Convergence (κ)** represents the surface mass density of a gravitational lens 
        normalized by the critical surface density. Key properties:
        
        - **κ < 1**: Weak lensing regime (small distortions)
        - **κ = 1**: Critical curve (infinite magnification)
        - **κ > 1**: Strong lensing regime (multiple images possible)
        
        **NFW Profile**: Navarro-Frenk-White profile describes the density distribution of 
        dark matter halos in cosmological simulations. It's the standard model for galaxy 
        cluster dark matter.
        
        **Elliptical NFW**: Extension that accounts for ellipticity in the projected mass 
        distribution, more realistic for observed galaxy clusters.
        """)


if __name__ == "__main__":
    main()
