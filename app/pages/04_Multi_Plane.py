"""
Multi-Plane Lensing - Advanced gravitational lensing with multiple lens planes

Simulate light propagation through multiple lens planes at different redshifts,
enabling realistic modeling of line-of-sight structure.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info
except ImportError:
    from utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info

# Import core modules
try:
    from src.lens_models.multi_plane import LensPlane, MultiPlaneLens
    from src.lens_models import NFWProfile
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Multi-Plane Lensing - Gravitational Lensing Platform",
    page_icon="🌌",
    layout="wide"
)

# Apply custom CSS
inject_custom_css()


def create_multi_plane_system(plane_configs, z_source):
    """Create multi-plane lens system from configurations."""
    planes = []
    
    for config in plane_configs:
        profile = NFWProfile(
            M_200=config['mass'],
            c_200=config['concentration'],
            z_lens=config['redshift'],
            z_source=z_source
        )
        
        plane = LensPlane(
            redshift=config['redshift'],
            mass_profile=profile,
            position=(config['x_offset'], config['y_offset'])
        )
        planes.append(plane)
    
    multi_lens = MultiPlaneLens(planes, z_source)
    return multi_lens


def plot_ray_tracing(source_grid, image_grid, title="Ray Tracing"):
    """Plot source and image plane grids."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Source plane
    ax1.scatter(source_grid[:, 0], source_grid[:, 1], 
               c='blue', alpha=0.5, s=10, label='Source positions')
    ax1.set_xlabel('β_x (arcsec)', fontsize=11)
    ax1.set_ylabel('β_y (arcsec)', fontsize=11)
    ax1.set_title('Source Plane', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Image plane
    ax2.scatter(image_grid[:, 0], image_grid[:, 1], 
               c='red', alpha=0.5, s=10, label='Image positions')
    ax2.set_xlabel('θ_x (arcsec)', fontsize=11)
    ax2.set_ylabel('θ_y (arcsec)', fontsize=11)
    ax2.set_title('Image Plane', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_convergence_comparison(convergences, redshifts):
    """Plot convergence for each plane."""
    n_planes = len(convergences)
    fig, axes = plt.subplots(1, n_planes, figsize=(5*n_planes, 4))
    
    if n_planes == 1:
        axes = [axes]
    
    for ax, conv, z in zip(axes, convergences, redshifts):
        im = ax.imshow(conv, cmap='viridis', origin='lower')
        ax.set_title(f'Plane at z = {z:.2f}', fontweight='bold')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        plt.colorbar(im, ax=ax, label='κ', fraction=0.046)
    
    plt.suptitle('Convergence Maps per Plane', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Main page function."""
    render_header(
        "🌌 Multi-Plane Gravitational Lensing",
        "Simulate complex lensing with multiple lens planes",
        "Advanced"
    )
    
    if not MODULES_AVAILABLE:
        show_error("Multi-plane modules not available. Please check installation.")
        with st.expander("📦 Setup Instructions"):
            st.code("""
# Install multi-plane lensing components
pip install -r requirements.txt

# Verify installation
python -c "from src.lens_models.multi_plane import MultiPlaneLens; print('✅ Available')"
            """)
        return
    
    st.markdown("""
    **Multi-plane gravitational lensing** accounts for the realistic 3D distribution of matter
    along the line of sight. Unlike single-plane lensing, this approach:
    
    - ✅ Models multiple lenses at different redshifts
    - ✅ Includes line-of-sight structure
    - ✅ Produces more realistic strong lensing predictions
    - ✅ Essential for accurate cosmological constraints
    """)
    
    # Configuration
    st.subheader("⚙️ Multi-Plane Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        z_source = st.slider(
            "Source Redshift",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Redshift of the background source"
        )
        
        n_planes = st.slider(
            "Number of Lens Planes",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of intervening lens planes"
        )
    
    with col2:
        grid_size = st.select_slider(
            "Grid Size",
            options=[32, 64, 128],
            value=64
        )
        
        field_size = st.slider(
            "Field Size (arcsec)",
            min_value=5.0,
            max_value=30.0,
            value=10.0,
            step=1.0
        )
    
    # Lens plane parameters
    st.markdown("---")
    st.subheader("🔭 Lens Plane Parameters")
    
    plane_configs = []
    
    cols = st.columns(n_planes)
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Plane {i+1}**")
            
            z_lens = st.number_input(
                f"Redshift",
                min_value=0.1,
                max_value=z_source - 0.1,
                value=0.3 + i * (z_source - 0.5) / max(n_planes - 1, 1),
                step=0.1,
                key=f"z_{i}"
            )
            
            mass = st.number_input(
                f"Mass (×10¹² M☉)",
                min_value=0.1,
                max_value=10.0,
                value=1.0 + i * 0.5,
                step=0.1,
                key=f"mass_{i}"
            )
            
            concentration = st.number_input(
                f"Concentration",
                min_value=2.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                key=f"conc_{i}"
            )
            
            x_offset = st.number_input(
                f"X Offset (arcsec)",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                key=f"x_{i}"
            )
            
            y_offset = st.number_input(
                f"Y Offset (arcsec)",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                key=f"y_{i}"
            )
            
            plane_configs.append({
                'redshift': z_lens,
                'mass': mass * 1e12,
                'concentration': concentration,
                'x_offset': x_offset,
                'y_offset': y_offset
            })
    
    # Run simulation
    st.markdown("---")
    st.subheader("🚀 Run Simulation")
    
    if st.button("▶️ Trace Rays", type="primary", use_container_width=True):
        with st.spinner("Tracing light rays through multiple planes..."):
            try:
                # Create multi-plane system
                multi_lens = create_multi_plane_system(plane_configs, z_source)
                
                # Generate source grid
                extent = field_size / 2
                x = np.linspace(-extent, extent, grid_size)
                y = np.linspace(-extent, extent, grid_size)
                X, Y = np.meshgrid(x, y)
                
                image_positions = np.column_stack([X.flatten(), Y.flatten()])
                
                # Trace rays
                source_positions = multi_lens.ray_trace(
                    image_positions[:, 0],
                    image_positions[:, 1]
                )
                
                # Store results
                st.session_state['multi_image_pos'] = image_positions
                st.session_state['multi_source_pos'] = source_positions
                st.session_state['multi_lens'] = multi_lens
                st.session_state['multi_planes'] = plane_configs
                
                show_success(f"Successfully traced {len(image_positions)} rays through {n_planes} planes!")
                
            except Exception as e:
                show_error(f"Error in ray tracing: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    # Display results
    if 'multi_source_pos' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Ray tracing visualization
        fig = plot_ray_tracing(
            st.session_state['multi_source_pos'],
            st.session_state['multi_image_pos'],
            f"Multi-Plane Ray Tracing ({n_planes} planes)"
        )
        st.pyplot(fig)
        plt.close()
        
        # Statistics
        st.markdown("### 📈 Statistics")
        col1, col2, col3 = st.columns(3)
        
        source_pos = st.session_state['multi_source_pos']
        image_pos = st.session_state['multi_image_pos']
        
        # Calculate magnification
        source_area = np.ptp(source_pos[:, 0]) * np.ptp(source_pos[:, 1])
        image_area = np.ptp(image_pos[:, 0]) * np.ptp(image_pos[:, 1])
        magnification = image_area / (source_area + 1e-10)
        
        col1.metric("Magnification", f"{magnification:.2f}×")
        col2.metric("Source RMS", f"{np.std(source_pos):.3f}\"")
        col3.metric("Image RMS", f"{np.std(image_pos):.3f}\"")
        
        # Convergence maps
        with st.expander("🗺️ View Convergence Maps per Plane"):
            st.info("Convergence map computation for multi-plane systems coming soon!")
        
        # Technical details
        with st.expander("🔬 Technical Details"):
            st.markdown("**Lens Plane Configuration:**")
            for i, config in enumerate(plane_configs):
                st.markdown(f"""
                **Plane {i+1}:**
                - Redshift: {config['redshift']:.2f}
                - Mass: {config['mass']:.2e} M☉
                - Concentration: {config['concentration']:.1f}
                - Position: ({config['x_offset']:.1f}\", {config['y_offset']:.1f}\")
                """)
    
    # Educational content
    with st.expander("📚 Learn More: Multi-Plane Lensing"):
        st.markdown("""
        ### Why Multi-Plane?
        
        Single-plane (thin-lens) approximation assumes all mass is concentrated at one redshift.
        While computationally efficient, it ignores:
        
        1. **Line-of-sight structure**: Matter is distributed along the line of sight
        2. **Multiple deflections**: Light can be deflected multiple times
        3. **Redshift evolution**: Lens properties change with cosmic time
        
        ### Mathematical Framework
        
        The multi-plane lens equation:
        
        $$\\beta = \\theta - \\sum_{i=1}^{N} D_{i,s} \\alpha_i(\\theta_i)$$
        
        Where:
        - $\\theta$ = observed image position
        - $\\beta$ = source position  
        - $\\alpha_i$ = deflection angle at plane $i$
        - $D_{i,s}$ = angular diameter distance ratio
        - $N$ = number of lens planes
        
        ### Applications
        
        - **Galaxy clusters**: Extended mass distributions
        - **Cosmological simulations**: Realistic structure
        - **Time delays**: Multiple images at different light-travel times
        - **Substructure detection**: Line-of-sight halos
        """)


if __name__ == "__main__":
    main()
