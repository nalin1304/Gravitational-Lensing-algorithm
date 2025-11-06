"""
Real Data Analysis - Upload and analyze real FITS astronomical data

Process and analyze real gravitational lensing observations from HST, JWST, 
and other telescopes.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import tempfile

# Configure page FIRST
st.set_page_config(
    page_title="Real Data Analysis - Gravitational Lensing Platform",
    page_icon="🔭",
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info
except ImportError:
    from utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info

# Apply custom CSS
inject_custom_css()

# Check for astropy
ASTROPY_AVAILABLE = False
astropy_error = None
try:
    import astropy
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval, ImageNormalize
    ASTROPY_AVAILABLE = True
except ImportError as e:
    astropy_error = str(e)

# Import core modules
REAL_DATA_AVAILABLE = False
import_error = None
try:
    from src.data.real_data_loader import FITSDataLoader, preprocess_real_data, PSFModel
    REAL_DATA_AVAILABLE = True
except ImportError as e:
    import_error = str(e)


def plot_fits_image(data, title="FITS Image", percentile=99.5):
    """Plot FITS image data with proper scaling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Apply ZScale normalization
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    
    im = ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Counts')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    return fig


def display_fits_header(header):
    """Display FITS header information."""
    important_keys = [
        'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME', 
        'DATE-OBS', 'OBJECT', 'RA_TARG', 'DEC_TARG'
    ]
    
    header_dict = {}
    for key in important_keys:
        if key in header:
            header_dict[key] = str(header[key])
    
    if header_dict:
        st.markdown("**Observation Metadata:**")
        for key, value in header_dict.items():
            st.text(f"{key}: {value}")
    else:
        st.info("No standard metadata found in FITS header.")


def main():
    """Main page function."""
    render_header(
        "Real Data Analysis",
        "Upload and analyze FITS astronomical data"
    )
    
    # Check dependencies
    if not ASTROPY_AVAILABLE:
        st.error("❌ Astropy not available. Cannot process FITS files.")
        if astropy_error:
            st.warning(f"⚠️ Import error: {astropy_error}")
        
        with st.expander("📦 Installation Instructions", expanded=True):
            st.markdown("""
            ### Install Astropy
            
            ```bash
            # Install astropy
            pip install astropy>=6.0.0

            # Verify installation
            python -c "import astropy; print(f'Astropy {astropy.__version__} installed')"
            ```
            """)
        return
    
    if not REAL_DATA_AVAILABLE:
        st.error("❌ Real data loader not available.")
        if import_error:
            st.warning(f"⚠️ Import error: {import_error}")
        return
    
    st.markdown("""
    Upload FITS files from telescopes like HST, JWST, or ground-based observatories
    to analyze gravitational lensing systems.
    
    **Supported Data:**
    - Hubble Space Telescope (HST) imaging
    - James Webb Space Telescope (JWST) imaging
    - Ground-based adaptive optics data
    - Simulated FITS files for testing
    """)
    
    # File upload
    st.subheader("📂 Upload FITS File")
    
    uploaded_file = st.file_uploader(
        "Choose a FITS file",
        type=['fits', 'fit', 'fts'],
        help="Upload a FITS format astronomical image"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading FITS file..."):
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Open FITS file
                with fits.open(tmp_path) as hdul:
                    # Display file structure
                    with st.expander("📋 FITS File Structure"):
                        st.text(f"Number of HDUs: {len(hdul)}")
                        for i, hdu in enumerate(hdul):
                            st.text(f"HDU {i}: {hdu.name} - {type(hdu).__name__}")
                            if hasattr(hdu, 'shape'):
                                st.text(f"  Shape: {hdu.shape}")
                    
                    # Get primary image
                    primary_hdu = None
                    for hdu in hdul:
                        if hasattr(hdu, 'data') and hdu.data is not None:
                            if len(hdu.data.shape) >= 2:
                                primary_hdu = hdu
                                break
                    
                    if primary_hdu is None:
                        show_error("No valid image data found in FITS file.")
                        return
                    
                    data = primary_hdu.data
                    header = primary_hdu.header
                    
                    # Handle 3D data (take first slice)
                    if len(data.shape) == 3:
                        data = data[0, :, :]
                        show_info(f"3D data detected. Using first slice. Shape: {data.shape}")
                    elif len(data.shape) > 3:
                        show_error(f"Unsupported data dimension: {len(data.shape)}D")
                        return
                    
                    # Store in session state
                    st.session_state['fits_data'] = data
                    st.session_state['fits_header'] = header
                    st.session_state['fits_filename'] = uploaded_file.name
                    
                    show_success(f"Loaded: {uploaded_file.name} ({data.shape[0]}×{data.shape[1]})")
                
            except Exception as e:
                show_error(f"Error loading FITS file: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                return
    
    # Display loaded data
    if 'fits_data' in st.session_state and 'fits_header' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Loaded Data")
        
        data = st.session_state['fits_data']
        header = st.session_state.get('fits_header', {})
        filename = st.session_state.get('fits_filename', 'Unknown')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            fig = plot_fits_image(data, title=f"FITS Image: {filename}")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Metadata
            st.markdown("**File Information:**")
            st.text(f"Filename: {filename}")
            st.text(f"Shape: {data.shape}")
            st.text(f"Data type: {data.dtype}")
            
            # Statistics
            st.markdown("---")
            st.markdown("**Image Statistics:**")
            st.metric("Min", f"{np.nanmin(data):.2f}")
            st.metric("Max", f"{np.nanmax(data):.2f}")
            st.metric("Mean", f"{np.nanmean(data):.2f}")
            st.metric("Median", f"{np.nanmedian(data):.2f}")
            st.metric("Std Dev", f"{np.nanstd(data):.2f}")
            
            # Display header
            st.markdown("---")
            with st.expander("📄 View FITS Header"):
                display_fits_header(header)
        
        # Preprocessing options
        st.markdown("---")
        st.subheader("⚙️ Preprocessing")
        
        col_pre1, col_pre2, col_pre3 = st.columns(3)
        
        with col_pre1:
            crop = st.checkbox("Crop to Center", value=False)
            if crop:
                crop_size = st.slider("Crop Size (px)", 64, min(data.shape), 256)
        
        with col_pre2:
            normalize = st.checkbox("Normalize", value=True)
            if normalize:
                norm_method = st.selectbox("Method", ["Min-Max", "Z-Score"])
        
        with col_pre3:
            denoise = st.checkbox("Denoise", value=False)
            if denoise:
                sigma = st.slider("Gaussian σ", 0.5, 5.0, 1.0, 0.5)
        
        if st.button("▶️ Apply Preprocessing", type="primary"):
            with st.spinner("Processing..."):
                try:
                    processed = data.copy()
                    
                    # Crop
                    if crop:
                        center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
                        half_size = crop_size // 2
                        processed = processed[
                            center_y - half_size:center_y + half_size,
                            center_x - half_size:center_x + half_size
                        ]
                    
                    # Normalize
                    if normalize:
                        if norm_method == "Min-Max":
                            processed = (processed - np.nanmin(processed)) / \
                                       (np.nanmax(processed) - np.nanmin(processed) + 1e-10)
                        else:  # Z-Score
                            processed = (processed - np.nanmean(processed)) / \
                                       (np.nanstd(processed) + 1e-10)
                    
                    # Denoise
                    if denoise:
                        from scipy.ndimage import gaussian_filter
                        processed = gaussian_filter(processed, sigma=sigma)
                    
                    # Store processed data
                    st.session_state['processed_data'] = processed
                    
                    show_success("Preprocessing complete!")
                    
                    # Display processed image
                    fig = plot_fits_image(processed, title="Processed Image")
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    show_error(f"Preprocessing error: {e}")
        
        # Export options
        if 'processed_data' in st.session_state:
            st.markdown("---")
            st.subheader("💾 Export")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                import io
                buf = io.BytesIO()
                np.save(buf, st.session_state['processed_data'])
                buf.seek(0)
                
                st.download_button(
                    label="Download Processed Data (.npy)",
                    data=buf,
                    file_name=f"processed_{filename}.npy",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            
            with col_exp2:
                # Use processed data for inference
                if st.button("🔬 Use for PINN Inference", use_container_width=True):
                    st.session_state['convergence_map'] = st.session_state['processed_data']
                    show_success("Data ready for inference! Go to PINN Inference page.")
    
    # Example datasets
    with st.expander("📚 Example Datasets"):
        st.markdown("""
        ### Public Gravitational Lensing Datasets
        
        **HST Archive:**
        - Einstein Cross (Q2237+0305)
        - Abell 1689 cluster
        - MACS J1149.5+2223 (Einstein Cross)
        
        **JWST Early Release:**
        - SMACS 0723 deep field
        - Stephan's Quintet
        
        **Download from:**
        - [MAST Archive](https://mast.stsci.edu/)
        - [ESA Hubble](https://esahubble.org/projects/fits_liberator/)
        - [JWST Archive](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)
        """)


if __name__ == "__main__":
    main()
