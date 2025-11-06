"""
PINN Inference - Physics-Informed Neural Network Predictions

Run the trained PINN model on convergence maps to predict lens parameters 
and classify dark matter model type.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info
except ImportError:
    from utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info

# Import core modules (fail fast if unavailable)
from src.ml.pinn import PhysicsInformedNN

# Configure page
st.set_page_config(
    page_title="PINN Inference - Gravitational Lensing Platform",
    page_icon="🔬",
    layout="wide"
)

# Apply custom CSS
inject_custom_css()


def load_pretrained_model(model_path=None):
    """Load pre-trained PINN model."""
    if model_path is None:
        model_path = project_root / "results" / "pinn_model_best.pth"
    
    if not model_path.exists():
        return None, f"Model file not found at: {model_path}"
    
    try:
        model = PhysicsInformedNN(
            input_size=64,
            dropout_rate=0.2
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, "Model loaded successfully!"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def create_demo_model():
    """Create a demo PINN model with random weights for testing."""
    model = PhysicsInformedNN(
        input_size=64,
        dropout_rate=0.2
    )
    model.eval()
    return model


def plot_classification_probs(class_names, probs, entropy):
    """Plot classification probabilities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(class_names, probs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Dark Matter Model Classification', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add entropy
    ax.text(0.95, 0.95, f'Entropy: {entropy:.3f}',
            transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    return fig


def main():
    """Main page function."""
    render_header(
        "🔬 PINN Inference",
        "Physics-Informed Neural Network for parameter prediction",
        "ML"
    )
    
    # Modules are required; if import failed, this file would not load
    
    st.markdown("""
    Run the trained Physics-Informed Neural Network (PINN) model on convergence maps to:
    - **Predict lens parameters** (mass, scale radius, source position, Hubble constant)
    - **Classify dark matter models** (CDM, WDM, SIDM)
    - **Quantify prediction uncertainty**
    
    The PINN incorporates gravitational lensing physics directly into the loss function,
    ensuring physically consistent predictions.
    """)
    
    # Model configuration
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_option = st.radio(
            "Model Type",
            ["Pre-trained PINN", "Demo Model (Random Weights)", "Custom Model Path"],
            help="Select the model to use for inference"
        )
    
    with col2:
        device = st.selectbox(
            "Device",
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
            help="Computation device"
        )
    
    # Custom model path
    custom_path = None
    if model_option == "Custom Model Path":
        custom_path = st.text_input("Model Path", "results/pinn_model_best.pth")
    
    # Load model
    if st.button("📥 Load Model", type="primary"):
        with st.spinner("Loading model..."):
            try:
                if model_option == "Demo Model (Random Weights)":
                    # Create demo model
                    model = create_demo_model()
                    model = model.to(device)
                    st.session_state['pinn_model'] = model
                    st.session_state['model_device'] = device
                    st.session_state['is_demo_model'] = True
                    show_success(f"✅ Demo model created successfully on {device}!")
                    show_warning("⚠️ This is a demo model with random weights. Predictions will not be accurate.")
                else:
                    # Load pre-trained model
                    if custom_path:
                        model_path = Path(custom_path)
                    else:
                        model_path = None
                    
                    model, message = load_pretrained_model(model_path)
                    
                    if model is None:
                        show_error(message)
                        st.info("💡 Try using 'Demo Model (Random Weights)' to test the interface.")
                    else:
                        model = model.to(device)
                        st.session_state['pinn_model'] = model
                        st.session_state['model_device'] = device
                        st.session_state['is_demo_model'] = False
                        show_success(f"✅ {message} on {device}!")
            except Exception as e:
                show_error(f"Error loading model: {e}")
    
    # Display model info
    if 'pinn_model' in st.session_state:
        st.success(f"✅ Model ready on {st.session_state.get('model_device', 'cpu')}")
        
        # Option to save demo model
        if st.session_state.get('is_demo_model', False):
            with st.expander("💾 Save Demo Model as Pre-trained"):
                st.markdown("Save the current demo model to use as a pre-trained model later.")
                save_path = st.text_input("Save Path", "results/pinn_model_best.pth")
                if st.button("Save Model"):
                    try:
                        save_path_obj = Path(save_path)
                        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'model_state_dict': st.session_state['pinn_model'].state_dict(),
                            'architecture': {
                                'input_size': 64,
                                'dropout_rate': 0.2
                            }
                        }, save_path_obj)
                        show_success(f"Model saved to {save_path}!")
                    except Exception as e:
                        show_error(f"Error saving model: {e}")
    
    # Data input
    st.markdown("---")
    st.subheader("📥 Input Data")
    
    data_source = st.radio(
        "Data Source",
        ["Use Session Data", "Upload .npy File", "Generate Random"],
        help="Select the source of convergence map data"
    )
    
    input_data = None
    
    if data_source == "Use Session Data":
        if 'convergence_map' in st.session_state:
            input_data = st.session_state['convergence_map']
            if input_data is not None and hasattr(input_data, 'shape'):
                show_info(f"Using convergence map from session: {input_data.shape}")
            else:
                show_warning("Session data is invalid. Please generate new data.")
                input_data = None
        else:
            show_warning("No data in session. Please generate or upload data first.")
    
    elif data_source == "Upload .npy File":
        uploaded = st.file_uploader("Upload convergence map (.npy)", type=['npy'])
        if uploaded:
            try:
                input_data = np.load(uploaded)
                show_success(f"Loaded: {input_data.shape}")
            except Exception as e:
                show_error(f"Error loading file: {e}")
    
    elif data_source == "Generate Random":
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            grid_size = st.select_slider("Grid Size", options=[32, 64, 128], value=64)
        with col_gen2:
            if st.button("² Generate Random Map"):
                input_data = np.random.randn(grid_size, grid_size) * 0.1 + 0.5
                input_data = np.clip(input_data, 0, 1)
                st.session_state['inference_input'] = input_data
                show_success("Generated random convergence map")
    
    # Display input data
    if input_data is not None:
        with st.expander("👁️ View Input Data"):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(input_data, cmap='viridis', origin='lower')
            ax.set_title("Input Convergence Map")
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            plt.colorbar(im, ax=ax, label='Îº')
            st.pyplot(fig)
            plt.close()
    
    # Run inference
    if input_data is not None and 'pinn_model' in st.session_state:
        st.markdown("---")
        st.subheader("🚀 Run Inference")
        
        if st.button("▶️ Predict Parameters", type="primary", use_container_width=True):
            with st.spinner("Running PINN inference..."):
                try:
                    # Prepare input
                    if input_data.shape[0] != 64:
                        from scipy.ndimage import zoom
                        scale = 64 / input_data.shape[0]
                        input_resized = zoom(input_data, scale, order=1)
                    else:
                        input_resized = input_data
                    
                    # Normalize
                    input_norm = (input_resized - input_resized.min()) / \
                                (input_resized.max() - input_resized.min() + 1e-10)
                    
                    # Convert to tensor
                    input_tensor = torch.from_numpy(input_norm).float()
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
                    input_tensor = input_tensor.to(st.session_state['model_device'])
                    
                    # Inference
                    model = st.session_state['pinn_model']
                    model.eval()
                    
                    with torch.no_grad():
                        params, classes = model(input_tensor)
                    
                    # Convert to numpy
                    params_np = params.cpu().numpy()[0]
                    classes_np = torch.softmax(classes, dim=1).cpu().numpy()[0]
                    
                    # Store results
                    st.session_state['pred_params'] = params_np
                    st.session_state['pred_classes'] = classes_np
                    
                    show_success("Inference complete!")
                    
                except Exception as e:
                    show_error(f"Error during inference: {e}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
        
        # Display results
        if 'pred_params' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔢 Predicted Parameters")
                
                param_names = ['M_vir', 'r_s', 'β_x', 'β_y', 'H₀']
                param_units = ['M☉', 'kpc', 'arcsec', 'arcsec', 'km/s/Mpc']
                params = st.session_state['pred_params']
                
                for name, unit, value in zip(param_names, param_units, params):
                    if name == 'M_vir':
                        st.metric(f"{name} ({unit})", f"{value:.2e}")
                    else:
                        st.metric(f"{name} ({unit})", f"{value:.4f}")
            
            with col2:
                st.markdown("#### 🏷️ Dark Matter Classification")
                
                class_names = ['CDM', 'WDM', 'SIDM']
                classes = st.session_state['pred_classes']
                
                for name, prob in zip(class_names, classes):
                    st.metric(name, f"{prob*100:.1f}%")
                
                predicted_class = class_names[np.argmax(classes)]
                confidence = classes[np.argmax(classes)]
                
                st.markdown("**Prediction:**")
                if confidence > 0.7:
                    st.success(f"… High confidence: **{predicted_class}**")
                elif confidence > 0.5:
                    st.info(f"ℹ„¹ï¸ Moderate confidence: **{predicted_class}**")
                else:
                    st.warning(f" ï¸ Low confidence: **{predicted_class}**")
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📈 Classification Visualization")
            
            entropy = -np.sum(classes * np.log(classes + 1e-10))
            fig = plot_classification_probs(class_names, classes, entropy)
            st.pyplot(fig)
            plt.close()
            
            # Interpretation
            with st.expander("ℹ️ Interpretation Guide"):
                st.markdown("""
                **Parameters:**
                - **M_vir**: Virial mass of the dark matter halo
                - **r_s**: Scale radius of the NFW profile
                - **β_x, β_y**: Source position in the source plane
                - **H₀**: Hubble constant
                
                **Dark Matter Models:**
                - **CDM (Cold Dark Matter)**: Standard model, no suppression of small-scale structure
                - **WDM (Warm Dark Matter)**: Suppresses structure below cutoff mass
                - **SIDM (Self-Interacting Dark Matter)**: Alters core density profiles
                
                **Entropy**: Lower entropy indicates higher confidence in classification.
                """)
    
    else:
        if 'pinn_model' not in st.session_state:
            show_info("Please load a model first.")
        if input_data is None:
            show_info("Please provide input data.")
    
    # Additional information
    with st.expander("🧠 About Physics-Informed Neural Networks"):
        st.markdown("""
        **Physics-Informed Neural Networks (PINNs)** incorporate known physics directly 
        into the neural network training process. For gravitational lensing:
        
        1. **Lens Equation**: β = θ - α(θ) must be satisfied
        2. **Mass Conservation**: Convergence relates to surface mass density
        3. **Physical Constraints**: Parameters must be physically reasonable
        
        **Advantages:**
        - Guaranteed physical consistency
        - Better generalization
        - Reduced training data requirements
        - Interpretable predictions
        
        **Architecture:**
        - Encoder: CNN for feature extraction
        - Regression Head: Predicts lens parameters
        - Classification Head: Identifies dark matter model
        - Physics Loss: Enforces lens equation residuals
        """)


if __name__ == "__main__":
    main()
