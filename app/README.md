# Gravitational Lensing Analysis Web App

Professional Streamlit web application for gravitational lensing analysis, convergence map generation, and PINN-based inference.

## Features

### 🏠 Home Page
- Project overview with key metrics
- Feature highlights
- Quick start guide
- System information

### 🎲 Generate Synthetic Data
- Real-time NFW and Elliptical NFW convergence maps
- Interactive parameter controls:
  - Mass: 10¹¹ - 10¹⁴ M☉
  - Scale radius: 50-500 kpc
  - Ellipticity: 0.0-0.5
  - Grid size: 32/64/128
- Download generated data as NumPy .npy files

### 📊 Analyze Real Data
- Upload FITS observations
- Display metadata (shape, pixel scale, coordinates)
- Preprocessing controls (resize, normalize, NaN handling)
- Before/after visualization

### 🧠 Model Inference
- Load pretrained PINN models
- Predict halo parameters (M_vir, r_s, ellipticity)
- Classification with confidence scores
- Interactive visualizations

### 📈 Uncertainty Analysis
- Monte Carlo Dropout (100 iterations)
- Parameter uncertainty (mean, std, 95% CI)
- Predictive entropy
- Confidence interval visualization

### ℹ️ About
- Project documentation
- References and citations
- Technical details

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Install Dependencies

```powershell
# Navigate to project root
cd d:\Coding projects\Collab\financial-advisor-tool

# Install all requirements
pip install -r requirements.txt
```

**Key dependencies**:
- `streamlit>=1.28.0` - Web framework
- `matplotlib` - Plotting
- `torch` - Neural networks
- `plotly>=5.17.0` - Interactive visualization
- `pillow>=10.0.0` - Image processing
- `astropy` - FITS file handling

## Usage

### Launch the Application

```powershell
# From project root
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

### Navigation

Use the **sidebar** to switch between pages:
1. 🏠 Home
2. 🎲 Generate Synthetic
3. 📊 Analyze Real Data
4. 🧠 Model Inference
5. 📈 Uncertainty Analysis
6. ℹ️ About

### Workflows

#### Generate Synthetic Convergence Maps
1. Go to **Generate Synthetic** page
2. Select profile type (NFW or Elliptical NFW)
3. Adjust parameters using sliders
4. Click **Generate Convergence Map**
5. View contour plot
6. Download results using **Download Data** button

#### Analyze Real Observations
1. Go to **Analyze Real Data** page
2. Upload FITS file using file uploader
3. View metadata and statistics
4. Adjust preprocessing parameters
5. Apply preprocessing
6. View before/after comparison

#### Run Model Inference
1. Generate or upload convergence map
2. Go to **Model Inference** page
3. Data automatically loaded from session state
4. Click **Run Inference**
5. View predicted parameters with uncertainties
6. Check classification results

#### Uncertainty Quantification
1. Run inference first (see above)
2. Go to **Uncertainty Analysis** page
3. Set number of MC samples (default 100)
4. Click **Compute Uncertainty**
5. View parameter distributions
6. Check confidence intervals and entropy

## Architecture

### File Structure

```
app/
├── main.py          # Streamlit UI (1,100+ lines)
├── utils.py         # Utility functions (382 lines)
└── README.md        # This file
```

### Modular Design

**main.py**: Pure UI layer
- Streamlit components
- Page routing
- Session state management
- User interactions

**utils.py**: Pure Python logic
- Data generation
- Visualization
- Model loading
- Calculations
- No Streamlit dependencies (easily testable)

### Performance Optimizations

**Caching**:
```python
@st.cache_resource  # Cache model loading
def load_pretrained_model(model_path):
    ...

@st.cache_data  # Cache expensive computations
def generate_synthetic_convergence(...):
    ...
```

**Session State**:
```python
# Persist data across page navigation
if 'convergence_map' not in st.session_state:
    st.session_state.convergence_map = None
```

## Configuration

### Model Paths
By default, the app looks for pretrained models in:
```
results/models/best_pinn_model.pth
```

To use a different model, modify `utils.py`:
```python
def load_pretrained_model(model_path='path/to/your/model.pth'):
    ...
```

### Visualization Settings
Customize plots in `utils.py`:
```python
# Change colormap
plot_convergence_map(..., cmap='plasma')

# Adjust figure size
fig, ax = plt.subplots(figsize=(12, 8))
```

### Page Configuration
Modify page settings in `main.py`:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## Troubleshooting

### Common Issues

**1. "No module named 'streamlit'"**
```powershell
pip install streamlit>=1.28.0
```

**2. "Model file not found"**
- Check model path exists: `results/models/best_pinn_model.pth`
- Or train a model first using Phase 5 scripts

**3. "FITS file upload fails"**
- Ensure file is valid FITS format
- Check file size (< 200 MB recommended)
- Verify astropy is installed: `pip install astropy`

**4. "Slow performance"**
- Enable caching (should be automatic)
- Use GPU if available
- Reduce MC samples for uncertainty analysis
- Use smaller grid sizes (32 or 64 instead of 128)

**5. "Port 8501 already in use"**
```powershell
# Use different port
streamlit run app/main.py --server.port 8502
```

## Testing

Run the test suite to verify installation:

```powershell
# All Phase 10 tests
pytest tests/test_web_interface.py -v

# Specific test class
pytest tests/test_web_interface.py::TestSyntheticGeneration -v

# With coverage
pytest tests/test_web_interface.py --cov=app
```

Expected: **37/37 tests passing**

## Development

### Adding New Features

1. **Add utility function** in `utils.py`:
```python
def new_feature(...):
    """Docstring with examples."""
    # Pure Python implementation
    return result
```

2. **Add tests** in `tests/test_web_interface.py`:
```python
def test_new_feature():
    result = new_feature(...)
    assert result == expected
```

3. **Add UI** in `main.py`:
```python
def show_new_page():
    st.title("New Feature")
    # Use functions from utils.py
    result = new_feature(...)
    st.write(result)
```

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests for new functions
- Keep UI and logic separated

## Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Generate 64×64 map | 2-3s | NFW profile |
| Model inference | 0.1s | Single forward pass |
| MC Dropout (100) | 10s | Uncertainty quantification |
| FITS file loading | 1-2s | Depends on file size |
| Plot generation | 0.5-1s | Matplotlib rendering |

## Resources

- **Full Documentation**: `docs/Phase10_COMPLETE.md`
- **Summary**: `docs/Phase10_SUMMARY.md`
- **Tests**: `tests/test_web_interface.py`
- **Project README**: `readme.md` (project root)

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Ensure all tests pass

## License

See project LICENSE file in root directory.

## Acknowledgments

Built as Phase 10 of the Gravitational Lensing Analysis toolkit, integrating:
- Phases 1-4: Core physics and ray tracing
- Phase 5: PINN neural networks
- Phase 6: Advanced profiles
- Phase 7: GPU acceleration (450-1217× speedup)
- Phase 8: Real data support
- Phase 9: Transfer learning

## Support

For issues or questions:
- Check documentation in `docs/`
- Review test examples in `tests/`
- Verify installation with `pytest tests/test_web_interface.py`

---

**Status**: ✅ Production-ready | **Tests**: 37/37 passing | **Version**: 1.0.0
