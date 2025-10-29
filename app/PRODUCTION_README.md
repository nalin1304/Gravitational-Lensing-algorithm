# Production-Ready Streamlit Application

## 🎯 Overview

This is the production-ready Streamlit web application for the Gravitational Lensing Analysis Platform. The application provides an intuitive interface for gravitational lensing parameter inference, scientific validation, and Bayesian uncertainty quantification.

## ✨ Features

### Core Functionality
- **Synthetic Data Generation**: Create convergence maps for NFW and Elliptical NFW profiles
- **Real Data Analysis**: Upload and analyze FITS files from HST/JWST telescopes
- **ML Inference**: Physics-Informed Neural Networks (PINNs) for parameter prediction
- **Transfer Learning**: Domain adaptation from synthetic to real observational data

### Phase 15 Enhancements (Production-Ready)
- **Scientific Validation**: Publication-ready metrics with statistical tests
- **Bayesian UQ**: Monte Carlo Dropout for calibrated uncertainty estimates
- **Professional UI**: Modern, responsive design with custom styling
- **Error Handling**: Comprehensive error management and user feedback
- **Logging**: Structured logging for debugging and monitoring

## 📁 File Structure

```
app/
├── main.py              # Main Streamlit application (2,327 lines)
├── styles.py            # Production-ready CSS styling
├── error_handler.py     # Error handling and validation
├── utils.py             # Utility functions
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- All dependencies from `requirements.txt`
- Trained PINN models (from Phase 5)

### Launch Application

```powershell
# From project root
streamlit run app/main.py
```

The application will open in your browser at `http://localhost:8501`

### Docker Launch

```powershell
docker-compose up streamlit
```

## 📖 Usage Guide

### 1. Home Page
- Overview of all features
- Phase 15 highlights
- Quick start guide
- Dependency status check

### 2. Generate Synthetic Data
**Purpose:** Create training/validation data

**Steps:**
1. Select lens profile (NFW or Elliptical NFW)
2. Set parameters:
   - Mass: 1e14 - 1e15 M☉
   - Scale radius: 100-500 kpc
   - Grid size: 64-512 pixels
3. Generate convergence map
4. View 2D map and radial profile
5. Download results (NPZ or PNG)

**Validation:** Automatic parameter validation ensures physical values

### 3. Analyze Real Data
**Purpose:** Upload and process observational data

**Steps:**
1. Upload FITS file (HST, JWST, etc.)
2. Preview data with WCS information
3. Optional: Apply PSF convolution
4. Extract convergence map
5. Save processed data

**Supported formats:** `.fits`, `.fit`

### 4. ML Inference
**Purpose:** Predict lens parameters from convergence maps

**Steps:**
1. Upload convergence map (NPZ or FITS)
2. Select model (Standard PINN or Bayesian)
3. Run inference
4. View predictions with confidence intervals
5. Download results

**Models Available:**
- `pinn_model_final.pth` - Standard PINN
- `pinn_model_final_bayesian.pth` - Bayesian PINN

### 5. Scientific Validation
**Purpose:** Validate model predictions with publication-ready metrics

**Features:**
- Quick validation (< 0.01s)
- Rigorous statistical analysis
- NFW profile-specific tests
- Automated scientific reports
- Export validation results

**Metrics:**
- RMSE (Root Mean Square Error)
- SSIM (Structural Similarity Index)
- Correlation coefficients
- Profile comparison plots
- Statistical significance tests

### 6. Bayesian Uncertainty Quantification
**Purpose:** Calibrated uncertainty estimates for predictions

**Features:**
- Monte Carlo Dropout (100-500 samples)
- Prediction intervals (68%, 95%, 99%)
- Calibration plots
- Uncertainty visualization
- Confidence assessment

**Outputs:**
- Mean predictions
- Standard deviations
- Prediction intervals
- Calibration curves

### 7. Transfer Learning
**Purpose:** Demonstrate domain adaptation

**Features:**
- Fine-tune on simulated data
- Test on real observations
- Compare pre/post transfer performance
- Visualize improvement

## 🎨 UI/UX Features

### Professional Styling
- Modern gradient backgrounds
- Animated hover effects
- Responsive card layouts
- Color-coded status indicators
- Professional typography

### User Feedback
- Success/warning/error messages with emojis
- Loading spinners for long operations
- Progress bars for multi-step processes
- Validation feedback in real-time

### Error Handling
- Graceful error recovery
- Detailed error messages
- Debug information in expandable sections
- Automatic logging to `logs/` directory

## 🔧 Configuration

### Environment Variables
See `.env.example` for required configuration:

```env
# Required
MODEL_PATH=./models
DATA_PATH=./data

# Optional
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_UPLOAD_SIZE=200
```

### Streamlit Configuration
Located in `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200
enableCORS = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e2130"
textColor = "#ffffff"
font = "sans serif"
```

## 📊 Performance

### Benchmarks
- **Synthetic generation:** < 1s for 256×256 grid
- **ML inference:** < 2s per image
- **Validation:** < 0.01s (quick), < 1s (rigorous)
- **Bayesian UQ:** 5-10s for 100 MC samples

### Optimization
- `@st.cache_data` for expensive computations
- Lazy loading of models
- Vectorized NumPy operations
- GPU acceleration (if available)

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Run from project root directory

**2. Model Not Found**
```
FileNotFoundError: Model file not found
```
**Solution:** Ensure models are trained (see Phase 5 docs)

**3. Unicode Encoding Errors** (Windows)
```
UnicodeEncodeError: 'charmap' codec can't encode
```
**Solution:** Already fixed with UTF-8 wrapper in test scripts

**4. Cache Issues**
```
Stale predictions after model update
```
**Solution:** Click "🔄 Clear Cache" button in sidebar

### Debug Mode
Enable detailed logging:

```python
# In main.py
logging.basicConfig(level=logging.DEBUG)
```

Logs are saved to `logs/app_YYYYMMDD.log`

## 📝 Development

### Adding New Pages

1. **Create page function:**
```python
@handle_errors
def show_new_page():
    render_header(
        title="New Feature",
        subtitle="Description",
        badge="Beta"
    )
    
    # Your code here
    pass
```

2. **Add to sidebar navigation:**
```python
page = st.sidebar.radio(
    "Navigation",
    [..., "New Feature"]
)

if page == "New Feature":
    show_new_page()
```

3. **Add validation:**
```python
from app.error_handler import validate_positive_number, handle_errors

@handle_errors
def process_input(value):
    validate_positive_number(value, "Input Value")
    # Process...
```

### Custom Styling

Add styles to `app/styles.py`:

```python
CUSTOM_CSS += """
.my-custom-class {
    background: var(--primary-blue);
    padding: 1rem;
    border-radius: 8px;
}
"""
```

Use in app:

```python
st.markdown('<div class="my-custom-class">Content</div>', unsafe_allow_html=True)
```

## 🧪 Testing

Run Streamlit page tests:

```powershell
python scripts/test_streamlit_pages.py
```

Expected output:
```
✅ Test 1/6: Home page loads successfully
✅ Test 2/6: Generate Synthetic page loads
✅ Test 3/6: Analyze Real Data page loads
...
✅ All 6 tests passed!
```

## 📈 Production Deployment

### Docker Deployment
```powershell
# Build image
docker build -f Dockerfile.streamlit -t lensing-app:latest .

# Run container
docker run -p 8501:8501 lensing-app:latest
```

### Cloud Deployment
See [CLOUD_NATIVE_ROADMAP.md](../docs/CLOUD_NATIVE_ROADMAP.md) for AWS/Azure/GCP deployment

### Security Considerations
- Input validation on all user inputs
- File type verification for uploads
- Size limits on uploaded files (200MB)
- No sensitive data in session state
- HTTPS recommended for production

## 📚 Additional Resources

- **Phase 15 Summary:** [docs/Phase15_COMPLETE.md](../docs/Phase15_COMPLETE.md)
- **Bug Fixes:** [docs/Phase15_BugFixes_Summary.md](../docs/Phase15_BugFixes_Summary.md)
- **Configuration Guide:** [CONFIG_SETUP.md](../CONFIG_SETUP.md)
- **Docker Setup:** [DOCKER_SETUP.md](../DOCKER_SETUP.md)
- **API Documentation:** [api/README.md](../api/README.md)

## 🤝 Contributing

See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines on:
- Code style
- Commit messages
- Pull requests
- Testing requirements

## 📄 License

See [LICENSE](../LICENSE) for details.

## 🎯 Roadmap

### Planned Enhancements
- [ ] Multi-language support (i18n)
- [ ] Advanced plotting options
- [ ] Batch processing interface
- [ ] Real-time collaboration features
- [ ] Mobile-optimized views

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email:** support@example.com

---

**Last Updated:** December 2024  
**Version:** 1.0.0 (Phase 15 Complete)  
**Status:** ✅ Production Ready
