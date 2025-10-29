# Real Gravitational Lensing Data Sources

**Last Updated**: October 11, 2025  
**Purpose**: Comprehensive guide to accessing real gravitational lensing observations for analysis

---

## 🌟 Major Data Archives

### 1. Hubble Space Telescope (HST) - **RECOMMENDED**

**Best for**: High-resolution strong lensing systems, galaxy clusters

#### HST Legacy Archive (HLA)
- **URL**: https://hla.stsci.edu/
- **Data Format**: FITS files
- **Resolution**: 0.03-0.1 arcsec/pixel
- **Filters**: Multiple (optical/IR)

**Quick Access**:
1. Go to: https://hla.stsci.edu/hlaview.html
2. Search by target name or coordinates
3. Filter by "Gravitational Lens" in keywords
4. Download FITS files directly

**Popular Targets**:
| Target | Coordinates | Description | HST Program |
|--------|-------------|-------------|-------------|
| **Abell 2744** | RA: 00:14:20, Dec: -30:23:50 | Pandora's Cluster - massive cluster lens | Multiple |
| **MACS J0416.1-2403** | RA: 04:16:09, Dec: -24:04:03 | Frontier Fields cluster | 13495, 13504 |
| **Einstein Cross** | RA: 22:40:30, Dec: +03:21:30 | Quad-lensed quasar | 5404, 6003 |
| **Horseshoe Lens** | RA: 11:49:36, Dec: +38:00:06 | SDSS J1148+3845 | 12190 |
| **Cosmic Eye** | RA: 21:35:12, Dec: -01:01:02 | J2135-0102 | 11591 |

**Direct Download Links**:
```bash
# Abell 2744 (Pandora's Cluster)
wget https://archive.stsci.edu/pub/hlsp/frontier/abell2744/images/hst_13495_07_acs_wfc_f606w_drz.fits

# Einstein Cross
wget https://hst.esac.esa.int/ehst-sl-server/servlet/data-action?RETRIEVAL_TYPE=PRODUCT&OBSERVATION_ID=o5ux01030
```

---

### 2. James Webb Space Telescope (JWST) - **LATEST**

**Best for**: High-redshift lensed galaxies, infrared imaging

#### MAST Archive
- **URL**: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
- **Data Format**: FITS files
- **Resolution**: 0.03-0.1 arcsec/pixel (NIRCam)
- **Wavelengths**: 0.6-28 μm

**Search Instructions**:
1. Go to: https://mast.stsci.edu/
2. Navigate to "JWST" missions
3. Search criteria:
   - Mission: JWST
   - Instrument: NIRCam or MIRI
   - Keyword: "gravitational lens" OR "lensing"
4. Download Level 2 (calibrated) or Level 3 (combined) data

**Featured Programs**:
- **GLASS-JWST** (Grism Lens-Amplified Survey): Abell 2744
- **PEARLS** (Prime Extragalactic Areas): Multiple clusters
- **CANUCS** (CAnadian NIRISS Unbiased Cluster Survey)

---

### 3. Sloan Digital Sky Survey (SDSS) - **LARGE SAMPLES**

**Best for**: Statistical studies, quad lenses, galaxy-scale lenses

#### SDSS Science Archive Server (SAS)
- **URL**: https://data.sdss.org/
- **Data Format**: FITS images, spectroscopy
- **Catalog URL**: https://www.sdss.org/dr17/

**Lens Catalog Access**:
```python
from astroquery.sdss import SDSS
from astropy import coordinates as coords

# Query known lens systems
ra = 180.0  # degrees
dec = 45.0
search_radius = '1d'  # 1 degree

result = SDSS.query_region(
    coords.SkyCoord(ra, dec, unit='deg'),
    radius=search_radius,
    spectro=True
)

# Download FITS images
images = SDSS.get_images(matches=result, band='r')
```

**Known SDSS Lens Catalogs**:
- **SLACS** (SDSS Legacy Survey): 85 galaxy-scale lenses
  - Catalog: http://www.slacs.org/
  - Download: http://www.slacs.org/DataRelease.html
  
- **BELLS** (BOSS Emission-Line Lens Survey): 25 lenses
  - Paper: https://arxiv.org/abs/1201.2988

---

### 4. Euclid Space Telescope - **NEW**

**Best for**: Wide-field surveys, weak lensing

- **URL**: https://www.cosmos.esa.int/web/euclid
- **Status**: Early Release Observations (ERO) available
- **Data**: https://www.euclid-ec.org/data-access/

---

### 5. ALMA (Atacama Large Millimeter Array)

**Best for**: Sub-millimeter lensed sources, high-z galaxies

- **URL**: https://almascience.org/
- **Archive**: https://almascience.nrao.edu/aq/
- **Search**: "gravitational lens" in science keywords

---

## 📊 Curated Lens Catalogs

### MasterLens Database
- **URL**: https://masterlens.org/
- **Content**: Comprehensive catalog of ~1000 known lenses
- **Features**: Images, redshifts, lens models
- **Format**: Downloadable FITS, CSV

### CASTLES (CfA-Arizona Space Telescope LEns Survey)
- **URL**: https://www.cfa.harvard.edu/castles/
- **Content**: 100+ lensed quasars and galaxies
- **Data**: HST images, lens models, literature
- **Status**: Legacy archive (pre-2010)

### BELLS GALLERY
- **URL**: https://www.slacs.org/bells.html
- **Content**: SDSS BOSS galaxy lenses
- **Format**: Images + spectra

---

## 🔬 Specific FITS File Examples

### Download Ready-to-Use Data

```bash
# Create data directory
mkdir -p data/raw/hst
mkdir -p data/raw/jwst
mkdir -p data/raw/sdss

# 1. Einstein Cross (HST/WFC3)
wget -O data/raw/hst/einstein_cross.fits \
  "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/o5ux01030_drz.fits"

# 2. Abell 2744 (JWST/NIRCam)
# Visit: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
# Search: Abell 2744, JWST, NIRCam, F200W
# Download via web interface (requires account)

# 3. SLACS Lens (SDSS)
wget -O data/raw/sdss/slacs_example.fits \
  "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/3063/6/frame-r-003063-6-0115.fits.bz2"
bunzip2 data/raw/sdss/slacs_example.fits.bz2
```

---

## 🐍 Python Code to Download and Load

### Example 1: HST Data via Astroquery

```python
from astroquery.mast import Observations
import astropy.units as u
from astropy.coordinates import SkyCoord

# Target: Einstein Cross
target = "Q2237+0305"  # Einstein Cross
coords = SkyCoord("22h40m30.3s +03d21m30.3s", frame='icrs')

# Query HST observations
obs_table = Observations.query_object(target, radius=0.1*u.deg)

# Filter for HST
hst_obs = obs_table[obs_table['obs_collection'] == 'HST']

# Get data products
data_products = Observations.get_product_list(hst_obs)

# Filter for science FITS files
fits_products = data_products[data_products['productType'] == 'SCIENCE']
fits_products = fits_products[fits_products['dataproduct_type'] == 'image']

# Download (first 5)
manifest = Observations.download_products(
    fits_products[:5],
    download_dir='data/raw/hst/'
)

print(f"Downloaded {len(manifest)} files to data/raw/hst/")
```

### Example 2: SDSS Data via Astroquery

```python
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.io import fits

# SLACS lens coordinates (example)
ra, dec = 177.669, 52.832  # SDSS J1150+5244
coord = coords.SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

# Get images
images = SDSS.get_images(coordinates=coord, band='r', radius=20*u.arcsec)

# Save FITS file
if images:
    fits_file = images[0]
    fits_file.writeto('data/raw/sdss/slacs_j1150.fits', overwrite=True)
    print("Downloaded SDSS image")
    
    # Display info
    with fits.open('data/raw/sdss/slacs_j1150.fits') as hdul:
        hdul.info()
        data = hdul[0].data
        print(f"Image shape: {data.shape}")
```

### Example 3: Load into Streamlit App

```python
# In your Streamlit app
from astropy.io import fits
import numpy as np
import streamlit as st

uploaded_file = st.file_uploader("Upload FITS file", type=['fits', 'fit'])

if uploaded_file:
    # Read FITS
    with fits.open(uploaded_file) as hdul:
        # Get primary image data
        data = hdul[0].data
        header = hdul[0].header
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Normalize
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Display
        st.image(data, caption="Loaded FITS Image", clamp=True)
        st.json(dict(header))
```

---

## 📖 Recommended Datasets for ISEF

### Beginner-Friendly

1. **Einstein Cross (Q2237+0305)**
   - Clear quad lens
   - Well-studied
   - Easy to model
   - **Download**: Search "Q2237" in HST Archive

2. **SDSS J1148+3845 (Horseshoe)**
   - Giant arc
   - Beautiful visual
   - **Download**: HST Program 12190

### Intermediate

3. **Abell 2744 (Pandora's Cluster)**
   - Multiple lenses
   - Complex mass distribution
   - JWST + HST data available
   - **Download**: Frontier Fields website

4. **MACS J0416.1-2403**
   - Merging cluster
   - Multiple arcs
   - **Download**: Frontier Fields

### Advanced

5. **SLACS Sample (85 lenses)**
   - Statistical study
   - Galaxy-scale lenses
   - Velocity dispersion data
   - **Download**: http://www.slacs.org/DataRelease.html

---

## 🛠️ Data Processing Tips

### Preprocessing Real Data

```python
from src.data.real_data_loader import FITSDataLoader, preprocess_real_data
import numpy as np

# Load FITS file
loader = FITSDataLoader('data/raw/hst/einstein_cross.fits')
data, metadata = loader.load_and_validate()

# Preprocess for ML
processed = preprocess_real_data(
    data,
    target_size=(64, 64),
    normalize=True,
    remove_nan=True,
    clip_outliers=True,
    clip_sigma=3.0
)

# Now ready for model inference
```

---

## 🌐 Online Resources

### Interactive Databases

1. **ESA Sky**: https://sky.esa.int/
   - Browse all major telescopes
   - Search by coordinates
   - Direct FITS download

2. **Aladin Desktop**: https://aladin.u-strasbg.fr/
   - Desktop app
   - Multi-wavelength viewer
   - Direct archive access

3. **DS9**: https://sites.google.com/cfa.harvard.edu/saoimageds9
   - Professional FITS viewer
   - Analysis tools

### Publications & Catalogs

1. **NASA/IPAC Extragalactic Database (NED)**
   - URL: https://ned.ipac.caltech.edu/
   - Search: "gravitational lens"
   - Get coordinates, references

2. **SIMBAD Astronomical Database**
   - URL: http://simbad.u-strasbg.fr/
   - Object type: "GravLens"

---

## 📥 Quick Start Commands

### Full Pipeline to Get HST Data

```bash
# 1. Install dependencies
pip install astropy astroquery

# 2. Download script
cat > download_lens_data.py << 'EOF'
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u

# Einstein Cross
coords = SkyCoord("22h40m30.3s +03d21m30.3s", frame='icrs')
obs = Observations.query_object("Einstein Cross", radius=0.1*u.deg)
hst = obs[obs['obs_collection'] == 'HST'][:5]
products = Observations.get_product_list(hst)
Observations.download_products(products, download_dir='data/raw/hst/')
print("✅ Download complete!")
EOF

# 3. Run download
python download_lens_data.py

# 4. Verify
ls -lh data/raw/hst/
```

---

## 🎯 For Your ISEF Project

**Recommended Approach**:

1. **Start Simple**: Einstein Cross (single FITS file)
2. **Demonstrate**: Load in Streamlit → Preprocess → Visualize
3. **Analyze**: Apply your PINN model for mass reconstruction
4. **Compare**: Model vs known lens parameters
5. **Present**: Show real data alongside synthetic examples

**Data Requirements**:
- ✅ 2-3 clear lensing examples (Einstein Cross, Horseshoe, Abell 2744)
- ✅ FITS files (HST preferred)
- ✅ Known lens parameters for validation
- ✅ High-resolution (>0.1 arcsec/pixel)

**Your app already supports FITS loading on the "Analyze Real Data" page! 🎉**

---

## 📞 Support & Help

**If download fails**:
1. Check internet connection
2. Create free account at: https://archive.stsci.edu/
3. Use web interface instead of API
4. Try alternative archives (ESA, SDSS)

**Data format issues**:
- Use `astropy.io.fits` for all FITS files
- Check `hdul.info()` for structure
- Try different extensions: `hdul[0].data`, `hdul[1].data`

**Questions?**
- HST Helpdesk: https://hsthelp.stsci.edu/
- Astroquery Docs: https://astroquery.readthedocs.io/
- SDSS Forum: https://www.sdss.org/contact/

---

**You now have access to terabytes of real gravitational lensing data! 🚀**
