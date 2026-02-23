"""Deprecated Streamlit entrypoint.

`app/main.py` is intentionally retained only to redirect users to:
    streamlit run app/Home.py

Current app organization:
app/
├── Home.py                 # Main launch page
├── main.py                 # Deprecated redirect page (this file)
├── styles.py               # Styling compatibility wrapper
├── core/                   # Reusable non-UI app services
│   ├── landing.py
│   └── web_utils.py
├── utils/                  # UI/session/demo helper modules
│   ├── ui.py
│   ├── session_state.py
│   └── demo_helpers.py
└── pages/                  # Feature pages
    ├── 02_Simple_Lensing.py
    ├── 03_PINN_Inference.py
    ├── 03_Results.py
    ├── 04_Multi_Plane.py
    ├── 05_Real_Data.py
    ├── 06_Training.py
    ├── 07_Validation.py
    ├── 08_Bayesian_UQ.py
    └── 09_Settings.py
"""

import streamlit as st

st.set_page_config(
    page_title="⚠️ Deprecated",
    page_icon="⚠️",
    layout="wide"
)

st.error("""
# ⚠️ DEPRECATION WARNING

This entry point (`app/main.py`) is **deprecated**.

## Use Instead:
```bash
streamlit run app/Home.py
```

## Why
The app now uses a modular multipage structure with shared core/services and reusable UI helpers.

## What to do
1. Stop the current Streamlit server (Ctrl+C)
2. Run: `streamlit run app/Home.py`
3. Update any scripts/bookmarks

See `README.md` for current architecture and usage.
""")

st.stop()
