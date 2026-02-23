# App Module Guide

Streamlit frontend for the Gravitational Lensing Toolkit.

Launch:

```bash
streamlit run app/Home.py
```

## Purpose

The app layer is organized to keep scientific logic testable and UI logic readable.

1. `app/core/` contains reusable, non-UI helpers.
2. `app/utils/` contains UI/session/demo wiring.
3. `app/pages/` contains feature-specific Streamlit pages.
4. `app/Home.py` is the main landing page and navigation hub.

## Current Structure

```text
app/
├── Home.py                    # Primary app entrypoint
├── main.py                    # Deprecated redirect entrypoint
├── styles.py                  # Backward-compatible style wrapper
├── error_handler.py           # Shared exception handling utilities
├── core/
│   ├── __init__.py
│   ├── landing.py             # Home stats + demo preview generation
│   └── web_utils.py           # Testable scientific helper functions
├── utils/
│   ├── __init__.py
│   ├── demo_helpers.py        # One-click demo pipeline + export
│   ├── helpers.py             # Dependency and validation helpers
│   ├── plotting.py            # Plot builders used across pages
│   ├── session_state.py       # Session state schema and setup
│   └── ui.py                  # Global CSS + shared UI components
└── pages/
    ├── 02_Simple_Lensing.py
    ├── 03_PINN_Inference.py
    ├── 03_Results.py
    ├── 04_Multi_Plane.py
    ├── 05_Real_Data.py
    ├── 06_Training.py
    ├── 07_Validation.py
    ├── 08_Bayesian_UQ.py
    └── 09_Settings.py
```

## Import Rules

1. New scientific helper logic should go to `app/core/`.
2. New visual components or CSS helpers should go to `app/utils/ui.py`.
3. `app/pages/*` should orchestrate user input and call reusable helpers.
4. Keep `app/utils.py` and `app/styles.py` as compatibility shims only.

## Testing Notes

- App utility tests use `tests/test_web_interface.py`.
- Global regression guard remains the full suite:

```bash
python3 -m pytest tests/ -q
```
