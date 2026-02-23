"""
UI Components and Styling Utilities.

Provides reusable UI components for the Streamlit application:
- Global CSS injection
- Header rendering
- Card components
- Styled status messages
- Download button helper
"""

from __future__ import annotations

from typing import Any

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --bg-0: #0A0E1A;
    --bg-1: #111A2C;
    --bg-2: #16213A;
    --card: rgba(17, 26, 44, 0.86);
    --card-strong: rgba(22, 33, 58, 0.92);

    --accent: #00D4FF;
    --accent-2: #7B2FBE;
    --accent-soft: rgba(0, 212, 255, 0.18);
    --success: #00FF88;

    --text-0: #EAF4FF;
    --text-1: #B6CAE6;
    --text-2: #7E94B3;

    --border: rgba(0, 212, 255, 0.28);
    --border-soft: rgba(123, 47, 190, 0.25);
    --shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    --shadow-soft: 0 8px 24px rgba(0, 0, 0, 0.28);

    --grad-main: linear-gradient(125deg, #00D4FF 0%, #4BA8FF 42%, #7B2FBE 100%);
    --grad-bg: radial-gradient(1200px 520px at -8% -12%, rgba(123, 47, 190, 0.28), transparent 60%),
               radial-gradient(980px 460px at 108% -6%, rgba(0, 212, 255, 0.24), transparent 62%),
               linear-gradient(180deg, #0A0E1A 0%, #0E1423 54%, #0B1020 100%);
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
    color: var(--text-0);
}

.stApp {
    background: var(--grad-bg);
    color: var(--text-0);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.45;
    background-image:
        radial-gradient(2px 2px at 14% 19%, rgba(255,255,255,0.7), transparent 65%),
        radial-gradient(1px 1px at 72% 16%, rgba(255,255,255,0.72), transparent 65%),
        radial-gradient(1px 1px at 80% 72%, rgba(255,255,255,0.6), transparent 64%),
        radial-gradient(2px 2px at 34% 76%, rgba(255,255,255,0.45), transparent 66%),
        radial-gradient(1px 1px at 50% 48%, rgba(255,255,255,0.52), transparent 62%);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    position: relative;
    z-index: 1;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(12, 20, 36, 0.98) 0%, rgba(10, 16, 30, 0.98) 100%);
    border-right: 1px solid var(--border-soft);
}

section[data-testid="stSidebar"] * {
    color: var(--text-0);
}

h1, h2, h3, h4 {
    font-family: "Space Grotesk", sans-serif;
    letter-spacing: 0.2px;
}

h1 {
    font-size: clamp(2rem, 3.6vw, 3rem);
}

h2 {
    font-size: clamp(1.35rem, 2.2vw, 2rem);
}

p, li, label, span {
    color: var(--text-1);
}

#MainMenu, footer, header {
    visibility: hidden;
}

@keyframes riseIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes haloPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.28); }
    50% { box-shadow: 0 0 0 10px rgba(0, 212, 255, 0.0); }
}

.app-header {
    border-radius: 22px;
    border: 1px solid var(--border);
    background:
        radial-gradient(720px 220px at 16% 14%, rgba(0, 212, 255, 0.26), transparent 58%),
        radial-gradient(640px 200px at 90% 90%, rgba(123, 47, 190, 0.28), transparent 60%),
        linear-gradient(145deg, rgba(14, 22, 39, 0.95) 0%, rgba(9, 15, 29, 0.95) 100%);
    box-shadow: var(--shadow);
    padding: 1.6rem 1.5rem;
    margin-bottom: 1.25rem;
    animation: riseIn 0.55s ease-out;
}

.app-header-content {
    position: relative;
}

.app-header-eyebrow {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    font-size: 0.74rem;
    margin-bottom: 0.65rem;
}

.app-header h1 {
    margin: 0;
    color: var(--text-0);
    line-height: 1.1;
}

.app-header-subtitle {
    margin-top: 0.65rem;
    max-width: 900px;
    font-size: 1rem;
    color: var(--text-1);
}

.app-header-badge {
    margin-top: 0.9rem;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.72rem;
    border-radius: 999px;
    border: 1px solid rgba(0, 212, 255, 0.46);
    background: rgba(0, 212, 255, 0.11);
    color: #D6F7FF;
    font-size: 0.82rem;
    font-weight: 600;
}

.custom-card {
    border-radius: 18px;
    border: 1px solid var(--border-soft);
    background: var(--card);
    backdrop-filter: blur(8px);
    padding: 1rem 1rem 0.95rem;
    box-shadow: var(--shadow-soft);
    transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    animation: riseIn 0.5s ease-out;
}

.custom-card:hover {
    transform: translateY(-3px);
    border-color: var(--border);
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.35);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-0);
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.card-body {
    color: var(--text-1);
    line-height: 1.6;
    font-size: 0.94rem;
}

div[data-testid="metric-container"] {
    border-radius: 15px;
    border: 1px solid var(--border-soft);
    background: linear-gradient(140deg, rgba(18, 30, 52, 0.85), rgba(13, 19, 35, 0.88));
    box-shadow: var(--shadow-soft);
    padding: 0.85rem 0.95rem;
}

div[data-testid="metric-container"] label {
    color: var(--text-2) !important;
    letter-spacing: 0.04em;
    font-weight: 700 !important;
    text-transform: uppercase;
    font-size: 0.74rem !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-0) !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

[data-testid="stButton"] > button {
    border-radius: 12px;
    border: 1px solid rgba(0, 212, 255, 0.45);
    background: linear-gradient(120deg, rgba(0, 212, 255, 0.2), rgba(123, 47, 190, 0.26));
    color: #EAF8FF;
    font-weight: 700;
    letter-spacing: 0.02em;
    transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    border-color: rgba(0, 212, 255, 0.9);
    box-shadow: 0 10px 24px rgba(0, 170, 255, 0.2);
}

[data-testid="stButton"] > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div,
[data-testid="stNumberInput"] > div,
[data-testid="stSlider"] > div,
[data-testid="stTextArea"] > div,
[data-testid="stFileUploader"] > div {
    border-radius: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    border: 1px solid var(--border-soft);
    background: rgba(15, 21, 37, 0.75);
    color: var(--text-1);
}

.stTabs [aria-selected="true"] {
    border-color: var(--accent) !important;
    background: rgba(0, 212, 255, 0.12) !important;
    color: var(--text-0) !important;
}

.streamlit-expanderHeader {
    border-radius: 12px !important;
    border: 1px solid var(--border-soft) !important;
    background: rgba(14, 20, 35, 0.8) !important;
}

.stAlert {
    border-radius: 12px;
}

.kpi-strip {
    border-radius: 14px;
    border: 1px solid rgba(0, 255, 136, 0.3);
    background: linear-gradient(110deg, rgba(0, 255, 136, 0.09), rgba(0, 212, 255, 0.08));
    padding: 0.7rem 0.95rem;
    margin: 0.25rem 0 1rem 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    align-items: center;
    justify-content: center;
    animation: haloPulse 2.8s ease-in-out infinite;
}

.kpi-pill {
    border: 1px solid rgba(0, 212, 255, 0.28);
    border-radius: 999px;
    padding: 0.3rem 0.68rem;
    font-size: 0.84rem;
    color: var(--text-0);
    background: rgba(12, 23, 39, 0.68);
}

.nebula-divider {
    height: 1px;
    width: 100%;
    margin: 1.2rem 0;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.46), rgba(123, 47, 190, 0.46), transparent);
}

@media (max-width: 900px) {
    .block-container {
        padding-top: 0.8rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .app-header {
        padding: 1.2rem 1rem;
        border-radius: 18px;
    }

    .app-header-subtitle {
        font-size: 0.95rem;
    }
}
</style>
"""


def inject_custom_css() -> None:
    """Inject shared application CSS into Streamlit pages."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "", badge: str = "") -> None:
    """Render a standardized hero header for app pages."""
    subtitle_html = f'<p class="app-header-subtitle">{subtitle}</p>' if subtitle else ""
    badge_html = f'<span class="app-header-badge">{badge}</span>' if badge else ""
    header_html = f"""
    <div class="app-header">
        <div class="app-header-content">
            <div class="app-header-eyebrow">Gravitational Lensing Toolkit</div>
            <h1>{title}</h1>
            {subtitle_html}
            {badge_html}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_card(title: str, content: str, icon: str = "") -> None:
    """Render a custom glass-style card component."""
    icon_html = f"<span>{icon}</span>" if icon else ""
    card_html = f"""
    <div class="custom-card">
        <div class="card-header">{icon_html}<span>{title}</span></div>
        <div class="card-body">{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def show_success(message: str) -> None:
    """Show success message with consistent styling."""
    st.success(message)


def show_warning(message: str) -> None:
    """Show warning message with consistent styling."""
    st.warning(message)


def show_info(message: str) -> None:
    """Show info message with consistent styling."""
    st.info(message)


def show_error(message: str) -> None:
    """Show error message with consistent styling."""
    st.error(message)


def create_download_button(
    data: Any,
    filename: str,
    button_text: str = "Download",
    mime_type: str = "text/plain",
) -> None:
    """Create a styled download button."""
    try:
        st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime=mime_type,
        )
    except Exception as exc:
        show_error(f"Failed to create download button: {exc}")


def create_parameter_summary(params: dict[str, Any]) -> str:
    """Create a formatted Markdown summary of parameters."""
    summary = "### Parameter Summary\n\n"
    for key, value in params.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            summary += f"- **{label}**: {value:.4f}\n"
        else:
            summary += f"- **{label}**: {value}\n"
    return summary
