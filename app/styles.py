"""
Enhanced Styling for Streamlit Application
Production-ready CSS with professional design
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* ========================================
       GLOBAL VARIABLES & THEME
       ======================================== */
    :root {
        --primary-blue: #667eea;
        --secondary-purple: #764ba2;
        --success-green: #10b981;
        --warning-yellow: #f59e0b;
        --danger-red: #ef4444;
        --info-cyan: #06b6d4;
        
        --bg-dark: #0e1117;
        --bg-card: #1e2130;
        --bg-hover: #262837;
        
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        
        --border-color: rgba(102, 126, 234, 0.2);
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* ========================================
       MAIN LAYOUT
       ======================================== */
    .main {
        padding: 1rem 2rem;
    }
    
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Better spacing for sections */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Cleaner headers */
    h1, h2, h3 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Better button spacing */
    .stButton {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Expander improvements */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--bg-hover) !important;
    }
    
    /* Info boxes - less intrusive */
    .stInfo, .stWarning, .stSuccess, .stError {
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Metrics improvements */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px !important;
        background-color: var(--bg-card) !important;
    }
    
    /* ========================================
       HEADER COMPONENTS
       ======================================== */
    .app-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-purple) 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.3;
        z-index: 0;
    }
    
    .app-header-content {
        position: relative;
        z-index: 1;
    }
    
    .app-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.5px;
    }
    
    .app-header-subtitle {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.15rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .app-header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        color: white;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* ========================================
       CARDS & CONTAINERS
       ======================================== */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-blue);
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-body {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* ========================================
       METRIC CARDS
       ======================================== */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid var(--border-color);
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary-blue);
    }
    
    div[data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* ========================================
       SIDEBAR STYLING
       ======================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-dark) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] .css-1d391kg {
        padding: 1rem 1rem 3rem 1rem;
    }
    
    /* Sidebar title */
    section[data-testid="stSidebar"] h1 {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.75rem !important;
        border-bottom: 2px solid var(--primary-blue);
    }
    
    /* Sidebar radio buttons */
    section[data-testid="stSidebar"] .row-widget.stRadio > div {
        gap: 0.5rem;
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label {
        background: rgba(102, 126, 234, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid transparent;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: var(--primary-blue);
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label[data-baseweb="radio"] {
        gap: 0.75rem;
    }
    
    /* ========================================
       BUTTONS
       ======================================== */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-purple) 100%);
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-md) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        filter: brightness(1.1);
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary buttons */
    .stButton.secondary button {
        background: rgba(102, 126, 234, 0.2) !important;
        border: 1px solid var(--primary-blue) !important;
    }
    
    /* Success buttons */
    .stButton.success button {
        background: linear-gradient(135deg, var(--success-green) 0%, #059669 100%) !important;
    }
    
    /* Danger buttons */
    .stButton.danger button {
        background: linear-gradient(135deg, var(--danger-red) 0%, #dc2626 100%) !important;
    }
    
    /* ========================================
       INPUTS & FORM ELEMENTS
       ======================================== */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Sliders */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary-blue) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 2px solid var(--primary-blue) !important;
        box-shadow: var(--shadow-md);
    }
    
    /* Checkboxes */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    
    .stCheckbox > label {
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .stCheckbox > label:hover {
        color: var(--primary-blue) !important;
    }
    
    /* ========================================
       TABS
       ======================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: var(--text-secondary);
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: var(--text-primary);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-purple) 100%);
        color: white;
    }
    
    /* ========================================
       ALERTS & MESSAGES
       ======================================== */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem 1.5rem !important;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border-left: 4px solid var(--success-green) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        border-left: 4px solid var(--warning-yellow) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid var(--danger-red) !important;
    }
    
    .stInfo {
        background: rgba(6, 182, 212, 0.15) !important;
        border-left: 4px solid var(--info-cyan) !important;
    }
    
    /* ========================================
       DATAFRAMES & TABLES
       ======================================== */
    .dataframe {
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead tr {
        background: var(--bg-card) !important;
    }
    
    .dataframe thead th {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        border-bottom: 2px solid var(--primary-blue) !important;
    }
    
    .dataframe tbody tr:hover {
        background: var(--bg-hover) !important;
    }
    
    .dataframe tbody td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    /* ========================================
       PLOTS & VISUALIZATIONS
       ======================================== */
    .stPlotlyChart, .stPyplot {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        background: var(--bg-card);
        box-shadow: var(--shadow-md);
    }
    
    /* ========================================
       EXPANDER
       ======================================== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-hover) !important;
        border-color: var(--primary-blue) !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        background: var(--bg-card) !important;
    }
    
    /* ========================================
       PROGRESS & SPINNERS
       ======================================== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--secondary-purple) 100%) !important;
    }
    
    .stSpinner > div {
        border-top-color: var(--primary-blue) !important;
    }
    
    /* ========================================
       FILE UPLOADER
       ======================================== */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        background: var(--bg-card);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-blue);
        background: var(--bg-hover);
    }
    
    .stFileUploader label {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* ========================================
       SCROLLBARS
       ======================================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-purple) 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-blue);
    }
    
    /* ========================================
       LOADING OVERLAYS
       ======================================== */
    .stSpinner {
        backdrop-filter: blur(5px);
    }
    
    /* ========================================
       RESPONSIVE DESIGN
       ======================================== */
    @media (max-width: 768px) {
        .app-header h1 {
            font-size: 1.75rem !important;
        }
        
        .app-header-subtitle {
            font-size: 1rem !important;
        }
        
        .block-container {
            padding: 1rem !important;
        }
        
        .custom-card {
            padding: 1rem !important;
        }
    }
    
    /* ========================================
       UTILITY CLASSES
       ======================================== */
    .text-center {
        text-align: center;
    }
    
    .text-muted {
        color: var(--text-muted) !important;
    }
    
    .text-primary {
        color: var(--primary-blue) !important;
    }
    
    .text-success {
        color: var(--success-green) !important;
    }
    
    .text-warning {
        color: var(--warning-yellow) !important;
    }
    
    .text-danger {
        color: var(--danger-red) !important;
    }
    
    .mb-1 { margin-bottom: 0.5rem; }
    .mb-2 { margin-bottom: 1rem; }
    .mb-3 { margin-bottom: 1.5rem; }
    .mb-4 { margin-bottom: 2rem; }
    
    .mt-1 { margin-top: 0.5rem; }
    .mt-2 { margin-top: 1rem; }
    .mt-3 { margin-top: 1.5rem; }
    .mt-4 { margin-top: 2rem; }
    
    .p-1 { padding: 0.5rem; }
    .p-2 { padding: 1rem; }
    .p-3 { padding: 1.5rem; }
    .p-4 { padding: 2rem; }
    
    /* ========================================
       ANIMATIONS
       ======================================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .animate-fadeIn {
        animation: fadeIn 0.5s ease-out;
    }
    
    .animate-slideIn {
        animation: slideIn 0.5s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* ========================================
       PRINT STYLES
       ======================================== */
    @media print {
        .stButton, section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        .block-container {
            max-width: 100% !important;
        }
    }
</style>
"""


def inject_custom_css():
    """Inject custom CSS into Streamlit app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "", badge: str = ""):
    """Render professional application header"""
    header_html = f"""
    <div class="app-header animate-fadeIn">
        <div class="app-header-content">
            <h1>🔭 {title}</h1>
            {f'<p class="app-header-subtitle">{subtitle}</p>' if subtitle else ''}
            {f'<span class="app-header-badge">{badge}</span>' if badge else ''}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_card(title: str, content: str, icon: str = "📊"):
    """Render custom card component"""
    card_html = f"""
    <div class="custom-card animate-fadeIn">
        <div class="card-header">
            <span>{icon}</span>
            <span>{title}</span>
        </div>
        <div class="card-body">
            {content}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
