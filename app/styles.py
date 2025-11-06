"""
🌟 ULTRA-MODERN STYLING FOR STREAMLIT APPLICATION 🌟
Featuring: Animations, Particle Effects, Glassmorphism, Smooth Transitions
100x Better UI with Professional Design
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* ========================================
       🎨 GLOBAL VARIABLES & THEME
       ======================================== */
    :root {
        --primary-blue: #667eea;
        --secondary-purple: #764ba2;
        --accent-pink: #f093fb;
        --accent-cyan: #4facfe;
        --success-green: #10b981;
        --warning-yellow: #f59e0b;
        --danger-red: #ef4444;
        --info-cyan: #06b6d4;
        
        --bg-dark: #0a0e1a;
        --bg-card: rgba(30, 33, 48, 0.95);
        --bg-hover: #262837;
        --bg-glass: rgba(255, 255, 255, 0.05);
        
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        
        --border-color: rgba(102, 126, 234, 0.3);
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 8px 16px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 30px rgba(102, 126, 234, 0.5);
        
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --gradient-cosmic: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    /* ========================================
       🌌 ANIMATED BACKGROUND WITH STARS
       ======================================== */
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes slideInFromLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInFromRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInFromTop {
        from { transform: translateY(-100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideInFromBottom {
        from { transform: translateY(100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeInScale {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5), 0 0 10px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 0 0 40px rgba(102, 126, 234, 0.5); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Animated Starfield Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(2px 2px at 20% 30%, white, transparent),
            radial-gradient(2px 2px at 60% 70%, white, transparent),
            radial-gradient(1px 1px at 50% 50%, white, transparent),
            radial-gradient(1px 1px at 80% 10%, white, transparent),
            radial-gradient(2px 2px at 90% 60%, white, transparent),
            radial-gradient(1px 1px at 33% 80%, white, transparent),
            radial-gradient(2px 2px at 10% 90%, white, transparent);
        background-size: 200% 200%;
        animation: twinkle 4s ease-in-out infinite;
        opacity: 0.5;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Cosmic Gradient Overlay */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.05) 0%, 
            rgba(118, 75, 162, 0.05) 25%,
            rgba(240, 147, 251, 0.05) 50%,
            rgba(79, 172, 254, 0.05) 75%,
            rgba(102, 126, 234, 0.05) 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    /* ========================================
       🎯 MAIN LAYOUT WITH ANIMATIONS
       ======================================== */
    .main {
        padding: 1rem 2rem;
        position: relative;
        z-index: 1;
        animation: fadeInScale 0.8s ease-out;
    }
    
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        animation: slideInFromBottom 0.6s ease-out;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Better spacing for sections with animations */
    .stMarkdown {
        margin-bottom: 1rem;
        animation: fadeInScale 0.5s ease-out;
    }
    
    /* Animated headers with gradient text */
    h1 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        background: var(--gradient-cosmic);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite, slideInFromTop 0.6s ease-out;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        color: var(--text-primary) !important;
        animation: slideInFromLeft 0.6s ease-out;
        font-weight: 700 !important;
        position: relative;
        padding-left: 1rem;
    }
    
    h2::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 70%;
        background: var(--gradient-1);
        border-radius: 2px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    h3 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        color: var(--text-secondary) !important;
        animation: fadeInScale 0.5s ease-out;
        font-weight: 600 !important;
    }
    
    /* Animated buttons with hover effects */
    .stButton {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        animation: fadeInScale 0.5s ease-out;
    }
    
    /* Enhanced expander with glassmorphism */
    .streamlit-expanderHeader {
        background: var(--bg-glass) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        animation: slideInFromRight 0.5s ease-out;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-hover) !important;
        border-color: var(--primary-blue) !important;
        transform: translateX(5px);
        box-shadow: var(--shadow-glow);
    }
    
    /* Animated info boxes with icons */
    .stInfo, .stWarning, .stSuccess, .stError {
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px);
        animation: slideInFromLeft 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .stInfo::before, .stWarning::before, .stSuccess::before, .stError::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        animation: shimmer 2s infinite;
    }
    
    .stSuccess::before {
        background: var(--success-green);
    }
    
    .stWarning::before {
        background: var(--warning-yellow);
    }
    
    .stError::before {
        background: var(--danger-red);
    }
    
    .stInfo::before {
        background: var(--info-cyan);
    }
    
    /* Animated metrics with pulse effect */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        animation: pulse 3s ease-in-out infinite;
    }
    
    /* Code blocks with glassmorphism */
    .stCodeBlock {
        border-radius: 12px !important;
        background: var(--bg-glass) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--border-color) !important;
        animation: fadeInScale 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .stCodeBlock:hover {
        border-color: var(--primary-blue);
        box-shadow: var(--shadow-glow);
    }
    
    /* ========================================
       ✨ EPIC HEADER WITH PARTICLES & ANIMATIONS
       ======================================== */
    .app-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-purple) 50%, var(--accent-pink) 100%);
        background-size: 400% 400%;
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg), 0 0 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
        animation: gradientShift 10s ease infinite, slideInFromTop 0.8s ease-out;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animated particle pattern */
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.05) 0%, transparent 50%);
        animation: float 6s ease-in-out infinite;
        opacity: 0.6;
        z-index: 1;
    }
    
    /* Geometric pattern overlay */
    .app-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(255,255,255,.05) 35px, rgba(255,255,255,.05) 70px),
            repeating-linear-gradient(-45deg, transparent, transparent 35px, rgba(255,255,255,.05) 35px, rgba(255,255,255,.05) 70px);
        opacity: 0.3;
        z-index: 1;
    }
    
    .app-header-content {
        position: relative;
        z-index: 2;
    }
    
    .app-header h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), 0 0 30px rgba(255, 255, 255, 0.3) !important;
        letter-spacing: -1px !important;
        animation: slideInFromLeft 0.8s ease-out, pulse 4s ease-in-out infinite !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: white !important;
    }
    
    .app-header-subtitle {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.25rem !important;
        margin-top: 0.75rem !important;
        font-weight: 400 !important;
        animation: slideInFromRight 0.8s ease-out !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    .app-header-badge {
        display: inline-block !important;
        background: rgba(255, 255, 255, 0.25) !important;
        padding: 0.5rem 1.25rem !important;
        border-radius: 25px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: white !important;
        margin-top: 1.5rem !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        animation: fadeInScale 1s ease-out, glow 3s ease-in-out infinite !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .app-header-badge:hover {
        transform: scale(1.05) !important;
        background: rgba(255, 255, 255, 0.35) !important;
    }
    
    /* ========================================
       🎴 GLASSMORPHIC CARDS WITH ANIMATIONS
       ======================================== */
    .custom-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px) saturate(180%);
        border: 2px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .custom-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .custom-card:hover::before {
        left: 100%;
    }
    
    .custom-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-lg), var(--shadow-glow), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: var(--primary-blue);
    }
    
    .card-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        animation: slideInFromLeft 0.6s ease-out;
    }
    
    .card-header::after {
        content: '';
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-blue), transparent);
        border-radius: 2px;
    }
    
    .card-body {
        color: var(--text-secondary);
        line-height: 1.7;
        animation: fadeInScale 0.7s ease-out;
    }
    
    /* Icon animation in cards */
    .card-header span:first-child {
        display: inline-block;
        animation: bounce 2s ease-in-out infinite;
        font-size: 1.75rem;
    }
    
    /* ========================================
       📊 ANIMATED METRIC CARDS WITH GLOW
       ======================================== */
    div[data-testid="metric-container"] {
        background: var(--bg-glass);
        backdrop-filter: blur(15px) saturate(180%);
        border: 2px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: spin 20s linear infinite;
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    div[data-testid="metric-container"]:hover::before {
        opacity: 1;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        border-color: var(--primary-blue);
    }
    
    div[data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        animation: slideInFromTop 0.5s ease-out;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: transparent !important;
        background: var(--gradient-cosmic);
        background-size: 200% 200%;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        animation: gradientShift 6s ease infinite, pulse 3s ease-in-out infinite !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-weight: 700 !important;
        animation: bounce 2s ease-in-out infinite;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* ========================================
       🎨 EPIC ANIMATED SIDEBAR
       ======================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 33, 48, 0.98) 0%, rgba(10, 14, 26, 0.98) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        border-right: 2px solid var(--border-color);
        animation: slideInFromLeft 0.6s ease-out;
        box-shadow: 0 0 40px rgba(0, 0, 0, 0.5);
    }
    
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(180deg, 
            rgba(102, 126, 234, 0.05) 0%, 
            transparent 50%, 
            rgba(118, 75, 162, 0.05) 100%);
        pointer-events: none;
        z-index: 0;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar navigation title */
    section[data-testid="stSidebar"] h1 {
        color: var(--text-primary) !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        margin-bottom: 2rem !important;
        padding-bottom: 1rem !important;
        border-bottom: 3px solid transparent;
        border-image: var(--gradient-cosmic) 1;
        animation: slideInFromLeft 0.7s ease-out, glow 3s ease-in-out infinite !important;
        background: var(--gradient-cosmic) !important;
        background-size: 200% 200% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    /* Enhanced navigation items */
    section[data-testid="stSidebar"] .row-widget.stRadio > div {
        gap: 0.75rem;
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border: 2px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        animation: fadeInScale 0.4s ease-out;
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent);
        transition: left 0.5s;
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label:hover::before {
        left: 100%;
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: var(--primary-blue);
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Active navigation item */
    section[data-testid="stSidebar"] .row-widget.stRadio > div label[data-baseweb="radio"][data-checked="true"] {
        background: var(--gradient-1);
        border-color: var(--primary-blue);
        transform: translateX(8px);
        box-shadow: var(--shadow-glow);
    }
    
    section[data-testid="stSidebar"] .row-widget.stRadio > div label[data-baseweb="radio"] {
        gap: 1rem;
        font-weight: 600;
    }
    
    /* ========================================
       🚀 EPIC BUTTONS WITH ANIMATIONS
       ======================================== */
    .stButton button {
        background: var(--gradient-cosmic);
        background-size: 200% 200%;
        color: white !important;
        border: none !important;
        padding: 1rem 2.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--shadow-md), 0 0 20px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        position: relative !important;
        overflow: hidden !important;
        animation: fadeInScale 0.5s ease-out, gradientShift 6s ease infinite !important;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton button:hover {
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: var(--shadow-lg), var(--shadow-glow), 0 0 40px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton button:active {
        transform: translateY(-2px) scale(1.02) !important;
    }
    
    /* Animated icon in button */
    .stButton button span {
        position: relative;
        z-index: 1;
    }
    
    /* Secondary button style */
    .stButton.secondary button {
        background: rgba(102, 126, 234, 0.2) !important;
        border: 2px solid var(--primary-blue) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Success button with green gradient */
    .stButton.success button {
        background: linear-gradient(135deg, var(--success-green) 0%, #059669 50%, #047857 100%) !important;
        background-size: 200% 200% !important;
    }
    
    /* Danger button with red gradient */
    .stButton.danger button {
        background: linear-gradient(135deg, var(--danger-red) 0%, #dc2626 50%, #b91c1c 100%) !important;
        background-size: 200% 200% !important;
    }
    
    /* Download button special animation */
    .stDownloadButton button {
        animation: pulse 2s ease-in-out infinite !important;
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
