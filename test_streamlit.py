"""
Minimal Streamlit test to verify basic functionality
"""
import streamlit as st

st.set_page_config(
    page_title="Test App",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 Test Application")
st.success("✅ Streamlit is working!")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select", ["Home", "Page 2"])

if page == "Home":
    st.write("This is the home page")
    st.metric("Test Metric", "100", "✓")
else:
    st.write("This is page 2")

st.button("Test Button")
