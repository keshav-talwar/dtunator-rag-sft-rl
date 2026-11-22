
import streamlit as st
from pathlib import Path
 
st.set_page_config(
    page_title="DTUnator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# Read and inject the HTML file
html_content = Path("index.html").read_text(encoding="utf-8")
st.markdown(html_content, unsafe_allow_html=True)
 
# Gives the page enough height for the HTML to breathe
st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
 