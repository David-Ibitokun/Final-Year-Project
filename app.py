"""
Climate Change Impact on Food Security - Crop Yield Prediction System
Streamlit Application
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Crop Yield Prediction System\nDeep Learning for Climate-Resilient Food Security Assessment"
    }
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #0E1117; /* dark background to match Streamlit dark theme */
        color: #E6EEF3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_data
def load_config():
    with open('config/crop_zone_suitability_5crops.json', 'r') as f:
        crop_suitability = json.load(f)
    with open('config/regions_and_state.json', 'r') as f:
        regions = json.load(f)
    return crop_suitability, regions

def main():
    # Load configurations
    try:
        crop_suitability, regions = load_config()
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        crop_suitability, regions = {}, {}
    
    # Show home page
    show_home(crop_suitability, regions)

def show_home(crop_suitability, regions):
    # Welcome Section
    st.markdown('<h1 class="main-header" style="text-align: center;">🌾 Welcome to the Crop Yield Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #888;">AI-Powered Crop Yield Forecasting for Nigeria</h3>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; padding: 2rem;">
        <p>Welcome to the <strong>Climate Change Impact on Food Security</strong> prediction system.</p>
        <p>This application uses advanced deep learning models to forecast crop yields across Nigeria's six geopolitical zones.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key capabilities
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌾 Crops", "3", help="Maize, Cassava, Yams")
    
    with col2:
        st.metric("🗺️ Zones", "6", help="All 6 geopolitical zones")
    
    with col3:
        st.metric("📅 Data Years", "24", help="2000-2023")
    
    with col4:
        st.metric("🤖 AI Models", "3", help="CNN, GRU, Hybrid")
    
    st.markdown("---")
    
    # Quick access cards
    st.markdown('<h2 style="text-align: center; color: #ffffff; margin: 2rem 0;">Get Started</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #1A2230; border-radius: 10px; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h3>🎯 Make Prediction</h3>
            <p>Get crop yield forecasts using AI models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #1A2230; border-radius: 10px; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h3>📊 Model Comparison</h3>
            <p>Compare CNN, GRU, and Hybrid models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #1A2230; border-radius: 10px; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h3>🔍 Explore Data</h3>
            <p>Analyze climate and crop yield trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p><strong>Deep Learning for Climate-Resilient Food Security Assessment</strong></p>
        <p>Final Year Project | Bells University of Technology</p>
    </div>
    """, unsafe_allow_html=True)
def show_about():
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Research Background
    
    This project addresses the critical challenge of food security in Nigeria in the context of climate change. 
    By leveraging deep learning models, we can predict crop yields across different geopolitical zones, 
    helping policymakers and farmers make informed decisions.
    
    ### Methodology
    
    **Data Sources:**
    - 🌡️ Climate Data: NASA POWER API (Temperature, Rainfall, Humidity)
    - 🌍 CO₂ Data: NOAA Global Monitoring Laboratory (Mauna Loa)
    - 🌾 Crop Yields: FAOSTAT (Food and Agriculture Organization)
    - 🏞️ Soil Data: ISDA Soil API & Open-Meteo (Elevation)
    
    **Regional Scaling Algorithm:**
    ```
    Regional_Yield = National_Yield × Scaling_Factor
    Scaling_Factor = (0.7 × Suitability + 0.3 × Climate) × Noise(0.95, 1.05)
    ```
    
    ### Model Performance
    
    All three models (CNN, GRU, Hybrid) have been trained on 34 years of historical data 
    with comprehensive climate and soil features.
    
    ### Technology Stack
    
    - **Framework**: TensorFlow/Keras
    - **Interface**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib
    
    ### Contact & Repository
    
    For more information about this research project, please refer to the documentation 
    in the project repository.
    
    ---
    
    **Developed as part of Final Year Project**  
    *Deep Learning for Climate-Resilient Food Security Assessment*
    """)

if __name__ == "__main__":
    main()
