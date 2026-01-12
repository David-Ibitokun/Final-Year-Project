"""
Climate Change Impact on Food Security - Crop Yield Prediction System
Streamlit Application
"""

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
    initial_sidebar_state="expanded"
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
    # Sidebar navigation
    st.sidebar.title("🌾 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Make Prediction", "Model Comparison", "Data Explorer", "About"]
    )
    
    # Load configurations
    try:
        crop_suitability, regions = load_config()
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        crop_suitability, regions = {}, {}
    
    # Route to selected page
    if page == "Home":
        show_home(crop_suitability, regions)
    elif page == "Make Prediction":
        show_prediction_page(crop_suitability, regions)
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "About":
        show_about()

def show_home(crop_suitability, regions):
    st.markdown('<h1 class="main-header">🌾 Climate Change Impact on Food Security in Nigeria</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Deep Learning-Based Crop Yield Prediction System</h3>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Crops Analyzed", "5", help="Millet, Sorghum, Groundnuts, Oil Palm, Cocoa")
    
    with col2:
        st.metric("Geopolitical Zones", "6", help="All 6 zones of Nigeria")
    
    with col3:
        st.metric("Years of Data", "34", help="1990-2023")
    
    with col4:
        st.metric("Model Types", "3", help="CNN, GRU, Hybrid")
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 Project Overview</h2>', unsafe_allow_html=True)
        st.write("""
        This system uses **deep learning** to predict crop yields across Nigeria's geopolitical zones, 
        helping assess the impact of climate change on food security.
        
        **Key Features:**
        - 🎯 **Regional Predictions**: Zone-specific yield forecasts
        - 🌡️ **Climate Integration**: Temperature, rainfall, humidity, CO₂
        - 🌱 **Multiple Crops**: 5 strategically selected crops
        - 🤖 **Advanced AI**: CNN, GRU, and Hybrid models
        - 📈 **34 Years Analysis**: Historical trends from 1990-2023
        """)
        
        st.info("👈 Use the sidebar to navigate to the **Make Prediction** page to get started!")
    
    with col2:
        st.markdown('<h2 class="sub-header">🌾 Selected Crops</h2>', unsafe_allow_html=True)
        crops = ["Yam", "Cassava", "Maize"]
        # Use crop names from configuration if available; fallback to defaults
        for crop in crops:
            st.write(f"✓ {crop}")
        
        st.markdown('<h2 class="sub-header" style="margin-top: 2rem;">🗺️ Coverage</h2>', unsafe_allow_html=True)
        zones = ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
        for zone in zones:
            st.write(f"• {zone}")
    
    st.markdown("---")
    
    # Model Architecture Overview
    st.markdown('<h2 class="sub-header">🧠 Model Architectures</h2>', unsafe_allow_html=True)
    
    model_col1, model_col2, model_col3 = st.columns(3)
    
    with model_col1:
        st.markdown("### 🔷 CNN Model")
        st.write("""
        **Convolutional Neural Network**
        - Spatial feature extraction
        - Best for: Pattern recognition
        - Layers: Conv1D + Dense
        """)
    
    with model_col2:
        st.markdown("### 🔶 GRU Model")
        st.write("""
        **Gated Recurrent Unit**
        - Temporal sequence learning
        - Best for: Time series trends
        - Layers: GRU + Dense
        """)
    
    with model_col3:
        st.markdown("### 🔸 Hybrid Model")
        st.write("""
        **Combined Architecture**
        - Spatial + Temporal features
        - Best for: Comprehensive analysis
        - Layers: CNN + GRU + Dense
        """)

def show_prediction_page(crop_suitability, regions):
    st.markdown('<h1 class="main-header">🎯 Make Crop Yield Prediction</h1>', unsafe_allow_html=True)
    
    st.write("Select your parameters to predict crop yield:")
    
    # Import the prediction page
    try:
        from pages import prediction
        prediction.show_prediction_interface(crop_suitability, regions)
    except ImportError:
        st.error("Prediction module not found. Please ensure all files are properly set up.")

def show_model_comparison():
    st.markdown('<h1 class="main-header">📊 Model Comparison</h1>', unsafe_allow_html=True)
    
    try:
        from pages import comparison
        comparison.show_comparison()
    except ImportError:
        st.info("Model comparison feature coming soon!")

def show_data_explorer():
    st.markdown('<h1 class="main-header">🔍 Data Explorer</h1>', unsafe_allow_html=True)
    
    try:
        from pages import explorer
        explorer.show_data_explorer()
    except ImportError:
        st.info("Data explorer feature coming soon!")

def show_about():
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Research Background
    
    This project addresses the critical challenge of food security in Nigeria in the context of climate change. 
    By leveraging deep learning models, we can predict crop yields across different geopolitical zones, 
    helping policymakers and farmers make informed decisions.
    
    ### Methodology
    
    **Data Sources:**
    - 🌡️ Climate Data: NASA POWER API
    - 🌍 CO₂ Data: NOAA ESRL
    - 🌾 Crop Yields: FAOSTAT
    - 🏞️ Soil Data: ISDA Soil API
    
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
