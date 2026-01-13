"""
Home Page - Crop Yield Prediction System
"""

import streamlit as st
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Home - Crop Yield Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load configuration
@st.cache_data
def load_config():
    try:
        with open('config/crop_zone_suitability_5crops.json', 'r') as f:
            crop_suitability = json.load(f)
        with open('config/regions_and_state.json', 'r') as f:
            regions = json.load(f)
        return crop_suitability, regions
    except:
        return {}, {}

crop_suitability, regions = load_config()

# Main content
st.markdown('<h1 class="main-header" style="text-align: center;">🌾 Climate Change Impact on Food Security in Nigeria</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Deep Learning-Based Crop Yield Prediction System</h3>', unsafe_allow_html=True)
    
st.markdown("---")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Crops Analyzed", "3", help="Yam, Cassava, Maize")

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
    st.markdown('<h2 style="font-size: 1.5rem; color: #ffffff;">📊 Project Overview</h2>', unsafe_allow_html=True)
    st.write("""
    This system uses **deep learning** to predict crop yields across Nigeria's geopolitical zones, 
    helping assess the impact of climate change on food security.
    
    **Key Features:**
    - 🎯 **Regional Predictions**: Zone-specific yield forecasts
    - 🌡️ **Climate Integration**: Temperature, rainfall, humidity, CO₂
    - 🌱 **Multiple Crops**: 3 strategically selected crops (Yam, Cassava, Maize)
    - 🤖 **Advanced AI**: CNN, GRU, and Hybrid models
    - 📈 **34 Years Analysis**: Historical trends from 1990-2023
    """)
    
    # st.info("👈 Use the sidebar above to navigate to **Make Prediction** to get started!")

with col2:    
    st.markdown('<h2 style="font-size: 1.5rem; color: #ffffff; margin-top: 2rem;">🗺️ Coverage</h2>', unsafe_allow_html=True)
    zones = ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
    for zone in zones:
        st.write(f"• {zone}")

st.markdown("---")

# Model Architecture Overview
st.markdown('<h2 style="font-size: 1.5rem; color: #ffffff;">🧠 Model Architectures</h2>', unsafe_allow_html=True)

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
    st.markdown("### ⚡ Hybrid Model")
    st.write("""
    **Combined Architecture**
    - Spatial + Temporal features
    - Best for: Comprehensive analysis
    - Layers: CNN + GRU + Dense
    """)

st.markdown("---")

# Quick Start Guide
st.markdown('<h2 style="font-size: 1.5rem; color: #ffffff;">🚀 Quick Start Guide</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 1️⃣ Make Predictions")
    st.write("""
    Navigate to **Make Prediction** page to:
    - Select crop and zone
    - Input climate variables
    - Get yield forecasts
    """)

with col2:
    st.markdown("#### 2️⃣ Compare Models")
    st.write("""
    Visit **Model Comparison** to:
    - View performance metrics
    - Compare predictions
    - Understand architectures
    """)

with col3:
    st.markdown("#### 3️⃣ Explore Data")
    st.write("""
    Check **Data Explorer** to:
    - View dataset statistics
    - Analyze climate trends
    - Explore crop patterns
    """)
