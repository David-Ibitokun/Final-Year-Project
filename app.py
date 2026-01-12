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
    st.markdown('<h3 style="text-align: center; color: #888;">Your AI-powered tool for predicting crop yields in Nigeria</h3>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Welcome message
    st.markdown("""
    ### 👋 Hello!
    
    Welcome to the **Climate Change Impact on Food Security** prediction system. This application uses advanced 
    deep learning models to help you forecast crop yields across Nigeria's six geopolitical zones.
    
    Whether you're a researcher, policymaker, farmer, or student, this tool provides data-driven insights 
    to support decision-making in agriculture and food security planning.
    """)
    
    st.markdown("---")
    
    # Key capabilities
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌾 Crops", "3", help="Yam, Cassava, Maize")
    
    with col2:
        st.metric("🗺️ Zones", "6", help="All 6 geopolitical zones")
    
    with col3:
        st.metric("📅 Data Years", "34", help="1990-2023")
    
    with col4:
        st.metric("🤖 AI Models", "3", help="CNN, GRU, Hybrid")
    
    st.markdown("---")
    
    # Tutorial Section
    st.markdown('<h2 class="sub-header">📖 How to Use This App</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Follow these simple steps to get crop yield predictions:
    """)
    
    # Step-by-step tutorial
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Navigate", "2️⃣ Input Data", "3️⃣ Get Predictions", "4️⃣ Explore More"])
    
    with tab1:
        st.markdown("### Step 1: Navigate to Make Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("""
            **Getting Started:**
            
            1. Look at the **sidebar** on the left (👈)
            2. You'll see several pages listed:
               - 🏠 Home (you are here!)
               - 🎯 Make Prediction
               - 📊 Model Comparison
               - 🔍 Data Explorer
               - ℹ️ About
            3. Click on **"🎯 Make Prediction"** to start
            
            This is where you'll input your data and get yield forecasts.
            """)
        
        with col2:
            st.info("""
            **💡 Tip:**
            
            If you don't see the sidebar, click the **">"** arrow 
            in the top-left corner of the screen to expand it.
            """)
    
    with tab2:
        st.markdown("### Step 2: Input Your Data")
        
        st.write("""
        On the **Make Prediction** page, you'll find several input fields:
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📍 Location Settings:**")
            st.write("""
            - **Select Crop**: Choose from Yam, Cassava, or Maize
            - **Select Zone**: Pick one of Nigeria's 6 geopolitical zones
              - North-Central, North-East, North-West
              - South-East, South-South, South-West
            - **Year**: Enter the prediction year (2024-2050)
            """)
        
        with col2:
            st.markdown("**🌡️ Climate Variables:**")
            st.write("""
            Use the sliders to set:
            - **Temperature** (°C): Average temperature
            - **Rainfall** (mm): Annual rainfall amount
            - **Humidity** (%): Relative humidity level
            - **CO₂** (ppm): Carbon dioxide concentration
            
            *Don't worry if you're unsure - the sliders have 
            reasonable default values!*
            """)
        
        st.markdown("**🤖 Choose Your AI Model:**")
        st.write("""
        Select which model to use for prediction:
        - **🔷 CNN**: Fast, good for pattern recognition
        - **🔶 GRU**: Best for time series trends
        - **🔸 Hybrid**: Most accurate, combines both (Recommended)
        """)
    
    with tab3:
        st.markdown("### Step 3: Get Your Predictions")
        
        st.write("""
        Once you've set all your inputs:
        
        1. **Click the "🎯 Predict Yield" button**
        2. The AI model will process your inputs
        3. You'll see three key results:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Predicted Yield**")
            st.write("The forecasted crop yield in tons per hectare")
        
        with col2:
            st.markdown("**✅ Model Confidence**")
            st.write("How confident the model is about its prediction")
        
        with col3:
            st.markdown("**🌱 Zone Suitability**")
            st.write("How suitable the zone is for that crop (1-10)")
        
        st.success("""
        💡 **Understanding Results:**
        - Higher yields indicate better growing conditions
        - Higher confidence means more reliable predictions
        - Higher suitability scores suggest the crop is well-matched to the zone
        """)
    
    with tab4:
        st.markdown("### Step 4: Explore Additional Features")
        
        st.write("""
        After making predictions, explore these pages to learn more:
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📊 Model Comparison**")
            st.write("""
            - Compare performance of CNN, GRU, and Hybrid models
            - View accuracy metrics (MAE, RMSE, R²)
            - See side-by-side predictions
            - Understand model architectures
            """)
            
            st.markdown("**🔍 Data Explorer**")
            st.write("""
            - View dataset statistics and summaries
            - Explore climate trends over 34 years
            - Analyze crop yield patterns by zone
            - Visualize temperature, rainfall, CO₂ trends
            """)
        
        with col2:
            st.markdown("**ℹ️ About**")
            st.write("""
            - Learn about the research methodology
            - Understand data sources (NASA, NOAA, FAOSTAT)
            - Read about the regional scaling algorithm
            - See technology stack details
            """)
            
            st.info("""
            **💡 Pro Tip:**
            
            Try making multiple predictions with different 
            climate scenarios to see how changes in temperature 
            or rainfall might affect crop yields!
            """)
    
    st.markdown("---")
    
    # Quick start CTA
    st.markdown('<h2 class="sub-header">🚀 Ready to Get Started?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.success("""
        ### 👈 Click on "🎯 Make Prediction" in the sidebar to begin!
        
        Start forecasting crop yields with our AI-powered models in just a few clicks.
        """)
    
    st.markdown("---")
    
    # Additional info cards
    st.markdown('<h2 class="sub-header">🎯 What Makes This Tool Special?</h2>', unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("### 🎓 Research-Based")
        st.write("""
        Built on 34 years of historical data and validated 
        scientific methodologies for accurate predictions.
        """)
    
    with feature_col2:
        st.markdown("### 🤖 AI-Powered")
        st.write("""
        Uses state-of-the-art deep learning models (CNN, GRU, Hybrid) 
        for comprehensive yield forecasting.
        """)
    
    with feature_col3:
        st.markdown("### 🗺️ Region-Specific")
        st.write("""
        Provides zone-specific predictions tailored to Nigeria's 
        diverse agro-ecological conditions.
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p>Developed for Climate-Resilient Food Security Assessment | Final Year Project</p>
        <p>Need help? Check the ℹ️ About page for more information</p>
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
