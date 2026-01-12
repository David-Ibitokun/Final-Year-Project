"""
Prediction Interface Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Make Prediction",
    page_icon="🎯",
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

def focal_loss_fixed(gamma=2.0, alpha=0.25):
    """Custom focal loss function used during training"""
    def focal_loss(y_true, y_pred):
        import tensorflow.keras.backend as K
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss

def load_model(model_type):
    """Load the specified model"""
    model_path = Path(f"models/{model_type}_model.keras")
    try:
        # Load model with custom objects
        custom_objects = {'focal_loss_fixed': focal_loss_fixed()}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading {model_type} model: {e}")
        return None

# Main content
st.markdown('<h1 style="text-align: center; color: #2E7D32;">🎯 Make Crop Yield Prediction</h1>', unsafe_allow_html=True)

st.write("Select your parameters to predict crop yield:")

crop_suitability, regions = load_config()

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🌾 Crop & Location Selection")
    
    # Crop selection
    crops = ["Yam", "Cassava", "Maize"]
    selected_crop = st.selectbox("Select Crop", crops, key="crop_select")
    
    # Zone selection
    zones = ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
    selected_zone = st.selectbox("Select Geopolitical Zone", zones, key="zone_select")
    
    # Year
    selected_year = st.number_input("Year", min_value=2024, max_value=2050, value=2024, key="year_input")
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    model_type = st.radio(
        "Choose Model",
        ["cnn", "gru", "hybrid"],
        format_func=lambda x: {
            "cnn": "🔷 CNN (Convolutional Neural Network)",
            "gru": "🔶 GRU (Gated Recurrent Unit)",
                "hybrid": "⚡ Hybrid (CNN + GRU)"
        }[x],
        key="model_select"
    )

with col2:
    st.markdown("### 🌡️ Climate Variables")
    
    # Temperature
    temperature = st.slider(
        "Average Temperature (°C)",
        min_value=15.0,
        max_value=40.0,
        value=27.5,
        step=0.5,
        key="temp_slider"
    )
    
    # Rainfall
    rainfall = st.slider(
        "Annual Rainfall (mm)",
        min_value=200.0,
        max_value=3000.0,
        value=1200.0,
        step=50.0,
        key="rain_slider"
    )
    
    # Humidity
    humidity = st.slider(
        "Relative Humidity (%)",
        min_value=20.0,
        max_value=95.0,
        value=65.0,
        step=1.0,
        key="humid_slider"
    )
    
    # CO2
    co2 = st.slider(
        "CO₂ Concentration (ppm)",
        min_value=380.0,
        max_value=550.0,
        value=420.0,
        step=5.0,
        key="co2_slider"
    )

st.markdown("---")

# Prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("🎯 Predict Yield", type="primary", use_container_width=True)

if predict_button:
    with st.spinner(f"Loading {model_type.upper()} model and generating prediction..."):
        # Load model
        model = load_model(model_type)
        
        if model is not None:
            # Display prediction result
            st.success("✅ Prediction Complete!")
            
            # Create result display
            result_col1, result_col2, result_col3 = st.columns(3)
            
            # Simulate prediction (replace with actual model prediction)
            predicted_yield = np.random.uniform(0.5, 5.0)  # tons/hectare
            
            with result_col1:
                st.metric(
                    "Predicted Yield",
                    f"{predicted_yield:.2f} tons/ha",
                    delta="+0.3 vs baseline"
                )
            
            with result_col2:
                confidence = np.random.uniform(75, 95)
                st.metric(
                    "Model Confidence",
                    f"{confidence:.1f}%"
                )
            
            with result_col3:
                # Get suitability score if available
                suitability = "N/A"
                if crop_suitability and selected_zone in crop_suitability.get(selected_crop, {}):
                    suitability = crop_suitability[selected_crop][selected_zone]
                
                st.metric(
                    "Zone Suitability",
                    f"{suitability}/10" if suitability != "N/A" else suitability
                )
            
            st.markdown("---")
            
            # Additional information
            st.info(f"""
            **Prediction Summary:**
            - **Crop**: {selected_crop}
            - **Zone**: {selected_zone}
            - **Year**: {selected_year}
            - **Model**: {model_type.upper()}
            - **Climate Conditions**: {temperature}°C, {rainfall}mm rainfall, {humidity}% humidity
            
            ⚠️ *Note: This is a demonstration. Full prediction implementation requires preprocessing pipeline integration.*
            """)
            
        else:
            st.error("Failed to load model. Please check that model files exist in the 'models/' directory.")

# Display additional information
with st.expander("ℹ️ How to Use This Tool"):
    st.write("""
    **Step-by-step Guide:**
    
    1. **Select Crop**: Choose from 3 climate-resilient crops
    2. **Select Zone**: Pick the geopolitical zone of interest
    3. **Set Year**: Enter the prediction year
    4. **Choose Model**: Select CNN, GRU, or Hybrid
    5. **Adjust Climate**: Use sliders to set climate variables
    6. **Predict**: Click the predict button to generate forecast
    
    **Understanding Results:**
    - **Predicted Yield**: Expected output in tons per hectare
    - **Model Confidence**: Prediction certainty percentage
    - **Zone Suitability**: Crop-zone compatibility score (1-10)
    
    **Tips:**
    - Higher suitability scores typically indicate better yields
    - Hybrid model often provides most comprehensive predictions
    - Extreme climate values may reduce prediction accuracy
    """)
