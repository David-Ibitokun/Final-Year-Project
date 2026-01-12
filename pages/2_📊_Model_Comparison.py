"""
Model Comparison Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.markdown('<h1 style="text-align: center; color: #ffffff;">📊 Model Comparison</h1>', unsafe_allow_html=True)

st.write("""
Compare the performance of CNN, GRU, and Hybrid models across different metrics and scenarios.
""")

# Tabs for different comparison views
tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "📈 Prediction Comparison", "⚙️ Model Details"])

with tab1:
    st.markdown("### Model Performance Comparison")
    
    # Simulated metrics (replace with actual metrics from your trained models)
    metrics_data = {
        "Model": ["CNN", "GRU", "Hybrid"],
        "MAE": [0.45, 0.38, 0.32],
        "RMSE": [0.62, 0.53, 0.47],
        "R² Score": [0.82, 0.87, 0.91],
        "Training Time (min)": [15, 22, 35]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(
        df_metrics.style.highlight_min(subset=["MAE", "RMSE", "Training Time (min)"], color="lightgreen")
                      .highlight_max(subset=["R² Score"], color="lightgreen"),
        use_container_width=True
    )
    
    # Visualize metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # MAE and RMSE comparison
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name="MAE", x=df_metrics["Model"], y=df_metrics["MAE"]))
        fig1.add_trace(go.Bar(name="RMSE", x=df_metrics["Model"], y=df_metrics["RMSE"]))
        fig1.update_layout(
            title="Error Metrics Comparison (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="Error Value",
            barmode="group"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # R² Score comparison
        fig2 = go.Figure(data=[
            go.Bar(x=df_metrics["Model"], y=df_metrics["R² Score"],
                   marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ])
        fig2.update_layout(
            title="R² Score Comparison (Higher is Better)",
            xaxis_title="Model",
            yaxis_title="R² Score"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    st.markdown("### 🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🔷 CNN Model**
        - Fast training time
        - Good for spatial patterns
        - Best for: Quick predictions
        """)
    
    with col2:
        st.info("""
        **🔶 GRU Model**
        - Excellent temporal modeling
        - Balanced performance
        - Best for: Time series trends
        """)
    
    with col3:
        st.success("""
        **⚡ Hybrid Model**
        - Highest accuracy
        - Comprehensive feature learning
        - Best for: Critical decisions
        """)

with tab2:
    st.markdown("### Comparative Prediction Analysis")
    
    st.write("Compare how different models predict yields under the same conditions.")
    
    # Sample comparison scenario
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Scenario Settings")
        crop = st.selectbox("Crop", ["Yam", "Cassava", "Maize"])
        zone = st.selectbox("Zone", ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"])
        temperature = st.slider("Temperature (°C)", 20.0, 35.0, 27.5)
        rainfall = st.slider("Rainfall (mm)", 500.0, 2500.0, 1200.0)
    
    with col2:
        st.markdown("#### Model Predictions")
        
        # Simulated predictions (replace with actual model predictions)
        predictions = {
            "CNN": np.random.uniform(2.0, 4.5),
            "GRU": np.random.uniform(2.0, 4.5),
            "Hybrid": np.random.uniform(2.0, 4.5)
        }
        
        # Create comparison chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(predictions.keys()),
                y=list(predictions.values()),
                text=[f"{v:.2f} tons/ha" for v in predictions.values()],
                textposition="auto",
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
        ])
        
        fig.update_layout(
            title=f"Yield Predictions for {crop} in {zone}",
            xaxis_title="Model",
            yaxis_title="Predicted Yield (tons/ha)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        avg_pred = np.mean(list(predictions.values()))
        std_pred = np.std(list(predictions.values()))
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Average Prediction", f"{avg_pred:.2f} tons/ha")
        with metric_col2:
            st.metric("Std Deviation", f"{std_pred:.3f}")

with tab3:
    st.markdown("### Model Architecture Details")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model to View",
        ["CNN", "GRU", "Hybrid"]
    )
    
    if model_choice == "CNN":
        st.markdown("#### 🔷 CNN (Convolutional Neural Network)")
        st.write("""
        **Architecture:**
        ```
        Input Layer (Features: 20-35)
        ↓
        Conv1D (64 filters, kernel_size=3, activation='relu')
        ↓
        MaxPooling1D (pool_size=2)
        ↓
        Conv1D (128 filters, kernel_size=3, activation='relu')
        ↓
        GlobalAveragePooling1D
        ↓
        Dense (64, activation='relu')
        ↓
        Dropout (0.3)
        ↓
        Dense (32, activation='relu')
        ↓
        Dense (1, activation='linear')
        ```
        
        **Strengths:**
        - Excellent at capturing spatial patterns in data
        - Fast training and inference
        - Efficient with structured tabular data
        
        **Best Use Cases:**
        - Quick yield estimates
        - Pattern-based predictions
        - Real-time applications
        """)
    
    elif model_choice == "GRU":
        st.markdown("#### 🔶 GRU (Gated Recurrent Unit)")
        st.write("""
        **Architecture:**
        ```
        Input Layer (Sequence: 12 months × Features)
        ↓
        GRU (128 units, return_sequences=True)
        ↓
        Dropout (0.3)
        ↓
        GRU (64 units)
        ↓
        Dropout (0.3)
        ↓
        Dense (32, activation='relu')
        ↓
        Dense (1, activation='linear')
        ```
        
        **Strengths:**
        - Captures temporal dependencies
        - Models seasonal patterns effectively
        - Memory of historical trends
        
        **Best Use Cases:**
        - Time series forecasting
        - Seasonal trend analysis
        - Multi-year predictions
        """)
    
    else:  # Hybrid
        st.markdown("#### ⚡ Hybrid (CNN + GRU)")
        st.write("""
        **Architecture:**
        ```
        Input Layer (Sequence: 12 months × Features)
        ↓
        ├─ CNN Branch                  ┬─ GRU Branch
        │  Conv1D (64 filters)         │  GRU (64 units)
        │  MaxPooling1D                │  Dropout (0.3)
        │  GlobalAveragePooling1D      │
        ↓                               ↓
        Concatenate [CNN + GRU outputs]
        ↓
        Dense (64, activation='relu')
        ↓
        Dropout (0.3)
        ↓
        Dense (32, activation='relu')
        ↓
        Dense (1, activation='linear')
        ```
        
        **Strengths:**
        - Combines spatial and temporal learning
        - Most comprehensive feature extraction
        - Highest prediction accuracy
        
        **Best Use Cases:**
        - Critical policy decisions
        - Comprehensive yield analysis
        - When accuracy is paramount
        """)
    
    st.markdown("---")
    
    # Training configuration
    with st.expander("⚙️ Training Configuration"):
        st.write("""
        **Common Training Settings:**
        - Optimizer: Adam
        - Loss Function: Mean Squared Error (MSE)
        - Batch Size: 32
        - Max Epochs: 100
        - Early Stopping: Patience 15
        - Learning Rate: 0.001
        - Validation Split: 15%
        """)
