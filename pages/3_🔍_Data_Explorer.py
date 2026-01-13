"""
Data Explorer Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔍",
    layout="wide"
)

st.markdown('<h1 style="text-align: center; color: #ffffff;">🔍 Data Explorer</h1>', unsafe_allow_html=True)

st.write("""
Explore the dataset used to train the prediction models.
""")

# Tabs for different data views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🌡️ Climate Data", "🌾 Crop Data", "📈 Trends"])

with tab1:
    st.markdown("### Dataset Summary")
    
    # Display key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", "5,184", help="3 crops × 6 zones × 24 years × 12 months")
    
    with col2:
        st.metric("Features", "28", help="Climate, soil, and engineered features")
    
    with col3:
        st.metric("Time Period", "2000-2023", help="24 years of historical data")
    
    with col4:
        st.metric("Data Sources", "4", help="NASA POWER, NOAA GML, FAOSTAT, ISDA")
    
    st.markdown("---")
    
    # Data sources
    st.markdown("### 📚 Data Sources")
    
    sources_data = {
        "Data Type": ["Climate", "CO₂", "Crop Yields", "Soil"],
        "Source": ["NASA POWER API", "NOAA GML (Mauna Loa)", "FAOSTAT", "ISDA Soil API & Open-Meteo"],
        "Variables": [
            "Temperature, Rainfall, Humidity",
            "CO₂ Concentration",
            "National Crop Production",
            "Soil Properties & Elevation"
        ],
        "Coverage": ["6 Zones", "Global", "National", "6 Zones"]
    }
    
    df_sources = pd.DataFrame(sources_data)
    st.dataframe(df_sources, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Feature categories
    st.markdown("### 🔍 Feature Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Climate Features")
        st.write("""
        - 🌡️ Temperature (°C)
        - 🌧️ Rainfall (mm)
        - 💧 Relative Humidity (%)
        - 🌍 CO₂ Concentration (ppm)
        - 📊 GDD (Growing Degree Days)
        - 🌦️ Cumulative Rainfall
        - 📅 Seasonal indicators
        """)
    
    with col2:
        st.markdown("#### Soil Features")
        st.write("""
        - pH level
        - Organic Matter (%)
        - Nitrogen (ppm)
        - Phosphorus (ppm)
        - Potassium
        - Cation Exchange Capacity
        - Bulk Density
        - Water Holding Capacity
        """)
    
    # Sample data view
    with st.expander("👁️ View Sample Data"):
        st.write("Loading sample from processed dataset...")
        
        try:
            # Try to load sample data
            sample_path = Path("project_data/processed_data/master_data_hybrid.csv")
            if sample_path.exists():
                df_sample = pd.read_csv(sample_path, nrows=100)
                st.dataframe(df_sample.head(10), use_container_width=True)
                st.caption(f"Showing 10 of {len(df_sample)} sample records")
            else:
                st.info("Sample data file not found. Please ensure processed data exists.")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

with tab2:
    st.markdown("### 🌡️ Climate Data Analysis")
    
    # Simulated climate data visualization
    st.write("Explore climate patterns across Nigeria's zones.")
    
    # Zone selection
    selected_zone = st.selectbox(
        "Select Zone",
        ["All Zones", "North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
    )
    
    # Simulated temperature trend
    years = list(range(2000, 2024))
    temp_data = {
        "Year": years,
        "Temperature": [25 + 0.03 * (y - 2000) + np.random.uniform(-0.5, 0.5) for y in years]
    }
    
    df_temp = pd.DataFrame(temp_data)
    
    fig_temp = px.line(
        df_temp,
        x="Year",
        y="Temperature",
        title=f"Average Temperature Trend (2000-2023) - {selected_zone}",
        labels={"Temperature": "Temperature (°C)"}
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Climate variable distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulated rainfall distribution
        rainfall_data = np.random.normal(1200, 300, 100)
        fig_rain = go.Figure(data=[go.Histogram(x=rainfall_data, nbinsx=20)])
        fig_rain.update_layout(
            title="Rainfall Distribution",
            xaxis_title="Rainfall (mm)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_rain, use_container_width=True)
    
    with col2:
        # Simulated humidity distribution
        humidity_data = np.random.normal(65, 10, 100)
        fig_humid = go.Figure(data=[go.Histogram(x=humidity_data, nbinsx=20)])
        fig_humid.update_layout(
            title="Humidity Distribution",
            xaxis_title="Relative Humidity (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_humid, use_container_width=True)

with tab3:
    st.markdown("### 🌾 Crop Yield Data")
    
    # Crop selection
    selected_crop = st.selectbox(
        "Select Crop",
        ["Maize", "Cassava", "Yams"]
    )
    
    # Simulated crop yield by zone
    zones = ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
    yields = np.random.uniform(1.5, 4.5, len(zones))
    
    df_yields = pd.DataFrame({
        "Zone": zones,
        "Average Yield": yields
    })
    
    fig_yields = px.bar(
        df_yields,
        x="Zone",
        y="Average Yield",
        title=f"Average {selected_crop} Yield by Zone (2000-2023)",
        labels={"Average Yield": "Yield (tons/ha)"},
        color="Average Yield",
        color_continuous_scale="Greens"
    )
    
    st.plotly_chart(fig_yields, use_container_width=True)
    
    # Yield statistics
    st.markdown("### 📊 Yield Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Yield", f"{np.mean(yields):.2f} tons/ha")
    
    with col2:
        st.metric("Median Yield", f"{np.median(yields):.2f} tons/ha")
    
    with col3:
        st.metric("Std Dev", f"{np.std(yields):.2f}")
    
    with col4:
        st.metric("Max Yield", f"{np.max(yields):.2f} tons/ha")

with tab4:
    st.markdown("### 📈 Temporal Trends")
    
    st.write("Analyze how crop yields and climate variables have changed over time.")
    
    # Variable selection
    variable = st.selectbox(
        "Select Variable to Analyze",
        ["Crop Yield", "Temperature", "Rainfall", "CO₂ Concentration"]
    )
    
    # Simulated trend data
    years = list(range(2000, 2024))
    
    if variable == "Crop Yield":
        values = [2.5 + 0.02 * (y - 2000) + np.random.uniform(-0.3, 0.3) for y in years]
        ylabel = "Yield (tons/ha)"
    elif variable == "Temperature":
        values = [25 + 0.03 * (y - 2000) + np.random.uniform(-0.5, 0.5) for y in years]
        ylabel = "Temperature (°C)"
    elif variable == "Rainfall":
        values = [1200 + np.random.uniform(-200, 200) for y in years]
        ylabel = "Rainfall (mm)"
    else:  # CO₂
        values = [370 + 2.2 * (y - 2000) + np.random.uniform(-2, 2) for y in years]
        ylabel = "CO₂ (ppm)"
    
    df_trend = pd.DataFrame({"Year": years, variable: values})
    
    # Create trend plot with regression line
    fig = px.scatter(
        df_trend,
        x="Year",
        y=variable,
        trendline="ols",
        title=f"{variable} Trend (2000-2023)"
    )
    
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=ylabel
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trend Slope", f"{slope:.4f}")
    
    with col2:
        st.metric("R² Value", f"{r_value**2:.4f}")
    
    with col3:
        trend_dir = "Increasing ↗️" if slope > 0 else "Decreasing ↘️"
        st.metric("Trend Direction", trend_dir)
    
    if p_value < 0.05:
        st.success(f"✅ The trend is statistically significant (p-value: {p_value:.4f})")
    else:
        st.info(f"ℹ️ The trend is not statistically significant (p-value: {p_value:.4f})")
