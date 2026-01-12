"""
About Page
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.markdown('<h1 style="text-align: center; color: #2E7D32;">ℹ️ About This Project</h1>', unsafe_allow_html=True)

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
