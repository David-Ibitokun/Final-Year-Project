# Climate Change Impact on Food Security in Nigeria

## Deep Learning-Based Crop Yield Prediction System

A comprehensive deep learning pipeline for assessing the impact of climate change on food security in Nigeria through regional crop yield prediction using Feedforward Neural Networks (FNN), Long Short-Term Memory (LSTM) networks, and Hybrid FNN-LSTM models.

---

## Project Overview

This research develops predictive models to forecast crop yields based on climate variables (rainfall, temperature, humidity, CO₂) across Nigeria's six geopolitical zones, covering five strategically selected crops optimized for climate resilience and food security.

### Key Features

- **5 Optimal Crops**: Millet, Sorghum, Groundnuts, Oil palm fruit, Cocoa beans (Average suitability: 9.5/10)
- **Regional Scaling Algorithm**: Novel methodology to transform national FAOSTAT data into zone-specific yields
- **34 Years of Data**: Historical analysis from 1990-2023
- **6 Geopolitical Zones**: North-Central, North-East, North-West, South-East, South-South, South-West
- **18 Representative States**: Comprehensive coverage of diverse agro-ecological zones
- **3 Deep Learning Architectures**: FNN, LSTM, and Hybrid models for comprehensive analysis

---

## Dataset

### Data Sources

- **Climate Data**: NASA POWER API (temperature, rainfall, humidity)
- **CO₂ Data**: NOAA ESRL Global Monitoring Laboratory
- **Crop Yields**: FAOSTAT (170 national records: 5 crops × 34 years)
- **Soil Data**: ISDA Soil API (18 state profiles, 15 properties)

### Data Volume

- **Raw Climate**: 7,344 monthly records (18 states × 12 months × 34 years)
- **Processed Regional**: 12,240 records (5 crops × 6 zones × 34 years × 12 months aggregated)
- **LSTM Sequences**: 146,880 monthly sequences (12,240 × 12 months)
- **FNN Features**: 20-35 features including climate aggregations, soil properties, encoded categoricals

### Regional Scaling Algorithm

**Problem**: FAOSTAT provides only national-level yield data, masking critical regional disparities.

**Solution**: Regional scaling methodology incorporating biophysical suitability and climate adjustments:

```
Regional_Yield = National_Yield × Scaling_Factor

Scaling_Factor = (0.7 × Suitability_Score + 0.3 × Climate_Adjustment) × Random_Noise(0.95, 1.05)
```

- **70% Weight**: Zone-crop suitability (soil, temperature, rainfall adequacy)
- **30% Weight**: Annual climate deviations from optimal conditions
- **±5% Noise**: Measurement uncertainty simulation

---

## Project Structure

```
Final_Year_Project/
├── data_prep_and_features.ipynb    # Phase 1-2: Data preparation & feature engineering
├── phase3_model_dev.ipynb          # Phase 3: Model training (FNN, LSTM, Hybrid)
├── phase4_validation.ipynb         # Phase 4: Validation & interpretation
├── requirements.txt                # Python dependencies
├── config/
│   └── crop_zone_suitability_5crops.json   # Suitability matrix
├── models/                         # Trained models & scalers (generated)
│   ├── fnn_model.keras
│   ├── lstm_model.keras
│   ├── hybrid_model.keras
│   └── *.pkl                      # Scalers & encoders
├── project_data/
│   ├── raw_data/
│   │   ├── agriculture/           # FAOSTAT crop yields
│   │   ├── climate/               # Temperature, rainfall, humidity, CO₂
│   │   └── soil/                  # Nigeria soil properties
│   ├── processed_data/            # Master datasets (generated)
│   │   ├── master_data_fnn.csv
│   │   ├── master_data_lstm.csv
│   │   └── master_data_hybrid.csv
│   └── train_test_split/          # Train/val/test splits (generated)
│       ├── fnn/
│       ├── lstm/
│       └── hybrid/
└── scripts/
    ├── download_climate_data.py   # NASA POWER API downloader
    └── download_soil_data.py      # ISDA Soil API downloader
```

---

## Installation

### Prerequisites

- Python 3.13+
- TensorFlow 2.18+
- CUDA-compatible GPU (optional, for faster training)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Final_Year_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation (Phase 1-2)

Run the data preparation notebook to generate regional datasets:

```bash
jupyter notebook data_prep_and_features.ipynb
```

**Outputs**:
- `project_data/processed_data/master_data_*.csv` (3 files)
- `project_data/train_test_split/*/` (train/val/test splits)
- `preprocessing_metadata.json`

**Key Steps**:
1. Load national crop yields from FAOSTAT
2. Load climate data (18 states × 34 years)
3. Load soil properties (18 states)
4. Apply regional scaling algorithm
5. Create FNN, LSTM, Hybrid datasets
6. Temporal split: Train (1990-2016), Val (2017-2019), Test (2020-2023)

### 2. Model Training (Phase 3)

Train deep learning models:

```bash
jupyter notebook phase3_model_dev.ipynb
```

**Models**:
- **FNN**: 3 hidden layers (128-64-32), processes aggregated annual features
- **LSTM**: 2 stacked layers (128-64), processes 12-month climate sequences
- **Hybrid**: Combines LSTM temporal branch + FNN static branch

**Outputs**:
- Trained models: `models/*.keras`
- Scalers: `models/*_scaler.pkl`
- Encoders: `models/*_encoder.pkl`

### 3. Validation & Interpretation (Phase 4)

Comprehensive model evaluation:

```bash
jupyter notebook phase4_validation.ipynb
```

**Analyses**:
- Regional performance (by zone and state)
- Crop-specific analysis
- Temporal trends (2020-2023)
- Climate scenario testing
- Feature importance (SHAP)
- Error analysis

---

## Model Architecture

### FNN Model
```
Input (20-35 features)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.3)
    ↓
Dense(32) + ReLU + Dropout(0.3)
    ↓
Output(1) - Linear
```

### LSTM Model
```
Input (12 timesteps × 4-6 features)
    ↓
LSTM(128, return_sequences=True)
    ↓
LSTM(64, return_sequences=False)
    ↓
Dense(32) + ReLU + Dropout(0.3)
    ↓
Output(1) - Linear
```

### Hybrid Model
```
Climate Sequences → LSTM Branch (64-32) ─┐
                                          ├→ Concatenate → Dense(64-32) → Output
Static Features → FNN Branch (64-32) ────┘
```

---

## Data Splits

| Split | Years | Records | Percentage |
|-------|-------|---------|------------|
| Train | 1990-2016 | ~9,800 | 80% |
| Val   | 2017-2019 | ~1,100 | 9% |
| Test  | 2020-2023 | ~1,340 | 11% |

**Note**: Temporal split ensures models are evaluated on genuinely future data.

---

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better, robust to outliers
- **R²** (Coefficient of Determination): Higher is better (0-1 scale)
- **MAPE** (Mean Absolute Percentage Error): Interpretable percentage error

---

## Key Innovations

1. **Regional Scaling Algorithm**: Transforms national data to zone-level yields using suitability factors and climate adjustments
2. **5-Crop Optimal Selection**: Strategically chosen for climate resilience (avg. 9.5/10 suitability)
3. **Hybrid Architecture**: Combines temporal dynamics (LSTM) with contextual factors (FNN)
4. **Comprehensive Validation**: Regional, crop-specific, and temporal analyses

---

## Results Preview

**Expected Model Performance** (pending training):
- Hybrid model anticipated to show superior performance (R² > 0.85)
- Regional patterns captured: Northern zones excel at cereals/legumes, Southern zones at tree crops
- Temporal sequences improve prediction of climate stress events

---

## Future Work

- Integration of satellite imagery (NDVI, soil moisture)
- Incorporation of socio-economic factors (fertilizer use, mechanization)
- Real-time prediction system with API
- Climate scenario projections (2030, 2050, 2100)
- Expansion to additional crops and West African countries

---

## References

- **FAOSTAT**: [https://www.fao.org/faostat/](https://www.fao.org/faostat/)
- **NASA POWER**: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)
- **NOAA CO₂**: [https://gml.noaa.gov/ccgg/trends/](https://gml.noaa.gov/ccgg/trends/)
- **ISDA Soil**: [https://www.isda-africa.com/](https://www.isda-africa.com/)

---

## License

[Specify your license]

---

## Author

Final Year Project - [Your Name/Institution]

**Contact**: [Your Email]

---

## Acknowledgments

- Food and Agriculture Organization (FAO) for FAOSTAT data
- NASA for POWER climate data
- NOAA for CO₂ monitoring data
- ISDA for African soil data
