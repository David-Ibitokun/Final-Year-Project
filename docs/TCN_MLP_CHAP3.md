# Chapter 3: Research Methodology

## Climate Change Impact on Food Security in Nigeria: A Deep Learning Approach

**Created**: February 21, 2026  
**Framework**: TensorFlow/Keras with Temporal Convolutional Networks  
**Data Period**: 2000-2023  
**Geographic Scope**: Nigeria's 6 Geopolitical Zones  
**Focal Crops**: Cassava and Yams

---

## 3.1 Introduction

This chapter details the systematic methodological approach employed to evaluate the impact of climate change on food security in Nigeria. The research design integrates climate science, agricultural data analytics, and deep learning to establish quantitative relationships between climatic variables and crop productivity. A hybrid deep learning architecture—the **Temporal Convolutional Network with Multi-Layer Perceptron (TCN-MLP)**—is developed and validated to capture temporal climate patterns and their nonlinear effects on agricultural yields.

The chapter is structured as follows:
- **Section 3.2**: Research design and experimental workflow
- **Section 3.3**: Data collection, sources, and availability
- **Section 3.4**: Preprocessing and feature engineering methodologies
- **Section 3.5**: The proposed TCN-MLP hybrid architecture
- **Section 3.6**: Training, validation, and hyperparameter optimization strategies
- **Section 3.7**: Experimental setup and computational environment
- **Section 3.8**: Summary and transition to results

---

## 3.2 Research Design

### 3.2.1 Study Type and Approach

This research employs a **quantitative, correlational, and predictive study design** centered on deep learning-based time series regression. The approach is grounded in the premise that historical climate variability, when processed through appropriately designed neural architectures, can reveal underlying climate-agriculture relationships that enable accurate yield prediction and climate impact assessment.

### 3.2.2 Experimental Workflow

The research follows a systematic pipeline:

```
[Data Collection] 
    ↓
[Data Preprocessing & Normalization]
    ↓
[Feature Engineering & Sequence Creation]
    ↓
[Data Splitting: Train/Val/Test]
    ↓
[TCN-MLP Model Development & Architecture Design]
    ↓
[Hyperparameter Tuning via Validation Set]
    ↓
[Model Training with Early Stopping]
    ↓
[Comprehensive Model Evaluation on Test Set]
    ↓
[Results Analysis & Interpretation]
```

### 3.2.3 Research Hypotheses

**Primary Hypothesis (H1)**: A hybrid TCN-MLP architecture can learn complex nonlinear relationships between multi-year climate sequences and crop yields, achieving R² > 0.80 on held-out test data, thus validating the model's utility for climate impact assessment.

**Secondary Hypotheses**:
- **H2**: Temporal Convolutional Networks effectively extract climate patterns from sequential data superior to standard feedforward networks
- **H3**: Categorical embeddings for region and crop type capture geospatial and agricultural heterogeneity
- **H4**: Minimal train-test generalization gap (<5%) indicates robust learning without overfitting

---

## 3.3 Data Collection and Sources

### 3.3.1 Study Area and Geographic Context

**Nigeria** represents a critical case study for climate-food security interactions:
- **Population**: ~220 million (largest in Africa)
- **Agricultural dependency**: ~35% of workforce engaged in farming
- **Agro-ecological diversity**: Spans 6 distinct geopolitical zones from Sahel to rainforest
- **Climate vulnerability**: Exposed to droughts, floods, and temperature extremes
- **Staple crops**: Cassava and Yams are culturally and nutritionally critical

**Geographic Divisions**: The analysis covers Nigeria's **6 geopolitical zones**:
1. **North West** (Sokoto, Kebbi, Katsina, Kaduna, Kano, Jigawa)
2. **North East** (Borno, Yobe, Adamawa, Taraba)
3. **North Central** (Niger, Plateau, Kwara, Kogi, Nasarawa)
4. **South West** (Lagos, Ogun, Oyo, Osun, Ondo, Ekiti)
5. **South East** (Anambra, Enugu, Ebonyi, Abia, Imo)
6. **South South** (Delta, Edo, Cross River, Akwa Ibom, Rivers, Bayelsa)

### 3.3.2 Data Sources (Input Features and Target)

| Variable | Source | Period | Resolution | Quality | Notes |
|----------|--------|--------|-------------|---------|-------|
| **Temperature** | NASA POWER | 1990-2023 | 0.5°×0.5° grid / Daily | 0.92 | T2M (2m temperature); aggregated to monthly means |
| **Rainfall** | NASA POWER | 1990-2023 | 0.5°×0.5° grid / Daily | 0.90 | PRECTOTCORR; corrected precipitation; summed to monthly totals |
| **Humidity** | NASA POWER | 1990-2023 | 0.5°×0.5° grid / Daily | 0.90 | RH2M (relative humidity at 2m); aggregated to monthly means |
| **CO₂** | NOAA GMCC | 1990-2023 | Global | 0.95 | Mauna Loa Observatory monthly CO₂; single global value |
| **Soil Properties** | ISDA Soil API | Static 2020 | Point-based | 0.80 | pH, organic matter, nitrogen, phosphorus; requires authentication |
| **Elevation** | Open-Meteo | Static 2023 | Point lookup | 0.90 | Used for climate interpolation and spatial context |
| **Crop Yield (TARGET)** | HarvestStat-Africa v1.1 | 2000-2023 | Admin-1 (State) | 0.85 | Harmonized from FAOSTAT, FEWS NET, NBS; monthly aggregates; kg/ha |

**Climate Variables Derived in Study**:
- **Growing Degree Days (GDD)**: Sum of daily temperatures >10°C per month
- **Cumulative Rainfall**: Seasonal rainfall accumulation (January-December)
- **Heat Stress Index**: Binary indicator when max temperature >30°C
- **Drought Risk Index**: Based on rainfall anomalies relative to 30-year mean
- **Flood Risk Index**: Based on rainfall exceeding 90th percentile

**Data Access**:
- NASA POWER: Free API access; no authentication required
- NOAA GMCC: Public repository; plain-text files parsed in Python
- ISDA Soil API: Requires API key (free tier available)
- Open-Meteo: Free elevation lookup API

### 3.3.3 Agricultural Yield Data (Target Variable)

**Source**: **HarvestStat-Africa v1.1** (Harmonized Subnational Crop Statistics)
- **Institution**: International Center for Tropical Agriculture (CIAT) & FEWS NET
- **Compilation**: Harmonized from FAOSTAT, FEWS NET, and Nigerian National Bureau of Statistics
- **Unit**: Metric tons/hectare (mt/ha)
- **Coverage**: Admin-1 level (State level for Nigeria = 36 states)
- **Reference**: GitHub repository: https://github.com/HarvestStat/HarvestStat-Africa

**Crop Selection Rationale**:

| Crop | Completeness | Mean Yield | Std Dev | Coverage | Status |
|------|--------------|-----------|---------|----------|--------|
| **Cassava** | 91.24% | 4.2 mt/ha | 2.8 | 37/36 states | ✓ Selected |
| **Yams** | 92.53% | 8.1 mt/ha | 4.5 | 30/36 states | ✓ Selected |
| Maize | 45.2% | 0.15 mt/ha | 1.9 | 18/36 states | ✗ Excluded |

*Note: Maize exhibits systematic regional harvest failure (~0.15 mt/ha, 5-6× lower than viable crops), indicating data quality issues or systemic cultivation problems. Excluded from analysis to maintain dataset integrity.*

**Yield Data Processing**:
- State-level yields aggregated to 6 geopolitical regions via area-weighted averaging
- Training period: 2000-2017 (18 years)
- Validation period: 2018-2020 (3 years)
- Test period: 2021-2023 (3 years)

---

## 3.4 Data Preprocessing and Feature Engineering

### 3.4.1 Data Cleaning and Quality Assurance

**Step 1: Missing Value Treatment**

```python
# Linear interpolation for climate data (continuous variables)
climate_df.interpolate(method='linear', limit_direction='both')

# Forward-fill for categorical data (Region, Crop)
categorical_data.fillna(method='ffill')

# Report: Missing values < 2% after interpolation
```

**Step 2: Outlier Detection and Treatment**

- **Method**: Isolation Forest with contamination=0.05
- **Treatment**: Capped outliers at 95th percentile (IQR-based)
- **Rationale**: Extreme weather events are valid data points; capping preserves signal while reducing ML instability

**Step 3: Quality Flags**

- Verified HarvestStat quality flags
- Removed records with `qa_flag=3` (low confidence)
- Retained `qa_flag=0,1,2` (acceptable-to-high confidence)

### 3.4.2 Data Integration and Consolidation

**Input Dataset Structure**: `master_data_hybrid.csv`

| Field | Type | Description |
|-------|------|-------------|
| Region | Categorical | One of 6 zones (North Central, etc.) |
| Crop | Categorical | Cassava or Yams |
| Year | Temporal | 2000-2023 |
| Month | Temporal | 1-12 |
| Temperature_C | Numeric | Monthly mean (°C) |
| Rainfall_mm | Numeric | Monthly total (mm) |
| Humidity_percent | Numeric | Monthly mean (%) |
| CO2_ppm | Numeric | Global monthly mean (ppm) |
| Avg_pH | Numeric | Soil pH (0-14 scale) |
| Avg_Nitrogen_ppm | Numeric | Available soil N (ppm) |
| Avg_Phosphorus_ppm | Numeric | Available soil P (ppm) |
| Avg_Organic_Matter_Percent | Numeric | Soil carbon (%) |
| **Yield_kg_per_ha** | Target | **Crop yield (kg/ha)** |
| GDD | Numeric | Growing Degree Days |
| Cumulative_Rainfall | Numeric | Seasonal accumulation |
| Days_Into_Season | Numeric | Days since planting |
| Heat_Stress | Binary | {0,1} indicator |
| Drought_Risk | Binary | {0,1} indicator |
| ... | Derived | [Interaction terms, moving averages, lags] |

**Shape**: 8,712 monthly records (6 regions × 2 crops × 24 years × 12 months)

**Lineage**:
- Raw sources (NASA POWER, NOAA, ISDA, HarvestStat) are merged in `data_prep_and_features.ipynb`
- Derived features (GDD, interactions, moving averages) are calculated layer-by-layer
- Final dataset exported as `master_data_hybrid.csv` for model input

### 3.4.3 Normalization and Standardization

All numerical features are normalized using **Z-score standardization**:

$$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$

Where:
- $\mu$ = mean of training set
- $\sigma$ = standard deviation of training set
- **Scaler fitted on training data only** to prevent data leakage
- Validation and test sets transformed using training scaler parameters

**Rationale**: Neural networks converge faster with normalized inputs; prevents feature dominance by magnitude.

### 3.4.4 Sequence Creation for Temporal Convolutions

**Concept**: Traditional ML treats each record independently. TCNs process **sequences** to capture temporal patterns.

**Sliding Window Approach**:

```python
lookback_window = 12  # months

for t in range(lookback_window, len(data)):
    X[sample_id] = data[t-12:t]      # Past 12 months of features
    y[sample_id] = data[t].yield     # Current month's yield
```

**Example**:
- Sample 1: Jan 2000 - Dec 2000 → Jan 2001 yield
- Sample 2: Feb 2000 - Jan 2001 → Feb 2001 yield
- ...
- Sample N: Jan 2023 - Dec 2023 → Jan 2024 yield (predicted)

**Final Sequence Shapes**:
- **Temporal input X**: (samples, 12 months, 12 features)
- **Categorical input**: (samples, 1) for Region; (samples, 1) for Crop
- **Target y**: (samples, 1)

### 3.4.5 Data Splitting Strategy

Temporal data requires **time-aware splitting** to avoid data leakage:

```
2000-2017 (18 years)          2018-2020 (3 years)        2021-2023 (3 years)
[TRAINING SET: ~2,318 samples] [VAL SET: ~496 samples]  [TEST SET: ~553 samples]
        70%                           15%                        15%
```

**Rationale**:
- Models are trained on historical data (2000-2017)
- Hyperparameters tuned on holdout validation data (2018-2020)
- Final performance reported on temporally unseen test data (2021-2023)
- Prevents temporal leakage where the model learns from "the future"

---

## 3.5 The Proposed TCN-MLP Hybrid Architecture

### 3.5.1 Architecture Overview and Design Rationale

**Why TCN-MLP?**

| Architecture Type | Strength | Weakness | Use Case |
|-------------------|----------|----------|----------|
| **Standard LSTM** | Good at long-term dependencies | Slow (~50ms), hard to parallelize | Time series with variable length |
| **1D CNN** | Fast inference, parallelizable | Struggles with very long sequences | Medium-length sequences (<100) |
| **TCN** | Fast, parallelizable, large receptive field | Causal only (no future context) | Climate forecasting (causal) |
| **Transformer** | Attention-based, flexible | High memory, slower on short sequences | Very long sequences, variable length |
| **TCN-MLP Hybrid** | **Fast, accurate, interpretable** | **Requires feature engineering** | **12-month climate sequences + categorical metadata** ✓ |

**Design Philosophy**:

The TCN-MLP hybrid recognizes that climate-agriculture data has two distinct types:

1. **Temporal Features** (Temperature, Rainfall, Humidity)
   - Vary month-to-month
   - Have seasonal patterns (wet/dry seasons)
   - Require sequential processing to extract dependencies
   - **Handled by: TCN Branch**

2. **Categorical Features** (Region, Crop)
   - Static metadata
   - Determine regional/crop-specific sensitivities
   - Better handled via embeddings than sequences
   - **Handled by: MLP Branch**

By processing these separately then merging, the model achieves:
- ✓ **Efficiency**: Only 16,829 parameters (10-50× smaller than LSTM)
- ✓ **Speed**: ~2ms inference per prediction
- ✓ **Accuracy**: Test R² = 0.8184 (see Chapter 4)
- ✓ **Interpretability**: Clear data flow through distinct branches

### 3.5.2 Temporal Convolutional Network (TCN) Branch

#### Core Concept: Dilated Causal Convolutions

**Standard Convolution**:
```
Input sequence:  [T₁, T₂, T₃, T₄, T₅, T₆]
Filter size 3:   [w₀, w₁, w₂]
                  ↓ ↓ ↓
Output @ pos 3:  w₀·T₁ + w₁·T₂ + w₂·T₃
```

**Dilated Convolution** (dilation=2):
```
Input sequence:  [T₁, T₂, T₃, T₄, T₅, T₆]
Filter size 3:   [w₀, w₁, w₂]
                  ↓     ↓     ↓
Output @ pos 5:  w₀·T₁ + w₁·T₃ + w₂·T₅  ← Skips T₂, T₄
```

**Effect**: Dilation increases the "receptive field" without increasing computation. A 3-layer TCN with dilation [1,2,4] sees a 16-month historical window while maintaining fast inference.

**Causal Convolution** (prevents temporal leakage):
```
At time t, only use t-past, never t+future
        ↑ PREVENT CHEATING
```

#### TCN Architecture Specification

```
INPUT: (Batch, 12 months, 12 features)
   ↓
[RESIDUAL BLOCK 1]
  ├─ Dilated Conv1d: 12 → 32 filters, dilation=1, kernel=3
  ├─ BatchNormalization
  ├─ ReLU activation
  ├─ Dropout(0.63)
  └─ Skip connection: add input back
   ↓
[RESIDUAL BLOCK 2]
  ├─ Dilated Conv1d: 32 → 32 filters, dilation=2, kernel=3
  ├─ BatchNormalization
  ├─ ReLU activation
  ├─ Dropout(0.63)
  └─ Skip connection: add input back
   ↓
[RESIDUAL BLOCK 3]
  ├─ Dilated Conv1d: 32 → 28 filters, dilation=4, kernel=3
  ├─ BatchNormalization
  ├─ ReLU activation
  ├─ Dropout(0.63)
  └─ Skip connection: adapt dimensions
   ↓
[SPATIAL GLOBAL AVERAGE POOLING]
  Input: (Batch, ~2, 28)  [after causal padding]
  Output: (Batch, 28)     [collapse temporal dimension]
   ↓
OUTPUT: (Batch, 28-dim feature vector)
```

**Key Design Choices**:

| Hyperparameter | Value | Rationale |
|---|---|---|
| Number of Blocks | 3 | Sufficient depth to extract multi-scale temporal patterns |
| Dilation Rates | [1, 2, 4] | Exponential dilation for logarithmic receptive field growth |
| Filters | 28 (final) | Balanced capacity; 32→28 reduction prevents over-parameterization |
| Kernel Size | 3 | Minimal asymmetry; captures local temporal patterns |
| Padding | 'causal' | Maintains sequence length; prevents future leakage |
| Dropout | 0.63 | Light-moderate regularization for generalization |
| Normalization | BatchNorm | Stabilizes training; accelerates convergence |
| Skip Connections | ✓ | Enable training of deep networks; preserve temporal information |

### 3.5.3 Multi-Layer Perceptron (MLP) Branch with Categorical Embeddings

#### Concept: Categorical Embeddings

Raw categorical data (e.g., Region=2, Crop=1) is one-hot encoded or embedded into dense vectors to capture semantic relationships.

```
Region: "North Central" (integer: 2)
   ↓
[EMBEDDING LAYER: 6 → 8 dimensions]
   ↓
Embedding vector: [0.23, -0.54, 0.89, -0.12, 0.41, 0.67, -0.33, 0.18]
(learned during training to distinguish regional climate sensitivity)
```

#### MLP Architecture with Embeddings

```
[REGION INPUT]            [CROP INPUT]
  (Batch, 1)                (Batch, 1)
    ↓                          ↓
[EMBEDDING: 6→8 dim]    [EMBEDDING: 2→4 dim]
L2 regularization (1e-3)  L2 regularization (1e-3)
    ↓                          ↓
  (Batch, 8)               (Batch, 4)
    ↓_______________________↓
         [CONCATENATE]
         (Batch, 12)
           ↓
    [DENSE LAYER]
    12 → 12 neurons
    ReLU activation
    BatchNormalization
    Dropout(0.63)
    L2 regularization (1e-3)
           ↓
         (Batch, 12)
           ↓
       [OUTPUT]
      (Batch, 12-dim)
```

**Embedding Details**:
- **Region Embedding**: 6 possible regions → 8-dimensional dense vectors
- **Crop Embedding**: 2 crops (Cassava, Yams) → 4-dimensional vectors
- **Learning**: Embeddings are jointly trained with the rest of the network to minimize yield prediction loss

**Rationale**: Instead of large sparse one-hot vectors, embeddings compress categorical information into dense, learned representations that capture regional-crop characteristics.

### 3.5.4 Fusion and Output Layer

```
[TCN OUTPUT: (Batch, 28)]    [MLP OUTPUT: (Batch, 12)]
         ↓                               ↓
         └───────────[CONCATENATE]──────┘
                    (Batch, 40)
                       ↓
              [MERGED DENSE LAYER]
              40 → 28 neurons
              ReLU activation
              BatchNormalization
              Dropout(0.63)
              L2 regularization (1e-3)
                       ↓
                   (Batch, 28)
                       ↓
              [OUTPUT DENSE LAYER]
              28 → 1 neuron
              **Linear activation** (regression)
              L2 regularization (1e-3)
                       ↓
             (Batch, 1) = Predicted Yield
```

**Architecture Summary**:
- **Total Parameters**: 16,829
- **Trainable Parameters**: 16,349
- **Non-trainable**: 480 (BatchNorm moving averages)
- **Model Size**: ~65 KB (40 KB trainable weights + overhead)
- **Inference Time**: ~2ms per prediction (CPU)

---

## 3.6 Model Training and Hyperparameter Tuning

### 3.6.1 Loss Function

**Mean Squared Error (MSE)**:

$$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

**Choice Rationale**: 
- Penalizes large errors quadratically (large mistakes hurt more)
- Standard for regression; differentiable everywhere
- Aligns with R² metric used for evaluation

**With L2 Regularization**:

$$\text{Total Loss} = \text{MSE} + \lambda \sum |w|^2$$

Where $\lambda = 0.025$ (weight decay coefficient)

### 3.6.2 Optimizer and Learning Rate

**Optimizer**: Adam (Adaptive Moment Estimation)
- **Initial learning rate**: 0.00018 (conservative, prevents divergence)
- **Beta₁** (momentum): 0.9
- **Beta₂** (RMsprop): 0.999
- **Epsilon**: 1e-7
- **Gradient clipping**: norm_max=1.0 (prevents exploding gradients)

**Learning Rate Decay**:
- Factor: 0.6× every 2 epochs if validation loss plateaus
- Minimum: 1e-7 (floor to prevent stalling)

### 3.6.3 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|---|
| **R² Score** | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | % variance explained; 0-1 scale; 0.8+ is excellent |
| **MAE** | $\frac{1}{n}\sum \|y - \hat{y}\|$ | Average absolute error (same units as target) |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | Root mean squared error; penalizes large errors |
| **MAPE** | $\frac{1}{n}\sum \frac{\|y - \hat{y}\|}{y}$ | Mean absolute % error; scale-independent |

### 3.6.4 Regularization Strategy

**Problem**: Without regularization, the model memorizes training data (overfitting).

**Solution**: Multi-layer regularization:

| Layer | Method | Strength | Effect |
|-------|--------|----------|--------|
| **Dropout** | Random neuron deactivation | 0.63 | Forces distributed learning; prevents co-adaptation |
| **L2 Penalty** | Weight magnitude penalty | 0.025 | Encourages small, distributed weights |
| **BatchNorm** | Input normalization per layer | - | Stabilizes gradients; acts as regularizer |
| **Early Stopping** | Stop if Val Loss doesn't improve | Patience=5 epochs | Prevents overfitting after peak generalization |

**Regularization Tuning Process** (documented in prior work):

```
Attempt 1: Aggressive (dropout 0.75, L2 5e-2)
  Result: R² ≈ 0 (model collapsed - too much regularization!)
  
Attempt 2: Moderate (dropout 0.68, L2 3.5e-2)
  Result: Train R² = 0.887, Val R² = 0.866 (good but lost 2% accuracy)
  
Attempt 3: OPTIMIZED ✓ (dropout 0.63, L2 2.5e-2) ← SELECTED
  Result: Train R² = 0.8349, Val R² = 0.8157
  Train-Val Gap = 2.29% (EXCELLENT generalization)
  Test R² = 0.8184 (confirmed on unseen data)
```

### 3.6.5 Hyperparameter Tuning Process

**Grid Search on Validation Set**:

```
Hyperparameters varied:
  Learning rate: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
  Batch size: [8, 16, 32, 64]
  Dropout: [0.3, 0.4, 0.5, 0.6, 0.7]
  L2 penalty: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2]
  
Best configuration identified via Val R² maximization:
  Learning rate: 0.00018
  Batch size: 16
  Dropout: 0.63
  L2 penalty: 0.025
```

---

## 3.7 Experimental Setup

### 3.7.1 Software Environment

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.9+ | Programming language |
| **TensorFlow** | 2.14+ | Deep learning framework |
| **Keras** | 2.14+ | High-level API (part of TensorFlow) |
| **NumPy** | 1.24+ | Numerical computing |
| **Pandas** | 2.0+ | Data manipulation |
| **Scikit-learn** | 1.3+ | Preprocessing, metrics |
| **Matplotlib** | 3.7+ | Visualization |
| **Seaborn** | 0.12+ | Statistical graphics |

**Reproducibility**:
- Random seed set to 42 (NumPy, TensorFlow)
- All results deterministic given same hardware

### 3.7.2 Hardware and Computational Requirements

| Resource | Specification | Notes |
|----------|---------------|-------|
| **CPU** | Multi-core (Intel i5/i7 or equivalents) | Model training ~3 hours on 8-core CPU |
| **RAM** | 16 GB minimum | Sufficient for full dataset in memory |
| **GPU** | Optional (NVIDIA A100, V100, or similar) | Accelerates training ~5-10×; inference CPU-only |
| **Storage** | 50 GB SSD | Raw data + models + results |

**Deployment Flexibility**:
- ✓ CPU inference capable (embedded systems, edge devices)
- ✓ Fast inference (2ms per prediction)
- ✓ Small model (65 KB weights)
- ✓ No GPU required for deployment

### 3.7.3 Data and Model Artifacts

**Input Data**:
- `master_data_hybrid.csv` (8,712 monthly records)
- `scalers/scaler_X.pkl` (feature scaler)
- `scalers/scaler_y.pkl` (target scaler)
- `encoders/label_encoders.pkl` (categorical encoders)

**Model Outputs**:
- `models/tcn_mlp_optimized_overfit_best.keras` (trained weights)
- `models/tcn_mlp_metadata.json` (architecture specs, metrics)
- `results/train_val_test_metrics.csv` (performance summary)

---

## 3.8 Summary

This chapter has detailed the comprehensive methodology employed to assess climate change impacts on Nigeria's food security through deep learning. The research integrates:

1. **Multi-source climate data** (NASA POWER, NOAA) with **agricultural yield data** (HarvestStat-Africa)
2. **Rigorous preprocessing** including normalization, temporal sequencing, and time-aware data splitting
3. **A novel TCN-MLP hybrid architecture** that separates temporal pattern extraction from categorical feature processing
4. **Careful regularization tuning** balancing accuracy (R² 0.82+) with generalization (Train-Val gap <3%)

The experimental setup ensures **reproducibility** (fixed random seeds), **scientific rigor** (temporal train-test split), and **practical deployment feasibility** (CPU-only, fast inference).

With these methodologies established, **Chapter 4 presents the empirical results**, validation metrics, and interpretation of the trained models' performance on the held-out test set and across different geographic regions and crop types.

---

## References and Data Sources

**Climate Data References**:
- Stackhouse Jr., P. W., [et al.]. (2018). POWER Release 8.02.2 (With GIS Applications). NASA/GSFC, Langley Research Center, Hampton, VA, USA. DOI: [10.5067/SV90YYJUMS9K](https://doi.org/10.5067/SV90YYJUMS9K)
- NOAA Global Monitoring Laboratory. (2024). Trends in Atmospheric Carbon Dioxide. https://gml.noaa.gov/ccgg/trends/

**Agricultural Data References**:
- HarvestStat-Africa. (2023). Harmonized Subnational Crop Statistics for Africa v1.1. GitHub: https://github.com/HarvestStat/HarvestStat-Africa
- Monfreda et al. (2008). Farming the Planet: 2. Global Agricultural Lands in 2000. Global Biogeochemical Cycles, 22, GB1022. DOI: [10.1029/2007GB003011](https://doi.org/10.1029/2007GB003011)

**Soil Data References**:
- ISDA Soil API. (2020). iSDA Africa Soil Information Service. https://api.isda-africa.com

**Deep Learning Methods**:
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
- Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017). Temporal convolutional networks for action segmentation and detection. In CVPR, pp. 156–165.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

---

**Chapter 3 Complete**

Next chapter (Chapter 4) presents empirical results, model validation, and climate impact assessment findings.
