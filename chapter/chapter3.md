# Chapter 3: Research Methodology

## Climate Change Impact on Food Security in Nigeria: A Deep Learning Approach

**Framework**: TensorFlow/Keras with Temporal Convolutional Networks  
**Data Period**: 2000–2023  
**Geographic Scope**: Nigeria's 6 Geopolitical Zones  
**Focal Crops**: Cassava and Yams

---

## 3.1 Introduction

This chapter details the systematic methodological approach employed to evaluate the impact of climate change on food security in Nigeria. The research design integrates climate science, agricultural data analytics, and deep learning to establish quantitative relationships between climatic variables and crop productivity. A hybrid deep learning architecture—the **Temporal Convolutional Network with Multi-Layer Perceptron (TCN-MLP)**—is developed and validated to capture temporal climate patterns and their nonlinear effects on agricultural yields.

In addition to predictive accuracy, this research emphasises **interpretability** through SHapley Additive exPlanations (SHAP) and **uncertainty quantification** via Monte Carlo Dropout (MC Dropout). These extensions are critical for translating deep learning predictions into policy-relevant insights.

The chapter is organised as follows:
- **Section 3.2**: Research design and experimental workflow
- **Section 3.3**: Data collection, sources, and availability
- **Section 3.4**: Preprocessing and feature engineering methodologies
- **Section 3.5**: The proposed TCN-MLP hybrid architecture
- **Section 3.6**: Training, validation, and hyperparameter optimisation strategies
- **Section 3.7**: Explainability and uncertainty quantification methods
- **Section 3.8**: Experimental setup and computational environment
- **Section 3.9**: Summary and transition to results

---

## 3.2 Research Design

### 3.2.1 Study Type and Approach

This research employs a **quantitative, correlational, and predictive study design** centred on deep learning-based time series regression. The approach is grounded in the premise that historical climate variability, when processed through appropriately designed neural architectures, can reveal underlying climate-agriculture relationships that enable accurate yield prediction and climate impact assessment.

### 3.2.2 Experimental Workflow

The research follows a systematic pipeline:

```
[Data Collection from NASA POWER, NOAA, ISDA, HarvestStat-Africa]
    ↓
[Data Preprocessing: Cleaning, Imputation, Quality Assurance]
    ↓
[Feature Engineering: GDD, Rainfall Anomaly, SPI-3, Interaction Terms]
    ↓
[Sequence Creation: 12-month rolling windows, 20 features]
    ↓
[Temporal Data Splitting: Train ≤2017 / Val 2018–2020 / Test 2021–2023]
    ↓
[TCN-MLP Architecture Design and Implementation]
    ↓
[Hyperparameter Tuning via Validation Set]
    ↓
[Model Training with Early Stopping and L2 Regularisation]
    ↓
[SHAP Feature Attribution + Permutation Importance]
    ↓
[MC Dropout Uncertainty Quantification]
    ↓
[Future Projection: 2024–2030 via Climate Trend Extrapolation]
    ↓
[Comprehensive Evaluation and Report Generation]
```

### 3.2.3 Research Hypotheses

**Primary Hypothesis (H1)**: A hybrid TCN-MLP architecture can learn complex nonlinear relationships between multi-year climate sequences and crop yields, achieving R² ≥ 0.75 on held-out test data and a train-test generalisation gap below 15%, validating the model's utility for climate impact assessment.

**Secondary Hypotheses**:
- **H2**: Temporal Convolutional Networks effectively extract seasonal climate patterns from sequential data, producing superior representations compared to standard feedforward networks applied to flattened sequences.
- **H3**: Categorical embeddings for geopolitical zone and crop type capture geospatial and agricultural heterogeneity, improving regional-level accuracy.
- **H4**: SHAP-attributed feature importances are consistent with established agronomic knowledge (e.g., seasonal rainfall as the dominant driver for cassava and yam yield).

---

## 3.3 Data Collection and Sources

### 3.3.1 Study Area and Geographic Context

**Nigeria** represents a critical case study for climate-food security interactions:
- **Population**: ~220 million (largest in Africa)
- **Agricultural dependency**: ~35% of workforce engaged in farming
- **Agro-ecological diversity**: Spans 6 distinct geopolitical zones from Sahel to equatorial rainforest
- **Climate vulnerability**: Exposed to droughts, floods, and temperature extremes
- **Strategic crops**: Cassava and Yams are culturally and nutritionally critical staples

**Geographic Divisions**: The analysis covers Nigeria's **6 geopolitical zones**:

| Zone | States Included | Climate Type |
|------|----------------|-------------|
| **North West** | Sokoto, Kebbi, Katsina, Kaduna, Kano, Jigawa | Semi-arid / Sahelian |
| **North East** | Borno, Yobe, Adamawa, Taraba | Semi-arid to sub-humid |
| **North Central** | Niger, Plateau, Kwara, Kogi, Nasarawa | Guinea savannah |
| **South West** | Lagos, Ogun, Oyo, Osun, Ondo, Ekiti | Tropical monsoon |
| **South East** | Anambra, Enugu, Ebonyi, Abia, Imo | Humid tropical |
| **South South** | Delta, Edo, Cross River, Akwa Ibom, Rivers, Bayelsa | Equatorial rainforest |

### 3.3.2 Data Sources

| Variable | Source | Period | Resolution | Quality Score | Notes |
|----------|--------|--------|-----------|---------------|-------|
| **Temperature** | NASA POWER | 1990–2023 | 0.5°×0.5° / Daily | 0.92 | T2M 2 m temperature; aggregated to monthly means |
| **Rainfall** | NASA POWER | 1990–2023 | 0.5°×0.5° / Daily | 0.90 | PRECTOTCORR corrected precipitation; monthly totals |
| **Humidity** | NASA POWER | 1990–2023 | 0.5°×0.5° / Daily | 0.90 | RH2M relative humidity; monthly means |
| **CO₂** | NOAA GMCC | 1990–2023 | Global monthly | 0.95 | Mauna Loa Observatory; single global value per month |
| **Soil Properties** | ISDA Soil API | Static 2020 | Point-based | 0.80 | pH, organic matter, nitrogen, phosphorus |
| **Elevation** | Open-Meteo | Static | Point lookup | 0.90 | Context for climate interpolation |
| **Crop Yield** | HarvestStat-Africa v1.1 | 2000–2023 | Admin-1 (State) | 0.85 | Harmonised from FAOSTAT, FEWS NET, NBS; kg/ha |

**Data Access**:
- NASA POWER: Free API access (https://power.larc.nasa.gov/)
- NOAA GMCC: Public repository plain-text files parsed in Python
- ISDA Soil API: Free tier API key required
- HarvestStat-Africa v1.1: GitHub repository (https://github.com/HarvestStat/HarvestStat-Africa)

### 3.3.3 Agricultural Yield Data

**Source**: HarvestStat-Africa v1.1, a harmonised subnational crop statistics database compiled by the International Center for Tropical Agriculture (CIAT) and FEWS NET from FAOSTAT and the Nigerian National Bureau of Statistics (NBS).

**Crop Selection Rationale**:

| Crop | Data Completeness | Mean Yield | State Coverage | Decision |
|------|------------------|-----------|----------------|---------|
| **Cassava** | 91.24% | 4.2 t/ha | 37/36 states | ✓ **Included** |
| **Yams** | 92.53% | 8.1 t/ha | 30/36 states | ✓ **Included** |
| Maize | 45.2% | 0.15 t/ha | 18/36 states | ✗ **Excluded** (inadequate coverage) |

Maize was excluded due to systematic regional harvest failure (mean yield ~0.15 t/ha, 5–6× below viable cultivation yields), indicating data quality issues. Cassava and yams were selected for their high data completeness, national production significance, and climate sensitivity.

**Yield Aggregation**: State-level yields were aggregated to geopolitical zone level using area-weighted averaging to align with the spatial resolution of climate data.

---

## 3.4 Data Preprocessing and Feature Engineering

### 3.4.1 Data Cleaning and Quality Assurance

**Missing Value Treatment**:
```python
# Linear interpolation for climate variables
climate_df.interpolate(method='linear', limit_direction='both')
```
Missing values constituted less than 2% of all climate records after interpolation. HarvestStat quality flags `qa_flag=0,1,2` (acceptable to high confidence) were retained; records with `qa_flag=3` (low confidence) were removed.

**Outlier Treatment**:
- Isolation Forest (contamination=0.05) identified multivariate outliers
- Extreme values capped at 95th percentile (IQR method)
- Rationale: Extreme weather events are legitimate observations; capping preserves signal while reducing ML instability

### 3.4.2 Feature Engineering

A total of **20 features** were engineered from raw climate and crop data for model input:

| Feature Category | Features (n) | Description |
|-----------------|-------------|-------------|
| **Base Climate** | 4 | Temperature_C, Rainfall_mm, Humidity_percent, CO2_ppm |
| **Soil Properties** | 4 | Avg_pH, Avg_Nitrogen_ppm, Avg_Phosphorus_ppm, Avg_Organic_Matter_Percent |
| **Agronomic Derived** | 4 | GDD (Growing Degree Days), Cumulative_Rainfall, Days_Into_Season, Heat_Stress |
| **Risk Indicators** | 3 | Drought_Risk, Flood_Risk, SPI_3 (3-month Standardised Precipitation Index) |
| **Temporal Signals** | 3 | Rainfall_Anomaly, Is_Rainy_Season, Is_Peak_Growing |
| **Interaction Terms** | 2 | Temp×Rainfall interaction, GDD normalised |

**Growing Degree Days (GDD)**:
$$\text{GDD} = \sum_{d=1}^{N_{\text{days}}} \max(0, T_d - T_{\text{base}})$$

where $T_d$ is daily mean temperature and $T_{\text{base}} = 10°C$.

**Standardised Precipitation Index (SPI-3)**:
$$\text{SPI}_3 = \frac{P_{t:t-3} - \mu_{P}}{\sigma_{P}}$$

where $P_{t:t-3}$ is the 3-month accumulated precipitation and the mean and standard deviation are calculated over the 2000–2017 training reference period.

**Rainfall Anomaly**:
$$\text{Rainfall\_Anomaly} = \frac{R_t - \bar{R}_{m}}{\bar{R}_{m}}$$

where $R_t$ is monthly rainfall and $\bar{R}_m$ is the long-run mean for that month across the training period.

### 3.4.3 Z-Score Normalisation

All numerical features were normalised using Z-score standardisation:

$$X_{\text{norm}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

The scaler was **fitted exclusively on training data** (2000–2017) to prevent data leakage. Validation and test sets were transformed using the training set statistics.

### 3.4.4 Temporal Sequence Construction

TCNs process sequences of consecutive time steps. A **sliding window** approach was employed with a lookback window of **12 months**:

```
For sample t (t ≥ 12):
    X[t] = features[t-12 : t]   → shape (12, 20)
    y[t] = yield[t]              → scalar (kg/ha)
```

**Example**:
- Sample at February 2015: Climate features from February 2014 – January 2015 → Yield for February 2015
- Final input tensor shape: **(N_samples, 12, 20)** — batch × timesteps × features

This formulation preserves temporal ordering and enables the TCN to detect seasonal accumulation patterns (e.g., cumulative pre-season rainfall) that single-record approaches miss.

### 3.4.5 Temporal Data Splitting

Temporal data requires **time-aware splitting** to prevent data leakage where future data influences past predictions:

```
2000–2017 (18 years)                2018–2020          2021–2023
[TRAINING SET — ~2,592 samples]  [VAL — ~576 samples] [TEST — ~576 samples]
        ≈ 75%                            ≈ 13%                ≈ 12%
```

All splits maintain strict temporal ordering: the model never sees validation or test data during training, nor validation data during final evaluation.

**Integrated Dataset Shape**: After sequence creation, the consolidated dataset `master_data_hybrid.csv` contains **8,712 monthly records** (6 regions × 2 crops × 24 years × 12 months).

---

## 3.5 The Proposed TCN-MLP Hybrid Architecture

### 3.5.1 Architecture Overview and Design Rationale

The core design insight is that climate-agriculture data contains two fundamentally different feature types requiring distinct processing strategies:

| Feature Type | Examples | Characteristics | Processing Strategy |
|-------------|----------|-----------------|---------------------|
| **Temporal sequences** | Temperature, Rainfall, GDD | Vary monthly; exhibit seasonal autocorrelation | TCN Branch — causal dilated convolutions |
| **Static categorical** | Region, Crop type | Fixed per record; capture site/crop characteristics | MLP Branch — learned embeddings |

Processing these types through parallel specialised branches before fusing enables the model to exploit both temporal dynamics and categorical structure without conflating them.

**Comparative Justification**:

| Architecture | Parameters | Parallelisable | Interpretable | Test R² | Notes |
|-------------|-----------|----------------|--------------|---------|-------|
| Standard LSTM | ~100 K | ❌ | ❌ | 0.72 | Baseline |
| 1D-CNN | ~50 K | ✓ | ✓ | 0.68 | Fixed receptive field |
| TCN | ~20 K | ✓ | ✓ | ~0.77 | Dilated, causal |
| **TCN-MLP (proposed)** | **~25 K** | **✓** | **✓** | **0.8863** | Dual-branch, efficient, interpretable |

### 3.5.2 Temporal Convolutional Network (TCN) Branch

**Core Operation: Dilated Causal Convolutions**

Standard 1D convolution at position $t$ with filter size $k$:

$$y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t-i}$$

Dilated convolution with dilation $d$:

$$y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t - d \cdot i}$$

Exponential dilation growth ($d = 1, 2, 4$) yields a receptive field of $2^L \cdot (k-1) + 1$ for $L$ layers, enabling capture of long-range seasonal dependencies without proportionally increasing parameters.

**Causal padding** ensures that the output at time $t$ depends only on inputs from the past ($t' \leq t$), preventing temporal leakage consistent with the operational constraint that predictions must not incorporate future observations.

**Residual connections** are added at each block to facilitate gradient flow:

$$\text{output}_{\text{block}} = F(x) + x$$

where $F(\cdot)$ is the convolutional transformation. Where dimensions differ between input and output (e.g., dimension change from block to block), a 1×1 convolution adapts the skip path.

**v4.1 TCN Branch Specification**:

```
INPUT: (Batch, 12 timesteps, 20 features)
   ↓
RESIDUAL BLOCK 1
  ├─ Conv1D: 20 → 32 filters, kernel=3, dilation=1, padding='causal'
  ├─ BatchNormalization
  ├─ ReLU activation
  ├─ Dropout(0.45)
  └─ Residual add (skip connection)
   ↓
RESIDUAL BLOCK 2
  ├─ Conv1D: 32 → 32 filters, kernel=3, dilation=2, padding='causal'
  ├─ BatchNormalization
  ├─ ReLU activation
  ├─ Dropout(0.45)
  └─ Residual add
   ↓
RESIDUAL BLOCK 3
  ├─ Conv1D: 32 → 32 filters, kernel=3, dilation=4, padding='causal'
  ├─ BatchNormalization
  ├─ ReLU activation
  ├─ Dropout(0.45)
  └─ Residual add
   ↓
GLOBAL AVERAGE POOLING  (Batch, 12, 32) → (Batch, 32)
   ↓
TCN OUTPUT: (Batch, 32)
```

The **effective receptive field** of this 3-block configuration is:

$$\text{RF} = (k - 1) \cdot (1 + 2 + 4) + 1 = 2 \cdot 7 + 1 = 15 \text{ months}$$

This exceeds the 12-month input window, ensuring that all past timesteps influence every output element.

**Hyperparameter Choices**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Dilation rates | [1, 2, 4] | Exponential growth; multi-scale temporal patterns |
| Filters | 32 | Adequate representational capacity without over-parameterisation |
| Kernel size | 3 | Minimal asymmetry; captures local dependencies |
| Padding | `causal` | Prevents temporal leakage |
| Dropout rate | 0.45 | Effective regularisation at this scale |
| BatchNorm | ✓ | Stabilises training; accelerates convergence |
| Skip connections | ✓ | Enable deep network training |

### 3.5.3 Multi-Layer Perceptron (MLP) Branch with Categorical Embeddings

Categorical variables—geopolitical zone and crop type—cannot be processed by convolutions. The MLP branch learns **dense embedding representations** for each categorical value:

- **Region Embedding**: 6 zones → 8-dimensional learned vector
- **Crop Embedding**: 2 crops (Cassava, Yams) → 4-dimensional learned vector

Embeddings are jointly trained with the rest of the network; the embedding vectors converge to represent each zone/crop's relative agro-climatic sensitivity.

**v4.1 MLP Branch Specification**:

```
REGION INPUT (B, 1) ──→ Embedding(6, 8) → (B, 8)
CROP INPUT   (B, 1) ──→ Embedding(2, 4) → (B, 4)
                                ↓
                        CONCATENATE → (B, 12)
                                ↓
                    Dense(12 → 64, ReLU)
                    BatchNormalization
                    Dropout(0.45)
                    L2(1e-4)
                                ↓
                    Dense(64 → 32, ReLU)
                    BatchNormalization
                    Dropout(0.45)
                    L2(1e-4)
                                ↓
                    MLP OUTPUT: (B, 32)
```

### 3.5.4 Fusion and Output Layer

The outputs of the TCN and MLP branches are concatenated and processed through a shared fusion head:

```
TCN OUTPUT (B, 32)    +    MLP OUTPUT (B, 32)
              ↓
    CONCATENATE → (B, 64)
              ↓
    Dense(64 → 32, ReLU)
    BatchNormalization
    Dropout(0.45)
    L2(1e-4)
              ↓
    Dense(32 → 1, Linear)        ← Regression output
    L2(1e-4)
              ↓
    PREDICTED YIELD (normalised)
              ↓
    INVERSE SCALER → kg/ha
```

**Linear activation** is used at the output layer since this is a regression task with an unbounded target range.

**v4.1 Model Summary**:

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| TCN Branch (Conv1D + BatchNorm) | ~12,500 | ~49% |
| MLP Branch (Embeddings + Dense) | ~3,200 | ~13% |
| Fusion Head (Dense layers) | ~7,800 | ~31% |
| BatchNorm non-trainable (moving stats) | ~1,765 | ~7% |
| **Total** | **~25,265** | 100% |

---

## 3.6 Model Training and Hyperparameter Optimisation

### 3.6.1 Loss Function

**Mean Squared Error (MSE)** was used as the primary loss function, appropriate for continuous regression:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

Combined with **L2 weight regularisation**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \sum_{w} \|w\|^2, \quad \lambda = 10^{-4}$$

### 3.6.2 Optimiser and Learning Rate Schedule

**Optimiser**: Adam (Adaptive Moment Estimation)

| Parameter | Value |
|-----------|-------|
| Learning rate | 1 × 10⁻³ |
| Beta₁ (momentum) | 0.9 |
| Beta₂ (RMSProp) | 0.999 |
| Epsilon | 1 × 10⁻⁷ |
| Gradient clipping | L2 norm ≤ 1.0 |

**Learning Rate Schedule**: ReduceLROnPlateau with factor=0.5, patience=5 epochs, minimum lr=1×10⁻⁶.

### 3.6.3 Regularisation Strategy

Multi-layer regularisation was applied to prevent overfitting:

| Method | Setting | Effect |
|--------|---------|--------|
| Dropout | Rate=0.45 | Randomly zeros activations; discourages co-adaptation |
| L2 Penalty | λ=1×10⁻⁴ | Penalises large weights; encourages sparse representations |
| BatchNormalization | Default | Normalises activations; acts as implicit regulariser |
| Early Stopping | Patience=10 epochs | Halts training when validation loss stops improving; saves best weights |
| ReduceLROnPlateau | Factor=0.5 | Anneals learning rate on plateau; fine-tunes convergence |

### 3.6.4 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **R² Score** | $1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | Proportion of variance explained; 1.0 = perfect |
| **MAE** | $\frac{1}{n}\sum \lvert y - \hat{y} \rvert$ | Mean absolute error in kg/ha |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Root mean squared error; more sensitive to large errors |
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ | Mean squared error (kg/ha)² |

---

## 3.7 Explainability and Uncertainty Quantification

### 3.7.1 SHAP Feature Attribution

SHapley Additive exPlanations (SHAP) [Lundberg & Lee, 2017] was applied to attribute model predictions to individual input features. SHAP values quantify each feature's **marginal contribution** to deviations from the expected prediction:

$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[f(S \cup \{i\}) - f(S)\right]$$

where $F$ is the full feature set and $S$ ranges over all subsets excluding feature $i$.

**Implementation**: Given the dual-input architecture of the TCN-MLP and API changes in SHAP 0.50+, a robust two-stage strategy was adopted:

1. **Primary**: `shap.DeepExplainer` was attempted with background samples from the training set.
2. **Fallback**: When array shapes were inconsistent (due to the multi-input Keras model), **Integrated Gradients** was computed via TensorFlow `GradientTape`:

$$\text{IG}_i(x) = (x_i - x_i') \cdot \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$

approximated with 50 interpolation steps.

**Outputs**: Global bar chart of mean |SHAP| values per feature; beeswarm plot showing feature impact distribution; cross-method comparison table (SHAP vs. permutation importance).

### 3.7.2 Permutation Feature Importance

As a model-agnostic complement to SHAP, permutation feature importance [Breiman, 2001] was computed:

$$\text{PFI}_j = \text{MAE}_{\text{permuted}_j} - \text{MAE}_{\text{baseline}}$$

For each feature $j$, values across all samples were randomly shuffled and the resulting MAE increase was measured over 5 permutation repeats. Features with higher MAE degradation are more important.

### 3.7.3 Monte Carlo Dropout Uncertainty Quantification

Standard neural networks produce deterministic predictions. To quantify predictive uncertainty, **Monte Carlo Dropout** [Gal & Ghahramani, 2016] was employed: dropout layers are retained active during inference, and $T$ stochastic forward passes are executed. The ensemble of $T$ predictions approximates the posterior predictive distribution:

$$p(y^* | x^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^{T} p(y^* | x^*, \hat{\omega}_t)$$

**Uncertainty decomposition**:
- **Predictive mean**: $\bar{y} = \frac{1}{T}\sum_t \hat{y}_t$
- **Predictive standard deviation**: $\sigma = \sqrt{\frac{1}{T}\sum_t (\hat{y}_t - \bar{y})^2}$
- **95% Prediction Interval**: $[\bar{y} - 1.96\sigma, \; \bar{y} + 1.96\sigma]$

Settings: $T = 100$ MC passes on the validation and test sets; $T = 50$ passes for future projections (2024–2030) to balance computational cost.

### 3.7.4 Future Yield Projections (2024–2030)

To demonstrate the model's practical utility for agricultural planning, climate trends were extrapolated to 2030 using **linear regression per (Region, Crop, Month, Feature)** fitted on 2000–2023 historical data:

```python
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(years_train, feature_values)
projected = intercept + slope * future_years
```

Binary seasonal features (Is\_Rainy\_Season, Is\_Peak\_Growing, etc.) were preserved from historical seasonal climatology rather than extrapolated. Rolling 12-month sequences were constructed from projected climate data, scaled, and passed through the model with MC Dropout to obtain yield projections with 95% prediction intervals for each Region-Crop-Year combination.

---

## 3.8 Experimental Setup

### 3.8.1 Software Environment

| Component | Version | Role |
|-----------|---------|------|
| Python | 3.9+ | Programming language |
| TensorFlow | 2.14+ | Deep learning framework |
| Keras | 2.14+ | High-level model API |
| NumPy | 1.24+ | Numerical computation |
| Pandas | 2.0+ | Data manipulation |
| Scikit-learn | 1.3+ | Preprocessing, metrics, permutation importance |
| SciPy | 1.10+ | Statistical functions (linregress for trend extrapolation) |
| SHAP | 0.50+ | Feature attribution |
| Matplotlib / Seaborn | 3.7+ / 0.12+ | Visualisation |

**Reproducibility**: Random seed fixed at 42 for NumPy and TensorFlow; all results are deterministic given the same hardware.

### 3.8.2 Hardware and Computational Environment

| Resource | Specification | Role |
|----------|--------------|------|
| CPU | Multi-core (Intel i5/i7 class) | Training and inference |
| RAM | 16 GB | Full dataset in memory |
| GPU | Optional (training accelerated 5–10×) | Not required for inference |
| Storage | 50 GB SSD | Raw data, model weights, results |

**Inference performance**: ~2 ms per prediction on CPU; all 8,712 records evaluated in ~17 seconds.

### 3.8.3 Model Artefacts

| Artefact | Path | Description |
|---------|------|-------------|
| Trained model | `models/v4_1_saved/tcn_mlp_final.keras` | TensorFlow SavedModel format |
| Feature scaler | `scalers/scaler_X.pkl` | `StandardScaler` fitted on training set |
| Target scaler | `scalers/scaler_y.pkl` | `StandardScaler` for yield normalisation |
| Label encoders | `encoders/label_encoders.pkl` | Region/crop integer encoding |
| Metrics report | `results/v4_1_complete_evaluation/comprehensive_metrics_report.json` | All performance metrics |
| SHAP values | `results/v4_1_complete_evaluation/shap_importance.csv` | Per-feature SHAP scores |
| Future projections | `results/v4_1_complete_evaluation/future_predictions_2024_2030.csv` | Annual yield projections |

---

## 3.9 Summary

This chapter has detailed the comprehensive methodology employed to assess climate change impacts on Nigeria's food security through deep learning. The research integrates:

1. **Multi-source climate and agricultural data** (NASA POWER, NOAA, ISDA, HarvestStat-Africa) harmonised at monthly, zone-level resolution across 24 years.
2. **20-feature engineering** incorporating base climate variables, soil properties, agronomic indicators (GDD, SPI-3), and risk indices.
3. **Rigorous preprocessing** including Z-score normalisation (training-set-only), temporal sequence construction (12-month windows), and time-aware data splitting.
4. **A novel TCN-MLP hybrid architecture** with ~25,265 parameters that separates temporal pattern extraction (TCN branch, filters=32, dilation=[1,2,4], dropout=0.45) from categorical feature processing (MLP branch with learned embeddings, units=64/32).
5. **Integrated explainability** via SHAP attribution and permutation importance, providing feature-level transparency.
6. **Uncertainty quantification** via MC Dropout (T=100 inference passes), producing 95% prediction intervals alongside point estimates.
7. **Future projections** (2024–2030) via climate trend extrapolation combined with MC Dropout, enabling forward-looking agricultural planning.

The experimental setup ensures scientific rigour (temporal split, no data leakage), reproducibility (fixed seeds, documented hyperparameters), and practical deployment feasibility (CPU-compatible, sub-2ms inference).

Chapter 4 presents the empirical results of this methodology, including model validation metrics, feature importance findings, uncertainty analysis, and future yield projections.

---

## References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML 2016*, PMLR.
- HarvestStat-Africa. (2023). Harmonized Subnational Crop Statistics for Africa v1.1. https://github.com/HarvestStat/HarvestStat-Africa
- Lea, C., et al. (2017). Temporal convolutional networks for action segmentation and detection. *CVPR 2017*, 156–165.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*, 4765–4774.
- NASA POWER. (2023). Prediction of Worldwide Energy Resources. https://power.larc.nasa.gov/
- NOAA Global Monitoring Laboratory. (2024). Trends in Atmospheric Carbon Dioxide. https://gml.noaa.gov/ccgg/trends/
- Stackhouse Jr., P. W., et al. (2018). POWER Release 8.02.2. NASA/GSFC. https://doi.org/10.5067/SV90YYJUMS9K
