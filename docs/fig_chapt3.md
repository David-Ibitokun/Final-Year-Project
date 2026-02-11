# Chapter 3: Methodology Figures and Diagrams

## Figure 3.1: Model Workflow

This figure illustrates the complete workflow for the TCN regression model development for crop yield prediction.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│               COLLECT AND LOAD DATASET                      │
│                                                             │
│  • Climate Data (NASA POWER API): Temperature, Rainfall,   │
│    Humidity - 18 states × 12 months × 34 years            │
│  • CO₂ Data (NOAA): Mauna Loa Observatory records          │
│  • Crop Yields (FAOSTAT): National-level data              │
│  • Soil Data (ISDA Soil API): pH, N, P, Organic Matter    │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│           DATA PREPROCESSING AND CLEANING                   │
│                                                             │
│  • Handle missing values and outliers                       │
│  • Regional scaling algorithm (national → zone-specific)   │
│  • Temporal aggregation to critical growth stages          │
│  • Feature standardization and normalization               │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│              FEATURE EXTRACTION & ENGINEERING               │
│                                                             │
│  TEMPORAL FEATURES (3 timesteps × 4 climate vars):         │
│  • Temperature, Rainfall, Humidity, CO₂                    │
│  • Establishment, Mid-season, Late Growth stages           │
│                                                             │
│  STATIC FEATURES (4 soil properties):                       │
│  • Soil pH, Nitrogen (ppm), Phosphorus (ppm), OM (%)      │
│                                                             │
│  INTERACTION FEATURES (8 engineered features):              │
│  • pH×Temperature, N×Rainfall, P×Rainfall, OM×Temperature  │
│  • Rainfall/N, Rainfall/P, CO₂×N, Humidity×OM             │
│                                                             │
│  CATEGORICAL FEATURES:                                      │
│  • Crop Type (Cassava, Yams)                               │
│  • Geopolitical Zone (6 zones)                             │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                 FEATURE FUSION & SCALING                    │
│                                                             │
│  • Standardization using StandardScaler                     │
│  • MinMaxScaler for specific features                      │
│  • Train-Test-Validation split preparation                 │
│  • Sequence creation for temporal modeling                 │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                  TRAIN-TEST-VAL SPLIT                       │
│                                                             │
│  • Training Set: 80% (2,745 sequences)                     │
│  • Validation Set: 10% (343 sequences)                     │
│  • Test Set: 10% (344 sequences)                           │
│  • Total: 3,432 sequences                                  │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│              MODEL TRAINING: TCN REGRESSION                 │
│                                                             │
│  • 4-Pathway Architecture (Temporal, Static, Interaction,  │
│    Categorical)                                             │
│  • Adam Optimizer (learning_rate=0.001)                    │
│  • Loss: Mean Absolute Error (MAE)                         │
│  • Early Stopping & Model Checkpointing                    │
│  • Training Time: ~52.5 seconds                            │
│  • Parameters: 23,025                                      │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                   MODEL EVALUATION                          │
│                                                             │
│  PERFORMANCE METRICS:                                       │
│  • R² Score: 0.6722 (67.2% variance explained)            │
│  • MAE: 0.3620 kg/ha                                       │
│  • RMSE: 0.5189 kg/ha                                      │
│  • Bias: +0.0091 (minimal bias)                            │
│                                                             │
│  EVALUATION METHODS:                                        │
│  • Cross-validation consistency check                      │
│  • Residual analysis                                       │
│  • Feature importance visualization                        │
│  • Prediction vs Actual plots                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Figure 3.2: Model Architecture

This figure presents the detailed architecture of the TCN Regression Model with 4-pathway design for crop yield prediction.

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER (4 PATHWAYS)                       │
└─────────┬─────────────────┬──────────────────┬───────────────┬─────────┘
          │                 │                  │               │
          │                 │                  │               │
    ┌─────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐   ┌────▼────┐
    │ TEMPORAL   │   │   STATIC    │   │INTERACTION  │   │CATEGOR- │
    │   INPUT    │   │   INPUT     │   │   INPUT     │   │  ICAL   │
    │            │   │             │   │             │   │  INPUT  │
    │ (3432,3,4) │   │  (3432,4)   │   │  (3432,8)   │   │(3432,2) │
    │            │   │             │   │             │   │         │
    │ 3 timesteps│   │ 4 soil      │   │ 8 engineered│   │Crop ID  │
    │ 4 climate  │   │ properties: │   │ features:   │   │Zone ID  │
    │ variables: │   │             │   │             │   │         │
    │            │   │ • pH        │   │• pH×Temp    │   │• Crop:  │
    │• Temp (°C) │   │ • Nitrogen  │   │• N×Rainfall │   │  0,1,2  │
    │• Rain (mm) │   │ • Phosphorus│   │• P×Rainfall │   │• Zone:  │
    │• Humidity  │   │ • Organic M │   │• OM×Temp    │   │  0-5    │
    │• CO₂ (ppm) │   │             │   │• Rain/N     │   │         │
    │            │   │             │   │• Rain/P     │   │         │
    │            │   │             │   │• CO₂×N      │   │         │
    │            │   │             │   │• Humidity×OM│   │         │
    └─────┬──────┘   └──────┬──────┘   └──────┬──────┘   └────┬────┘
          │                 │                  │               │
          │                 │                  │               │
    ┌─────▼──────────┐ ┌────▼─────────┐ ┌─────▼──────────┐ ┌──▼──────┐
    │ TEMPORAL       │ │ STATIC       │ │ INTERACTION    │ │EMBEDDING│
    │ BRANCH         │ │ BRANCH       │ │ BRANCH         │ │ BRANCH  │
    │                │ │              │ │                │ │         │
    │ Conv1D Block 1 │ │ Dense(32)    │ │ Dense(32)      │ │Crop     │
    │ • 64 filters   │ │ • ReLU       │ │ • ReLU         │ │Embed    │
    │ • kernel=3     │ │ • L2(1e-4)   │ │ • L2(1e-4)     │ │(3→4)    │
    │ • causal pad   │ │ • Dropout    │ │ • Dropout      │ │         │
    │ • ReLU         │ │   (0.3)      │ │   (0.3)        │ │Zone     │
    │ • L2(1e-4)     │ │              │ │                │ │Embed    │
    │ • SpatialDrop  │ │ Output:      │ │ Output:        │ │(6→4)    │
    │   (0.3)        │ │ 8-D vector   │ │ 8-D vector     │ │         │
    │                │ │              │ │                │ │Output:  │
    │ Conv1D Block 2 │ │              │ │                │ │8-D      │
    │ • 64 filters   │ │              │ │                │ │vector   │
    │ • kernel=3     │ │              │ │                │ │         │
    │ • causal pad   │ │              │ │                │ │         │
    │ • ReLU         │ │              │ │                │ │         │
    │ • L2(1e-4)     │ │              │ │                │ │         │
    │ • SpatialDrop  │ │              │ │                │ │         │
    │   (0.3)        │ │              │ │                │ │         │
    │                │ │              │ │                │ │         │
    │GlobalAvgPool1D │ │              │ │                │ │         │
    │                │ │              │ │                │ │         │
    │ Output:        │ │              │ │                │ │         │
    │ 16-D vector    │ │              │ │                │ │         │
    └────────┬───────┘ └──────┬───────┘ └────────┬───────┘ └────┬────┘
             │                │                   │              │
             │                │                   │              │
             └────────────────┴───────────────────┴──────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   CONCATENATE LAYER  │
                           │                      │
                           │  16 + 8 + 8 + 8 = 40D│
                           │                      │
                           │  Temporal (16-D)     │
                           │  + Static (8-D)      │
                           │  + Interaction (8-D) │
                           │  + Categorical (8-D) │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   FUSION LAYER       │
                           │                      │
                           │  Dense(64)           │
                           │  • ReLU activation   │
                           │  • L2 reg (1e-4)     │
                           │  • Dropout (0.3)     │
                           │                      │
                           │  Output: 64-D        │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   OUTPUT LAYER       │
                           │                      │
                           │  Dense(1)            │
                           │  • ReLU activation   │
                           │  • Non-negative      │
                           │                      │
                           │  Output:             │
                           │  Yield (kg/ha)       │
                           └──────────────────────┘
```

**Architecture Summary:**
- **Total Parameters:** 23,025
- **Input Dimensions:** 40-D after concatenation
- **Training Strategy:** Adam optimizer, MAE loss, Early stopping
- **Regularization:** L2 regularization, Spatial/Standard Dropout, Causal padding

---

## Figure 3.3: Data Flow and Transformations

This figure shows how data flows through the preprocessing pipeline to create the final input for the TCN model.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RAW DATA SOURCES                                    │
├────────────────────┬───────────────────┬────────────────┬───────────────┤
│ NASA POWER API     │ FAOSTAT Database  │ ISDA Soil API  │ NOAA Mauna Loa│
│ • Temperature      │ • National Yields │ • pH           │ • CO₂ (ppm)   │
│ • Rainfall         │ • Crop Types      │ • Nitrogen     │               │
│ • Humidity         │ • Annual Data     │ • Phosphorus   │               │
│                    │                   │ • Organic M.   │               │
│ 18 states × 408 mo.│ 5 crops × 34 yrs │ 18 states      │ Monthly 1990- │
│                    │                   │                │ 2023          │
└──────────┬─────────┴─────────┬─────────┴────────┬───────┴───────┬───────┘
           │                   │                  │               │
           │                   │                  │               │
           ▼                   ▼                  ▼               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     DATA PREPROCESSING STAGE                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. REGIONAL SCALING ALGORITHM:                                        │
│     National_Yield → Regional_Yield using:                            │
│     • Zone-crop suitability scores (70% weight)                       │
│     • Climate deviation adjustments (30% weight)                      │
│     • Random noise (±5%) for uncertainty                              │
│                                                                        │
│  2. TEMPORAL AGGREGATION:                                              │
│     Monthly data → 3 Critical Growth Stages:                          │
│     • Stage 1: Establishment (months 1-4)                             │
│     • Stage 2: Mid-season/Flowering (months 5-8)                      │
│     • Stage 3: Late Growth/Maturation (months 9-12)                   │
│                                                                        │
│  3. FEATURE ENGINEERING:                                               │
│     Create 8 interaction features:                                    │
│     • Nutrient availability: pH×Temp, N×Rain, P×Rain, OM×Temp        │
│     • Balance ratios: Rain/N, Rain/P                                  │
│     • Growth factors: CO₂×N, Humidity×OM                              │
│                                                                        │
│  4. DATA CLEANING:                                                     │
│     • Remove outliers (IQR method)                                    │
│     • Handle missing values (forward/backward fill)                   │
│     • Validate data ranges and consistency                            │
│                                                                        │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    FEATURE SCALING & NORMALIZATION                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  • StandardScaler for temporal features (mean=0, std=1)               │
│  • StandardScaler for static soil features                            │
│  • StandardScaler for interaction features                            │
│  • LabelEncoder for categorical features (Crop, Zone)                 │
│                                                                        │
│  Output: Normalized feature matrices ready for model input            │
│                                                                        │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    DATASET SPLITTING                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Stratified split by Crop and Zone to ensure representation:          │
│                                                                        │
│  ┌─────────────────┬──────────────┬───────────────────┐              │
│  │   Dataset       │   Samples    │   Percentage      │              │
│  ├─────────────────┼──────────────┼───────────────────┤              │
│  │ Training        │   2,745      │      80%          │              │
│  │ Validation      │     343      │      10%          │              │
│  │ Test            │     344      │      10%          │              │
│  ├─────────────────┼──────────────┼───────────────────┤              │
│  │ TOTAL           │   3,432      │     100%          │              │
│  └─────────────────┴──────────────┴───────────────────┘              │
│                                                                        │
│  Note: Data represents 2 high-performing crops (Cassava, Yams)        │
│        across 6 geopolitical zones, 34 years (1990-2023)              │
│                                                                        │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    FINAL MODEL INPUT FORMAT                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  X_temporal:     (3432, 3, 4)  - 3 timesteps × 4 climate vars        │
│  X_static:       (3432, 4)     - 4 soil properties                   │
│  X_interaction:  (3432, 8)     - 8 engineered features               │
│  X_crop:         (3432,)       - Crop IDs (0-2)                      │
│  X_zone:         (3432,)       - Zone IDs (0-5)                      │
│                                                                        │
│  y:              (3432,)       - Target yields (kg/ha)                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Figure 3.4: Performance Evaluation Framework

This figure illustrates the comprehensive evaluation methodology used to assess model performance.

```
┌───────────────────────────────────────────────────────────────────┐
│                    TRAINED TCN MODEL                              │
│                 (tcn_regression_phase3_final.keras)               │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────┐
│  TRAINING SET    │              │  VALIDATION SET      │
│  Predictions     │              │  Predictions         │
│  (2,745 samples) │              │  (343 samples)       │
└────────┬─────────┘              └──────────┬───────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
┌────────────────────────────────────────────────────────────┐
│              CALCULATE METRICS                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  REGRESSION METRICS:                                       │
│  ┌──────────────────────────────────────────────┐        │
│  │ • R² Score (Coefficient of Determination)     │        │
│  │   - Measures variance explained by model      │        │
│  │   - Range: 0 to 1 (higher is better)         │        │
│  │                                               │        │
│  │ • MAE (Mean Absolute Error)                  │        │
│  │   - Average prediction error magnitude        │        │
│  │   - Units: kg/ha (lower is better)           │        │
│  │                                               │        │
│  │ • RMSE (Root Mean Square Error)              │        │
│  │   - Penalizes larger errors more heavily      │        │
│  │   - Units: kg/ha (lower is better)           │        │
│  │                                               │        │
│  │ • Bias                                        │        │
│  │   - Systematic over/under-prediction          │        │
│  │   - Close to 0 indicates unbiased model       │        │
│  └──────────────────────────────────────────────┘        │
│                                                            │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│              CROSS-VALIDATION CONSISTENCY                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Compare metrics across splits:                            │
│                                                            │
│  ┌─────────────┬──────────┬──────────┬──────────┐        │
│  │   Split     │    R²    │   MAE    │  Samples │        │
│  ├─────────────┼──────────┼──────────┼──────────┤        │
│  │ Training    │  0.7653  │  0.295   │  2,745   │        │
│  │ Validation  │  0.6724  │  0.356   │   343    │        │
│  │ Test        │  0.6722  │  0.362   │   344    │        │
│  └─────────────┴──────────┴──────────┴──────────┘        │
│                                                            │
│  ✓ Val and Test R² nearly identical → Good generalization │
│  ✓ Training R² slightly higher → No severe overfitting    │
│                                                            │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│                 VISUALIZATION ANALYSIS                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. PREDICTION vs ACTUAL SCATTER PLOT                      │
│     • Visual assessment of prediction accuracy             │
│     • Identify systematic biases                           │
│     • Check for heteroscedasticity                         │
│                                                            │
│  2. RESIDUAL ANALYSIS                                      │
│     • Distribution of prediction errors                    │
│     • Check for normality assumption                       │
│     • Identify outliers and anomalies                      │
│                                                            │
│  3. LEARNING CURVES                                        │
│     • Training vs Validation loss over epochs              │
│     • Assess convergence and overfitting                   │
│     • Validate early stopping effectiveness                │
│                                                            │
│  4. FEATURE IMPORTANCE                                     │
│     • Identify most influential features                   │
│     • Validate domain knowledge alignment                  │
│     • Guide future feature engineering                     │
│                                                            │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│                  FINAL TEST SET EVALUATION                 │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ╔════════════════════════════════════════════════╗       │
│  ║     FINAL TEST RESULTS (344 samples)           ║       │
│  ╠════════════════════════════════════════════════╣       │
│  ║  R² Score:        0.6722  ✓                    ║       │
│  ║  MAE:             0.3620 kg/ha                 ║       │
│  ║  RMSE:            0.5189 kg/ha                 ║       │
│  ║  Bias:            +0.0091 (minimal)            ║       │
│  ║  Confidence:      67.2% variance explained     ║       │
│  ╚════════════════════════════════════════════════╝       │
│                                                            │
│  CONCLUSION:                                               │
│  • Model achieves target R² > 0.60                        │
│  • Low MAE indicates good practical accuracy              │
│  • Minimal bias ensures unbiased predictions              │
│  • Consistent performance across all data splits          │
│  • Production-ready for deployment                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Table 3.1: Model Performance Summary

| Metric | Training | Validation | Test | Target | Status |
|--------|----------|------------|------|--------|--------|
| **R² Score** | 0.7653 | 0.6724 | 0.6722 | > 0.60 | ✓ Exceeded |
| **MAE (kg/ha)** | 0.295 | 0.356 | 0.362 | < 0.50 | ✓ Achieved |
| **RMSE (kg/ha)** | 0.422 | 0.518 | 0.519 | < 0.70 | ✓ Achieved |
| **Bias** | +0.003 | +0.007 | +0.009 | ≈ 0 | ✓ Minimal |
| **Samples** | 2,745 | 343 | 344 | - | - |

---

## Table 3.2: Architecture Specifications

| Component | Configuration | Parameters | Purpose |
|-----------|--------------|------------|---------|
| **Temporal Branch** | Conv1D(64)×2 + GlobalAvgPool | 4,224 | Climate sequence processing |
| **Static Branch** | Dense(32) → Dense(8) | 160 | Soil properties encoding |
| **Interaction Branch** | Dense(32) → Dense(8) | 288 | Engineered feature processing |
| **Categorical Branch** | Embedding(3,4) + Embedding(6,4) | 36 | Crop & Zone representation |
| **Fusion Layer** | Dense(64) + Dropout(0.3) | 2,624 | Multi-pathway integration |
| **Output Layer** | Dense(1, ReLU) | 65 | Yield prediction |
| **Total** | 4-pathway architecture | **23,025** | End-to-end model |

---

## Table 3.3: Input Feature Summary

| Feature Category | Features | Dimension | Description |
|------------------|----------|-----------|-------------|
| **Temporal** | Temperature, Rainfall, Humidity, CO₂ | (3432, 3, 4) | 3 growth stages × 4 climate variables |
| **Static** | pH, Nitrogen, Phosphorus, Organic Matter | (3432, 4) | Time-invariant soil properties |
| **Interaction** | pH×Temp, N×Rain, P×Rain, OM×Temp, Rain/N, Rain/P, CO₂×N, Humidity×OM | (3432, 8) | Engineered agronomic interactions |
| **Categorical** | Crop Type, Geopolitical Zone | (3432, 2) | Crop (2 types) + Zone (6 regions) |
| **Total Input** | Combined multi-pathway input | 40-D | After concatenation |

---

## Summary

This chapter presented the complete methodology for developing a TCN-based regression model for crop yield prediction. The approach integrates:

1. **Comprehensive Data Pipeline**: Multi-source data collection (climate, soil, yield) with regional scaling algorithms
2. **Advanced Feature Engineering**: 8 agronomic interaction features capturing nutrient availability, balance ratios, and growth factors
3. **4-Pathway Architecture**: Specialized processing branches for temporal, static, interaction, and categorical features
4. **Robust Regularization**: L2 regularization, spatial/standard dropout, and early stopping prevent overfitting
5. **Strong Performance**: R² = 0.6722 on test set, explaining 67.2% of yield variance with minimal bias

The model successfully addresses the challenge of predicting crop yields from complex climate-soil interactions, achieving production-ready performance with only 23,025 parameters and training in ~52.5 seconds.
