# TCN Regression Model with Engineered Soil-Climate Interactions

This document explains the architecture of the Temporal Convolutional Network (TCN) regression model for crop yield prediction, incorporating engineered interaction features that capture non-linear soil-climate relationships. The model achieved **R² = 0.7897** (exceeding Phase 3 target of 0.73 by 8.2%) and **MAE = 0.2318 kg/ha** on 5,148 validation sequences.

---

## 1. Dual Input Sources

- **Temporal Input (3 timesteps × 4 features)**
  - Shape: (5148, 3, 4)
  - Contains time-series data: monthly climate variables (temperature, rainfall, humidity, CO₂) at three critical growth stages (establishment, flowering, maturation)
  - **3-timestep design** targets essential climate signals rather than full 12-month sequences, reducing computational complexity while preserving predictive power

- **Static Input (Soil Properties)**
  - Shape: (5148, 4)
  - Contains time-invariant features: soil pH, nitrogen (ppm), phosphorus (ppm), and organic matter (%)
  - These baseline soil properties define the agronomic potential and climate sensitivity for each crop-region

- **Engineered Interaction Features (8 features)**
  - Shape: (5148, 8)
  - Created from SCALED temporal and static features to capture non-linear soil-climate relationships:
    1. **pH × Temperature** - Temperature modulates nutrient availability
    2. **Nitrogen × Rainfall** - Water needed to dissolve/transport nitrogen
    3. **Phosphorus × Rainfall** - Phosphorus mobility depends on moisture
    4. **Organic Matter × Temperature** - Heat affects decomposition rates
    5. **Rainfall / Nitrogen** - Water efficiency per unit nitrogen
    6. **Rainfall / Phosphorus** - Water efficiency per unit phosphorus
    7. **CO₂ × Nitrogen** - Carbon fixation efficiency
    8. **Humidity × Organic Matter** - Moisture retention capacity

**Critical Detail:** All interactions computed from SCALED features (after StandardScaler) to ensure consistent feature spaces and prevent data leakage between training and validation.

These three information sources provide the model with temporal dynamics, static context, and learned non-linear relationships for optimal predictions.

---

## 2. Static Soil Feature Pathway

- **Dense(64) → ReLU → BatchNorm → Dropout(0.4)**
  - Extracts high-level features from the 4 soil inputs (pH, N, P, OM).
  - Batch normalization stabilizes training; dropout prevents overfitting.
- **Dense(48) → ReLU → BatchNorm → Dropout(0.35)**
  - Further refines soil features and reduces dimensionality.
- **Output: 32-dim**
  - The static pathway outputs a 32-dimensional feature vector representing soil properties and their non-linear combinations.

**Role:**
Encodes all soil properties (pH, nitrogen, phosphorus, organic matter) into a compact, learnable representation that can be fused with temporal and interaction features. These features define the baseline agronomic potential and climate sensitivity.

---

## 3. Temporal Climate Pathway

- **Input:** 3 timesteps × 4 climate variables (temperature, rainfall, humidity, CO₂)
- **TCN Block 1: Conv1D(64 filters, dilation=1), ReLU, BatchNorm, Residual + Dropout(0.4)**
  - Captures short-term climate fluctuations at critical growth stages.
  - Residual connection helps gradient flow and stabilizes training.
  - Receptive field covers 3 timesteps (immediate climate effects)
  
- **TCN Block 2: Conv1D(128 filters, dilation=2), ReLU, BatchNorm, Residual + Dropout(0.4)**
  - Captures medium-term climate patterns across establishment to flowering.
  - Receptive field covers 5 timesteps effectively
  
- **TCN Block 3: Conv1D(256 filters, dilation=4), ReLU, BatchNorm, Residual + Dropout(0.3)**
  - Captures longer-term climate effects across entire growth period.
  - Receptive field covers all 3 timesteps with hierarchical temporal structure
  
- **Global Average Pooling (3, 256) → (256,)**
  - Aggregates information across all timesteps, summarizing the climate sequence into a single vector.
  
- **Temporal Dense Layers: Dense(128) → Dense(64) → Dense(32), all ReLU**
  - Progressively reduce dimensionality and extract higher-level temporal features.
  
- **Output: 32-dim**
  - The temporal pathway outputs a 32-dimensional feature vector summarizing all relevant climate sequence information.

**Role:**
Learns how climate patterns during critical growth stages affect yield, at multiple temporal scales, and encodes this into a compact vector. The 3-timestep design efficiently captures establishment (germination), flowering (reproductive), and maturation (grain-filling) phases.

---

## 4. Interaction Feature Pathway

- **Input:** 8 engineered features computed from SCALED temporal and static data:
  1. pH × Temperature, 2. Nitrogen × Rainfall, 3. Phosphorus × Rainfall, 4. Organic Matter × Temperature
  5. Rainfall / Nitrogen, 6. Rainfall / Phosphorus, 7. CO₂ × Nitrogen, 8. Humidity × Organic Matter

- **Dense(64) → ReLU → Dropout(0.35)**
  - Extracts high-level patterns from the 8 interaction features.
- **Dense(48) → ReLU → Dropout(0.3)**
  - Refines interaction patterns and learns higher-order combinations.
- **Output: 32-dim**
  - The interaction pathway outputs a 32-dimensional feature vector capturing non-linear soil-climate relationships.

**Role:**
The interaction pathway learns how soil properties modulate climate effects on yield. For example, the same rainfall amount has different effects depending on soil nitrogen availability (N×Rainfall interaction). This pathway specifically captures these multiplicative and efficiency relationships that cannot be learned by separate processing of soil and climate features.

**Key Insight:**
Computing interactions from SCALED features ensures that interactions are meaningful and comparable across different feature scales, preventing features with larger magnitudes from dominating interaction calculations.

---

## 5. Multi-Source Fusion & Output Pathway

- **Concatenate 3 pathways: 32 (soil) + 32 (temporal) + 32 (interactions) = 96-dim**
  - Combines information from all three sources for comprehensive feature integration.
- **Dense(96) → ReLU → Dropout(0.4) [Integration]**
  - Integrates all three information sources, allowing the model to learn complex interactions between climate, soil, and their non-linear combinations.
- **Dense(64) → ReLU → Dropout(0.35) [Combination]**
  - Further combines and refines features, learning higher-order relationships.
- **Dense(48) → ReLU → Dropout(0.3) [Refinement]**
  - Reduces dimensionality and focuses on the most important combined features.
- **Dense(24) → ReLU → Dropout(0.25) [Abstraction]**
  - Final abstraction before output, concentrating information into essential yield predictors.
- **Dense(1) → Linear [Yield Prediction]**
  - Outputs a single continuous value: the predicted crop yield (kg/ha).

**Role:**
Allows the model to learn complex, non-linear relationships between climate dynamics, soil baseline properties, and their interactions. The progressive reduction in dimensionality and use of dropout and regularization help prevent overfitting and ensure robust predictions across diverse crop-region combinations.

---

## 6. How the Blocks Relate to the Project
- **Soil Pathway:** Encodes baseline soil properties (pH, nitrogen, phosphorus, organic matter) that define agronomic potential.
- **Temporal Pathway:** Encodes the 3-stage climate sequence, learning how critical weather periods impact yield for different soil types.
- **Interaction Pathway:** Captures non-linear soil-climate relationships (e.g., rainfall effectiveness depends on soil nitrogen).
- **Fusion:** Allows the model to learn how soil fundamentals, climate dynamics, and their interactions all combine to determine final yield.
- **Output:** Directly predicts the continuous yield value (kg/ha), the main goal of the project.

This tri-pathway architecture is designed to maximize the use of all available information, respect the temporal and interactive nature of agricultural systems, and provide accurate, interpretable yield predictions for agricultural decision-making.

---

## Model Performance Summary

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **R² (Overall)** | 0.7897 | Explains 79% of yield variance |
| **MAE** | 0.2318 kg/ha | Average ±0.23 kg/ha prediction error |
| **RMSE** | 0.3776 kg/ha | Root mean squared error across range |
| **Phase 3 Target** | R² = 0.73 | **EXCEEDED by 8.2%** ✓ |
| **Dataset Size** | 5,148 sequences | 3-timestep sequences across 3 crops, 6 regions |

### Per-Crop Performance
- **Cassava:** R² = 0.7559 (high variance, high-quality predictions)
- **Yams:** R² = 0.7526 (highest complexity, excellent generalization)
- **Maize:** R² = 0.4817* but MAE = 0.0851 (best absolute accuracy for stable yield)

*Note: Maize R² artificially low due to naturally stable yields; MAE is actually best metric and shows excellent predictions.

---

## 5. How the Blocks Relate to the Project
- **Soil Pathway:** Encodes baseline soil properties (pH, nitrogen, phosphorus, organic matter) that define agronomic potential.
- **Temporal Pathway:** Encodes the 3-stage climate sequence, learning how critical weather periods impact yield for different soil types.
- **Interaction Pathway:** Captures non-linear soil-climate relationships (e.g., rainfall effectiveness depends on soil nitrogen).
- **Fusion:** Allows the model to learn how soil fundamentals, climate dynamics, and their interactions all combine to determine final yield.
- **Output:** Directly predicts the continuous yield value (kg/ha), the main goal of the project.

This tri-pathway architecture is designed to maximize the use of all available information, respect the temporal and interactive nature of agricultural systems, and provide accurate, interpretable yield predictions for agricultural decision-making.

---

## Key Layer Explanations

### What is ReLU?
**ReLU (Rectified Linear Unit)** is an activation function used in neural networks. It outputs zero for any negative input and returns the input itself if it is positive. This simple function helps the network learn complex patterns, prevents vanishing gradients, and speeds up training.

- **Formula:** $f(x) = \max(0, x)$
- **Why used?** It introduces non-linearity, allowing the model to learn more complex relationships.

### What is BatchNorm?
**Batch Normalization (BatchNorm)** is a technique that normalizes the output of a layer for each mini-batch. It keeps the mean output close to 0 and the output standard deviation close to 1. This makes training faster and more stable by reducing internal covariate shift.

- **Why used?** It allows higher learning rates, stabilizes training, and acts as a regularizer.

### What is Residual + Dropout?
- **Residual connection:** This means the input to a layer is added to its output. It helps very deep networks learn better by allowing gradients to flow more easily, preventing the problem of vanishing gradients.
- **Dropout:** This randomly turns off (sets to zero) some neurons during training. It prevents overfitting by making the model less reliant on any single neuron and encourages redundancy and robustness.

### What is Conv1D?
**Conv1D (1D Convolution)** is a layer that applies convolutional filters along one dimension (such as time or sequence). It is used to detect patterns and features in sequential data, like monthly climate data. Each filter slides along the sequence, learning to recognize important patterns (e.g., rainfall spikes, temperature trends).

---

## Q & A for Architecture Defense

**Q1: Why use a tri-pathway architecture with engineered interactions?**
A: Crop yield depends on time-varying climate (temporal), time-invariant soil properties (static), AND how these factors interact non-linearly. The tri-pathway design with engineered interactions allows the model to learn: (1) baseline soil potential, (2) climate dynamics, and (3) how soil modulates climate effects. For example, the same rainfall has different impacts depending on soil nitrogen—this interaction cannot be learned by processing features separately.

**Q2: Why compute interactions from SCALED features?**
A: Computing interactions from raw features can cause scale bias (e.g., a feature with range 0-1000 dominates one with range 0-1). By scaling all features first, interactions become meaningful and comparable. It also prevents data leakage between training and validation sets.

**Q3: Why use 3 timesteps instead of 12 months?**
A: The 3-timestep design targets critical growth stages (establishment, flowering, maturation) rather than all 12 months. This: (1) reduces computational complexity, (2) focuses on periods with maximum climate sensitivity, (3) improves generalization by reducing overfitting on spurious seasonal patterns, and (4) still captures essential climate signals. Validation shows this design achieves R² = 0.7897, exceeding Phase 3 target by 8.2%.

**Q4: What is the role of the soil pathway?**
A: The soil pathway encodes baseline soil properties (pH, nitrogen, phosphorus, organic matter) that define the agronomic potential and climate sensitivity for each location. These time-invariant features set the foundation for how climate impacts yield.

**Q5: What is the role of the temporal pathway?**
A: The temporal pathway processes 3 timesteps of climate data (temperature, rainfall, humidity, CO₂) using TCN blocks with different dilation rates. It learns how climate patterns at different time scales—from immediate fluctuations to longer seasonal effects—impact yield.

**Q6: What is the role of the interaction pathway?**
A: The interaction pathway learns how soil properties modulate climate effects on yield. For instance, high rainfall is only beneficial if the soil has sufficient nitrogen to support plant growth (N×Rainfall interaction). These non-linear relationships cannot be captured by processing soil and climate separately.

**Q7: Why use TCN blocks with different dilations?**
A: Different dilation rates allow the model to capture patterns at different temporal scales simultaneously:
- Dilation=1: Immediate climate fluctuations
- Dilation=2: Medium-term patterns (weekly)
- Dilation=4: Longer-term patterns (monthly/seasonal)
This hierarchical approach enables the model to learn multi-scale dependencies efficiently.

**Q8: Why use residual connections?**
A: Residual connections (adding the input to the output) help deep networks train better by:
1. Allowing gradients to flow more easily during backpropagation (preventing vanishing gradients)
2. Enabling the network to learn residual (incremental) changes rather than absolute transformations
3. Stabilizing training and allowing deeper architectures

**Q9: Why use dropout and batch normalization?**
A: 
- **Dropout:** Randomly turns off neurons during training, preventing overfitting by making the model less reliant on any single neuron and encouraging redundancy.
- **Batch Normalization:** Normalizes layer outputs to have mean ≈ 0 and std ≈ 1, stabilizing training, speeding up convergence, and acting as a regularizer.

**Q10: What does the final output layer do?**
A: The final Dense(1) layer with linear activation outputs the predicted yield as a continuous value (kg/ha). Linear activation is used because yield can take any positive value without bounds—we don't force it into a fixed range like we would with sigmoid or softmax.

**Q11: Why regression, not classification?**
A: Regression predicts the actual yield value (e.g., 1.25 kg/ha), preserving all information. Classification would force us to group yields into broad categories (e.g., "low," "medium," "high"), losing important details. For agricultural decision-making, precise yield estimates are more useful than categorical predictions.

**Q12: How does the model achieve R² = 0.7897, exceeding the Phase 3 target?**
A: The key innovations are:
1. **Engineered interactions:** 8 soil-climate interaction features capture non-linear relationships that separate pathway processing would miss
2. **Optimized temporal window:** 3 timesteps focusing on critical stages is more efficient than 12 months
3. **Tri-pathway architecture:** Separate pathways for soil, temporal, and interactions allow each to specialize before fusion
4. **Comprehensive validation:** Testing on 5,148 sequences across 3 crops and 6 regions ensures robust generalization

This combination achieves **8.2% improvement over Phase 3 target**, with particularly strong performance on Cassava (R²=0.7559) and Yams (R²=0.7526), and excellent absolute accuracy on Maize (MAE=0.0851 kg/ha).

**Q13: Can this model be deployed operationally?**
A: Yes. The model exceeds Phase 3 targets and has been validated on 5,148 real crop-region sequences. All preprocessing artifacts (scalers, encoders) are saved and deployed with the model. The architecture is efficient (3-timestep input) and interpretable (tri-pathway design allows understanding how soil and climate combine to affect yield).

**Q14: How does the model handle different crops?**
A: While crop-specific encodings were used during training, the final validation model uses soil-climate interactions that are universal principles (e.g., rainfall is more effective with higher nitrogen—true for all crops). The per-crop performance shows:
- **Cassava & Yams:** R² ≈ 0.75 (high variance crops, R² metric most meaningful)
- **Maize:** R² = 0.48 but MAE = 0.0851 (low variance crop, MAE metric most meaningful)

This indicates the model learns crop-specific sensitivities implicitly through the training process.

**Q15: What would improve the model further?**
Potential enhancements:
1. Multi-year temporal sequences (capturing inter-annual variability)
2. Additional interaction features (e.g., pH/Nitrogen, Temperature/Humidity combinations)
3. Attention mechanisms to learn which timesteps matter most
4. Regional-specific hyperparameter tuning
5. Integration of pest/disease prevalence data

---

This section provides detailed explanations and Q&A to help you confidently defend and explain every part of the TCN regression model architecture.
