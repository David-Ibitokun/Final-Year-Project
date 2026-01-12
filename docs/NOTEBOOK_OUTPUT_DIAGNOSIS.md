# Phase 4 Validation Notebook - Output Analysis

**Date**: January 11, 2026  
**Issue**: "The output is not right" - Models show as None

---

## Current State

### ✅ What's Working
- All model files exist in `models/` directory (cnn_model.keras, gru_model.keras, hybrid_model.keras)
- All scaler files exist (.pkl files)
- All test data files exist (train_test_split/*/test.csv)
- Predictions have been generated previously (arrays exist in kernel variables)
- The notebook cells have all been executed (execution counts 1-33)

### ❌ What's Wrong
- **Current kernel shows:** `cnn_model = None`, `gru_model = None`, `hybrid_model = None`  
- **But:** Prediction arrays exist (`cnn_predictions`, `gru_predictions`, `hybrid_predictions`)

**This means:** The notebook was run successfully at some point, but the current kernel session doesn't have the models loaded in memory anymore.

---

## Root Cause

### Most Likely Scenario
The kernel was **restarted** or **interrupted** after loading models but the prediction results remained in the output cells. The model objects themselves are large and aren't persisted in notebook outputs - only the computed results (predictions, metrics) are saved.

### Evidence
1. ✅ Models exist on disk
2. ✅ Test data loaded successfully (shapes shown in outputs)
3. ✅ Scalers loaded successfully
4. ✅ Predictions exist as numpy arrays
5. ❌ Model objects are None (not in memory)
6. ✅ All cells show execution counts (were run previously)

---

## Solution

### Option 1: Restart and Re-run (RECOMMENDED)
**Step 1:** Restart the kernel completely
```
Kernel → Restart Kernel
```

**Step 2:** Run all cells from the beginning
```
Cell → Run All
```

**Expected outcome:**
- Cell 5 will load all 3 models successfully
- All subsequent cells will run with loaded models
- Full validation results will be generated

**Time required:** 5-10 minutes (model loading takes ~30-60 seconds per model)

### Option 2: Just Re-run Cell 5
**Step 1:** Re-run just the model loading cell (Cell 5, lines 47-220)

**Step 2:** Verify models loaded:
```python
print(f"CNN: {type(cnn_model)}")
print(f"GRU: {type(gru_model)}")  
print(f"Hybrid: {type(hybrid_model)}")
```

**Expected output:**
```
CNN: <class 'keras.src.models.functional.Functional'>
GRU: <class 'keras.src.models.functional.Functional'>
Hybrid: <class 'keras.src.models.functional.Functional'>
```

**Step 3:** Continue running remaining cells

---

## Why Models Might Not Load

If re-running Cell 5 still shows models as None, check for:

### 1. TensorFlow/Keras Version Mismatch
**Training version:** Check what version was used in phase3  
**Current version:** `keras==3.13.0`, `tensorflow==2.20.0`

**Fix:** Ensure same versions between training and validation

### 2. Focal Loss Definition Mismatch
The models were trained with a custom `focal_loss` function. If the definition doesn't match exactly, loading will fail.

**Current definition:**
```python
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed
```

**Fix:** Verify this matches phase3_model_dev.ipynb exactly

### 3. Corrupted .keras Files
If files got corrupted during save/transfer:

**Test:**
```bash
ls -lh models/*.keras
```

**Expected:** All files should be 5-50 MB

**Fix:** Re-run phase3_model_dev.ipynb to regenerate models

### 4. Memory Issues
Loading 3 large deep learning models requires ~2-4 GB RAM

**Check:** Task Manager → Python process memory usage

**Fix:** Close other applications, restart VS Code

---

## Expected Validation Results

Once models load successfully, you should see:

###CNN Model (Cell 10-11)
- Accuracy: ~54-62%
- Confusion matrix heatmap
- Classification report

### GRU Model (Cell 13-14)
- Accuracy: ~58-66%
- Confusion matrix heatmap
- Classification report

### Hybrid Model (Cell 38-40)
- Accuracy: ~65-75%
- Better than CNN/GRU alone
- Confusion matrix
- Per-crop and per-zone metrics

### Feature Importance (Cell 26-30)
- SHAP values for each model
- Top 10 most important features
- Visualization plots

### Cross-Model Comparison (Cell 33-35)
- Side-by-side metrics
- Best model identification
- Performance summary table

---

## Troubleshooting Commands

### Check model files exist:
```python
from pathlib import Path
for model in ['cnn', 'gru', 'hybrid']:
    path = f'models/{model}_model.keras'
    exists = Path(path).exists()
    size = Path(path).stat().st_size / (1024**2) if exists else 0
    print(f"{model:8s}: {exists} ({size:.1f} MB)")
```

### Test direct loading:
```python
import tensorflow as tf
custom_objects = {'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25)}
model = tf.keras.models.load_model('models/cnn_model.keras', 
                                   custom_objects=custom_objects,
                                   compile=False)
print(f"Loaded: {model is not None}")
```

### Check kernel variables:
```python
print("Models in memory:")
print(f"  cnn_model: {cnn_model is not None}")
print(f"  gru_model: {gru_model is not None}")
print(f"  hybrid_model: {hybrid_model is not None}")
```

---

## Next Steps

1. ✅ **Restart kernel** - Clear all variables
2. ✅ **Run Cell 1-3** - Import libraries
3. ✅ **Run Cell 5** - Load models (watch for errors)
4. ✅ **Run Cell 38** - Check model status
5. ✅ **Run remaining cells** - Generate full validation report

If models still don't load after restart, run `diagnose_model_loading.py` script to get detailed error messages.

---

**Status**: Ready to restart and re-run  
**Expected fix time**: 5-10 minutes  
**Risk**: Low (all files verified present)
