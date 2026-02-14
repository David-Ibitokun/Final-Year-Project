# Gap Reduction Strategy - Current Analysis & Action Plan

## Current Model Performance Analysis

### GRU Model - **CRITICAL ISSUE: Val-Test Misalignment**
```
Train Accuracy:       91.36%
Validation Accuracy:  62.96%  ⚠️ UNUSUALLY LOW
Test Accuracy:        77.78%

Gaps:
- Train-Val Gap:      28.4%   (too high)
- Val-Test Gap:      -14.82%  (validation LOWER than test - unusual pattern)
- Train-Test Gap:     13.58%  (acceptable)
```

**Problem**: Validation accuracy is significantly lower than test accuracy. This suggests:
1. Validation set might have harder samples OR
2. Model is underfitting (too much regularization during training)
3. Class distribution mismatch between val/test

### TCN Model - **High Train-Test Gap**
```
Train Accuracy:  98.15%
Validation Accuracy: 74.07%
Test Accuracy:   77.78%

Gaps:
- Train-Val Gap:  24.08%  (too high - overfitting)
- Val-Test Gap:   -3.71%  (close - good alignment)
- Train-Test Gap: 20.37%  (high - overfitting)
```

**Problem**: Model memorizes training data (98% train accuracy) but generalizes acceptably to test (78%).

---

## Root Cause Analysis

### Why GRU has Val-Test Mismatch
1. **Current GRU Config**:
   - Units: 28 → 22 (moderate)
   - L2: 0.012 (light)
   - Dropout: 0.40/0.30 in-layer, 0.60 post-layer
   - Label smoothing: 0.05

2. **Why Validation is Low**:
   - Model NOT trained aggressively enough during training
   - Early stopping based on validation loss stops too early
   - Validation set might be harder than test set
   - Under-regularization doesn't prevent BUT under-training

### Why TCN has High Train-Test Gap
1. **Current TCN Config**:
   - Filters: 23
   - L2: 0.035
   - SpatialDropout: 0.58
   - Dense Dropout: 0.65/0.55
   - Label smoothing: 0.05

2. **Why Training is Too High**:
   - Model capacity still sufficient for small dataset
   - Early stopping lets training continue until validation plateaus
   - L2 and dropout not aggressive enough
   - Need stronger mechanisms

---

## Solution Strategy

### For GRU: Ensure Better Val-Test Alignment

**Root Issue**: Current model underfitting on validation set
**Solution**: Train longer before early stopping kicks in

**Implementation**:
```python
# Change early stopping to be less aggressive
EarlyStopping(
    monitor='val_loss',
    patience=5,  # → increase from 2
    min_delta=0.001,  # → relax from 0.003
    restore_best_weights=True
)
```

**Rationale**: 
- Let model train more epochs to better fit validation
- Higher patience = more chances to improve
- Smaller min_delta = accept smaller improvements

**Expected Result**:
- Validation accuracy should rise (62.96% → 70-75%)
- Val-Test gap should converge (close to 0-2%)
- Train-Val gap should shrink

### For TCN: Reduce Training Accuracy (Close Gap)

**Root Issue**: Model memorizes training data
**Solution**: Increase regularization aggressively

**Implementation**:
```python
# Option 1: Extreme Dropout
SpatialDropout1D: 0.58 → 0.70
Dense Dropout: 0.65/0.55 → 0.75/0.65

# Option 2: Extreme L2
L2 Regularization: 0.035 → 0.050

# Option 3: Reduce Capacity
Filters: 23 → 18
Dense units: 20/10 → 16/8

# Option 4: Aggressive Early Stopping
EarlyStopping(
    monitor='val_loss',
    patience=1,  # stop immediately
    min_delta=0.005,  # require improvement
)
```

**Rationale**:
- Dropout prevents co-adaptation (neurons relying on each other)
- L2 forces simpler solutions
- Smaller model has less capacity to memorize
- Early stopping prevents prolonged training

**Expected Result**:
- Training accuracy drops to 82-85%
- Train-Test gap reduces to 10-12%
- Test accuracy might drop to 72-75% (but gap improves significantly)

---

## Recommended Next Steps

### Priority 1: Fix GRU Val-Test Mismatch (CRITICAL)
```python
# Modify cell with GRU training callbacks
gru_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,      # more patience for longer training
        min_delta=0.001,  # accept smaller improvements
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]
```

### Priority 2: Reduce TCN Overfitting
```python
# Option A: Increase Dropout (least disruptive)
SpatialDropout1D(0.70)  # was 0.58
Dense dropout: [0.75, 0.65]  # was [0.65, 0.55]

# Option B: Increase L2 (if dropout alone insufficient)
L2: 0.045  # was 0.035

# Option C: Reduce Capacity (if other methods insufficient)
Filters: 20  # was 23
Dense: [16, 8]  # was [20, 10]
```

---

## Expected Outcomes After Changes

| Model | Metric | Current | Target | Status |
|-------|--------|---------|--------|--------|
| **GRU** | Train | 91.36% | 85-88% | Expect ↓ |
| | Val | 62.96% | 70-75% | Expect ↑ |
| | Test | 77.78% | 75-78% | Expected stable |
| | Train-Val Gap | 28.4% | <15% | Expect ↓ |
| | Val-Test Gap | -14.82% | 0-3% | Expect closer |
| **TCN** | Train | 98.15% | 82-85% | Expect ↓ |
| | Val | 74.07% | 73-76% | Expected stable |
| | Test | 77.78% | 73-76% | Expect ↓ |
| | Train-Test Gap | 20.37% | 10-12% | Expect ↓ |

---

## Why These Changes Work

### GRU Analysis
- **Problem**: Val loss is higher than test loss - unusual
- **Cause**: Early stopping stops training too soon
- **Fix**: Train longer with more patience
- **Result**: Validation performance converges to test performance

### TCN Analysis
- **Problem**: Train accuracy 98% vs Test 78% - severe gap
- **Cause**: Model capacity exceeds data size, enables memorization
- **Fix**: Increase dropout to force learning of robust features
- **Result**: Train accuracy drops but test accuracy stabilizes, gap closes

---

## Implementation Checklist

- [ ] Modify GRU training callbacks (patience=5, min_delta=0.001)
- [ ] Retrain GRU and evaluate new val/test gap
- [ ] If GRU val-test gap still >5%, apply additional dropout to GRU
- [ ] Modify TCN regularization (increase dropout/L2 OR reduce capacity)
- [ ] Retrain TCN and evaluate train-test gap
- [ ] Compare final results with targets above
- [ ] Document final performance metrics

---

## Key Metrics to Monitor

After each change, track:
1. **Training vs Validation Loss Curves** (should converge)
2. **Per-Class Recall** (ensure balanced performance)
3. **Epoch Count** (early stopping should trigger around epoch 20-30)
4. **Final Gaps** (train-val and val-test)

---

## Notes

- Both models are currently performing well on TEST data (~74-78%)
- The problem is **gap reduction**, not accuracy improvement
- Gap reduction requires sacrificing some training accuracy
- Target is: **Test ~73-76%** with **Gap <10-12%**
- This trade-off is acceptable because lower gaps = better generalization

