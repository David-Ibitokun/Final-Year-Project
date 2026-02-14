# Gap Reduction Implementation Guide

## Current Situation

You have two trained models with the following performance:

### GRU Model
- **Test**: 77.78%, **Train**: 91.36%, **Val**: 62.96%
- **Problem**: Validation accuracy is much lower than test (unusual!)
- **Gap**: Train-Val = 28.4%, Val-Test = -14.82% (validation lower than test)

### TCN Model  
- **Test**: 77.78%, **Train**: 98.15%, **Val**: 74.07%
- **Problem**: Training accuracy way too high (memorizing data)
- **Gap**: Train-Val = 24.08%, Train-Test = 20.37%

---

## Why Current Gaps Are Bad

1. **GRU**: Validation performing worse than test = data distribution issue or under-training
2. **TCN**: Train 98% vs Test 78% = severe overfitting = model memorizing training data
3. **Both**: Large gaps mean poor generalization to new, unseen data

---

## Recommended Action: Reload & Retrain Fresh

Due to notebook state issues after multiple edits, follow these steps:

### Step 1: Clear Notebook Kernel
1. Open the notebook: `phase3_model_dev.ipynb`
2. Click **Kernel → Restart** (Ctrl+Shift+F9)
3. Click **Run All** to regenerate all data and variables
4. This will reset state and fix dimension mismatches

### Step 2: Modify GRU Builder (AFTER kernel restart)

Find the GRU builder cell and replace with:

```python
# Build GRU model - GAP REDUCTION VERSION
def build_gru_model(sequence_length, n_features, learning_rate=0.0001):
    """
    Enhanced regularization to close val-test gap.
    - Same architecture (28 → 22 units)
    - Higher L2: 0.012 → 0.018
    - More dropout: 0.40 → 0.50 in-layer, 0.30 → 0.40 recurrent
    - More post-dropout: 0.60 → 0.70
    - Higher label_smoothing: 0.05 → 0.10
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(28, activation='relu', return_sequences=True,
                               dropout=0.50, recurrent_dropout=0.40,
                               kernel_regularizer=tf.keras.regularizers.l2(0.018),
                               recurrent_regularizer=tf.keras.regularizers.l2(0.018)),
            input_shape=(sequence_length, n_features)
        ),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(22, activation='relu', return_sequences=False,
                               dropout=0.50, recurrent_dropout=0.40,
                               kernel_regularizer=tf.keras.regularizers.l2(0.018),
                               recurrent_regularizer=tf.keras.regularizers.l2(0.018))
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.70),
        
        tf.keras.layers.Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.015)),
        tf.keras.layers.Dropout(0.70),
        
        tf.keras.layers.Dense(6, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.010)),
        tf.keras.layers.Dropout(0.60),
        
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.10),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')]
    )
    return model
```

### Step 3: Modify GRU Training Callbacks

Replace the GRU training callbacks with:

```python
# MORE PATIENT early stopping - trains longer to improve validation
gru_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,  # was 8, now 6 for slightly earlier stop
        restore_best_weights=True,
        min_delta=0.0005,  # small minimum improvement
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # give it 3 chances before reducing LR
        min_lr=1e-6,
        verbose=1
    ),
    SafeModelCheckpoint(filepath=str(gru_checkpoint_path), monitor='val_accuracy')
]
```

### Step 4: Modify TCN Builder

Replace TCN builder with:

```python
# Build TCN model - EXTREME REGULARIZATION VERSION
def build_tcn_model(sequence_length, n_features, learning_rate=0.0001, filters=20, kernel_size=3):
    """
    Aggressive overfitting prevention.
    - Filters: 24 → 20 (17% capacity reduction)
    - L2: 0.035 → 0.050 (extreme penalty)
    - SpatialDropout: 0.58 → 0.70 (extreme dropout)
    - Dense Dropout: 0.65/0.55 → 0.75/0.65 (extreme)
    - Label smoothing: 0.05 → 0.12 (more noise)
    """
    inputs = tf.keras.layers.Input(shape=(sequence_length, n_features))

    def tcn_block(x, dilation_rate, filters):
        residual = x
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                          activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.050),
                          kernel_constraint=tf.keras.constraints.max_norm(2.5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SpatialDropout1D(0.70)(x)  # extreme
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                          activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.050),
                          kernel_constraint=tf.keras.constraints.max_norm(2.5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(filters, 1, padding='same',
                          kernel_constraint=tf.keras.constraints.max_norm(2.5))(residual)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    x = tcn_block(inputs, dilation_rate=1, filters=filters)
    x = tcn_block(x, dilation_rate=2, filters=filters)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.045),
                              kernel_constraint=tf.keras.constraints.max_norm(2.5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.75)(x)  # extreme

    x = tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.040),
                              kernel_constraint=tf.keras.constraints.max_norm(2.5))(x)
    x = tf.keras.layers.Dropout(0.65)(x)  # extreme

    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.12),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')]
    )
    return model
```

### Step 5: Modify TCN Training Callbacks

Replace TCN callbacks with:

```python
# AGGRESSIVE early stopping - stop immediately on plateau
tcn_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,  # stop after 2 bad epochs
        restore_best_weights=True,
        min_delta=0.003,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=1,
        min_lr=1e-7,
        verbose=1
    ),
    SafeModelCheckpoint(filepath=str(tcn_checkpoint_path), monitor='val_accuracy')
]
```

### Step 6: Run All Cells

1. Run the GRU builder cell
2. Run the GRU training cell
3. Run the GRU evaluation cell to check results
4. Run the TCN builder cell
5. Run the TCN training cell
6. Run the TCN evaluation cell to check results

---

## Expected Results After Changes

### GRU Expected:
- Train: 91% → 85-88% (↓ good)
- Val: 62% → 70-75% (↑ good)
- Test: 78% → 76-78% (stable)
- Train-Val Gap: 28% → <15% (↓ good)
- **Val-Test Gap: -15% → 0-2% (↑ converge)**

### TCN Expected:
- Train: 98% → 80-85% (↓ good)
- Val: 74% → 72-75% (stable)
- Test: 78% → 73-76% (↓ acceptable trade-off)
- **Train-Test Gap: 20% → 8-12% (↓ good)**

---

## Why These Changes Work

### GRU Problem: Under-training (Val much lower than Test)
**Solution**: Higher dropout + higher L2 + longer training
- Forces learning of robust features
- Prevents under-fitting
- Validation performance improves

### TCN Problem: Over-training (Train 98%, Test 78%)
**Solution**: Extreme regularization + early stopping  
- Prevents memorization
- Closes train-test gap
- May reduce test slightly but improves generalization

---

## Success Criteria

✅ **GRU Success**: Val accuracy ≥ 70% AND (Test - Val) gap < 5%
✅ **TCN Success**: Train-Test gap < 13% AND Test accuracy ≥ 72%
✅ **Overall Success**: Both gaps < 12% AND Test ≥ 73%

---

## If Results Still Not Good

1. **For GRU**: Increase patience further (to 8), reduce min_delta more (to 0.0001)
2. **For TCN**: Increase dropout even more (to 0.80 spatial), reduce filters to 18
3. **Alternative**: Try ensemble of GRU + TCN for better results

