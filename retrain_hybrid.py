"""
Retrain Hybrid Model with Updated Dropout Regularization
This script retrains only the hybrid model with stronger regularization to address overfitting.
"""

import os
os.chdir(r'c:\Users\ibito\Documents\Final_Year_Project')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.optimizers import Adam

print("="*80)
print("RETRAINING HYBRID MODEL WITH STRONGER REGULARIZATION")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
splits_path = Path('project_data/train_test_split/hybrid')

# Load data
print("\nLoading data splits...")
hybrid_train = pd.read_csv(splits_path / 'train.csv')
hybrid_val = pd.read_csv(splits_path / 'val.csv')
hybrid_test = pd.read_csv(splits_path / 'test.csv')

print(f"  Train: {hybrid_train.shape}")
print(f"  Val:   {hybrid_val.shape}")
print(f"  Test:  {hybrid_test.shape}")

# Define feature columns
temporal_features = [
    'Temperature_C', 'Rainfall_mm', 'Humidity_percent', 'CO2_ppm',
    'GDD', 'Cumulative_Rainfall', 'Days_Into_Season',
    'Is_Rainy_Season', 'Is_Peak_Growing',
    'Heat_Stress', 'Cold_Stress', 'Rainfall_Anomaly',
    'Drought_Risk', 'Flood_Risk',
    'Yield_Lag_1', 'Yield_MA_3yr', 'Yield_YoY_Change'
]

static_features = [
    'Avg_pH', 'Avg_Nitrogen_ppm', 'Avg_Phosphorus_ppm', 'Avg_Organic_Matter_Percent'
]

# Encode categorical features
print("\nEncoding categorical features...")
le_crop = LabelEncoder()
le_region = LabelEncoder()

# Fit on train, transform all splits
hybrid_train['Crop_encoded'] = le_crop.fit_transform(hybrid_train['Crop'])
hybrid_val['Crop_encoded'] = le_crop.transform(hybrid_val['Crop'])
hybrid_test['Crop_encoded'] = le_crop.transform(hybrid_test['Crop'])

hybrid_train['Region_encoded'] = le_region.fit_transform(hybrid_train['Region'])
hybrid_val['Region_encoded'] = le_region.transform(hybrid_val['Region'])
hybrid_test['Region_encoded'] = le_region.transform(hybrid_test['Region'])

static_features.extend(['Crop_encoded', 'Region_encoded'])

print(f"  Temporal features: {len(temporal_features)}")
print(f"  Static features: {len(static_features)}")

# Create yield categories for classification
print("\nCreating yield categories...")
def create_yield_categories(df):
    """Create Low/Medium/High categories based on yield percentiles"""
    df = df.copy()
    yield_col = 'Yield_kg_per_ha'
    
    # Calculate percentiles per crop for balanced categories
    def categorize_yield(group):
        q33 = group[yield_col].quantile(0.33)
        q67 = group[yield_col].quantile(0.67)
        
        conditions = [
            group[yield_col] <= q33,
            (group[yield_col] > q33) & (group[yield_col] <= q67),
            group[yield_col] > q67
        ]
        choices = [0, 1, 2]  # 0=Low, 1=Medium, 2=High
        group['Yield_Category_encoded'] = np.select(conditions, choices, default=1)
        return group
    
    df = df.groupby('Crop', group_keys=False).apply(categorize_yield)
    return df

hybrid_train = create_yield_categories(hybrid_train)
hybrid_val = create_yield_categories(hybrid_val)
hybrid_test = create_yield_categories(hybrid_test)

print(f"  Category distribution (train): {hybrid_train['Yield_Category_encoded'].value_counts().sort_index().to_dict()}")

# Create sequences
sequence_length = 12

def create_hybrid_sequences(data, temporal_cols, static_cols, sequence_length=12):
    """Create sequences for Hybrid model"""
    X_temporal = []
    X_static = []
    y_targets = []
    
    grouped = data.groupby(['Year', 'Region', 'Crop'])
    
    for name, group in grouped:
        group = group.sort_values('Month')
        if len(group) >= sequence_length:
            temporal = group[temporal_cols].values[:sequence_length]
            static = group[static_cols].iloc[0].values
            target = group['Yield_Category_encoded'].iloc[0]
            
            X_temporal.append(temporal)
            X_static.append(static)
            y_targets.append(target)
    
    return (np.array(X_temporal), np.array(X_static), 
            tf.keras.utils.to_categorical(np.array(y_targets), num_classes=3))

print("\nCreating sequences...")
X_hybrid_temp_train, X_hybrid_stat_train, y_hybrid_train = create_hybrid_sequences(
    hybrid_train, temporal_features, static_features, sequence_length
)
X_hybrid_temp_val, X_hybrid_stat_val, y_hybrid_val = create_hybrid_sequences(
    hybrid_val, temporal_features, static_features, sequence_length
)
X_hybrid_temp_test, X_hybrid_stat_test, y_hybrid_test = create_hybrid_sequences(
    hybrid_test, temporal_features, static_features, sequence_length
)

print(f"  Train - Temporal: {X_hybrid_temp_train.shape}, Static: {X_hybrid_stat_train.shape}")
print(f"  Val   - Temporal: {X_hybrid_temp_val.shape}, Static: {X_hybrid_stat_val.shape}")
print(f"  Test  - Temporal: {X_hybrid_temp_test.shape}, Static: {X_hybrid_stat_test.shape}")

# Scale features
print("\nScaling features...")
scaler_hybrid_temp = StandardScaler()
scaler_hybrid_stat = StandardScaler()

# Reshape for scaling
n_samples, seq_len, n_temp = X_hybrid_temp_train.shape
X_temp_2d = X_hybrid_temp_train.reshape(-1, n_temp)
X_temp_scaled_2d = scaler_hybrid_temp.fit_transform(X_temp_2d)
X_hybrid_temp_train_scaled = X_temp_scaled_2d.reshape(n_samples, seq_len, n_temp)

# Val
n_samples_val, _, _ = X_hybrid_temp_val.shape
X_temp_val_2d = X_hybrid_temp_val.reshape(-1, n_temp)
X_temp_val_scaled_2d = scaler_hybrid_temp.transform(X_temp_val_2d)
X_hybrid_temp_val_scaled = X_temp_val_scaled_2d.reshape(n_samples_val, seq_len, n_temp)

# Test
n_samples_test, _, _ = X_hybrid_temp_test.shape
X_temp_test_2d = X_hybrid_temp_test.reshape(-1, n_temp)
X_temp_test_scaled_2d = scaler_hybrid_temp.transform(X_temp_test_2d)
X_hybrid_temp_test_scaled = X_temp_test_scaled_2d.reshape(n_samples_test, seq_len, n_temp)

# Scale static
X_hybrid_stat_train_scaled = scaler_hybrid_stat.fit_transform(X_hybrid_stat_train)
X_hybrid_stat_val_scaled = scaler_hybrid_stat.transform(X_hybrid_stat_val)
X_hybrid_stat_test_scaled = scaler_hybrid_stat.transform(X_hybrid_stat_test)

print("  ✓ Scaling complete")

# Define focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

# Build Hybrid CNN-GRU model
print("\nBuilding Hybrid CNN-GRU model with INCREASED dropout (0.4)...")

n_temporal_features = X_hybrid_temp_train_scaled.shape[2]
n_static_features = X_hybrid_stat_train_scaled.shape[1]

# Temporal input: Conv1D → GRU
temporal_input = layers.Input(shape=(sequence_length, n_temporal_features), name='temporal_input')

# Conv1D layers with INCREASED DROPOUT
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(0.001))(temporal_input)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)  # INCREASED to reduce overfitting

x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)  # INCREASED to reduce overfitting
x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Bidirectional GRU layers
x = layers.Bidirectional(layers.GRU(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                                   kernel_regularizer=regularizers.l2(0.01),
                                   recurrent_regularizer=regularizers.l2(0.01)))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Bidirectional(layers.GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2,
                                   kernel_regularizer=regularizers.l2(0.01),
                                   recurrent_regularizer=regularizers.l2(0.01)))(x)
x = layers.BatchNormalization()(x)
temporal_out = x

# Static branch
static_input = layers.Input(shape=(n_static_features,), name='static_input')
y = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(static_input)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.4)(y)

y = layers.Dense(96, activation='relu', kernel_regularizer=regularizers.l2(0.01))(y)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.35)(y)

y = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(y)
y = layers.BatchNormalization()(y)
static_out = layers.Dropout(0.3)(y)

# Fusion with residual
combined = layers.concatenate([temporal_out, static_out])

# Add residual shortcut
residual = layers.Dense(192, activation='linear', kernel_regularizer=regularizers.l2(0.01))(combined)

z = layers.Dense(192, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
z = layers.BatchNormalization()(z)
z = layers.Dropout(0.4)(z)

z = layers.add([z, residual])  # Residual connection

z = layers.Dense(96, activation='relu', kernel_regularizer=regularizers.l2(0.01))(z)
z = layers.BatchNormalization()(z)
z = layers.Dropout(0.35)(z)

z = layers.Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.01))(z)
z = layers.BatchNormalization()(z)

output = layers.Dense(3, activation='softmax', name='output')(z)

hybrid_model = models.Model(inputs=[temporal_input, static_input], outputs=output)

# Compile with focal loss
optimizer = Adam(learning_rate=0.0002, clipnorm=1.0)
hybrid_model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=3.0, alpha=0.25),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
             tf.keras.metrics.Recall(name='recall')]
)

print("\nModel Summary:")
hybrid_model.summary()

# Compute class weights
print("\nComputing class weights...")
y_train_labels = np.argmax(y_hybrid_train, axis=1)
class_counts = np.bincount(y_train_labels)
total = len(y_train_labels)
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
# Amplify weights for minority classes
class_weights = {k: v * 1.5 for k, v in class_weights.items()}
print(f"  Class weights: {class_weights}")

# Define callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    'models/hybrid_best.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Train model
print("\nTraining Hybrid model...")
print("="*80)

history = hybrid_model.fit(
    [X_hybrid_temp_train_scaled, X_hybrid_stat_train_scaled],
    y_hybrid_train,
    validation_data=([X_hybrid_temp_val_scaled, X_hybrid_stat_val_scaled], y_hybrid_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

print("\n✓ Training complete!")

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = hybrid_model.evaluate(
    [X_hybrid_temp_test_scaled, X_hybrid_stat_test_scaled],
    y_hybrid_test,
    verbose=0
)

print(f"\nTest Results:")
print(f"  Loss:      {test_results[0]:.4f}")
print(f"  Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
print(f"  Precision: {test_results[2]:.4f}")
print(f"  Recall:    {test_results[3]:.4f}")

# Get predictions for detailed analysis
y_pred_probs = hybrid_model.predict([X_hybrid_temp_test_scaled, X_hybrid_stat_test_scaled], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_hybrid_test, axis=1)

# Per-class accuracy
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Low', 'Medium', 'High']))

# Save model and scalers
print("\nSaving model and artifacts...")
Path('models').mkdir(exist_ok=True)

hybrid_model.save('models/hybrid_model.keras')
print("  ✓ models/hybrid_model.keras")

joblib.dump(scaler_hybrid_temp, 'models/hybrid_temp_scaler.pkl')
joblib.dump(scaler_hybrid_stat, 'models/hybrid_stat_scaler.pkl')
print("  ✓ Scalers saved")

joblib.dump(le_crop, 'models/crop_encoder.pkl')
joblib.dump(le_region, 'models/region_encoder.pkl')
print("  ✓ Encoders saved")

print("\n" + "="*80)
print("RETRAINING COMPLETE!")
print("="*80)
print(f"\nExpected: Accuracy should be 75-82% (more realistic than previous 90.74%)")
print(f"Actual: {test_results[1]*100:.2f}%")
print("\nThe model now has stronger regularization to prevent overfitting.")
