"""Quick model loading test with full error output"""
import os
os.chdir(r'c:\Users\ibito\Documents\Final_Year_Project')

print("="*80)
print("TESTING MODEL LOADING")
print("="*80)

# Check files exist
from pathlib import Path
print("\n1. Checking files...")
for model in ['cnn_model.keras', 'gru_model.keras', 'hybrid_model.keras']:
    path = Path('models') / model
    print(f"  {model}: {'EXISTS' if path.exists() else 'MISSING'} ({path})")

# Try loading
print("\n2. Importing TensorFlow...")
import tensorflow as tf
print(f"  TensorFlow: {tf.__version__}")

print("\n3. Defining focal loss...")
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

custom_objects = {'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25)}
print("  ✓ Focal loss defined")

print("\n4. Loading CNN model...")
try:
    cnn = tf.keras.models.load_model('models/cnn_model.keras', custom_objects=custom_objects, compile=False)
    print(f"  ✓ CNN loaded: {type(cnn)}")
except Exception as e:
    print(f"  ✗ CNN FAILED: {type(e).__name__}: {e}")

print("\n5. Loading GRU model...")
try:
    gru = tf.keras.models.load_model('models/gru_model.keras', custom_objects=custom_objects, compile=False)
    print(f"  ✓ GRU loaded: {type(gru)}")
except Exception as e:
    print(f"  ✗ GRU FAILED: {type(e).__name__}: {e}")

print("\n6. Loading Hybrid model...")
try:
    hybrid = tf.keras.models.load_model('models/hybrid_model.keras', custom_objects=custom_objects, compile=False)
    print(f"  ✓ Hybrid loaded: {type(hybrid)}")
except Exception as e:
    print(f"  ✗ Hybrid FAILED: {type(e).__name__}: {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
