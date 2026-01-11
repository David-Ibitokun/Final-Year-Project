"""
Test model loading directly
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import sys

print("="*80)
print("TESTING MODEL LOADING")
print("="*80)

# Define focal loss
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

# Try loading hybrid model
model_path = 'models/hybrid_model.keras'
print(f"\nAttempting to load: {model_path}")
print(f"File exists: {Path(model_path).exists()}")

if Path(model_path).exists():
    try:
        print("Loading with compile=False...")
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ SUCCESS! Model loaded")
        print(f"   Model type: {type(model)}")
        print(f"   Model inputs: {len(model.inputs)}")
        print(f"   Input shapes: {[inp.shape for inp in model.inputs]}")
        print(f"   Output shape: {model.output.shape}")
    except Exception as e:
        print(f"❌ FAILED!")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("❌ File not found!")

print("\n" + "="*80)
