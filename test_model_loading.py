"""
Test if models can be loaded with focal_loss custom_objects
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Define focal loss function (must match training definition)
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

# Create custom objects dict for model loading
custom_objects = {'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25)}

print("Testing model loading...")
print("=" * 80)

# Try loading each model
models_to_test = {
    'CNN': 'models/cnn_model.keras',
    'GRU': 'models/gru_model.keras',
    'Hybrid': 'models/hybrid_model.keras'
}

for name, path in models_to_test.items():
    print(f"\nTesting {name} model...")
    if Path(path).exists():
        print(f"  ✓ File exists: {path}")
        try:
            model = keras.models.load_model(path, custom_objects=custom_objects)
            print(f"  ✓ Successfully loaded {name} model")
            print(f"  ✓ Model summary: {model.count_params()} parameters")
        except Exception as e:
            print(f"  ✗ Failed to load {name} model")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ✗ File not found: {path}")

print("\n" + "=" * 80)
print("Test complete")
