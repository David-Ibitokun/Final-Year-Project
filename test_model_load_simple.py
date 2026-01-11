"""Simple test to check if models can be loaded"""
import sys
print("Python:", sys.version)

print("\nImporting TensorFlow...")
import tensorflow as tf
print("TensorFlow:", tf.__version__)

print("\nDefining focal loss...")
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

print("\nAttempting to load CNN model...")
try:
    cnn_model = tf.keras.models.load_model('models/cnn_model.keras', custom_objects=custom_objects, compile=False)
    print("✓ CNN model loaded successfully!")
    print(f"  Model type: {type(cnn_model)}")
    print(f"  Input shape: {cnn_model.input_shape}")
    print(f"  Output shape: {cnn_model.output_shape}")
except Exception as e:
    print(f"✗ Failed to load CNN model: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to load GRU model...")
try:
    gru_model = tf.keras.models.load_model('models/gru_model.keras', custom_objects=custom_objects, compile=False)
    print("✓ GRU model loaded successfully!")
    print(f"  Model type: {type(gru_model)}")
    print(f"  Input shape: {gru_model.input_shape}")
    print(f"  Output shape: {gru_model.output_shape}")
except Exception as e:
    print(f"✗ Failed to load GRU model: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to load Hybrid model...")
try:
    hybrid_model = tf.keras.models.load_model('models/hybrid_model.keras', custom_objects=custom_objects, compile=False)
    print("✓ Hybrid model loaded successfully!")
    print(f"  Model type: {type(hybrid_model)}")
    if hasattr(hybrid_model, 'input_shape'):
        print(f"  Input shape: {hybrid_model.input_shape}")
    else:
        print(f"  Inputs: {[inp.shape for inp in hybrid_model.inputs]}")
    print(f"  Output shape: {hybrid_model.output_shape}")
except Exception as e:
    print(f"✗ Failed to load Hybrid model: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete!")
