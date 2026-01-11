"""Direct test to diagnose model loading issues"""
import sys
import os
os.chdir('c:/Users/ibito/Documents/Final_Year_Project')

# Minimal imports
import tensorflow as tf
from tensorflow import keras
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")

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

print("\n" + "="*80)
print("LOADING CNN MODEL")
print("="*80)
try:
    cnn_model = keras.models.load_model('models/cnn_model.keras', 
                                        custom_objects=custom_objects, 
                                        compile=False)
    print(f"✅ SUCCESS! CNN model loaded")
    print(f"  Type: {type(cnn_model)}")
    print(f"  Input: {cnn_model.input_shape}")
    print(f"  Output: {cnn_model.output_shape}")
except Exception as e:
    print(f"❌ FAILED! Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("LOADING GRU MODEL")
print("="*80)
try:
    gru_model = keras.models.load_model('models/gru_model.keras',
                                        custom_objects=custom_objects,
                                        compile=False)
    print(f"✅ SUCCESS! GRU model loaded")
    print(f"  Type: {type(gru_model)}")
    print(f"  Input: {gru_model.input_shape}")
    print(f"  Output: {gru_model.output_shape}")
except Exception as e:
    print(f"❌ FAILED! Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("LOADING HYBRID MODEL")
print("="*80)
try:
    hybrid_model = keras.models.load_model('models/hybrid_model.keras',
                                          custom_objects=custom_objects,
                                          compile=False)
    print(f"✅ SUCCESS! Hybrid model loaded")
    print(f"  Type: {type(hybrid_model)}")
    print(f"  Inputs: {[inp.shape for inp in hybrid_model.inputs]}")
    print(f"  Output: {hybrid_model.output_shape}")
except Exception as e:
    print(f"❌ FAILED! Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
