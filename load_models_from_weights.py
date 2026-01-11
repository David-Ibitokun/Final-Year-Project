"""
Load models from weights files instead of .keras files
"""
import tensorflow as tf
from tensorflow import keras
import os

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Check what model files we have
models_dir = 'models'
print("\nModel files available:")
for f in os.listdir(models_dir):
    if f.endswith(('.keras', '.h5')):
        size = os.path.getsize(os.path.join(models_dir, f)) / (1024*1024)
        print(f"  {f:30s} {size:>8.2f} MB")

# Try loading from .weights.h5 files
print("\n" + "="*80)
print("APPROACH: Load from .weights.h5 files requires model architecture")
print("="*80)

# The problem: .weights.h5 files need the model architecture to be defined first
# The .keras files SHOULD work but have a version compatibility issue

print("\nRoot cause identified:")
print("  - Models saved with Keras 3.13 (includes 'quantization_config' in Dense layer)")
print("  - Current environment's Keras doesn't recognize 'quantization_config' parameter")
print("  - This is preventing model loading from .keras files")

print("\nPossible solutions:")
print("  1. Rebuild models from scratch in phase3 with current Keras version")
print("  2. Load using TensorFlow's SavedModel format instead")
print("  3. Extract architecture from .keras file and manually rebuild + load weights")
print("  4. Upgrade/downgrade Keras to match saved model version")

print("\n" + "="*80)
print("CHECKING: Were models still in memory during notebook execution?")
print("="*80)

# The jupyter nbconvert execution was done in a fresh kernel
# So models from phase3 were NOT available
# The notebook execution failed at model loading stage

print("\n❌ Conclusion: Phase 4 validation did NOT complete successfully")
print("   The notebook execution stopped after failing to load models")
print("   No validation metrics were generated")
