"""Save models from phase3 notebook kernel to disk"""
import sys
import os
os.chdir(r'c:\Users\ibito\Documents\Final_Year_Project')

# This script will be run in the phase3 notebook kernel
# to save the trained models that are currently in memory

import joblib
from pathlib import Path

print("="*80)
print("SAVING MODELS TO DISK")
print("="*80)

# Ensure models directory exists
Path('models').mkdir(exist_ok=True)

# Check if models are in memory
try:
    print("\n1. Checking if models exist in kernel...")
    print(f"  CNN model: {type(cnn_model)}")
    print(f"  GRU model: {type(gru_model)}")
    print(f"  Hybrid model: {type(hybrid_model)}")
    
    print("\n2. Saving models to .keras files...")
    cnn_model.save('models/cnn_model.keras')
    print("  ✓ CNN model saved")
    
    gru_model.save('models/gru_model.keras')
    print("  ✓ GRU model saved")
    
    hybrid_model.save('models/hybrid_model.keras')
    print("  ✓ Hybrid model saved")
    
    print("\n3. Saving scalers...")
    joblib.dump(scaler_cnn, 'models/cnn_scaler.pkl')
    print("  ✓ CNN scaler saved")
    
    joblib.dump(scaler_gru, 'models/gru_scaler.pkl')
    print("  ✓ GRU scaler saved")
    
    joblib.dump(scaler_hybrid_temp, 'models/hybrid_temp_scaler.pkl')
    print("  ✓ Hybrid temp scaler saved")
    
    joblib.dump(scaler_hybrid_stat, 'models/hybrid_stat_scaler.pkl')
    print("  ✓ Hybrid stat scaler saved")
    
    print("\n4. Saving encoders...")
    joblib.dump(crop_encoder, 'models/crop_encoder.pkl')
    print("  ✓ Crop encoder saved")
    
    joblib.dump(region_encoder, 'models/region_encoder.pkl')
    print("  ✓ Region encoder saved")
    
    print("\n" + "="*80)
    print("✅ ALL MODELS SAVED SUCCESSFULLY!")
    print("="*80)
    print("\nFiles saved:")
    import os
    for file in sorted(os.listdir('models')):
        size = os.path.getsize(f'models/{file}') / (1024*1024)
        print(f"  {file:<30} {size:>8.2f} MB")
    
except NameError as e:
    print(f"\n❌ ERROR: Variable not found - {e}")
    print("\nThis script must be run in the phase3_model_dev.ipynb kernel")
    print("where the models have been trained and are in memory.")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
