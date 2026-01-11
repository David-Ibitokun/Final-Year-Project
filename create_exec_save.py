"""
Direct save using exec - run this in Jupyter environment
"""
import os
os.chdir('c:/Users/ibito/Documents/Final_Year_Project')

# Import what we need
from pathlib import Path
import joblib

# Exec code to save (this will run in the phase3 kernel context if imported there)
save_script = """
from pathlib import Path
import joblib
import os

print("="*80)
print("SAVING MODELS")
print("="*80)

Path('models').mkdir(exist_ok=True)

print("\\n💾 Saving CNN model...")
cnn_model.save('models/cnn_model.keras')
print("  ✓ models/cnn_model.keras")

print("\\n💾 Saving GRU model...")
gru_model.save('models/gru_model.keras')
print("  ✓ models/gru_model.keras")

print("\\n💾 Saving Hybrid model...")
hybrid_model.save('models/hybrid_model.keras')
print("  ✓ models/hybrid_model.keras")

print("\\n💾 Saving scalers...")
joblib.dump(scaler_cnn, 'models/cnn_scaler.pkl')
joblib.dump(scaler_gru, 'models/gru_scaler.pkl')
joblib.dump(scaler_hybrid_temp, 'models/hybrid_temp_scaler.pkl')
joblib.dump(scaler_hybrid_stat, 'models/hybrid_stat_scaler.pkl')
print("  ✓ All scalers saved")

print("\\n💾 Saving encoders...")
joblib.dump(crop_encoder, 'models/crop_encoder.pkl')
joblib.dump(region_encoder, 'models/region_encoder.pkl')
print("  ✓ All encoders saved")

print("\\n📁 Files created:")
for f in sorted(os.listdir('models')):
    size = os.path.getsize(f'models/{f}') / (1024*1024)
    print(f"  {f:<35} {size:>8.2f} MB")

print("\\n" + "="*80)
print("✅ ALL FILES SAVED!")
print("="*80)
"""

# Save as a file that can be imported or exec'd in notebook
with open('EXEC_SAVE_MODELS.py', 'w', encoding='utf-8') as f:
    f.write(save_script)

print("✓ Created EXEC_SAVE_MODELS.py")
print("\nTo save models, run this in phase3_model_dev.ipynb:")
print("  exec(open('EXEC_SAVE_MODELS.py').read())")
