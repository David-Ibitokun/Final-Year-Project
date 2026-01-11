"""Extract validation results from executed notebook"""
import json

with open('phase4_validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 80)
print("PHASE 4 VALIDATION RESULTS SUMMARY")
print("=" * 80)

# Find cells with specific outputs
for i, cell in enumerate(nb['cells'], 1):
    if cell.get('cell_type') != 'code':
        continue
    
    outputs = cell.get('outputs', [])
    if not outputs:
        continue
    
    # Extract text output
    for output in outputs:
        if output.get('output_type') == 'stream':
            text = ''.join(output.get('text', []))
            
            # Print key results
            if 'models loaded successfully' in text.lower():
                print(f"\nCell {i} (Model Loading):")
                print(text)
            
            if 'CNN predictions:' in text or 'GRU predictions:' in text or 'Hybrid predictions:' in text:
                print(f"\nCell {i} (Predictions):")
                print(text)
            
            if 'Best Performing Model:' in text:
                print(f"\nCell {i} (Best Model):")
                print(text)

print("\n" + "=" * 80)
print("Extraction complete!")
