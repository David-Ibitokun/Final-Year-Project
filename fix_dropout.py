"""Fix dropout values in phase3_model_dev.ipynb"""
import json
from pathlib import Path

notebook_path = Path('phase3_model_dev.ipynb')

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Track changes
changes_made = 0

# Iterate through cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this cell contains the dropout lines we need to change
        for i, line in enumerate(source):
            # Look for the specific dropout lines in hybrid model
            if 'x = layers.Dropout(0.25)(x)  # Reduced from 0.4' in line:
                # Replace with increased dropout
                source[i] = line.replace(
                    'x = layers.Dropout(0.25)(x)  # Reduced from 0.4',
                    'x = layers.Dropout(0.4)(x)  # INCREASED to reduce overfitting'
                )
                changes_made += 1
                print(f"Changed dropout in line: {line.strip()}")

print(f"\nTotal changes made: {changes_made}")

if changes_made > 0:
    # Write back to notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Updated {notebook_path}")
else:
    print("⚠ No matching lines found to change")
