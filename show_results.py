"""Show phase4 validation results"""
import json

with open('phase4_validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells_with_output = [c for c in nb['cells'] if c['cell_type'] == 'code' and c.get('outputs')]

print("=" * 80)
print("PHASE 4 VALIDATION RESULTS")
print("=" * 80)

# Cell 5: Model Loading
print("\n### CELL 5: MODEL LOADING ###")
for output in code_cells_with_output[1]['outputs']:
    if output['output_type'] == 'stream':
        print(''.join(output['text']))

# Cell 6: Quick Status Check
print("\n### CELL 6: MODEL STATUS CHECK ###")
for output in code_cells_with_output[2]['outputs']:
    if output['output_type'] == 'stream':
        print(''.join(output['text']))

# Cell 8: CNN Predictions
print("\n### CELL 8: CNN MODEL PREDICTIONS ###")
for output in code_cells_with_output[3]['outputs']:
    if output['output_type'] == 'stream':
        text = ''.join(output['text'])
        if 'CNN predictions:' in text or 'Accuracy:' in text:
            print(text)

#  Cell 11: GRU Predictions
print("\n### CELL 11: GRU MODEL PREDICTIONS ###")
for output in code_cells_with_output[4]['outputs']:
    if output['output_type'] == 'stream':
        text = ''.join(output['text'])
        if 'GRU predictions:' in text or 'Accuracy:' in text:
            print(text)

# Cell 14: Hybrid Predictions
print("\n### CELL 14: HYBRID MODEL PREDICTIONS ###")
for output in code_cells_with_output[5]['outputs']:
    if output['output_type'] == 'stream':
        text = ''.join(output['text'])
        if 'Hybrid predictions:' in text or 'Accuracy:' in text:
            print(text)

# Cell 17: Model Comparison
print("\n### CELL 17: MODEL COMPARISON ###")
if len(code_cells_with_output) > 7:
    for output in code_cells_with_output[7]['outputs']:
        if output['output_type'] == 'stream':
            text = ''.join(output['text'])
            if 'Comparison' in text or 'Best' in text:
                print(text)

print("\n" + "=" * 80)
