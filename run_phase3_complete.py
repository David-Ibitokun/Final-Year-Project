"""
Run phase3_model_dev.ipynb cells programmatically
This script extracts and executes all code cells from the notebook
"""
import json
import sys
from pathlib import Path

# Load the notebook
notebook_path = Path(__file__).parent / "phase3_model_dev.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Found {len(notebook['cells'])} cells in notebook")
print("=" * 80)

# Execute each code cell
for i, cell in enumerate(notebook['cells'], 1):
    if cell['cell_type'] == 'code':
        print(f"\n{'='*80}")
        print(f"Executing Cell {i} (lines {cell.get('metadata', {}).get('line', 'unknown')})")
        print(f"{'='*80}")
        
        code = ''.join(cell['source'])
        
        try:
            exec(code, globals())
            print(f"✓ Cell {i} executed successfully")
        except Exception as e:
            print(f"✗ Cell {i} failed with error:")
            print(f"  {type(e).__name__}: {e}")
            print(f"\nCode:")
            print(code[:500])
            
            response = input("\nContinue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    elif cell['cell_type'] == 'markdown':
        print(f"\nSkipping markdown cell {i}")

print("\n" + "="*80)
print("All cells executed!")
print("="*80)
