"""
Script to execute phase4_validation.ipynb and capture key results
"""
import subprocess
import sys

print("=" * 80)
print("EXECUTING PHASE 4 VALIDATION NOTEBOOK")
print("=" * 80)

# Execute the notebook using jupyter nbconvert
result = subprocess.run([
    sys.executable, '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook',
    '--execute',
    '--inplace',
    '--ExecutePreprocessor.timeout=3600',
    'phase4_validation.ipynb'
], capture_output=True, text=True)

if result.returncode == 0:
    print("\n✅ Notebook executed successfully!")
    print("Check phase4_validation.ipynb for results.")
else:
    print("\n❌ Notebook execution failed:")
    print(result.stderr)
    sys.exit(1)
