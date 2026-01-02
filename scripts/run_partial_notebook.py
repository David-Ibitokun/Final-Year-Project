import nbformat
import sys
import subprocess
import tempfile
import os
from pathlib import Path

NBPATH = sys.argv[1] if len(sys.argv) > 1 else 'phase4_validation.ipynb'
STOP_MARKER = '## 3.3 LSTM Model Validation'
TIMEOUT = 120  # seconds per cell
LOGFILE = 'partial_exec_log.txt'

print(f"Partial executor starting for notebook: {NBPATH}")
sys.stdout.flush()

nb = nbformat.read(NBPATH, as_version=4)
project_dir = Path(NBPATH).resolve().parent
print(f"Notebook loaded. Total cells: {len(nb.cells)}")
sys.stdout.flush()

with open(LOGFILE, 'w', encoding='utf-8') as log:
    log.write(f'Executing notebook: {NBPATH}\n')
    # Collect code cells up to stop marker and concatenate into one script
    code_blocks = []
    cell_idx = 0
    for cell in nb.cells:
        cell_idx += 1
        if cell.get('cell_type') == 'markdown':
            src = ''.join(cell.get('source', []))
            log.write(f"\n---\nCell {cell_idx} (markdown)\n{src[:200]}\n---\n")
            if STOP_MARKER in src:
                log.write(f"Encountered stop marker at cell {cell_idx}.\n")
                break
            continue
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if not src.strip():
            continue
        # Skip heavy training cells heuristically
        if ('model.fit(' in src or '.fit(' in src) and 'predict' not in src:
            log.write(f"Skipping potential training cell {cell_idx}.\n")
            continue
        # Add cell source with separator
        code_blocks.append(f"# ---- cell {cell_idx} ----\n{src}\n\n")

    if not code_blocks:
        log.write('No code blocks collected. Exiting.\n')
    else:
        combined = '\n'.join(code_blocks)
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as tf:
            tf.write('# Combined notebook cells (partial) - auto-generated\n')
            tf.write('import warnings\nwarnings.filterwarnings(\'ignore\')\n')
            tf.write(combined)
            combined_py = tf.name
        log.write(f"\nExecuting combined script -> {combined_py}\n")
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            res = subprocess.run([sys.executable, combined_py], cwd=project_dir, capture_output=True, text=True, timeout=TIMEOUT*10, env=env)
            log.write('--- STDOUT ---\n')
            log.write(res.stdout or '')
            log.write('\n--- STDERR ---\n')
            log.write(res.stderr or '')
            log.write('\nReturn code: ' + str(res.returncode) + '\n')
        except subprocess.TimeoutExpired as e:
            log.write(f"Combined script timed out after {TIMEOUT*10}s.\n")
            log.write(str(e) + '\n')
        finally:
            try:
                os.remove(combined_py)
            except Exception:
                pass

print('Partial execution complete. See', LOGFILE)
