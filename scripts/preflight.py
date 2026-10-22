"""Run the organiser's public evaluator before data is available.

No private tracking data is transmitted or uploaded.
"""
from __future__ import annotations
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_FILES = (
    'evaluation/run_evaluation.py', 'evaluation/validate_submission.py',
    'evaluation/possession_metrics.py', 'starter/baseline.py',
    'games.py', 'DATA.md', 'RULES.md',
)
DATA_FILES = (
    'tactical/BAR-ATH_tactical_atd.xml',
    'tactical/BAR-ATH_tactical_atd.txt',
    'tactical/BAR-ATH_tactical_smart_tagging.xml',
    'tactical/BAR-ATH_tactical_homography.csv',
)

def main():
    root = Path(__file__).resolve().parent.parent
    print(f'PulsoXI preflight | Python: {sys.version.split()[0]}')
    print(f'Official-repository root: {root}')
    for name in EXPECTED_FILES:
        print(f'{"OK" if (root / name).is_file() else "MISSING"}  {name}')
    for module in ('numpy', 'pandas', 'kloppy'):
        print(f'{module}: {"installed" if importlib.util.find_spec(module) else "not installed"}')

    data_root = Path(os.environ.get('MTAH_DATA_DIR') or (root / 'data'))
    print(f'Local licensed data root: {data_root}')
    for name in DATA_FILES:
        print(f'{"present" if (data_root / name).is_file() else "not yet available"}: {name}')

    evaluator = root / 'evaluation/run_evaluation.py'
    if evaluator.is_file():
        print('\nRunning the OFFICIAL scorer on the publicly supplied example CSVs:')
        proc = subprocess.run([sys.executable, str(evaluator)], cwd=root, check=False)
        print(f'Official public scorer exit code: {proc.returncode}')
        return proc.returncode
    print('\nNo official scorer found. Put this kit inside a clone of the official repo.')
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
