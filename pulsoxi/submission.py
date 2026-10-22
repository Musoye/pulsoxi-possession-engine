"""Write a predictions CSV compliant with the official format.

Only use with real algorithm-generated frame labels; do not manually edit
final predictions. The official validator remains authoritative.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Sequence
from .temporal import N_FRAMES

ALLOWED = {"H", "A", "D", "X"}


def write_submission(path: str | Path, labels: Sequence[str]) -> None:
    if len(labels) != N_FRAMES:
        raise ValueError(f"required exactly {N_FRAMES} labels, received {len(labels)}")
    if any(str(y) not in ALLOWED for y in labels):
        raise ValueError("invalid label: only H, A, D, X are accepted")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(("frame", "label"))
        writer.writerows((i, str(y)) for i, y in enumerate(labels))
