from __future__ import annotations

import csv
import io
from .registry import register_extension


def _extract_csv(file_bytes: bytes) -> str:
    # Try utf-8-sig first to strip BOM if present
    text_io = io.StringIO(file_bytes.decode("utf-8-sig", errors="replace"))
    reader = csv.reader(text_io)
    rows = [", ".join(cell.strip() for cell in row) for row in reader]
    return "\n".join(rows).strip()


for _ext in [".csv", ".tsv"]:
    register_extension(_ext, _extract_csv)


