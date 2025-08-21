from __future__ import annotations

from .registry import register_extension


def _extract_text(file_bytes: bytes) -> str:
    # Attempt to decode as UTF-8; if it fails, fall back to latin-1
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


# Register for common plain text extensions
for _ext in [".txt", ".text", ".log"]:
    register_extension(_ext, _extract_text)


