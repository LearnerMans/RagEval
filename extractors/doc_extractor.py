from __future__ import annotations

from .registry import register_extension


def _extract_doc(file_bytes: bytes) -> str:
    # Try textract (may require system deps), otherwise return empty
    try:
        import textract  # type: ignore
        return textract.process(None, input_encoding=None, extension='doc', stream=file_bytes).decode('utf-8', errors='replace').strip()
    except Exception:
        # As a fallback, try antiword via textract if present
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix='.doc', delete=True) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                out = subprocess.check_output(['antiword', tmp.name], stderr=subprocess.DEVNULL)
                return out.decode('utf-8', errors='replace').strip()
        except Exception:
            return ""


register_extension(".doc", _extract_doc)


