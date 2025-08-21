from __future__ import annotations

from .registry import register_extension


def _extract_pdf(file_bytes: bytes) -> str:
    # Prefer pypdf (lightweight); fall back to pdfminer.six if unavailable
    text_parts: list[str] = []
    try:
        from pypdf import PdfReader  # type: ignore
        import io

        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception:
        # Fallback to pdfminer.six
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract  # type: ignore
            import io

            return (pdfminer_extract(io.BytesIO(file_bytes)) or "").strip()
        except Exception:
            return ""


register_extension(".pdf", _extract_pdf)


