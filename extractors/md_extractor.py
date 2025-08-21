from __future__ import annotations

from .registry import register_extension


def _extract_md(file_bytes: bytes) -> str:
    # Decode and strip Markdown to text by removing basic formatting
    try:
        txt = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        txt = file_bytes.decode("latin-1", errors="replace")

    # Very light cleanup: remove code fences and markdown headers/emphasis
    import re
    txt = re.sub(r"```[\s\S]*?```", "", txt)  # remove fenced code blocks
    txt = re.sub(r"^\s{0,3}#{1,6}\s*", "", txt, flags=re.MULTILINE)  # headers
    txt = re.sub(r"\*\*([^*]+)\*\*", r"\1", txt)  # bold
    txt = re.sub(r"\*([^*]+)\*", r"\1", txt)      # italics
    txt = re.sub(r"_([^_]+)_", r"\1", txt)          # underscores
    txt = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", txt)  # images
    txt = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", txt)  # links -> url
    return txt.strip()


for _ext in [".md", ".markdown", ".mdown"]:
    register_extension(_ext, _extract_md)


