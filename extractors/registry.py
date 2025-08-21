from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional

# Type alias for extractor functions
Extractor = Callable[[bytes], str]


def _ext(path: str | os.PathLike[str]) -> str:
    return Path(path).suffix.lower()


_REGISTRY: Dict[str, Extractor] = {}


def register_extension(ext: str, extractor: Extractor) -> None:
    if not ext.startswith("."):
        raised = ValueError(f"Extension must start with '.', got: {ext}")
        raise raised
    _REGISTRY[ext.lower()] = extractor


def get_extractor_for_path(path: str | os.PathLike[str]) -> Optional[Extractor]:
    return _REGISTRY.get(_ext(path))


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a file's bytes by its extension.

    Args:
        file_bytes: The raw file content as bytes.
        filename: The original filename (used to determine extension).

    Returns:
        Extracted text as a UTF-8 string.

    Raises:
        ValueError: If no extractor is registered for the file extension.
    """

    ext = _ext(filename)
    extractor = _REGISTRY.get(ext)
    if extractor is None:
        raise ValueError(f"No extractor registered for extension: {ext}")
    return extractor(file_bytes)


# Import side-effect registrations
from . import text_extractor  # noqa: F401
from . import pdf_extractor  # noqa: F401
from . import csv_extractor  # noqa: F401
from . import excel_extractor  # noqa: F401
from . import docx_extractor  # noqa: F401
from . import doc_extractor  # noqa: F401
from . import md_extractor  # noqa: F401

