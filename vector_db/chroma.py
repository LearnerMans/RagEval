# chroma_singleton.py
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import chromadb

# ---- Globals ----
_CHROMA_CLIENT: Optional["chromadb.api.client.ClientAPI"] = None
_LOCK = threading.RLock()
_PERSIST_DIR: Optional[str] = None


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _make_client(persist_dir: str) -> "chromadb.api.client.ClientAPI":
    # Support both 0.5+ and older Chroma APIs
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=persist_dir)
    from chromadb.config import Settings  # type: ignore
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))


def open_client(persist_dir: str) -> "chromadb.api.client.ClientAPI":
    """
    Create (if needed) and return a singleton Chroma client bound to `persist_dir`.
    Safe to call multiple times; returns the same instance.
    """
    global _CHROMA_CLIENT, _PERSIST_DIR
    with _LOCK:
        # Already open?
        if _CHROMA_CLIENT is not None:
            # Safety: if caller passed a different directory, surface it early.
            if _PERSIST_DIR and os.path.abspath(persist_dir) != os.path.abspath(_PERSIST_DIR):
                raise RuntimeError(
                    f"Chroma client already open at {_PERSIST_DIR}, but requested {persist_dir}."
                )
            return _CHROMA_CLIENT

        persist_dir = _ensure_dir(persist_dir)
        client = _make_client(persist_dir)

        # Optional sanity/health check: make a cheap API call
        try:
            _ = client.list_collections()  # proves the backend is reachable/ready
        except Exception as e:
            # If the backend is mid-migration or locked, fail fast with context
            raise RuntimeError(f"Failed to initialize Chroma at {persist_dir}: {e}") from e

        _CHROMA_CLIENT = client
        _PERSIST_DIR = persist_dir
        return _CHROMA_CLIENT


def get_client() -> "chromadb.api.client.ClientAPI":
    """Return the active client; raise if not opened yet."""
    if _CHROMA_CLIENT is None:
        raise RuntimeError("Chroma client not initialized. Call open_client(persist_dir) first.")
    return _CHROMA_CLIENT


def close_client() -> None:
    """
    Best-effort close/reset. Chroma doesn't need an explicit close for
    DuckDB/Parquet, but we null out the handle so a fresh open is possible.
    """
    global _CHROMA_CLIENT, _PERSIST_DIR
    with _LOCK:
        # Some versions expose ._client or ._db with close methods, but it's not stable.
        _CHROMA_CLIENT = None
        _PERSIST_DIR = None
