# embedded_chroma.py
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TypedDict

# Use your singleton
from chroma import get_client
import chromadb


# ---------- Types ----------
class UpsertItem(TypedDict, total=False):
    id: str
    embedding: Sequence[float]
    document: Optional[str]
    metadata: Dict[str, Any]

QueryRow = Dict[str, Any]  # {id, distance, document, metadata, collection_name}
QueryResult = List[QueryRow]


# ---------- Internal helpers ----------
def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()

def _derive_name(project_id: str, test_id: str, name: Optional[str]) -> str:
    base = f"proj={_slug(project_id)}__test={_slug(test_id)}"
    return f"{base}__{_slug(name)}" if name else base


# ---------- Collection helpers ----------
def get_or_create_collection(
    *,
    project_id: str,
    test_id: str,
    name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    client: Optional["chromadb.api.client.ClientAPI"] = None,
) -> "chromadb.api.models.Collection.Collection":
    """
    Get or create a collection bound to (project_id, test_id[, name]).
    Defaults to the singleton client if `client` is not provided.
    """
    client = client or get_client()
    coll_name = _derive_name(project_id, test_id, name)
    metadata = {"project_id": project_id, "test_id": test_id}
    if extra_metadata:
        metadata.update(extra_metadata)

    # Try fetch first; sanity-check metadata if present
    try:
        coll = client.get_collection(coll_name)
        meta = getattr(coll, "metadata", {}) or {}
        if meta.get("project_id") not in (None, project_id) or meta.get("test_id") not in (None, test_id):
            raise ValueError(
                f"Existing collection '{coll_name}' has conflicting metadata: {meta}. "
                f"Expected project_id={project_id}, test_id={test_id}."
            )
        return coll
    except Exception:
        # Not found or incompatible → create
        return client.get_or_create_collection(name=coll_name, metadata=metadata)


# ---------- Bulk Upsert ----------
def bulk_upsert(
    *,
    project_id: str,
    test_id: str,
    items: Sequence[UpsertItem],
    name: Optional[str] = None,
    default_metadata: Optional[Dict[str, Any]] = None,
    client: Optional["chromadb.api.client.ClientAPI"] = None,
) -> None:
    """
    Upsert vectors into the (project_id, test_id[, name]) collection.
    Each item must include: id (str), embedding (Sequence[float]).
    """
    if not items:
        return

    coll = get_or_create_collection(
        project_id=project_id, test_id=test_id, name=name, client=client
    )

    ids: List[str] = []
    embeddings: List[Sequence[float]] = []
    documents: List[Optional[str]] = []
    metadatas: List[Dict[str, Any]] = []

    for it in items:
        if "id" not in it or "embedding" not in it:
            raise ValueError("Each item must include 'id' and 'embedding'.")
        ids.append(it["id"])
        embeddings.append(it["embedding"])
        documents.append(it.get("document"))
        md = dict(default_metadata or {})
        md.update(it.get("metadata") or {})
        md.setdefault("project_id", project_id)
        md.setdefault("test_id", test_id)
        metadatas.append(md)

    coll.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


# ---------- Query (single collection) ----------
def query_collection(
    *,
    project_id: str,
    test_id: str,
    query_embeddings: Sequence[Sequence[float]],
    name: Optional[str] = None,
    n_results: int = 10,
    where: Optional[Mapping[str, Any]] = None,
    where_document: Optional[Mapping[str, Any]] = None,
    client: Optional["chromadb.api.client.ClientAPI"] = None,
) -> QueryResult:
    """
    Query a specific (project_id, test_id[, name]) collection with precomputed embeddings.
    Returns: [{id, distance, document, metadata, collection_name}, ...]
    """
    coll = get_or_create_collection(
        project_id=project_id, test_id=test_id, name=name, client=client
    )
    coll_name = _derive_name(project_id, test_id, name)

    out = coll.query(
        query_embeddings=list(query_embeddings),
        n_results=n_results,
        where=dict(where) if where else None,
        where_document=dict(where_document) if where_document else None,
    )

    results: QueryResult = []
    for ids, dists, docs, mds in zip(
        out.get("ids", []),
        out.get("distances", []),
        out.get("documents", []),
        out.get("metadatas", []),
    ):
        for i, _id in enumerate(ids):
            results.append(
                {
                    "id": _id,
                    "distance": dists[i] if dists else None,
                    "document": docs[i] if docs else None,
                    "metadata": mds[i] if mds else None,
                    "collection_name": coll_name,
                }
            )
    return results


# ---------- Query (all collections under project/test) ----------
def query_project_test(
    *,
    project_id: str,
    test_id: str,
    query_embeddings: Sequence[Sequence[float]],
    n_results_per_collection: int = 10,
    where: Optional[Mapping[str, Any]] = None,
    where_document: Optional[Mapping[str, Any]] = None,
    client: Optional["chromadb.api.client.ClientAPI"] = None,
) -> QueryResult:
    """
    Query across *all* collections belonging to (project_id, test_id) and merge results.
    Results are sorted by ascending distance (when available).
    """
    client = client or get_client()
    merged: QueryResult = []

    # Prefer metadata match, but also allow name prefix (backward-compat)
    prefix = _derive_name(project_id, test_id, None)
    for c in client.list_collections():
        try:
            meta = getattr(c, "metadata", {}) or {}
            is_match = (meta.get("project_id") == project_id and meta.get("test_id") == test_id) \
                       or c.name.startswith(prefix)
            if not is_match:
                continue

            out = c.query(
                query_embeddings=list(query_embeddings),
                n_results=n_results_per_collection,
                where=dict(where) if where else None,
                where_document=dict(where_document) if where_document else None,
            )
            for ids, dists, docs, mds in zip(
                out.get("ids", []),
                out.get("distances", []),
                out.get("documents", []),
                out.get("metadatas", []),
            ):
                for i, _id in enumerate(ids):
                    merged.append(
                        {
                            "id": _id,
                            "distance": dists[i] if dists else None,
                            "document": docs[i] if docs else None,
                            "metadata": mds[i] if mds else None,
                            "collection_name": c.name,
                        }
                    )
        except Exception:
            # Be permissive if a collection has unexpected shape
            continue

    merged.sort(key=lambda r: (float("inf") if r["distance"] is None else r["distance"]))
    return merged


# ---------- Utilities ----------
def list_collections_for(
    *,
    project_id: str,
    test_id: str,
    client: Optional["chromadb.api.client.ClientAPI"] = None,
) -> List[str]:
    """List collection names associated with (project_id, test_id)."""
    client = client or get_client()
    names: List[str] = []
    prefix = _derive_name(project_id, test_id, None)

    for c in client.list_collections():
        meta = getattr(c, "metadata", {}) or {}
        if meta.get("project_id") == project_id and meta.get("test_id") == test_id:
            names.append(c.name)
        elif c.name.startswith(prefix):
            names.append(c.name)

    return sorted(set(names))
