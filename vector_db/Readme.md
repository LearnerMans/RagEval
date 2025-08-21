Here’s a compact, production-ready helper you can drop into your codebase to run an **embedded ChromaDB** (on disk), create/get collections **tied to `project_id` and `test_id`**, bulk-upsert vectors, and query a specific collection (or all collections for a project/test).

```python
# embedded_chroma.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TypedDict

try:
    import chromadb
except ImportError as e:
    raise RuntimeError(
        "chromadb is required. Install with:  pip install chromadb"
    ) from e


# ---------- Types ----------
class UpsertItem(TypedDict, total=False):
    id: str
    embedding: Sequence[float]
    document: Optional[str]
    metadata: Dict[str, Any]

QueryResult = List[Dict[str, Any]]  # [{id, distance, document, metadata, collection_name}]


# ---------- Internal helpers ----------
def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()

def _derive_name(project_id: str, test_id: str, name: Optional[str]) -> str:
    # A deterministic, readable collection name
    base = f"proj={_slug(project_id)}__test={_slug(test_id)}"
    return f"{base}__{_slug(name)}" if name else base

def _ensure_dir(p: Union[str, Path]) -> str:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# ---------- Client / Collection ----------
def init_chroma(persist_dir: str) -> "chromadb.api.client.ClientAPI":
    """
    Create (if needed) and open a local, embedded Chroma database in `persist_dir`.
    Returns a PersistentClient.
    """
    path = _ensure_dir(persist_dir)
    # Chroma 0.5+ API
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=path)
    # Fallback for older versions
    return chromadb.Client(chromadb.config.Settings(chroma_db_impl="duckdb+parquet", persist_directory=path))


def get_or_create_collection(
    client: "chromadb.api.client.ClientAPI",
    *,
    project_id: str,
    test_id: str,
    name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> "chromadb.api.models.Collection.Collection":
    """
    Get or create a collection bound to (project_id, test_id).
    We encode (project_id, test_id) both in the collection metadata and in the collection name.
    """
    coll_name = _derive_name(project_id, test_id, name)
    metadata = {"project_id": project_id, "test_id": test_id}
    if extra_metadata:
        metadata.update(extra_metadata)

    # If collection exists, Chroma merges metadata (doesn't enforce equality), so we sanity-check.
    try:
        coll = client.get_collection(coll_name)
        # Best-effort consistency check:
        meta = getattr(coll, "metadata", {}) or {}
        if meta.get("project_id") not in (None, project_id) or meta.get("test_id") not in (None, test_id):
            raise ValueError(
                f"Existing collection '{coll_name}' has conflicting metadata: {meta}. "
                f"Expected project_id={project_id}, test_id={test_id}."
            )
        # Optionally update metadata if empty (Chroma doesn't expose update; ignore if not supported)
        return coll
    except Exception:
        pass

    # Create if missing
    return client.get_or_create_collection(name=coll_name, metadata=metadata)  # embedding_function is optional


# ---------- Bulk Upsert ----------
def bulk_upsert(
    client: "chromadb.api.client.ClientAPI",
    *,
    project_id: str,
    test_id: str,
    items: Sequence[UpsertItem],
    name: Optional[str] = None,
    default_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Bulk insert or update vectors into the (project_id, test_id[, name]) collection.
    Each item should include: id (str), embedding (list[float]),
    and optionally document (str) and metadata (dict).

    Notes:
      - If you already generate embeddings elsewhere, pass them here (recommended).
      - This function does not depend on a collection-level embedding function.
    """
    if not items:
        return

    coll = get_or_create_collection(client, project_id=project_id, test_id=test_id, name=name)

    ids: List[str] = []
    embeddings: List[Sequence[float]] = []
    documents: List[Optional[str]] = []
    metadatas: List[Dict[str, Any]] = []

    for it in items:
        if "id" not in it or "embedding" not in it:
            raise ValueError("Each item must include at least 'id' and 'embedding'.")
        ids.append(it["id"])
        embeddings.append(it["embedding"])
        documents.append(it.get("document"))
        md = dict(default_metadata or {})
        md.update(it.get("metadata") or {})
        # Ensure association is present on every row:
        md.setdefault("project_id", project_id)
        md.setdefault("test_id", test_id)
        metadatas.append(md)

    coll.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


# ---------- Query (single collection) ----------
def query_collection(
    client: "chromadb.api.client.ClientAPI",
    *,
    project_id: str,
    test_id: str,
    name: Optional[str] = None,
    query_embeddings: Sequence[Sequence[float]],
    n_results: int = 10,
    where: Optional[Mapping[str, Any]] = None,
    where_document: Optional[Mapping[str, Any]] = None,
) -> QueryResult:
    """
    Query a specific (project_id, test_id[, name]) collection using precomputed `query_embeddings`.
    Returns a flat list of results: [{id, distance, document, metadata, collection_name}, ...]
    """
    coll = get_or_create_collection(client, project_id=project_id, test_id=test_id, name=name)
    coll_name = _derive_name(project_id, test_id, name)

    out = coll.query(
        query_embeddings=list(query_embeddings),
        n_results=n_results,
        where=dict(where) if where else None,
        where_document=dict(where_document) if where_document else None,
    )

    results: QueryResult = []
    # Chroma returns lists per query vector; flatten them:
    for ids, dists, docs, mds in zip(out.get("ids", []), out.get("distances", []), out.get("documents", []), out.get("metadatas", [])):
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


# ---------- Query (all collections for a project/test) ----------
def query_project_test(
    client: "chromadb.api.client.ClientAPI",
    *,
    project_id: str,
    test_id: str,
    query_embeddings: Sequence[Sequence[float]],
    n_results_per_collection: int = 10,
    where: Optional[Mapping[str, Any]] = None,
    where_document: Optional[Mapping[str, Any]] = None,
) -> QueryResult:
    """
    Query across *all* collections that belong to (project_id, test_id) and merge results.
    Results are sorted by ascending distance (when available).
    """
    # Find matching collections by metadata/name convention
    matching = []
    prefix = _derive_name(project_id, test_id, None)  # "proj=..__test=.."
    for c in client.list_collections():
        try:
            meta = getattr(c, "metadata", {}) or {}
            if meta.get("project_id") == project_id and meta.get("test_id") == test_id:
                matching.append(c)
            elif c.name.startswith(prefix):
                matching.append(c)
        except Exception:
            # Be permissive; skip on any unexpected shape
            continue

    merged: QueryResult = []
    for coll in matching:
        out = coll.query(
            query_embeddings=list(query_embeddings),
            n_results=n_results_per_collection,
            where=dict(where) if where else None,
            where_document=dict(where_document) if where_document else None,
        )
        for ids, dists, docs, mds in zip(out.get("ids", []), out.get("distances", []), out.get("documents", []), out.get("metadatas", [])):
            for i, _id in enumerate(ids):
                merged.append(
                    {
                        "id": _id,
                        "distance": dists[i] if dists else None,
                        "document": docs[i] if docs else None,
                        "metadata": mds[i] if mds else None,
                        "collection_name": coll.name,
                    }
                )

    # Sort when distances are present
    merged.sort(key=lambda r: (float("inf") if r["distance"] is None else r["distance"]))
    return merged


# ---------- (Optional) Utilities ----------
def list_collections_for(
    client: "chromadb.api.client.ClientAPI", *, project_id: str, test_id: str
) -> List[str]:
    """List collection names associated with (project_id, test_id)."""
    names: List[str] = []
    prefix = _derive_name(project_id, test_id, None)
    for c in client.list_collections():
        meta = getattr(c, "metadata", {}) or {}
        if meta.get("project_id") == project_id and meta.get("test_id") == test_id:
            names.append(c.name)
        elif c.name.startswith(prefix):
            names.append(c.name)
    return sorted(set(names))
```

### How to use

```python
from embedded_chroma import init_chroma, get_or_create_collection, bulk_upsert, query_collection, query_project_test

# 1) Start embedded DB (stored under ./vectorstore)
client = init_chroma("./vectorstore")

# 2) Create/get a collection for a project & test (name is optional)
coll = get_or_create_collection(client, project_id="proj_123", test_id="test_A", name="chunks_v1")

# 3) Bulk upsert (you provide embeddings)
items = [
    {"id": "doc-1#0", "embedding": [0.01, 0.2, ...], "document": "First chunk text", "metadata": {"source": "pdf", "chunk_index": 0}},
    {"id": "doc-1#1", "embedding": [0.05, 0.1, ...], "document": "Second chunk text", "metadata": {"source": "pdf", "chunk_index": 1}},
]
bulk_upsert(client, project_id="proj_123", test_id="test_A", name="chunks_v1", items=items)

# 4) Query a specific collection (using your own query embeddings)
q_emb = [[0.02, 0.18, ...]]
hits = query_collection(client, project_id="proj_123", test_id="test_A", name="chunks_v1", query_embeddings=q_emb, n_results=5)

# 5) Or query across all collections for (project_id, test_id)
hits_all = query_project_test(client, project_id="proj_123", test_id="test_A", query_embeddings=q_emb, n_results_per_collection=3)
```

**Notes & design choices**

* Uses **embedded** Chroma (on-disk) via `PersistentClient(path=...)`. No server required.
* Each collection is **namespaced** and **tagged** by `project_id` and `test_id`:

  * In the **collection name** (e.g., `proj=proj_123__test=test_A__chunks_v1`)
  * In **collection metadata** (`{"project_id": "...", "test_id": "..."}`)
  * Also stamped onto every row’s metadata on upsert, so you can filter later if you merge data.
* You supply embeddings (fits your pipeline that already stores configs/embeddings in SQLite).
* `query_project_test(...)` optionally aggregates results **across multiple collections** for the same `(project_id, test_id)` and sorts by distance.

If you want, I can add a thin adapter that reads chunk embeddings out of your `chunk` table (by `index_build_id`) and bulk-upserts them straight into the appropriate Chroma collection.
