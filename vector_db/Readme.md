Here’s a clean, copy-pasteable guide for using your **embedded Chroma** utilities (and the singleton wrapper) end-to-end.

# Embedded Chroma — Usage Guide

## 1) Install & Setup

```bash
pip install chromadb
```

Project layout (suggested):

```
your_app/
  embedded_chroma.py         # your helpers (init/get_or_create/bulk_upsert/query/…)
  chroma_singleton.py        # the singleton client wrapper I shared
  run_example.py             # your app/tests
```

---

## 2) Quick Start (TL;DR)

```python
# run_example.py
from chroma_singleton import open_client, get_client, close_client
from embedded_chroma import (
    get_or_create_collection,
    bulk_upsert,
    query_collection,
    query_project_test,
    list_collections_for,
)

# 1) Start Chroma (once per process)
client = open_client("./.chroma_store")

# 2) Upsert some vectors into a (project_id, test_id[, name]) collection
items = [
    {
        "id": "doc-1",
        "embedding": [0.1, 0.2, 0.3],  # your precomputed embedding
        "document": "Hello world",
        "metadata": {"type": "greeting", "lang": "en"},
    },
    {
        "id": "doc-2",
        "embedding": [0.0, 0.1, 0.4],
        "document": "Merhaba dünya",
        "metadata": {"type": "greeting", "lang": "tr"},
    },
]
bulk_upsert(
    get_client(),
    project_id="mohap",
    test_id="t1",
    items=items,
    name="default",  # optional sub-collection name
)

# 3) Query a single collection with a query embedding
q = [[0.09, 0.19, 0.31]]  # 1 query vector (same dimension as your items)
hits = query_collection(
    get_client(),
    project_id="mohap",
    test_id="t1",
    name="default",
    query_embeddings=q,
    n_results=5,
)
print("Single-collection hits:", hits)

# 4) Or query across all collections under (project_id, test_id)
hits_all = query_project_test(
    get_client(),
    project_id="mohap",
    test_id="t1",
    query_embeddings=q,
    n_results_per_collection=3,
)
print("Cross-collection hits:", hits_all)

# 5) List collection names for a project/test
names = list_collections_for(get_client(), project_id="mohap", test_id="t1")
print("Collections:", names)

# 6) Shutdown (optional; resets the singleton)
close_client()
```

---

## 3) Concepts

* **Persist directory**: local folder Chroma uses (DuckDB + Parquet). One per environment is typical (e.g., `./.chroma_store`).
* **Collections**: logical buckets; here they’re **namespaced by** `project_id` + `test_id` (+ optional `name`).
  Your helper builds stable names like `proj=mohap__test=t1__default`.
* **Embeddings**: you pass **precomputed** vectors (recommended). All vectors in a collection must share the **same dimension**.

---

## 4) Lifecycle: Open / Get / Close (Singleton)

Use the singleton so you don’t accidentally open multiple clients to the same store.

```python
from chroma_singleton import open_client, get_client, close_client

# Open (idempotent)
open_client("./.chroma_store")

# Get the active client anywhere in your code
client = get_client()

# Close/reset when your app ends or between tests
close_client()
```

**Why singleton?**

* Avoids corruption/races from multiple writers to the same `persist_dir`.
* Central place to do health checks.
* Thread-safe initialization with a lock.

---

## 5) Creating / Fetching a Collection

```python
from embedded_chroma import get_or_create_collection

coll = get_or_create_collection(
    client=get_client(),
    project_id="mohap",
    test_id="t1",
    name="default",  # optional (sub-collection)
    extra_metadata={"owner": "abdullah"},  # optional metadata
)
```

* The helper verifies metadata consistency if the collection already exists.
* The `(project_id, test_id)` are also stamped into **each row’s metadata** during upserts (defensive).

---

## 6) Bulk Upsert

```python
from embedded_chroma import bulk_upsert, UpsertItem

items: list[UpsertItem] = [
    {"id": "a", "embedding": [0.1, 0.2, 0.3], "document": "A doc", "metadata": {"topic": "alpha"}},
    {"id": "b", "embedding": [0.2, 0.1, 0.0], "document": "B doc", "metadata": {"topic": "beta"}},
]

bulk_upsert(
    client=get_client(),
    project_id="mohap",
    test_id="t1",
    items=items,
    name="default",
    default_metadata={"source": "ingest_v1"},  # merged into each item’s metadata
)
```

**Rules**

* Each `item` must have `id` and `embedding`.
* `default_metadata` is merged with `item["metadata"]`.
* `project_id` and `test_id` are auto-added if missing.

---

## 7) Querying

### 7.1 Single Collection

```python
from embedded_chroma import query_collection

query_vecs = [[0.09, 0.19, 0.31]]  # one or more query vectors

results = query_collection(
    client=get_client(),
    project_id="mohap",
    test_id="t1",
    name="default",
    query_embeddings=query_vecs,
    n_results=5,
    where={"lang": "en"},                # filter by metadata (optional)
    where_document={"$contains": "Hello"}  # filter by document text (optional)
)

# Result shape (flattened):
# [
#   {
#     "id": "doc-1",
#     "distance": 0.123,                 # lower is closer (depending on metric)
#     "document": "Hello world",
#     "metadata": {"type": "greeting", "lang": "en", "project_id": "...", "test_id": "..."},
#     "collection_name": "proj=...__test=...__default",
#   },
#   ...
# ]
```

### 7.2 Across All Collections of (project\_id, test\_id)

```python
from embedded_chroma import query_project_test

results = query_project_test(
    client=get_client(),
    project_id="mohap",
    test_id="t1",
    query_embeddings=[[0.09, 0.19, 0.31]],
    n_results_per_collection=3,
    where={"type": "greeting"},
)
# Returns a unified, distance-sorted list across all matching collections.
```

---

## 8) Listing Collections

```python
from embedded_chroma import list_collections_for

names = list_collections_for(get_client(), project_id="mohap", test_id="t1")
print(names)
# -> ['proj=mohap__test=t1__default', 'proj=mohap__test=t1__rules', ...]
```

---

## 9) Filters (Metadata & Document)

* `where` works on the **metadata** dict you upserted with each item. Examples:

  * Exact match: `{"lang": "en"}`
  * In-list: `{"lang": {"$in": ["en", "tr"]}}`
  * Not equal: `{"type": {"$ne": "spam"}}`
  * Numeric compare: `{"score": {"$gt": 0.5}}`
* `where_document` works on the **document text** (if stored). Examples:

  * Contains substring: `{"$contains": "invoice"}`
  * Regex (if supported by your Chroma version).

> Filtering operators vary a bit by Chroma version—keep it simple (eq/in/contains) for portability.

---

## 10) Best Practices

* **Consistent embedding model** per collection. Don’t mix dimensions or models.
* **Stable IDs**: use deterministic IDs (e.g., hash of file path + version) so re-ingests upsert cleanly.
* **Chunking strategy**: if storing long docs, chunk and upsert per chunk with shared metadata (`doc_id`, `page`, etc.).
* **Backups**: the persist dir is just files—snapshot/backup like any app data.
* **Concurrency**: one client per process; don’t share the Python object across processes. Within a process the singleton is thread-safe on open/close.
* **Testing**: point the singleton to a temp dir per test; call `close_client()` between tests.

---

## 11) Minimal API Reference

### `chroma_singleton.py`

* `open_client(persist_dir: str) -> ClientAPI`
  Idempotent; initializes (and verifies) the store at `persist_dir`.
* `get_client() -> ClientAPI`
  Returns the active client or raises if unopened.
* `close_client() -> None`
  Resets the global handle (tidy shutdown / test isolation).

### `embedded_chroma.py`

* `init_chroma(persist_dir: str) -> ClientAPI`
  (If you prefer not to use the singleton.)
* `get_or_create_collection(client, *, project_id, test_id, name=None, extra_metadata=None) -> Collection`
* `bulk_upsert(client, *, project_id, test_id, items, name=None, default_metadata=None) -> None`
* `query_collection(client, *, project_id, test_id, name=None, query_embeddings, n_results=10, where=None, where_document=None) -> list[dict]`
* `query_project_test(client, *, project_id, test_id, query_embeddings, n_results_per_collection=10, where=None, where_document=None) -> list[dict]`
* `list_collections_for(client, *, project_id, test_id) -> list[str]`

---

## 12) Common Pitfalls & Fixes

* **Dimension mismatch**:

  > `ValueError: Expected embedding dimension X but got Y`
  > Ensure your query vectors and upserted vectors use the same model/dimension.

* **Empty query results** with filters:
  Loosen filters (start with no `where`/`where_document`), then add back gradually.

* **Multiple stores accidentally**:
  If you see odd “missing data,” confirm you’re opening the same `persist_dir` everywhere. The singleton blocks mismatched paths.

* **Performance**:
  Batch upserts (you already do), and prefer fewer, larger calls over many tiny ones.

---

That’s it. If you want, I can add a tiny wrapper so your call sites don’t need to pass `client=` at all (each helper can default to `get_client()` internally), while keeping an override for tests.
