from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
    )
except Exception as import_error:  # pragma: no cover - import-time guidance
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None  # type: ignore
    _QDRANT_IMPORT_ERROR = import_error
else:
    _QDRANT_IMPORT_ERROR = None


_DISTANCE_ALIASES = {
    "cosine": "COSINE",
    "cos": "COSINE",
    "dot": "DOT",
    "dotproduct": "DOT",
    "euclid": "EUCLID",
    "l2": "EUCLID",
}


def _ensure_qdrant_installed() -> None:
    if _QDRANT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "qdrant-client is required. Install with: pip install qdrant-client"
        ) from _QDRANT_IMPORT_ERROR


def _normalize_distance(distance: Any) -> Any:
    _ensure_qdrant_installed()
    if isinstance(distance, str):
        key = distance.strip().lower()
        mapped = _DISTANCE_ALIASES.get(key, key).upper()
        try:
            return getattr(Distance, mapped)
        except Exception:
            raise ValueError(
                f"Unsupported distance '{distance}'. Use one of: cosine, dot, euclid/l2."
            )
    return distance


class QdrantVectorDB:
    """Minimal Qdrant wrapper for managing collections and vectors.

    Features:
    - Create/Delete collections
    - Bulk upsert points with payloads
    - Vector search (retrieval)
    - Retrieve by IDs
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = True,
        timeout: Optional[float] = 30.0,
    ) -> None:
        _ensure_qdrant_installed()

        client_kwargs: Dict[str, Any] = {"prefer_grpc": prefer_grpc}
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        if url:
            client_kwargs.update({"url": url, "api_key": api_key})
        else:
            # Default to local if not provided
            client_kwargs.update(
                {
                    "host": host or "localhost",
                    "port": port or 6333,
                    "api_key": api_key,
                }
            )
            if grpc_port is not None:
                client_kwargs["grpc_port"] = grpc_port

        self.client = QdrantClient(**client_kwargs)

    # ---------- Collection Ops ----------
    def create_collection(
        self,
        collection_name: str,
        *,
        vector_size: int,
        distance: str = "cosine",
        on_disk: bool = True,
        shard_number: Optional[int] = None,
        recreate: bool = False,
    ) -> None:
        """Create a collection for single-vector usage.

        Args:
            collection_name: Target collection name.
            vector_size: Dimension of the vector embeddings.
            distance: Distance metric ('cosine'|'dot'|'euclid' or Distance enum).
            on_disk: Whether to store vectors on disk.
            shard_number: Optional shard count.
            recreate: If True, delete if exists then create.
        """

        metric = _normalize_distance(distance)

        if recreate:
            try:
                self.delete_collection(collection_name)
            except Exception:
                pass

        # If already exists, return quietly
        try:
            exists = self.client.get_collection(collection_name) is not None
        except Exception:
            exists = False
        if exists and not recreate:
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=metric, on_disk=on_disk),
            shard_number=shard_number,
        )

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists."""
        try:
            self.client.delete_collection(collection_name=collection_name)
        except Exception:
            # Ignore if it doesn't exist
            pass

    # ---------- Upsert Ops ----------
    def upsert_vectors(
        self,
        collection_name: str,
        items: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 512,
        wait: bool = True,
    ) -> None:
        """Upsert vectors in bulk.

        Each item must contain:
            - 'id': unique id (int | str | uuid str)
            - 'vector': embedding list[float]
            - 'payload': optional dict payload
        """

        if not items:
            return

        def to_point(item: Dict[str, Any]):
            point_id = item.get("id")
            vector = item.get("vector")
            payload = item.get("payload")
            if point_id is None or vector is None:
                raise ValueError("Each item must include 'id' and 'vector'")
            return PointStruct(id=point_id, vector=vector, payload=payload)

        # chunked upsert to avoid oversized requests
        chunk: List[Any] = []
        for item in items:
            chunk.append(to_point(item))
            if len(chunk) >= batch_size:
                self.client.upsert(collection_name=collection_name, points=chunk, wait=wait)
                chunk = []
        if chunk:
            self.client.upsert(collection_name=collection_name, points=chunk, wait=wait)

    # ---------- Retrieval Ops ----------
    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        qdrant_filter: Optional[Any] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search nearest neighbors and return list of {id, score, payload} dicts."""

        results = self.client.search(
            collection_name=collection_name,
            query_vector=list(query_vector),
            limit=top_k,
            with_payload=with_payload,
            with_vectors=with_vectors,
            filter=qdrant_filter,
            score_threshold=score_threshold,
        )

        output: List[Dict[str, Any]] = []
        for r in results:
            output.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": getattr(r, "payload", None) if with_payload else None,
                    "vector": getattr(r, "vector", None) if with_vectors else None,
                }
            )
        return output

    def retrieve_by_ids(
        self,
        collection_name: str,
        ids: Sequence[Any],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve points by IDs."""
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=list(ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        output: List[Dict[str, Any]] = []
        for p in points:
            output.append(
                {
                    "id": p.id,
                    "payload": getattr(p, "payload", None) if with_payload else None,
                    "vector": getattr(p, "vector", None) if with_vectors else None,
                }
            )
        return output

    # ---------- Helpers ----------
    @classmethod
    def from_env(cls) -> "QdrantVectorDB":
        """Initialize client using environment variables.

        Recognized variables:
            - QDRANT_URL
            - QDRANT_HOST (default: localhost)
            - QDRANT_PORT (default: 6333)
            - QDRANT_GRPC_PORT (optional)
            - QDRANT_API_KEY (optional)
            - QDRANT_PREFER_GRPC ("true"/"false", default: true)
            - QDRANT_TIMEOUT_SECS (float, default: 30)
        """

        url = os.getenv("QDRANT_URL")
        host = os.getenv("QDRANT_HOST")
        port_str = os.getenv("QDRANT_PORT")
        grpc_port_str = os.getenv("QDRANT_GRPC_PORT")
        api_key = os.getenv("QDRANT_API_KEY")
        prefer_grpc_str = os.getenv("QDRANT_PREFER_GRPC", "true").strip().lower()
        timeout_str = os.getenv("QDRANT_TIMEOUT_SECS", "30")

        port = int(port_str) if port_str else None
        grpc_port = int(grpc_port_str) if grpc_port_str else None
        prefer_grpc = prefer_grpc_str in {"1", "true", "yes", "y"}
        try:
            timeout = float(timeout_str)
        except Exception:
            timeout = 30.0

        return cls(
            url=url,
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )
        
    def list_collections(self) -> List[str]:
        """Return all collection names in Qdrant."""
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Return metadata/config for a collection."""
        info = self.client.get_collection(collection_name)
        return {
            "status": info.status,
            "vectors_count": info.vectors_count,
            "config": info.config.dict() if hasattr(info.config, "dict") else info.config,
        }

    # ---------- Retrieval Helpers ----------
    def retrieve_all(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[int] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        qdrant_filter: Optional[Any] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[int]]:
        """Retrieve points from a collection with optional filtering.
        Returns (points, next_offset).
        """
        points, next_page = self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            filter=qdrant_filter,
        )

        output: List[Dict[str, Any]] = []
        for p in points:
            output.append(
                {
                    "id": p.id,
                    "payload": getattr(p, "payload", None) if with_payload else None,
                    "vector": getattr(p, "vector", None) if with_vectors else None,
                }
            )

        return output, next_page

