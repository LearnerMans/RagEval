import logging
import os
import uuid
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel

from db.db import get_conn
from store.project_repo import ProjectRepo
from store.corpus_repo import CorpusRepo
from store.corpus_item_repo import CorpusItemRepo



logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpus", tags=["corpus"])


class IngestResponse(BaseModel):
    corpus_id: str
    created_file_items: int
    created_url_items: int


def _ensure_upload_dir(corpus_id: str) -> Path:
    base_dir = Path(__file__).resolve().parent.parent / "data" / "uploads" / corpus_id
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _safe_filename(name: str) -> str:
    keep = [c for c in name if c.isalnum() or c in (".", "-", "_", " ")]
    return "".join(keep).strip().replace(" ", "_") or "file"



@router.post("/ingest", response_model=IngestResponse)
async def ingest_corpus(
    project_id: str = Form(...),
    urls: Optional[str] = Form(None),  # JSON-encoded list of strings
    files: Optional[List[UploadFile]] = File(None),
):
    """Create a corpus for a project and add items from uploaded files and URLs."""
    try:
        conn = get_conn()
        project_repo = ProjectRepo(conn)
        corpus_repo = CorpusRepo(conn)
        corpus_item_repo = CorpusItemRepo(conn)

        # Validate project exists
        if not project_repo.get_project(project_id):
            raise HTTPException(status_code=404, detail="Project not found")

        # Create corpus
        corpus_id = str(uuid.uuid4())
        corpus_repo.create_corpus(corpus_id, project_id)
        logger.info(f"Created corpus {corpus_id} for project {project_id}")

        created_file_items = 0
        created_url_items = 0

        # Handle files
        if files:
            upload_dir = _ensure_upload_dir(corpus_id)
            for up in files:
                try:
                    content = await up.read()
                    content_hash = hashlib.sha256(content).hexdigest()
                   

                    # Save file
                    fname = f"{uuid.uuid4()}_{_safe_filename(up.filename or 'file')}"
                    path_on_disk = upload_dir / fname
                    with open(path_on_disk, "wb") as f:
                        f.write(content)

                    # Insert corpus_item
                    item_id = str(uuid.uuid4())
                    uri = str(path_on_disk)
                    corpus_item_repo.create_corpus_item(
                        item_id=item_id,
                        corpus_id=corpus_id,
                        uri=uri,
                        content_hash=content_hash,
                        metadata_json=None,
                    )

                    # Insert file metadata
                    file_size_bytes = len(content)
                    mime_type = up.content_type or None
                    fs_modified_time = datetime.utcfromtimestamp(path_on_disk.stat().st_mtime).isoformat() + "Z"
                    corpus_item_repo.create_corpus_item_file(
                        corpus_item_id=item_id,
                        file_path=str(path_on_disk),
                        original_filename=up.filename,
                        file_size_bytes=file_size_bytes,
                        mime_type=mime_type,
                        fs_modified_time=fs_modified_time,
                        storage_backend="local",
                        storage_key=None,
                        ingested_at=datetime.utcnow().isoformat() + "Z",
                    )
                    created_file_items += 1
                finally:
                    await up.close()

        # Handle URLs
        if urls:
            try:
                url_list = json.loads(urls)
                if not isinstance(url_list, list):
                    raise ValueError
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid 'urls' payload; must be JSON array")

            for href in url_list:
                if not isinstance(href, str):
                    continue
                href = href.strip()
                if not href:
                    continue
                # Use a hash of the URL as content hash placeholder
                content_hash = hashlib.sha256(href.encode("utf-8")).hexdigest()
                item_id = str(uuid.uuid4())
                corpus_item_repo.create_corpus_item(
                    item_id=item_id,
                    corpus_id=corpus_id,
                    uri=href,
                    content_hash=content_hash,
                    metadata_json=None,
                )
                # Insert URL record
                corpus_item_repo.create_corpus_item_url(
                    corpus_item_id=item_id,
                    url=href,
                    url_normalized=href,
                    http_status_last=None,
                    http_etag=None,
                    http_last_modified=None,
                    fetched_at=None,
                    fetch_error=None,
                )
                created_url_items += 1

        return IngestResponse(
            corpus_id=corpus_id,
            created_file_items=created_file_items,
            created_url_items=created_url_items,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting corpus: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest corpus")

@router.get("/projects/{project_id}")
async def get_corpus(
    project_id: str,
    
):
    try:
        conn = get_conn()
        corpus_repo = CorpusRepo(conn)
        logger.info(f"Fetching corpus: {project_id}")
        
        corpus =corpus_repo.get_corpus_by_project_id(project_id)
        
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")
        
        return corpus
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching corpus {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch corpus: {str(e)}")