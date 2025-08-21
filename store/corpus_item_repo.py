from store.base_repo import BaseRepo


class CorpusItemRepo(BaseRepo):
    def __init__(self, conn):
        super().__init__(conn)

    def create_corpus_item(self, *, item_id: str, corpus_id: str, uri: str, content_hash: str, metadata_json: str | None):
        self.execute(
            "INSERT INTO corpus_item (id, corpus_id, uri, content_hash, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (item_id, corpus_id, uri, content_hash, metadata_json),
        )

    def create_corpus_item_file(
        self,
        *,
        corpus_item_id: str,
        file_path: str,
        original_filename: str | None,
        file_size_bytes: int | None,
        mime_type: str | None,
        fs_modified_time: str | None,
        storage_backend: str | None,
        storage_key: str | None,
        ingested_at: str | None,
    ):
        self.execute(
            """
            INSERT INTO corpus_item_file (
                id, corpus_item_id, file_path, original_filename, file_size_bytes,
                mime_type, fs_modified_time, storage_backend, storage_key, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._generate_id(),
                corpus_item_id,
                file_path,
                original_filename,
                file_size_bytes,
                mime_type,
                fs_modified_time,
                storage_backend,
                storage_key,
                ingested_at,
            ),
        )

    def create_corpus_item_url(
        self,
        *,
        corpus_item_id: str,
        url: str,
        url_normalized: str,
        http_status_last: int | None,
        http_etag: str | None,
        http_last_modified: str | None,
        fetched_at: str | None,
        fetch_error: str | None,
    ):
        self.execute(
            """
            INSERT INTO corpus_item_url (
                id, corpus_item_id, url, url_normalized, http_status_last,
                http_etag, http_last_modified, fetched_at, fetch_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._generate_id(),
                corpus_item_id,
                url,
                url_normalized,
                http_status_last,
                http_etag,
                http_last_modified,
                fetched_at,
                fetch_error,
            ),
        )

    def get_corpus_item(self, corpus_item_id):
        return self.fetch_one("SELECT * FROM corpus_item WHERE id = ?", (corpus_item_id,))

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())