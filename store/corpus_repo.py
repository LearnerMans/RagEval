from store.base_repo import BaseRepo


class CorpusRepo(BaseRepo):
    def __init__(self, conn):
        super().__init__(conn)

    def create_corpus(self, corpus_id: str, project_id: str):
        self.execute(
            "INSERT INTO corpus (id, project_id) VALUES (?, ?)",
            (corpus_id, project_id),
        )

    def get_corpus(self, corpus_id: str):
        return self.fetch_one("SELECT * FROM corpus WHERE id = ?", (corpus_id,))

    def get_all_corpora(self):
        return self.fetch_all("SELECT * FROM corpus")

    def delete_corpus(self, corpus_id: str):
        self.execute("DELETE FROM corpus WHERE id = ?", (corpus_id,))

    def get_corpus_by_project_id(self, project_id: str):
        return self.fetch_all("SELECT * FROM corpus WHERE project_id = ?", (project_id,))
