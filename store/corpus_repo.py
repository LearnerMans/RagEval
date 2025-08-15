from store.base_repo import BaseRepo

class CorpusRepo(BaseRepo):
    def __init__(self, conn):
        super().__init__(conn)
    
    def create_corpus(self, project_id, corpus):
        self.execute("INSERT INTO corpus (project_id, corpus) VALUES (?, ?)", (project_id, corpus))
    
    def get_corpus(self, corpus_id):
        return self.fetch_one("SELECT * FROM corpus WHERE id = ?", (corpus_id,))
    
    def get_all_corpora(self):
        return self.fetch_all("SELECT * FROM corpus")
    
    def update_corpus(self, corpus_id, corpus):
        self.execute("UPDATE corpus SET corpus = ? WHERE id = ?", (corpus, corpus_id))
    
    def delete_corpus(self, corpus_id):
        self.execute("DELETE FROM corpus WHERE id = ?", (corpus_id,))