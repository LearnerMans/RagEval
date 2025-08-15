from store.base_repo import BaseRepo

class CorpusItemRepo(BaseRepo):
    def __init__(self, conn):
        super().__init__(conn)
    
    def create_corpus_item(self, corpus_id, item_type, item_data):
        self.execute("INSERT INTO corpus_item (corpus_id, item_type, item_data) VALUES (?, ?, ?)", (corpus_id, item_type, item_data))
    
    def get_corpus_item(self, corpus_item_id):
        return self.fetch_one("SELECT * FROM corpus_item WHERE id = ?", (corpus_item_id,))