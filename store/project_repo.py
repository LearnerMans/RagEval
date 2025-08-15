from store.base_repo import BaseRepo

class ProjectRepo(BaseRepo):
    def __init__(self, conn):
        super().__init__(conn)
    
    def create_project(self, project_id, name, description):
        self.execute("INSERT INTO project (id, name, description) VALUES (?, ?, ?)", (project_id, name, description))
    
    def get_project(self, project_id):
        return self.fetch_one("SELECT * FROM project WHERE id = ?", (project_id,))
    
    def get_all_projects(self):
        return self.fetch_all("SELECT * FROM project")
    
    def update_project(self, project_id, name, description):
        self.execute("UPDATE project SET name = ?, description = ? WHERE id = ?", (name, description, project_id))
    
    def delete_project(self, project_id):
        self.execute("DELETE FROM project WHERE id = ?", (project_id,))