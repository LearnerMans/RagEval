class BaseRepo:
    def __init__(self, conn):
        self._conn = conn
    
    def execute(self, sql, params=None):
       self._conn.execute(sql, params)
       self._conn.commit()
    
    def fetch_one(self, sql, params=None):
        if params is None:
            return self._conn.execute(sql).fetchone()
        return self._conn.execute(sql, params).fetchone()
    
    def fetch_all(self, sql, params=None):
        if params is None:
            return self._conn.execute(sql).fetchall()
        return self._conn.execute(sql, params).fetchall()