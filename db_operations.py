import psycopg2
from db_settings import DB_NAME, USER, PASSWORD, HOST, PORT


class DB_Operations():
    def __init__(self, total_epochs=0):
        self._total_epochs = total_epochs

    def __enter__(self):
        # Establish the database connection in __enter__
        self._conn = psycopg2.connect(
            dbname=DB_NAME, 
            user=USER, 
            password=PASSWORD, 
            host=HOST, 
            port=PORT,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the connection in __exit__
        if self.conn is not None:
            self.conn.close()