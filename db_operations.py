import psycopg2

from db_settings import DB_NAME, USER, PASSWORD, HOST, PORT
from game_settings import LR, SNAKE_GAMMA

class DB_Operations():
    def __init__(self, total_epochs=0):
        self._total_epochs = total_epochs

    def add_era(self, description):
        self._conn.cursor().execute(
            "INSERT INTO era (description, alpha, gamma) VALUES (%s, %s, %s)", 
            (description, LR, SNAKE_GAMMA))
        self._conn.commit()

    def add_epoch(self, score, time, reward, epsilon, bumps, steps, rotations, fk_era_id):
        self._conn.cursor().execute(
            "INSERT INTO era (score, time, reward, epsilon, bumps, steps, rotations, fk_era_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
            (score, time, reward, epsilon, bumps, steps, rotations, fk_era_id))
        self._conn.commit()

    # For using with 'with' statement
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
        if self._conn is not None:
            self._conn.close()
