DB_NAME="snake"
USER="agent"
PASSWORD="5271"
HOST="localhost"
PORT=5432

"""
CREATE TABLE era (
    fk_era_id SERIAL PRIMARY KEY,
    description TEXT,
    alpha FLOAT,
    gamma FLOAT
);
"""

"""
CREATE TABLE epoch (
    epoch_id SERIAL PRIMARY KEY,
    score INT,
    time FLOAT,
    reward FLOAT,
    epsilon FLOAT,
    bumps INT,
    steps INT,
    rotations INT,
    fk_era_id INT,
    FOREIGN KEY (fk_era_id) REFERENCES era(fk_era_id)
);
"""

"GRANT ALL PRIVILEGES ON DATABASE snake TO agent;"
"GRANT ALL PRIVILEGES ON TABLE epoch TO agent;"
"GRANT ALL PRIVILEGES ON TABLE era TO agent;"
"GRANT USAGE, SELECT ON SEQUENCE epoch_epoch_id_seq TO agent;"
"GRANT USAGE, SELECT ON SEQUENCE era_fk_era_id_seq TO agent;"

"""
ALTER TABLE epoch ADD CONSTRAINT fk_era_id FOREIGN KEY (fk_era_id) REFERENCES era(fk_era_id) ON DELETE CASCADE;
"""

"ALTER SEQUENCE era_fk_era_id_seq RESTART WITH 1;"
"ALTER SEQUENCE epoch_epoch_id_seq RESTART WITH 1;"
"ALTER TABLE era OWNER TO agent;"
"ALTER TABLE epoch OWNER TO agent;"
