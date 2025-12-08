import sqlite3
import csv

DB_PATH = r"onto_db/ontology_sample_new.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS taxonomy_edges;")  # deletes the table
conn.commit()
conn.close()