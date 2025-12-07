import sqlite3
import csv

DB_PATH = r"onto_db/olaf_sample_llm.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS llm_terms;")  # deletes the table
conn.commit()
conn.close()