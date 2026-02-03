import sqlite3
import csv

DB_PATH = r"onto_db/sample3.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS non_taxonomic_edges_llm_binary;")  
conn.commit()
conn.close()