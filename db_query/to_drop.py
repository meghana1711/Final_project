import sqlite3
import csv

DB_PATH = r"onto_db/onto_new.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS non_taxonomic_edges;")  
conn.commit()
conn.close()