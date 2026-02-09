import sqlite3
import csv

DB_PATH = r"onto_db/slurm_llm_final.db"
OUT_CSV = r"llm_is_a_edges_core.csv"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT * FROM llm_is_a_edges_core")
rows = cur.fetchall()
col_names = [d[0] for d in cur.description]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    writer.writerows(rows)
conn.close()