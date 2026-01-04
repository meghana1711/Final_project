import sqlite3
import csv

DB_PATH = r"onto_db/onto_new.db"
OUT_CSV = r"output/llm_enrich_new.csv"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT * FROM llm_enrich")
rows = cur.fetchall()
col_names = [d[0] for d in cur.description]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    writer.writerows(rows)
conn.close()