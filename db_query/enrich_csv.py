import sqlite3
import csv

DB_PATH = r"onto_db/onto_new.db"
OUT_CSV = r"output/term_enrichment_v2_22.csv"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT * FROM term_enrichment_v2")
rows = cur.fetchall()
col_names = [d[0] for d in cur.description]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    writer.writerows(rows)
conn.close()