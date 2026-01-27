import sqlite3
import csv

DB_PATH = r"onto_db/sample2.db"
OUT_CSV = r"new_output/taxonomy_is_a_final.csv"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT * FROM taxonomy_is_a_final")
rows = cur.fetchall()
col_names = [d[0] for d in cur.description]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    writer.writerows(rows)
conn.close()