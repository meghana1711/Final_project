import sqlite3
import csv

DB_PATH = r"final_db/lsf_new.db"
OUT_CSV = r"non_taxonomic_edges_accep_lsf.csv"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT * FROM non_taxonomic_edges_accept")
rows = cur.fetchall()
col_names = [d[0] for d in cur.description]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    writer.writerows(rows)
conn.close()