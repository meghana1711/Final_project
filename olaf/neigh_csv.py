import sqlite3
import csv

DB_PATH = r"onto_db/ontology_sample_new.db"
OUT_CSV = r"output/skipgram_neighbors.csv"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    SELECT
        c1.term_id       AS term_id,
        c1.term_text     AS term,
        c2.term_id       AS neighbor_id,
        c2.term_text     AS neighbor,
        n.similarity
    FROM skipgram_neighbors n
    JOIN term_candidates c1 ON n.term_id = c1.term_id
    JOIN term_candidates c2 ON n.neighbor_term_id = c2.term_id
    ORDER BY c1.term_id, n.similarity DESC
""")

rows = cur.fetchall()
col_names = [d[0] for d in cur.description]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    for row in rows:
        term_id, term, neighbor_id, neighbor, sim = row
        writer.writerow([term_id, term, neighbor_id, neighbor, f"{sim:.4f}"])

conn.close()
print(f"Wrote neighbors to {OUT_CSV}")
