import sqlite3
import csv

DB_PATH = r"onto_db/olaf_sample.db"
OUT_CSV = r"output/term_scores_new2.csv"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    SELECT
        c.term_id,
        c.term_text,
        s.tf,
        s.idf,
        s.tf_idf,
        s.c_value,
        s.score
    FROM term_candidates AS c
    JOIN term_scores AS s
        ON s.term_id = c.term_id
    ORDER BY s.tf_idf DESC
""")

rows = cur.fetchall()
col_names = [d[0] for d in cur.description]

def fmt(x):
    """Format numbers to 2 decimal places; leave others as-is."""
    if isinstance(x, (int, float)):
        return f"{x:.2f}"
    return x

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)

    for row in rows:
        term_id = row[0]
        term_text = row[1]
        tf = fmt(row[2])
        idf = fmt(row[3])
        tf_idf = fmt(row[4])
        c_value = fmt(row[5])
        score = fmt(row[6])

        writer.writerow([term_id, term_text, tf, idf, tf_idf, c_value, score])

conn.close()

