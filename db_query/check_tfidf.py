import sqlite3

db_path = "onto_db/new_db.db"

sql = """
SELECT
  COUNT(*) AS total_terms,
  SUM(CASE WHEN tf_idf >= 5  THEN 1 ELSE 0 END) AS ge_5,
  SUM(CASE WHEN tf_idf >= 10 THEN 1 ELSE 0 END) AS ge_10,
  SUM(CASE WHEN tf_idf >= 20 THEN 1 ELSE 0 END) AS ge_20,
  SUM(CASE WHEN tf_idf >= 40 THEN 1 ELSE 0 END) AS ge_40
FROM term_candidates
WHERE length_tokens <= 3;
"""

with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    row = conn.execute(sql).fetchone()

print(dict(row))
