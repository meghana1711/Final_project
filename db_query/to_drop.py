import sqlite3

db_path = "onto_db/slurm_llm_final.db"
table_name = "llm_is_a_runs"

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute(f"DROP TABLE IF EXISTS {table_name}")
conn.commit()
conn.close()

print(f"[OK] Dropped table: {table_name}")