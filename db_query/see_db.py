import sqlite3, os

db_path = "onto_db/olaf_sample_llm.db"
print("Using DB:", os.path.abspath(db_path))

conn = sqlite3.connect(db_path)
c = conn.cursor()

# See if tables even exist
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", c.fetchall())

# Count chunks
try:
    c.execute("SELECT COUNT(*) FROM contextual_chunk")
    print("contextual_chunk rows =", c.fetchone()[0])
except Exception as e:
    print("Error reading contextual_chunk:", e)

# Count llm_terms
try:
    c.execute("SELECT COUNT(*) FROM llm_terms")
    print("llm_terms rows =", c.fetchone()[0])
except Exception as e:
    print("Error reading llm_terms:", e)

conn.close()

