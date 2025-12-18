import sqlite3

DB_PATH = r"onto_db/ontology_sample.db"  # adjust if needed

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# --- term_candidates ---
print("=== term_candidates ===")
cur.execute("SELECT COUNT(*) FROM term_candidates;")
print("Total terms:", cur.fetchone()[0])

cur.execute("""
    SELECT term_id, term_text, term_lemma, length_tokens, freq_total, freq_docs
    FROM term_candidates
    ORDER BY freq_total DESC, length_tokens DESC
    LIMIT 10
""")
rows = cur.fetchall()
print("\nTop 10 terms:")
for r in rows:
    print(r)

# --- term_occurrences ---
print("\n=== term_occurrences ===")
cur.execute("SELECT COUNT(*) FROM term_occurrences;")
print("Total occurrences:", cur.fetchone()[0])

cur.execute("""
    SELECT term_id, doc_id, sent_idx, token_start, token_end, cleaned_version
    FROM term_occurrences
    LIMIT 10
""")
rows = cur.fetchall()
print("\nSample 10 occurrences:")
for r in rows:
    print(r)

conn.close()

