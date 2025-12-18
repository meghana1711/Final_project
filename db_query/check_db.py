import sqlite3

db_path = "C:/Users/20236193/Final_project/onto_db/ontology_sample.db"  

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("PRAGMA table_info(sentence_lemmatized);")
print("Columns:")
for row in cur.fetchall():
    print(row)

print("\nSample rows:")
cur.execute("""  
    SELECT doc_id, sent_idx, pos_tags_json , cleaned_version, created_at
    FROM sentence_lemmatized
    ORDER BY doc_id, sent_idx
    LIMIT 10
""")
for row in cur.fetchall():
    print(row)

cur.execute("SELECT COUNT(*) FROM sentence_lemmatized;")
print("\nTotal rows:", cur.fetchone()[0])

conn.close()


