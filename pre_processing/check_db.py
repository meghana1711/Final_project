import sqlite3

db_path = "ontology_workspace.db"  # path to your .db file

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
    SELECT doc_id, title, cleaned_text
    FROM cleaned_documents
    WHERE cleaned_version = 1
    ORDER BY created_at DESC
""")

rows = cur.fetchall()

#cur.execute("DROP TABLE IF EXISTS cleaned_documents;")
conn.close()

# optional: look at the rows
for row in rows:
    doc_id, title, cleaned_text = row
    print(doc_id, title, len(cleaned_text), "chars")


