import sqlite3

DB_PATH = r"onto_db/onto_new.db"  # adjust if needed

def show_top_relations(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT rel_text, COUNT(*) AS freq
        FROM non_taxonomic_edges_clean
        GROUP BY rel_text
        ORDER BY freq DESC
        LIMIT 30;
    """

    for row in cur.execute(query):
        print(f"{row['rel_text']}\t{row['freq']}")

    conn.close()

if __name__ == "__main__":
    show_top_relations(DB_PATH)
