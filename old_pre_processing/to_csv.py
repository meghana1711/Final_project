import sqlite3
import csv
from pathlib import Path

DB_PATH = Path("onto_db/ontology_sample.db")   # adjust if needed

def export_table(db_path: Path, table: str, out_csv: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()

    # get column names
    col_names = [desc[0] for desc in cur.description]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(rows)

    conn.close()
    print(f"✓ Exported {table} -> {out_csv}")

if __name__ == "__main__":
    export_table(DB_PATH, "term_candidates", Path("term_candidates.csv"))
    export_table(DB_PATH, "term_occurrences", Path("term_occurrences.csv"))
