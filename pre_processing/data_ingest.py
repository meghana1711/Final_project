import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict

from .db_utils import init_db, normalize_whitespace, hash_text, utc_now


def ingest_folder(
    folder_path: str,
    db_path: str,
    raw_table: str,
    version: int = 1,
) -> List[Dict[str, str]]:
    init_db(db_path, raw_table=raw_table)

    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        print(f"Warning: No .txt files found in {folder_path}")
        return []

    print(f"Found {len(txt_files)} .txt files in {folder_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    inserted_docs: List[Dict[str, str]] = []
    count_new = 0

    for file_path in txt_files:
        try:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            if not raw_text.strip():
                print(f"Skipping empty file: {file_path.name}")
                continue

            cleaned_for_hash = normalize_whitespace(raw_text).strip()
            if not cleaned_for_hash:
                print(f"Skipping empty-after-normalize file: {file_path.name}")
                continue

            h = hash_text(cleaned_for_hash)
            doc_id = f"doc_{h[:4]}"
            title = file_path.stem
            created_at = utc_now()

            before = conn.total_changes
            cur.execute(
                f"""
                INSERT OR IGNORE INTO {raw_table}
                    (doc_id, title, text, content_hash, version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, title, cleaned_for_hash, h, version, created_at),
            )
            inserted = (conn.total_changes > before)

            if inserted:
                count_new += 1
                inserted_docs.append(
                    {"doc_id": doc_id, "title": title, "version": version, "created_at": created_at}
                )
                print(f"[NEW] {file_path.name} -> {doc_id} ({len(cleaned_for_hash)} chars)")
            else:
                print(f"[SKIP exists] {file_path.name} (duplicate content)")

        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

    conn.commit()
    conn.close()

    print(f"\nDone. Inserted {count_new} new document(s) into {db_path} (table={raw_table}) version={version}.")
    return inserted_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--input", required=True, help="Folder containing .txt files")
    ap.add_argument("--raw_table", default="raw_documents")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    ingest_folder(args.input, args.db, args.raw_table, version=args.version)


if __name__ == "__main__":
    main()
