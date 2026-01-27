import argparse
import csv
import os
import re
import sqlite3
from typing import List, Tuple, Optional


def safe_filename(name: str) -> str:
    # Keep it filesystem-safe and readable
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name[:180] if len(name) > 180 else name


def list_objects(conn: sqlite3.Connection, include_views: bool) -> List[Tuple[str, str]]:
    """
    Returns list of (name, type) where type in {'table','view'}.
    Excludes SQLite internal tables.
    """
    types = ("table", "view") if include_views else ("table",)
    q = f"""
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ({",".join(["?"] * len(types))})
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
    """
    return conn.execute(q, types).fetchall()


def get_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    # PRAGMA table_info works for both tables and views
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    # row format: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]


def export_one(
    conn: sqlite3.Connection,
    name: str,
    out_path: str,
    delimiter: str,
    limit: Optional[int],
) -> int:
    cols = get_columns(conn, name)
    if not cols:
        # Fallback: attempt select and infer columns
        cur = conn.execute(f'SELECT * FROM "{name}" LIMIT 1')
        cols = [d[0] for d in cur.description] if cur.description else []

    sql = f'SELECT * FROM "{name}"'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cur = conn.execute(sql)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        if cols:
            writer.writerow(cols)
        row_count = 0
        for row in cur:
            writer.writerow(row)
            row_count += 1
    return row_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite .db file")
    ap.add_argument("--out_dir", required=True, help="Directory to write CSV files")
    ap.add_argument("--include_views", action="store_true", help="Also export views")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--limit", type=int, default=None, help="Max rows per table (optional)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = None

    objects = list_objects(conn, args.include_views)

    # Write a summary file
    summary_path = os.path.join(args.out_dir, "__tables__.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "type", "csv_file", "rows_exported"])
        for name, obj_type in objects:
            csv_name = safe_filename(f"{obj_type}__{name}.csv")
            out_path = os.path.join(args.out_dir, csv_name)
            try:
                rows = export_one(conn, name, out_path, args.delimiter, args.limit)
            except sqlite3.Error as e:
                # still log it
                writer.writerow([name, obj_type, csv_name, f"ERROR: {e}"])
                continue
            writer.writerow([name, obj_type, csv_name, rows])

    conn.close()
    print(f"Exported {len(objects)} object(s) to: {args.out_dir}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
