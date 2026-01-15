import hashlib
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


def normalize_whitespace(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[^\S\n\r\t]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_db(
    db_path: str,
    raw_table: str = "raw_documents",
    cleaned_table: str = "cleaned_documents",
    segmented_table: str = "sentence_segmented",
    lemmatized_table: str = "sentence_lemmatized",
) -> None:
    """
    Create DB and required tables if missing.
    Table names are configurable.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {raw_table} (
            doc_id        TEXT PRIMARY KEY,
            title         TEXT,
            text          TEXT,
            content_hash  TEXT UNIQUE,
            version       INTEGER,
            created_at    TEXT
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {cleaned_table} (
            cleaned_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id          TEXT NOT NULL,
            title           TEXT,
            cleaned_text    TEXT,
            raw_version     INTEGER,
            cleaned_version INTEGER,
            created_at      TEXT,
            stats_json      TEXT,
            FOREIGN KEY (doc_id) REFERENCES {raw_table}(doc_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {segmented_table} (
            sentence_id     TEXT PRIMARY KEY,
            doc_id          TEXT NOT NULL,
            sent_idx        INTEGER NOT NULL,
            sentence        TEXT NOT NULL,
            start_char      INTEGER,
            end_char        INTEGER,
            length          INTEGER,
            cleaned_version INTEGER,
            created_at      TEXT,
            FOREIGN KEY (doc_id) REFERENCES {raw_table}(doc_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {lemmatized_table} (
            lemma_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id                      TEXT NOT NULL,
            sent_idx                    INTEGER NOT NULL,
            sentence                    TEXT NOT NULL,
            tokens_json                 TEXT,
            lemmas_json                 TEXT,
            lemmas_with_case_json       TEXT,
            lemmatized_text             TEXT,
            lemmatized_text_with_case   TEXT,
            pos_tags_json               TEXT,
            cleaned_version             INTEGER,
            created_at                  TEXT,
            FOREIGN KEY (doc_id) REFERENCES {raw_table}(doc_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
