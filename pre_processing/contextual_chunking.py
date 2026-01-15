import argparse
import json
import sqlite3
from typing import List, Dict, Optional
from collections import defaultdict
import statistics


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------

def init_chunks_table(conn: sqlite3.Connection, table_name: str) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            chunk_id          TEXT PRIMARY KEY,
            doc_id            TEXT    NOT NULL,
            text              TEXT    NOT NULL,
            sentence_ids_json TEXT    NOT NULL,
            num_sentences     INTEGER NOT NULL,
            start_char        INTEGER,
            end_char          INTEGER,
            estimated_tokens  INTEGER NOT NULL
        )
        """
    )
    conn.commit()


# -------------------------------------------------------------------
# Chunker
# -------------------------------------------------------------------

class SentenceChunker:
    def __init__(
        self,
        min_sentences: int,
        max_sentences: int,
        min_tokens: int,
        max_tokens: int,
        overlap_sentences: int,
    ):
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

        print("Chunker configuration:")
        print(f"  Sentences/chunk: {min_sentences}–{max_sentences}")
        print(f"  Tokens/chunk:    {min_tokens}–{max_tokens}")
        print(f"  Overlap:         {overlap_sentences}")

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def create_chunk(
        self,
        sentences: List[Dict],
        chunk_id: str,
        doc_id: str,
    ) -> Dict:
        combined_text = " ".join(s["sentence"] for s in sentences)

        sentence_ids = [
            s.get("sentence_id") or f"{doc_id}_sent_{s['sent_idx']:04d}"
            for s in sentences
        ]

        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": combined_text,
            "sentence_ids": sentence_ids,
            "num_sentences": len(sentences),
            "start_char": sentences[0].get("start_char"),
            "end_char": sentences[-1].get("end_char"),
            "estimated_tokens": self.estimate_tokens(combined_text),
        }

    def chunk_document_sentences(self, sentences: List[Dict], doc_id: str) -> List[Dict]:
        if not sentences:
            return []

        sentences = sorted(sentences, key=lambda s: s["sent_idx"])
        chunks, i, chunk_num = [], 0, 1

        while i < len(sentences):
            current, tokens, start_i = [], 0, i

            while i < len(sentences) and len(current) < self.max_sentences:
                sent = sentences[i]
                sent_tokens = self.estimate_tokens(sent["sentence"])

                if current and tokens + sent_tokens > self.max_tokens:
                    break

                current.append(sent)
                tokens += sent_tokens
                i += 1

                if (
                    len(current) >= self.min_sentences
                    and tokens >= self.min_tokens
                ):
                    if i < len(sentences):
                        next_tokens = self.estimate_tokens(sentences[i]["sentence"])
                        if tokens + next_tokens > self.max_tokens:
                            break

            if current:
                chunk_id = f"{doc_id}_chunk_{chunk_num:04d}"
                chunks.append(self.create_chunk(current, chunk_id, doc_id))
                chunk_num += 1

                if self.overlap_sentences > 0:
                    i = max(i - self.overlap_sentences, start_i + 1)
            else:
                i += 1

        return chunks


# -------------------------------------------------------------------
# DB Processing
# -------------------------------------------------------------------

def process_sentences_from_db(
    db_path: str,
    sentence_table: str,
    chunks_table: str,
    cleaned_version: Optional[int],
    chunker: SentenceChunker,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        init_chunks_table(conn, chunks_table)
        cur = conn.cursor()

        where = ""
        params = []
        if cleaned_version is not None:
            where = "WHERE cleaned_version = ?"
            params.append(cleaned_version)

        query = f"""
            SELECT sentence_id, doc_id, sent_idx, sentence, start_char, end_char
            FROM {sentence_table}
            {where}
            ORDER BY doc_id, sent_idx
        """

        cur.execute(query, params)
        rows = cur.fetchall()
        print(f"Loaded {len(rows)} sentences from {sentence_table}")

        docs = defaultdict(list)
        for sid, doc_id, idx, sent, sc, ec in rows:
            docs[doc_id].append({
                "sentence_id": sid,
                "doc_id": doc_id,
                "sent_idx": idx,
                "sentence": sent,
                "start_char": sc,
                "end_char": ec,
            })

        insert_sql = f"""
            INSERT OR IGNORE INTO {chunks_table}
            (chunk_id, doc_id, text, sentence_ids_json,
             num_sentences, start_char, end_char, estimated_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        stats = []

        for doc_id, sents in docs.items():
            chunks = chunker.chunk_document_sentences(sents, doc_id)
            stats.append(len(chunks))

            for c in chunks:
                cur.execute(
                    insert_sql,
                    (
                        c["chunk_id"],
                        c["doc_id"],
                        c["text"],
                        json.dumps(c["sentence_ids"]),
                        c["num_sentences"],
                        c["start_char"],
                        c["end_char"],
                        c["estimated_tokens"],
                    ),
                )

            conn.commit()
            print(f"{doc_id}: {len(chunks)} chunks")

        if stats:
            print("\nChunking summary:")
            print(f"  Docs: {len(stats)}")
            print(f"  Avg chunks/doc: {sum(stats)/len(stats):.1f}")

    finally:
        conn.close()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--db", required=True)
    ap.add_argument("--sentence_table", default="sentence_segmented")
    ap.add_argument("--chunks_table", default="contextual_chunk")
    ap.add_argument("--cleaned_version", type=int, default=1)

    ap.add_argument("--min_sentences", type=int, default=5)
    ap.add_argument("--max_sentences", type=int, default=12)
    ap.add_argument("--min_tokens", type=int, default=400)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--overlap_sentences", type=int, default=1)

    args = ap.parse_args()

    chunker = SentenceChunker(
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_sentences=args.overlap_sentences,
    )

    process_sentences_from_db(
        db_path=args.db,
        sentence_table=args.sentence_table,
        chunks_table=args.chunks_table,
        cleaned_version=args.cleaned_version,
        chunker=chunker,
    )


if __name__ == "__main__":
    main()
