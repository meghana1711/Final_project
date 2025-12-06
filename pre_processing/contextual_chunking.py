import json
import sqlite3
from typing import List, Dict, Optional
from collections import defaultdict
import statistics

DB_PATH = "onto_db/olaf_trial.db"          
CHUNKS_TABLE = "contextual_chunk"      
SENTENCE_TABLE = "sentence_segmented"

def init_chunks_table(conn: sqlite3.Connection, table_name: str = CHUNKS_TABLE) -> None:
    """
    Create the chunks table if it doesn't exist.
    One row per LLM chunk.
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            chunk_id          TEXT PRIMARY KEY,   -- e.g. doc123_chunk_0001
            doc_id            TEXT    NOT NULL,
            text              TEXT    NOT NULL,
            sentence_ids_json TEXT    NOT NULL,   -- JSON array of sentence_ids
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
        min_sentences: int = 7 ,
        max_sentences: int = 15,
        min_tokens: int = 300,
        max_tokens: int = 900,
        overlap_sentences: int = 1,
    ):
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

        print("Chunker initialized:")
        print(f"  Sentences per chunk: {min_sentences}-{max_sentences}")
        print(f"  Token range: {min_tokens}-{max_tokens}")
        print(f"  Overlap: {overlap_sentences} sentence(s)")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: words * 1.3)."""
        words = len(text.split())
        return int(words * 1.3)

    def create_chunk(
        self,
        sentences: List[Dict],
        chunk_id: str,
        doc_id: str,
        section: Optional[str] = None,   # ignored, kept only for signature compatibility
    ) -> Dict:
        """
        Create a chunk from a list of sentence dicts from sentence_segmented.
        Each sentence dict must contain: sentence_id (or fallback), sentence, start_char, end_char, sent_idx.
        """
        combined_text = " ".join(s["sentence"] for s in sentences)

        # Robust sentence IDs: use sentence_id if present, otherwise doc_id_sent_<sent_idx>
        sentence_ids = []
        for s in sentences:
            sid = s.get("sentence_id")
            if not sid:
                sid = f"{doc_id}_sent_{s['sent_idx']:04d}"
            sentence_ids.append(sid)

        start_char = sentences[0].get("start_char")
        end_char = sentences[-1].get("end_char")
        token_count = self.estimate_tokens(combined_text)

        return {
            "chunk_id": chunk_id,            # TEXT primary key in DB
            "doc_id": doc_id,
            "text": combined_text,
            "sentence_ids": sentence_ids,
            "num_sentences": len(sentences),
            "start_char": start_char,
            "end_char": end_char,
            "estimated_tokens": token_count,
        }

    def chunk_document_sentences(self, sentences: List[Dict], doc_id: str) -> List[Dict]:
        """
        Chunk sentences from a single document.
        sentences: list of dicts with at least sentence_id, sent_idx, sentence, start_char, end_char.
        """
        if not sentences:
            return []

        # Make sure they are ordered
        sentences = sorted(sentences, key=lambda s: s["sent_idx"])

        chunks = []
        i = 0
        chunk_num = 1
        max_iterations = len(sentences) * 2
        iterations = 0

        while i < len(sentences):
            iterations += 1
            if iterations > max_iterations:
                print(
                    f"  WARNING: {doc_id} - Breaking infinite loop at sentence {i}/{len(sentences)}"
                )
                break

            current_chunk_sents = []
            current_tokens = 0
            start_i = i

            while i < len(sentences) and len(current_chunk_sents) < self.max_sentences:
                sent = sentences[i]

                if "sentence" not in sent:
                    print(
                        f"  WARNING: {doc_id} - sentence at index {i} missing 'sentence', skipping"
                    )
                    i += 1
                    continue

                sent_tokens = self.estimate_tokens(sent["sentence"])

                if current_tokens + sent_tokens > self.max_tokens and current_chunk_sents:
                    break

                current_chunk_sents.append(sent)
                current_tokens += sent_tokens
                i += 1

                if (
                    len(current_chunk_sents) >= self.min_sentences
                    and current_tokens >= self.min_tokens
                ):
                    if len(current_chunk_sents) >= self.max_sentences:
                        break
                    if i < len(sentences):
                        next_tokens = self.estimate_tokens(sentences[i]["sentence"])
                        if current_tokens + next_tokens > self.max_tokens:
                            break

            if current_chunk_sents:
                # e.g. "doc_0117_chunk_0003"
                chunk_id = f"{doc_id}_chunk_{chunk_num:04d}"
                chunk = self.create_chunk(current_chunk_sents, chunk_id, doc_id)
                chunks.append(chunk)
                chunk_num += 1

                # overlap logic
                if self.overlap_sentences > 0 and i < len(sentences):
                    overlap = min(self.overlap_sentences, len(current_chunk_sents))
                    new_i = i - overlap
                    if new_i > start_i:  # avoid infinite loop by going backwards too far
                        i = new_i
            else:
                if i == start_i:
                    print(
                        f"  WARNING: {doc_id} - No progress at sentence {i}, forcing skip"
                    )
                    i += 1

        return chunks

    # ---------------------------------------------------------------
    # Read from sentence_segmented and write chunks to llm_chunks
    # ---------------------------------------------------------------

    def process_sentences_from_db(
        self,
        db_path: str = DB_PATH,
        sentence_table: str = SENTENCE_TABLE,
        chunks_table: str = CHUNKS_TABLE,
        cleaned_only: bool = True,
    ):
        """
        Read preprocessed sentences from sentence_segmented and insert chunks into llm_chunks.
        """
        conn = sqlite3.connect(db_path)
        try:
            init_chunks_table(conn, chunks_table)
            cur = conn.cursor()

            where_clause = ""
            params = []
            if cleaned_only:
                # assumes cleaned_version = 1 marks the cleaned version
                where_clause = "WHERE cleaned_version = 1"

            query = f"""
                SELECT
                    sentence_id,
                    doc_id,
                    sent_idx,
                    sentence,
                    start_char,
                    end_char
                FROM {sentence_table}
                {where_clause}
                ORDER BY doc_id, sent_idx
            """

            print(f"Loading sentences from {sentence_table} in {db_path} ...")
            cur.execute(query, params)
            rows = cur.fetchall()
            print(f"  Loaded {len(rows)} sentences")

            docs = defaultdict(list)
            for sentence_id, doc_id, sent_idx, sentence, start_char, end_char in rows:
                # Fallback if sentence_id is NULL/None
                if sentence_id is None:
                    sentence_id = f"{doc_id}_sent_{sent_idx:04d}"

                docs[doc_id].append(
                    {
                        "sentence_id": sentence_id,
                        "doc_id": doc_id,
                        "sent_idx": sent_idx,
                        "sentence": sentence,
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                )

            print(f"Found {len(docs)} documents with sentences")

            insert_sql = f"""
                INSERT OR IGNORE INTO {chunks_table} (
                    chunk_id,
                    doc_id,
                    text,
                    sentence_ids_json,
                    num_sentences,
                    start_char,
                    end_char,
                    estimated_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            stats = {
                "total_chunks": 0,
                "total_tokens": 0,
                "total_sentences": 0,
                "chunks_per_doc": [],
            }

            for idx, doc_id in enumerate(sorted(docs.keys()), 1):
                doc_sentences = docs[doc_id]
                print(
                    f"  [{idx}/{len(docs)}] {doc_id}: {len(doc_sentences)} sentences...",
                    end=" ",
                )

                try:
                    doc_chunks = self.chunk_document_sentences(doc_sentences, doc_id)
                    print(f"→ {len(doc_chunks)} chunks")
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

                stats["total_chunks"] += len(doc_chunks)
                stats["chunks_per_doc"].append(len(doc_chunks))

                for chunk in doc_chunks:
                    stats["total_tokens"] += chunk["estimated_tokens"]
                    stats["total_sentences"] += chunk["num_sentences"]

                    sentence_ids_json = json.dumps(chunk["sentence_ids"])

                    cur.execute(
                        insert_sql,
                        (
                            chunk["chunk_id"],
                            chunk["doc_id"],
                            chunk["text"],
                            sentence_ids_json,
                            chunk["num_sentences"],
                            chunk["start_char"],
                            chunk["end_char"],
                            chunk["estimated_tokens"],
                        ),
                    )

                conn.commit()

            avg_tokens = (
                stats["total_tokens"] / stats["total_chunks"]
                if stats["total_chunks"]
                else 0
            )
            avg_sents = (
                stats["total_sentences"] / stats["total_chunks"]
                if stats["total_chunks"]
                else 0
            )
            avg_chunks_per_doc = (
                sum(stats["chunks_per_doc"]) / len(stats["chunks_per_doc"])
                if stats["chunks_per_doc"]
                else 0
            )

            print("\n✓ Chunking finished and written to DB.")
            print(f"  DB:                {db_path}")
            print(f"  Sentence table:    {sentence_table}")
            print(f"  Chunks table:      {chunks_table}")
            print(f"  Total docs:        {len(docs)}")
            print(f"  Total chunks:      {stats['total_chunks']}")
            print(f"  Avg chunks/doc:    {avg_chunks_per_doc:.1f}")
            print(f"  Avg sentences/chunk: {avg_sents:.1f}")
            print(f"  Avg tokens/chunk:    {avg_tokens:.1f}")

        finally:
            conn.close()


# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------

def preview_chunks_from_db(
    db_path: str = DB_PATH,
    table_name: str = CHUNKS_TABLE,
    num_chunks: int = 3,
):
    """Preview first few chunks directly from the database."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT chunk_id, doc_id, num_sentences, estimated_tokens,
                   sentence_ids_json, text
            FROM {table_name}
            ORDER BY doc_id, chunk_id
            LIMIT ?
            """,
            (num_chunks,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"\nPreview of up to {num_chunks} chunks from {table_name}:")
    print("=" * 80)

    for row in rows:
        chunk_id, doc_id, num_sents, tokens, sent_ids_json, text = row
        sentence_ids = json.loads(sent_ids_json)

        print(f"\nChunk ID:    {chunk_id}")
        print(f"  Doc ID:      {doc_id}")
        print(f"  Sentences:   {num_sents}")
        print(f"  Tokens est.: {tokens}")
        print(f"  Sentence IDs: {', '.join(map(str, sentence_ids))}")
        preview_text = text[:300] + "..." if len(text) > 300 else text
        print(f"\n  Text preview:\n  {preview_text}")
        print("-" * 80)


def analyze_chunk_distribution_db(
    db_path: str = DB_PATH,
    table_name: str = CHUNKS_TABLE,
):
    """Analyze token and sentence distribution across chunks in the DB."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT estimated_tokens, num_sentences
            FROM {table_name}
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("\nNo chunks found in DB for analysis.")
        return

    token_counts = [r[0] for r in rows]
    sent_counts = [r[1] for r in rows]

    print("\nChunk Distribution Analysis (from DB):")
    print("=" * 80)

    print("\nToken Distribution:")
    print(f"  Min:    {min(token_counts)}")
    print(f"  Max:    {max(token_counts)}")
    print(f"  Mean:   {sum(token_counts) / len(token_counts):.1f}")
    print(f"  Median: {statistics.median(token_counts):.1f}")

    print("\nSentence Distribution:")
    print(f"  Min:    {min(sent_counts)}")
    print(f"  Max:    {max(sent_counts)}")
    print(f"  Mean:   {sum(sent_counts) / len(sent_counts):.1f}")
    print(f"  Median: {statistics.median(sent_counts):.1f}")

    print(f"\nToken Range Distribution:")
    ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 10000)]
    total = len(token_counts)

    for low, high in ranges:
        count = sum(1 for t in token_counts if low <= t < high)
        pct = 100 * count / total
        # no need for :4d, just keep it simple
        print(f"  {low:3}-{high:4}: {count:4} chunks ({pct:5.1f}%)")

if __name__ == "__main__":
    chunker = SentenceChunker(
        min_sentences=5,
        max_sentences=12,
        min_tokens=400,
        max_tokens=800,
        overlap_sentences=1,
    )

    # 1) Build chunks from sentence_segmented -> llm_chunks
    chunker.process_sentences_from_db(
        db_path=DB_PATH,
        sentence_table=SENTENCE_TABLE,
        chunks_table=CHUNKS_TABLE,
        cleaned_only=True,
    )

    # 2) Quick evaluation
    preview_chunks_from_db(DB_PATH, CHUNKS_TABLE, num_chunks=3)
    analyze_chunk_distribution_db(DB_PATH, CHUNKS_TABLE)