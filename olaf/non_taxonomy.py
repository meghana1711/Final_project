"""
olaf/non_taxonomy.py  (simple version)

- Extract non-taxonomic (subject, relation, object) triples from sentences using spaCy
- Canonicalize subject/object spans to term_enrichment canonical IDs
- Store raw triples in non_taxonomic_edges
- Store filtered triples in non_taxonomic_edges_clean
- DOES NOT drop tables every run
- DOES NOT use run_id/rule/score/support columns (back to "usual")

CLI:
  python -m olaf.non_taxonomy \
    --db onto_db/onto_new.db \
    --spacy_model en_core_web_sm \
    --stopwords stop_word/stop_words.txt \
    --sentence_table sentence_segmented \
    --term_candidates_table term_candidates \
    --term_enrichment_table term_enrichment \
    --raw_edges_table non_taxonomic_edges \
    --clean_edges_table non_taxonomic_edges_clean \
    --method openie_spacy

Notes:
- Requires spaCy model installed: python -m spacy download en_core_web_sm
"""

import argparse
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import spacy


# -------------------------------------------------------------------
# DEFAULTS
# -------------------------------------------------------------------

DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_METHOD = "openie_spacy"

DEFAULT_STOP_WORDS_FILE = "stop_word/stop_words.txt"

DEFAULT_SENTENCE_TABLE = "sentence_segmented"
DEFAULT_TERM_CANDIDATES_TABLE = "term_candidates"
DEFAULT_TERM_ENRICHMENT_TABLE = "term_enrichment"

DEFAULT_RAW_EDGES_TABLE = "non_taxonomic_edges"
DEFAULT_CLEAN_EDGES_TABLE = "non_taxonomic_edges_clean"

# Fully automatic filters
MAX_REL_LEN = 80        # "very long" relation text; adjust if needed
MIN_REL_TOTAL = 3       # ReVerb-style: relation must appear at least this many times
MIN_REL_SUBJ = 2        # ... with at least this many distinct subjects
MIN_REL_OBJ = 2         # ... and at least this many distinct objects

ATTRIBUTE_HEADS = {"time", "size", "value"}

STOP_WORDS: set[str] = set()


# -------------------------------------------------------------------
# DB HELPERS
# -------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_non_taxonomic_edges_table(conn: sqlite3.Connection, table_name: str) -> None:
    """
    Raw OpenIE-style triples with canonical mapping.
    No drop; upserts prevented via UNIQUE + INSERT OR IGNORE.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id                TEXT,
            sentence_id           TEXT,
            sentence_text         TEXT,

            subj_text             TEXT NOT NULL,
            subj_canonical_id     INTEGER NOT NULL,
            subj_canonical_term   TEXT    NOT NULL,

            rel_text              TEXT NOT NULL,

            obj_text              TEXT NOT NULL,
            obj_canonical_id      INTEGER NOT NULL,
            obj_canonical_term    TEXT    NOT NULL,

            method                TEXT NOT NULL,
            created_at            TEXT    NOT NULL,

            UNIQUE(subj_canonical_id, rel_text, obj_canonical_id, sentence_id, method)
        )
        """
    )
    conn.commit()
    print(f"[INFO] Ensured {table_name} exists (no drop).")


def init_non_taxonomic_edges_clean_table(conn: sqlite3.Connection, table_name: str) -> None:
    """
    Cleaned triples after automatic filtering.
    Do NOT drop table every run.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id                TEXT,
            sentence_id           TEXT,
            sentence_text         TEXT,

            subj_text             TEXT NOT NULL,
            subj_canonical_id     INTEGER NOT NULL,
            subj_canonical_term   TEXT    NOT NULL,

            rel_text              TEXT NOT NULL,

            obj_text              TEXT NOT NULL,
            obj_canonical_id      INTEGER NOT NULL,
            obj_canonical_term    TEXT    NOT NULL,

            method                TEXT NOT NULL,
            created_at            TEXT    NOT NULL,

            UNIQUE(subj_canonical_id, rel_text, obj_canonical_id, sentence_id, method)
        )
        """
    )
    conn.commit()
    print(f"[INFO] Ensured {table_name} exists (no drop).")


# -------------------------------------------------------------------
# STOP WORDS
# -------------------------------------------------------------------

def load_stop_words(path: str) -> set:
    """
    Load stop words from a file (one word per line).
    Lines starting with '#' or empty lines are ignored.
    """
    words: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if not w or w.startswith("#"):
                    continue
                words.add(w)
        print(f"[INFO] Loaded {len(words)} stop words from '{path}'.")
    except FileNotFoundError:
        fallback = ["a", "an", "the", "this", "that", "these", "those", "it", "its"]
        words.update(fallback)
        print(f"[WARN] Stop-word file '{path}' not found. Using fallback list: {fallback}")
    return words


# -------------------------------------------------------------------
# CANONICAL TERM MAPPING
# -------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_canonical_maps(
    conn: sqlite3.Connection,
    term_enrichment_table: str,
    term_candidates_table: str,
):
    """
    Build:
      id2term: canonical_id -> canonical_term
      label2id: normalized canonical_term -> canonical_id
      surface2canonical: normalized term_text -> canonical_id
                         via member_term_ids_json + term_candidates
    """
    rows = conn.execute(
        f"""
        SELECT canonical_id, canonical_term, member_term_ids_json
        FROM {term_enrichment_table}
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
        """
    ).fetchall()

    id2term: Dict[int, str] = {}
    label2id: Dict[str, int] = {}
    termid2canonical: Dict[int, int] = {}

    for r in rows:
        cid = int(r["canonical_id"])
        label = (r["canonical_term"] or "").strip()
        if not label:
            continue
        id2term[cid] = label
        label2id[normalize_text(label)] = cid

        mjson = r["member_term_ids_json"]
        if not mjson:
            continue
        try:
            member_ids = json.loads(mjson)
        except json.JSONDecodeError:
            continue
        if isinstance(member_ids, list):
            for tid in member_ids:
                try:
                    termid2canonical[int(tid)] = cid
                except Exception:
                    continue

    surface2canonical: Dict[str, int] = {}
    trows = conn.execute(f"SELECT term_id, term_text FROM {term_candidates_table}").fetchall()
    for tr in trows:
        term_id = int(tr["term_id"])
        text = (tr["term_text"] or "").strip()
        if not text:
            continue
        cid = termid2canonical.get(term_id)
        if cid is None:
            continue
        surface2canonical[normalize_text(text)] = cid

    print(
        f"[INFO] Canonical maps: id2term={len(id2term)}, "
        f"label2id={len(label2id)}, surface2canonical={len(surface2canonical)}"
    )
    return id2term, label2id, surface2canonical


def canonicalize_span(
    span_text: str,
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int],
) -> Optional[int]:
    """
    Map a raw NP span string to canonical_id, if possible.

    Steps:
      1) exact match on canonical_term
      2) exact match on term_candidates.term_text
      3) strip leading 'the', 'a', 'an' and retry
      4) take head word (last token) and retry
    """
    if not span_text:
        return None

    norm = normalize_text(span_text)

    cid = label2id.get(norm)
    if cid is not None:
        return cid

    cid = surface2canonical.get(norm)
    if cid is not None:
        return cid

    norm2 = re.sub(r"^(the|a|an)\s+", "", norm)
    if norm2 != norm:
        cid = label2id.get(norm2)
        if cid is not None:
            return cid
        cid = surface2canonical.get(norm2)
        if cid is not None:
            return cid

    tokens = norm2.split()
    if len(tokens) > 1:
        head = tokens[-1]
        cid = label2id.get(head)
        if cid is not None:
            return cid
        cid = surface2canonical.get(head)
        if cid is not None:
            return cid

    return None


# -------------------------------------------------------------------
# TEXT HELPERS
# -------------------------------------------------------------------

def is_generic_span(span_text: str) -> bool:
    """
    Return True if the span is only made of stop-words.
    """
    if not span_text:
        return True
    tokens = re.findall(r"[A-Za-z0-9_]+", span_text.lower())
    if not tokens:
        return True
    return all(tok in STOP_WORDS for tok in tokens)


def get_head(term: str) -> str:
    term = (term or "").strip().lower()
    if not term:
        return ""
    tokens = term.split()
    return tokens[-1]


def is_valid_relation_text(rel_text: str) -> bool:
    """
    Relation should be:
      - not empty
      - not too long
      - contain at least one alphabetic character
      - composed only of letters and spaces
    """
    if not rel_text:
        return False
    rel = rel_text.strip()
    if not rel:
        return False
    if len(rel) > MAX_REL_LEN:
        return False
    if not re.search(r"[A-Za-z]", rel):
        return False
    if re.search(r"[^A-Za-z\s]", rel):
        return False
    return True


# -------------------------------------------------------------------
# spaCy HELPERS: simple OpenIE-style triple extraction
# -------------------------------------------------------------------

def get_noun_span(token) -> str:
    """
    Return a reasonably complete NP span for a token (subject/object).
    Prefer noun chunks; fallback to subtree.
    """
    doc = token.doc
    for nc in doc.noun_chunks:
        if token.i >= nc.start and token.i < nc.end:
            return nc.text.strip()

    subtree_tokens = [t for t in token.subtree if not t.is_space]
    if not subtree_tokens:
        return token.text
    return doc[subtree_tokens[0].i: subtree_tokens[-1].i + 1].text.strip()


def extract_triples_from_sentence(doc) -> List[Tuple[str, str, str]]:
    """
    Very simple OpenIE-like extractor:

    For each VERB:
      - subjects (nsubj / nsubjpass)
      - objects (dobj / attr / oprd)
      - prep objects: verb -> prep -> pobj
      - rel_text = verb lemma + optional preposition (e.g. 'submit to')

    Returns list of (subj_text, rel_text, obj_text).
    """
    triples: List[Tuple[str, str, str]] = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        verb = token
        subjects = [w for w in verb.children if w.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            continue

        direct_objs = [w for w in verb.children if w.dep_ in ("dobj", "attr", "oprd")]

        prep_objs = []
        preps = [w for w in verb.children if w.dep_ == "prep"]
        for p in preps:
            for o in [c for c in p.children if c.dep_ == "pobj"]:
                prep_objs.append((p, o))

        base_rel = verb.lemma_.lower()

        for subj in subjects:
            subj_text = get_noun_span(subj)

            for obj in direct_objs:
                obj_text = get_noun_span(obj)
                rel_text = base_rel
                if is_valid_relation_text(rel_text):
                    triples.append((subj_text, rel_text, obj_text))

            for p, pobj in prep_objs:
                obj_text = get_noun_span(pobj)
                rel_text = f"{base_rel} {p.text.lower()}"
                if is_valid_relation_text(rel_text):
                    triples.append((subj_text, rel_text, obj_text))

    return triples


# -------------------------------------------------------------------
# MAIN EXTRACTION LOGIC (RAW)
# -------------------------------------------------------------------

def induce_openie_edges(
    conn: sqlite3.Connection,
    nlp,
    id2term: Dict[int, str],
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int],
    sentence_table: str,
    raw_edges_table: str,
    method: str,
) -> None:
    """
    Scan sentence_table, run spaCy-based OpenIE, store triples into raw_edges_table.

    Only keep triples where BOTH subject and object map to canonical terms.
    """
    try:
        sents = conn.execute(f"SELECT * FROM {sentence_table}").fetchall()
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Could not read {sentence_table}:", e)
        return

    print(f"[DEBUG] Loaded {len(sents)} rows from {sentence_table}.")

    init_non_taxonomic_edges_table(conn, raw_edges_table)
    cur = conn.cursor()

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    total_triples = 0
    canonicalized = 0
    inserted = 0
    debug_examples = 0

    for row in sents:
        cols = row.keys()

        # doc_id
        if "doc_id" in cols:
            doc_id = row["doc_id"]
        elif "document_id" in cols:
            doc_id = row["document_id"]
        else:
            doc_id = None

        # sentence_id
        if "sentence_id" in cols:
            sentence_id = row["sentence_id"]
        elif "sent_id" in cols:
            sentence_id = row["sent_id"]
        else:
            sentence_id = None

        # sentence text
        text = ""
        for cand in ("sent_text", "sentence_text", "sentence", "text"):
            if cand in cols:
                text = row[cand] or ""
                break

        sent = (text or "").strip()
        if not sent:
            continue

        doc = nlp(sent)
        triples = extract_triples_from_sentence(doc)
        if not triples:
            continue

        total_triples += len(triples)

        for subj_text, rel_text, obj_text in triples:
            if is_generic_span(subj_text) or is_generic_span(obj_text):
                continue

            subj_cid = canonicalize_span(subj_text, label2id, surface2canonical)
            obj_cid = canonicalize_span(obj_text, label2id, surface2canonical)

            if subj_cid is None or obj_cid is None:
                continue

            subj_term = id2term.get(subj_cid)
            obj_term = id2term.get(obj_cid)
            if not subj_term or not obj_term:
                continue

            if subj_cid == obj_cid:
                continue

            canonicalized += 1

            if debug_examples < 15:
                print(
                    f"[TRIPLE] subj='{subj_text}' → {subj_term}  "
                    f"rel='{rel_text}'  obj='{obj_text}' → {obj_term}"
                )
                debug_examples += 1

            cur.execute(
                f"""
                INSERT OR IGNORE INTO {raw_edges_table}
                (doc_id,
                 sentence_id,
                 sentence_text,
                 subj_text,
                 subj_canonical_id,
                 subj_canonical_term,
                 rel_text,
                 obj_text,
                 obj_canonical_id,
                 obj_canonical_term,
                 method,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    None if sentence_id is None else str(sentence_id),
                    sent,
                    subj_text,
                    subj_cid,
                    subj_term,
                    rel_text,
                    obj_text,
                    obj_cid,
                    obj_term,
                    method,
                    now,
                ),
            )
            inserted += cur.rowcount

    conn.commit()
    print(f"[STATS] total raw triples extracted (before filters): {total_triples}")
    print(f"[STATS] triples with BOTH canonical arguments:       {canonicalized}")
    print(f"[STATS] NEW rows inserted into {raw_edges_table}:     {inserted}")


# -------------------------------------------------------------------
# RELATION FREQUENCY & CLEANING
# -------------------------------------------------------------------

def build_good_relation_set(conn: sqlite3.Connection, raw_edges_table: str) -> set:
    """
    ReVerb-style frequency & diversity filter:
      - min total count
      - min distinct subjects
      - min distinct objects
    """
    rows = conn.execute(
        f"""
        SELECT
            rel_text,
            COUNT(*) AS n_total,
            COUNT(DISTINCT subj_canonical_id) AS n_subj,
            COUNT(DISTINCT obj_canonical_id) AS n_obj
        FROM {raw_edges_table}
        GROUP BY rel_text
        """
    ).fetchall()

    good = set()
    for r in rows:
        if (r["n_total"] >= MIN_REL_TOTAL and
            r["n_subj"]  >= MIN_REL_SUBJ  and
            r["n_obj"]   >= MIN_REL_OBJ):
            good.add(r["rel_text"])

    print(f"[INFO] Good relation phrases (freq/diversity filter): {len(good)}")
    return good


def apply_automatic_filters(
    conn: sqlite3.Connection,
    raw_edges_table: str,
    clean_edges_table: str,
) -> None:
    """
    Append into clean_edges_table from raw_edges_table using automatic filters.
    Does NOT drop clean table.
    """
    init_non_taxonomic_edges_clean_table(conn, clean_edges_table)

    good_relations = build_good_relation_set(conn, raw_edges_table)
    rows = conn.execute(f"SELECT * FROM {raw_edges_table}").fetchall()
    cur = conn.cursor()

    inserted = 0

    for r in rows:
        rel_text = r["rel_text"]
        if rel_text not in good_relations:
            continue
        if not is_valid_relation_text(rel_text):
            continue

        subj_id = r["subj_canonical_id"]
        obj_id = r["obj_canonical_id"]
        if subj_id == obj_id:
            continue

        subj_text = r["subj_text"]
        obj_text = r["obj_text"]
        if is_generic_span(subj_text) or is_generic_span(obj_text):
            continue

        subj_term = r["subj_canonical_term"]
        if get_head(subj_term) in ATTRIBUTE_HEADS:
            continue

        cur.execute(
            f"""
            INSERT OR IGNORE INTO {clean_edges_table}
            (doc_id,
             sentence_id,
             sentence_text,
             subj_text,
             subj_canonical_id,
             subj_canonical_term,
             rel_text,
             obj_text,
             obj_canonical_id,
             obj_canonical_term,
             method,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["doc_id"],
                r["sentence_id"],
                r["sentence_text"],
                r["subj_text"],
                subj_id,
                subj_term,
                rel_text,
                r["obj_text"],
                obj_id,
                r["obj_canonical_term"],
                r["method"],
                r["created_at"],
            ),
        )
        inserted += cur.rowcount

    conn.commit()
    print(f"[STATS] NEW rows inserted into {clean_edges_table}: {inserted}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-taxonomic relation extraction (simple OpenIE via spaCy).")

    p.add_argument("--db", required=True, help="Path to SQLite DB.")
    p.add_argument("--spacy_model", default=DEFAULT_SPACY_MODEL, help="spaCy model name (e.g., en_core_web_sm).")
    p.add_argument("--stopwords", default=DEFAULT_STOP_WORDS_FILE, help="Path to stopwords file.")

    p.add_argument("--sentence_table", default=DEFAULT_SENTENCE_TABLE, help="Input sentence table (segmented sentences).")
    p.add_argument("--term_candidates_table", default=DEFAULT_TERM_CANDIDATES_TABLE, help="term_candidates table name.")
    p.add_argument("--term_enrichment_table", default=DEFAULT_TERM_ENRICHMENT_TABLE, help="term_enrichment table name.")

    p.add_argument("--raw_edges_table", default=DEFAULT_RAW_EDGES_TABLE, help="Output raw edges table name.")
    p.add_argument("--clean_edges_table", default=DEFAULT_CLEAN_EDGES_TABLE, help="Output clean edges table name.")

    p.add_argument("--method", default=DEFAULT_METHOD, help="Method name to store in DB.")

    return p.parse_args()


def main() -> None:
    global STOP_WORDS

    args = parse_args()

    print(f"[INFO] Loading spaCy model '{args.spacy_model}'...")
    nlp = spacy.load(args.spacy_model)

    STOP_WORDS = load_stop_words(args.stopwords)

    conn = get_connection(args.db)
    try:
        id2term, label2id, surface2canonical = load_canonical_maps(
            conn,
            term_enrichment_table=args.term_enrichment_table,
            term_candidates_table=args.term_candidates_table,
        )

        induce_openie_edges(
            conn=conn,
            nlp=nlp,
            id2term=id2term,
            label2id=label2id,
            surface2canonical=surface2canonical,
            sentence_table=args.sentence_table,
            raw_edges_table=args.raw_edges_table,
            method=args.method,
        )

        apply_automatic_filters(
            conn=conn,
            raw_edges_table=args.raw_edges_table,
            clean_edges_table=args.clean_edges_table,
        )

    finally:
        conn.close()

    print("[INFO] OpenIE-style non-taxonomic extraction + automatic filtering completed.")


if __name__ == "__main__":
    main()
