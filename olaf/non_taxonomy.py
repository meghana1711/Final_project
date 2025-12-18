import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import spacy


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DB_PATH = r"onto_db/onto_new.db"          # <-- adjust if needed
SPACY_MODEL = "en_core_web_sm"

METHOD_NAME = "openie_spacy"

# Path to your custom stop-word list (one word per line, lowercase recommended)
STOP_WORDS_FILE = r"stop_word/stop_words.txt"          # <-- set to your file

# Fully automatic filters
MAX_REL_LEN = 80        # "very long" relation text; adjust if needed
MIN_REL_TOTAL = 3       # ReVerb-style: relation must appear at least this many times
MIN_REL_SUBJ = 2        # ... with at least this many distinct subjects
MIN_REL_OBJ = 2         # ... and at least this many distinct objects

# Heads that are likely attribute-like and not good as subjects of actions
ATTRIBUTE_HEADS = {"time", "size", "value"}

# Global stop-word set (filled at runtime)
STOP_WORDS: set[str] = set()


# -------------------------------------------------------------------
# DB HELPERS
# -------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_non_taxonomic_edges_table(conn: sqlite3.Connection) -> None:
    """
    Raw OpenIE-style triples with canonical mapping.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS non_taxonomic_edges (
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
    print("[INFO] Ensured non_taxonomic_edges table exists.")


def init_non_taxonomic_edges_clean_table(conn: sqlite3.Connection) -> None:
    """
    Cleaned triples after automatic filtering.
    """
    conn.execute("DROP TABLE IF EXISTS non_taxonomic_edges_clean;")

    conn.execute(
        """
        CREATE TABLE non_taxonomic_edges_clean (
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
    print("[INFO] Created non_taxonomic_edges_clean table.")


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
        # Fallback minimal set if file is missing
        fallback = ["a", "an", "the", "this", "that", "these", "those", "it", "its"]
        words.update(fallback)
        print(
            f"[WARN] Stop-word file '{path}' not found. "
            f"Using fallback list: {fallback}"
        )
    return words


# -------------------------------------------------------------------
# CANONICAL TERM MAPPING
# -------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_canonical_maps(conn: sqlite3.Connection):
    """
    Build:
      id2term: canonical_id -> canonical_term
      label2id: normalized canonical_term -> canonical_id
      surface2canonical: normalized term_text -> canonical_id
                         via member_term_ids_json + term_candidates
    """
    rows = conn.execute(
        """
        SELECT canonical_id, canonical_term, member_term_ids_json
        FROM term_enrichment
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
        for tid in member_ids:
            termid2canonical[int(tid)] = cid

    surface2canonical: Dict[str, int] = {}
    trows = conn.execute("SELECT term_id, term_text FROM term_candidates").fetchall()
    for tr in trows:
        term_id = int(tr["term_id"])
        text = (tr["term_text"] or "").strip()
        if not text:
            continue
        cid = termid2canonical.get(term_id)
        if cid is None:
            continue
        key = normalize_text(text)
        surface2canonical[key] = cid

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

    Steps (all automatic):
      1) exact match on canonical_term
      2) exact match on term_candidates.term_text
      3) strip leading 'the', 'a', 'an' and retry
      4) take head word (last token) and retry
    """
    if not span_text:
        return None

    norm = normalize_text(span_text)

    # 1) exact match on canonical labels
    cid = label2id.get(norm)
    if cid is not None:
        return cid

    # 2) exact match on surface terms
    cid = surface2canonical.get(norm)
    if cid is not None:
        return cid

    # 3) strip leading determiners and retry
    norm2 = re.sub(r"^(the|a|an)\s+", "", norm)
    if norm2 != norm:
        cid = label2id.get(norm2)
        if cid is not None:
            return cid
        cid = surface2canonical.get(norm2)
        if cid is not None:
            return cid

    # 4) fall back to head word (last token)
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
# TEXT HELPERS (stop words, heads, relation validity)
# -------------------------------------------------------------------

def is_generic_span(span_text: str) -> bool:
    """
    Return True if the span is only made of stop-words
    (e.g. 'the', 'a', 'this'), so we don't want it as subject/object.
    Uses STOP_WORDS loaded from file.
    """
    if not span_text:
        return True

    tokens = re.findall(r"[A-Za-z0-9_]+", span_text.lower())
    if not tokens:
        return True

    # If *all* tokens are stop-words, treat as generic/junk.
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
      - composed only of letters and spaces (no symbols like '=')
    """
    if not rel_text:
        return False

    rel = rel_text.strip()
    if not rel:
        return False

    if len(rel) > MAX_REL_LEN:
        return False

    # must contain at least one letter
    if not re.search(r"[A-Za-z]", rel):
        return False

    # must NOT contain non-letter, non-space characters (so no '=' etc.)
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
    # If token is inside a noun chunk, use that chunk
    for nc in doc.noun_chunks:
        if token.i >= nc.start and token.i < nc.end:
            return nc.text.strip()
    # else use subtree
    subtree_tokens = list(token.subtree)
    subtree_tokens = [t for t in subtree_tokens if not t.is_space]
    if not subtree_tokens:
        return token.text
    return doc[subtree_tokens[0].i : subtree_tokens[-1].i + 1].text.strip()


def extract_triples_from_sentence(doc) -> List[Tuple[str, str, str]]:
    """
    Very simple OpenIE-like extractor:

    For each VERB:
      - find subjects (nsubj / nsubjpass)
      - find objects (dobj / attr / oprd)
      - also objects from prepositions: verb -> prep -> pobj
      - relation text = verb lemma + optional preposition (e.g. 'submit to', 'run on')

    Returns list of (subj_text, rel_text, obj_text).
    """
    triples: List[Tuple[str, str, str]] = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        verb = token

        # subjects
        subjects = [w for w in verb.children if w.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            continue  # we want at least a subject

        # direct / attribute objects
        direct_objs = [w for w in verb.children if w.dep_ in ("dobj", "attr", "oprd")]

        # prepositional objects: verb -> prep -> pobj
        prep_objs = []
        preps = [w for w in verb.children if w.dep_ == "prep"]
        for p in preps:
            pobj = [c for c in p.children if c.dep_ == "pobj"]
            for o in pobj:
                prep_objs.append((p, o))

        base_rel = verb.lemma_.lower()

        # subject × direct object
        for subj in subjects:
            subj_text = get_noun_span(subj)
            for obj in direct_objs:
                obj_text = get_noun_span(obj)
                rel_text = base_rel
                if is_valid_relation_text(rel_text):
                    triples.append((subj_text, rel_text, obj_text))

            # subject × prepositional object
            for p, pobj in prep_objs:
                obj_text = get_noun_span(pobj)
                rel_text = f"{base_rel} {p.text.lower()}"  # e.g. 'submit to'
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
    method: str = METHOD_NAME,
) -> None:
    """
    Scan sentence_segmentation, run spaCy-based OpenIE, and store triples
    into non_taxonomic_edges.

    We only keep triples where BOTH subject and object are mapped
    to canonical terms.
    """
    try:
        sents = conn.execute("SELECT * FROM sentence_segmented").fetchall()
    except sqlite3.OperationalError as e:
        print("[ERROR] Could not read sentence_segmentation:", e)
        return

    print(f"[DEBUG] Loaded {len(sents)} sentences from sentence_segmented.")

    init_non_taxonomic_edges_table(conn)
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
            # Skip if subject/object span is just stop-words
            if is_generic_span(subj_text) or is_generic_span(obj_text):
                continue

            # Map spans to canonical terms
            subj_cid = canonicalize_span(subj_text, label2id, surface2canonical)
            obj_cid = canonicalize_span(obj_text, label2id, surface2canonical)

            # Only keep triples where BOTH subject and object are canonical
            if subj_cid is None or obj_cid is None:
                continue

            subj_term = id2term.get(subj_cid)
            obj_term = id2term.get(obj_cid)
            if not subj_term or not obj_term:
                continue

            # Remove self-loops
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
                """
                INSERT OR IGNORE INTO non_taxonomic_edges
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
            inserted += 1

    conn.commit()
    print(f"[STATS] total raw triples extracted (before filters): {total_triples}")
    print(f"[STATS] triples with BOTH canonical arguments:       {canonicalized}")
    print(f"[STATS] rows inserted into non_taxonomic_edges:      {inserted}")


# -------------------------------------------------------------------
# RELATION FREQUENCY & CLEANING
# -------------------------------------------------------------------

def build_good_relation_set(conn: sqlite3.Connection) -> set:
    """
    ReVerb-style frequency & diversity filter:
      - min total count
      - min distinct subjects
      - min distinct objects
    """
    rows = conn.execute(
        """
        SELECT
            rel_text,
            COUNT(*) AS n_total,
            COUNT(DISTINCT subj_canonical_id) AS n_subj,
            COUNT(DISTINCT obj_canonical_id) AS n_obj
        FROM non_taxonomic_edges
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


def apply_automatic_filters(conn: sqlite3.Connection) -> None:
    """
    Build non_taxonomic_edges_clean from non_taxonomic_edges
    applying:
      - relation frequency & diversity filter
      - remove self-loops (defensive)
      - remove long rel_text
      - stop-word spans for subj/obj
      - attribute-like subjects (* time, * size, * value)
      - relation must be only alphabetic + spaces (no '=' etc.)
    """
    init_non_taxonomic_edges_clean_table(conn)

    good_relations = build_good_relation_set(conn)
    rows = conn.execute("SELECT * FROM non_taxonomic_edges").fetchall()
    cur = conn.cursor()

    kept = 0

    for r in rows:
        rel_text = r["rel_text"]
        if rel_text not in good_relations:
            continue

        if not is_valid_relation_text(rel_text):
            continue

        subj_id = r["subj_canonical_id"]
        obj_id = r["obj_canonical_id"]

        # self-loop check (again, just in case)
        if subj_id == obj_id:
            continue

        subj_text = r["subj_text"]
        obj_text = r["obj_text"]

        # stop-word span check
        if is_generic_span(subj_text) or is_generic_span(obj_text):
            continue

        # attribute-head subject filter
        subj_term = r["subj_canonical_term"]
        subj_head = get_head(subj_term)
        if subj_head in ATTRIBUTE_HEADS:
            continue

        cur.execute(
            """
            INSERT OR IGNORE INTO non_taxonomic_edges_clean
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
                subj_text,
                subj_id,
                subj_term,
                rel_text,
                obj_text,
                obj_id,
                r["obj_canonical_term"],
                r["method"],
                r["created_at"],
            ),
        )
        kept += 1

    conn.commit()
    print(f"[STATS] rows kept in non_taxonomic_edges_clean: {kept}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    global STOP_WORDS

    print(f"[INFO] Loading spaCy model '{SPACY_MODEL}'...")
    nlp = spacy.load(SPACY_MODEL)

    # Load stop words from your file
    STOP_WORDS = load_stop_words(STOP_WORDS_FILE)

    conn = get_connection(DB_PATH)
    id2term, label2id, surface2canonical = load_canonical_maps(conn)

    # 1) extract raw canonical–canonical triples
    induce_openie_edges(conn, nlp, id2term, label2id, surface2canonical)

    # 2) apply fully automatic filters into the _clean table
    apply_automatic_filters(conn)

    conn.close()
    print("[INFO] OpenIE-style non-taxonomic extraction + automatic filtering completed.")


if __name__ == "__main__":
    main()
