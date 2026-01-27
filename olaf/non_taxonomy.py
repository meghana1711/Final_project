from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Set

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
DEFAULT_TERM_ENRICHMENT_EXT_TABLE = "term_enrichment_exten"
DEFAULT_TAXONOMY_TABLE = "taxonomy_is_a_clean"

DEFAULT_RAW_EDGES_TABLE = "non_taxonomic_edges"
DEFAULT_CLEAN_EDGES_TABLE = "non_taxonomic_edges_clean"

STOP_WORDS: set[str] = set()

# Minimal universal lists (not domain seeds; needed to prevent taxonomy leakage)
ISA_LIKE_KEYS = {"be", "become", "remain", "represent"}
TAXO_LIKE_KEYS = { "type_of", "kind_of", "include", "contain", "comprise", "consist_of","classify_as", "categorize_as",
}

# -------------------------------------------------------------------
# DB HELPERS
# -------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return r is not None


def table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    cols: Set[str] = set()
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        for r in rows:
            cols.add(str(r["name"]))
    except sqlite3.OperationalError:
        pass
    return cols

# -------------------------------------------------------------------
# TABLES
# -------------------------------------------------------------------

def init_non_taxonomic_edges_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id                TEXT,
            sentence_id           TEXT,
            sentence_text         TEXT,

            subj_text             TEXT NOT NULL,
            subj_canonical_id     INTEGER NOT NULL,
            subj_canonical_term   TEXT NOT NULL,

            rel_text_raw          TEXT NOT NULL,
            rel_key               TEXT NOT NULL,

            obj_text              TEXT NOT NULL,
            obj_canonical_id      INTEGER NOT NULL,
            obj_canonical_term    TEXT NOT NULL,

            method                TEXT NOT NULL,
            created_at            TEXT NOT NULL,

            UNIQUE(subj_canonical_id, rel_key, obj_canonical_id, sentence_id, method)
        )
        """
    )
    conn.commit()
    print(f"[INFO] Ensured {table_name} exists (no drop).")


def init_non_taxonomic_edges_clean_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id                TEXT,
            sentence_id           TEXT,
            sentence_text         TEXT,

            subj_text             TEXT NOT NULL,
            subj_canonical_id     INTEGER NOT NULL,
            subj_canonical_term   TEXT NOT NULL,

            rel_text_raw          TEXT NOT NULL,
            rel_key               TEXT NOT NULL,

            obj_text              TEXT NOT NULL,
            obj_canonical_id      INTEGER NOT NULL,
            obj_canonical_term    TEXT NOT NULL,

            method                TEXT NOT NULL,
            created_at            TEXT NOT NULL,

            UNIQUE(subj_canonical_id, rel_key, obj_canonical_id, sentence_id, method)
        )
        """
    )
    conn.commit()
    print(f"[INFO] Ensured {table_name} exists (no drop).")


# -------------------------------------------------------------------
# STOP WORDS
# -------------------------------------------------------------------

def load_stop_words(path: str) -> set[str]:
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
# CANONICAL TERM MAPPING (auto-detect ext JSON surfaces)
# -------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_json_load(x) -> Optional[object]:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _add_surface(label2id: Dict[str, int], cid: int, surface: str) -> None:
    surface = (surface or "").strip()
    if not surface:
        return
    label2id[normalize_text(surface)] = cid


def _extract_surfaces_from_any_value(v) -> List[str]:
    out: List[str] = []
    parsed = _safe_json_load(v)
    if isinstance(parsed, list):
        out.extend([x.strip() for x in parsed if isinstance(x, str) and x.strip()])
    elif isinstance(parsed, dict):
        for _, vv in parsed.items():
            if isinstance(vv, str) and vv.strip():
                out.append(vv.strip())
            elif isinstance(vv, list):
                out.extend([x.strip() for x in vv if isinstance(x, str) and x.strip()])
    elif isinstance(v, str):
        s = v.strip()
        if ";" in s:
            out.extend([x.strip() for x in s.split(";") if x.strip()])
        elif "," in s:
            out.extend([x.strip() for x in s.split(",") if x.strip()])
    return out


def load_canonical_maps(
    conn: sqlite3.Connection,
    term_enrichment_table: str,
    term_candidates_table: str,
    term_enrichment_ext_table: Optional[str] = None,
):
    """
    Returns:
      id2term: canonical_id -> canonical_term (ext wins)
      label2id: normalized surface -> canonical_id (canonical + ext surfaces + base surfaces)
      surface2canonical: normalized term_candidates.term_text -> canonical_id via member_term_ids_json
    """
    if not table_exists(conn, term_enrichment_table):
        raise RuntimeError(f"term_enrichment_table '{term_enrichment_table}' not found.")

    base_cols = table_columns(conn, term_enrichment_table)
    ext_exists = bool(term_enrichment_ext_table and table_exists(conn, term_enrichment_ext_table))
    ext_cols = table_columns(conn, term_enrichment_ext_table) if ext_exists else set()

    id2term: Dict[int, str] = {}
    label2id: Dict[str, int] = {}
    termid2canonical: Dict[int, int] = {}

    def load_table(table: str, cols: Set[str], prefer: bool) -> None:
        if "canonical_id" not in cols or "canonical_term" not in cols:
            return

        select_cols = ["canonical_id", "canonical_term"]
        if "member_term_ids_json" in cols:
            select_cols.append("member_term_ids_json")

        # auto include surface-ish columns (still automated)
        for c in cols:
            lc = c.lower()
            if c in select_cols:
                continue
            if any(k in lc for k in ("syn", "alias", "acronym", "surface", "variant", "label")):
                select_cols.append(c)

        rows = conn.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM {table}
            WHERE canonical_term IS NOT NULL AND TRIM(canonical_term) != ''
            """
        ).fetchall()

        for r in rows:
            cid = int(r["canonical_id"])
            canon = (r["canonical_term"] or "").strip()
            if not canon:
                continue

            if prefer or cid not in id2term:
                id2term[cid] = canon

            _add_surface(label2id, cid, canon)

            if "member_term_ids_json" in r.keys():
                obj = _safe_json_load(r["member_term_ids_json"])
                if isinstance(obj, list):
                    for tid in obj:
                        try:
                            termid2canonical[int(tid)] = cid
                        except Exception:
                            pass

            for c in r.keys():
                if c in ("canonical_id", "canonical_term", "member_term_ids_json"):
                    continue
                for s in _extract_surfaces_from_any_value(r[c]):
                    _add_surface(label2id, cid, s)

    # ext first (wins)
    if ext_exists:
        load_table(term_enrichment_ext_table, ext_cols, prefer=True)

    # base second
    load_table(term_enrichment_table, base_cols, prefer=False)

    # term_candidates mapping
    surface2canonical: Dict[str, int] = {}
    if table_exists(conn, term_candidates_table):
        tcols = table_columns(conn, term_candidates_table)
        if "term_id" in tcols and "term_text" in tcols:
            trows = conn.execute(f"SELECT term_id, term_text FROM {term_candidates_table}").fetchall()
            for tr in trows:
                try:
                    tid = int(tr["term_id"])
                except Exception:
                    continue
                text = (tr["term_text"] or "").strip()
                if not text:
                    continue
                cid = termid2canonical.get(tid)
                if cid is None:
                    continue
                surface2canonical[normalize_text(text)] = cid
        else:
            print(f"[WARN] {term_candidates_table} missing (term_id, term_text). surface2canonical limited.")
    else:
        print(f"[WARN] term_candidates_table '{term_candidates_table}' not found. surface2canonical empty.")

    print(
        f"[INFO] Canonical maps: id2term={len(id2term)}, "
        f"label2id(surfaces)={len(label2id)}, surface2canonical={len(surface2canonical)}"
    )
    return id2term, label2id, surface2canonical


def canonicalize_span(span_text: str, label2id: Dict[str, int], surface2canonical: Dict[str, int]) -> Optional[int]:
    """
    Automated-only (no head fallback):
      1) label2id
      2) surface2canonical
      3) strip determiners
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

    norm2 = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", norm)
    if norm2 != norm:
        cid = label2id.get(norm2)
        if cid is not None:
            return cid
        cid = surface2canonical.get(norm2)
        if cid is not None:
            return cid

    return None

# -------------------------------------------------------------------
# TAXONOMY
# -------------------------------------------------------------------

def load_taxonomy(conn: sqlite3.Connection, taxonomy_table: str) -> Tuple[Set[Tuple[int, int]], Dict[int, Set[int]]]:
    edges: Set[Tuple[int, int]] = set()
    parents: Dict[int, Set[int]] = {}

    if not taxonomy_table or not table_exists(conn, taxonomy_table):
        print(f"[WARN] taxonomy table '{taxonomy_table}' not found.")
        return edges, parents

    cols = table_columns(conn, taxonomy_table)
    candidates = [
        ("child_canonical_id", "parent_canonical_id"),
        ("child_id", "parent_id"),
        ("child", "parent"),
        ("child_term_id", "parent_term_id"),
    ]
    pair = None
    for a, b in candidates:
        if a in cols and b in cols:
            pair = (a, b)
            break
    if pair is None:
        print(f"[WARN] taxonomy table '{taxonomy_table}' has no recognizable (child,parent) id columns.")
        return edges, parents

    a, b = pair
    rows = conn.execute(f"SELECT {a} AS child, {b} AS parent FROM {taxonomy_table}").fetchall()
    for r in rows:
        try:
            c = int(r["child"])
            p = int(r["parent"])
        except Exception:
            continue
        if c == p:
            continue
        edges.add((c, p))
        parents.setdefault(c, set()).add(p)

    print(f"[INFO] Loaded taxonomy edges: {len(edges)} from {taxonomy_table}")
    return edges, parents


def build_ancestor_map(parents: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    ancestor_map: Dict[int, Set[int]] = {}

    def dfs(x: int, seen: Set[int]) -> Set[int]:
        if x in ancestor_map:
            return ancestor_map[x]
        acc: Set[int] = set()
        for p in parents.get(x, set()):
            if p in seen:
                continue
            seen.add(p)
            acc.add(p)
            acc |= dfs(p, seen)
        ancestor_map[x] = acc
        return acc

    for c in parents.keys():
        dfs(c, set())
    return ancestor_map


def is_taxonomy_like_triple(
    subj_id: int,
    rel_key: str,
    obj_id: int,
    taxo_edges: Set[Tuple[int, int]],
    ancestor_map: Dict[int, Set[int]],
) -> bool:
    """
    Only drop as taxonomy-like if taxonomy supports linkage (direct edge or ancestor).
    No unconditional 'type_of'/'kind_of' dropping here.
    """
    if subj_id == obj_id:
        return True

    if (subj_id, obj_id) in taxo_edges:
        return True

    if subj_id in ancestor_map and obj_id in ancestor_map[subj_id]:
        if rel_key in ISA_LIKE_KEYS or rel_key in TAXO_LIKE_KEYS:
            return True

    if obj_id in ancestor_map and subj_id in ancestor_map[obj_id]:
        if rel_key in ISA_LIKE_KEYS:
            return True

    return False


# -------------------------------------------------------------------
# TEXT HELPERS
# -------------------------------------------------------------------

def is_generic_span(span_text: str) -> bool:
    if not span_text:
        return True
    tokens = re.findall(r"[A-Za-z0-9_]+", span_text.lower())
    if not tokens:
        return True
    return all(tok in STOP_WORDS for tok in tokens)


def is_valid_relation_text(rel_text: str, max_len: int) -> bool:
    """
    Relaxed for HPC:
      - allow letters, digits, spaces, underscores, hyphens, slashes
      - still reject very long and empty
    """
    if not rel_text:
        return False
    rel = rel_text.strip()
    if not rel:
        return False
    if len(rel) > max_len:
        return False
    if not re.search(r"[A-Za-z]", rel):
        return False
    # allow: A-Za-z0-9 _ - /
    if re.search(r"[^A-Za-z0-9_\-\s/]", rel):
        return False
    return True


def normalize_rel_key(base_lemma: str, prep: Optional[str] = None) -> str:
    base = normalize_text(base_lemma).replace(" ", "_")
    base = re.sub(r"[^a-z0-9_]", "", base)
    if not base:
        return ""
    if prep:
        p = normalize_text(prep).replace(" ", "_")
        p = re.sub(r"[^a-z0-9_]", "", p)
        if p:
            return f"{base}_{p}"
    return base


# -------------------------------------------------------------------
# spaCy OpenIE-style extractor
# -------------------------------------------------------------------

def get_noun_span(token) -> str:
    doc = token.doc
    for nc in doc.noun_chunks:
        if token.i >= nc.start and token.i < nc.end:
            return nc.text.strip()
    subtree_tokens = [t for t in token.subtree if not t.is_space]
    if not subtree_tokens:
        return token.text
    return doc[subtree_tokens[0].i: subtree_tokens[-1].i + 1].text.strip()


def extract_triples_from_sentence(doc, max_rel_len: int) -> List[Tuple[str, str, str, str]]:
    """
    Returns (subj_text, rel_text_raw, rel_key, obj_text)
    """
    triples: List[Tuple[str, str, str, str]] = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        verb = token
        subjects = [w for w in verb.children if w.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            continue

        direct_objs = [w for w in verb.children if w.dep_ in ("dobj", "attr", "oprd")]

        prep_objs = []
        for p in [w for w in verb.children if w.dep_ == "prep"]:
            for o in [c for c in p.children if c.dep_ == "pobj"]:
                prep_objs.append((p, o))

        base_rel_raw = (verb.lemma_ or verb.text).lower().strip()
        if not is_valid_relation_text(base_rel_raw, max_rel_len):
            continue

        for subj in subjects:
            subj_text = get_noun_span(subj)

            for obj in direct_objs:
                obj_text = get_noun_span(obj)
                rel_text_raw = base_rel_raw
                rel_key = normalize_rel_key(base_rel_raw, None)
                if rel_key:
                    triples.append((subj_text, rel_text_raw, rel_key, obj_text))

            for p, pobj in prep_objs:
                obj_text = get_noun_span(pobj)
                rel_text_raw = f"{base_rel_raw} {p.text.lower()}".strip()
                if is_valid_relation_text(rel_text_raw, max_rel_len):
                    rel_key = normalize_rel_key(base_rel_raw, p.text.lower())
                    if rel_key:
                        triples.append((subj_text, rel_text_raw, rel_key, obj_text))

    return triples


# -------------------------------------------------------------------
# SENTENCE SELECTION
# -------------------------------------------------------------------

def select_sentences(conn: sqlite3.Connection, sentence_table: str, cleaned_version: Optional[int]) -> List[sqlite3.Row]:
    if not table_exists(conn, sentence_table):
        raise RuntimeError(f"Sentence table '{sentence_table}' not found.")

    cols = table_columns(conn, sentence_table)
    if cleaned_version is not None and "cleaned_version" in cols:
        rows = conn.execute(
            f"SELECT * FROM {sentence_table} WHERE cleaned_version = ?",
            (int(cleaned_version),),
        ).fetchall()
        print(f"[INFO] Loaded {len(rows)} rows from {sentence_table} (cleaned_version={cleaned_version}).")
        return rows

    rows = conn.execute(f"SELECT * FROM {sentence_table}").fetchall()
    print(f"[INFO] Loaded {len(rows)} rows from {sentence_table} (no cleaned_version filter).")
    return rows


# -------------------------------------------------------------------
# RAW EXTRACTION
# -------------------------------------------------------------------

def induce_openie_edges(
    conn: sqlite3.Connection,
    nlp,
    id2term: Dict[int, str],
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int],
    sentence_table: str,
    cleaned_version: Optional[int],
    raw_edges_table: str,
    method: str,
    max_rel_len: int,
    use_taxonomy_filter: bool,
    taxo_edges: Set[Tuple[int, int]],
    ancestor_map: Dict[int, Set[int]],
    debug_k: int,
) -> None:
    sents = select_sentences(conn, sentence_table, cleaned_version)
    init_non_taxonomic_edges_table(conn, raw_edges_table)
    cur = conn.cursor()

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    total_triples = 0
    canonicalized = 0
    inserted = 0
    debug_examples = 0

    for row in sents:
        cols = row.keys()

        doc_id = row["doc_id"] if "doc_id" in cols else (row["document_id"] if "document_id" in cols else None)
        sentence_id = row["sentence_id"] if "sentence_id" in cols else (row["sent_id"] if "sent_id" in cols else None)

        text = ""
        for cand in ("sent_text", "sentence_text", "sentence", "text"):
            if cand in cols:
                text = row[cand] or ""
                break

        sent = (text or "").strip()
        if not sent:
            continue

        doc = nlp(sent)
        triples = extract_triples_from_sentence(doc, max_rel_len)
        if not triples:
            continue

        total_triples += len(triples)

        for subj_text, rel_text_raw, rel_key, obj_text in triples:
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

            if use_taxonomy_filter and is_taxonomy_like_triple(subj_cid, rel_key, obj_cid, taxo_edges, ancestor_map):
                continue

            canonicalized += 1

            if debug_examples < debug_k:
                print(
                    f"[TRIPLE] subj='{subj_text}' → {subj_term}  "
                    f"rel_raw='{rel_text_raw}' rel_key='{rel_key}'  "
                    f"obj='{obj_text}' → {obj_term}"
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
                 rel_text_raw,
                 rel_key,
                 obj_text,
                 obj_canonical_id,
                 obj_canonical_term,
                 method,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    None if sentence_id is None else str(sentence_id),
                    sent,
                    subj_text,
                    subj_cid,
                    subj_term,
                    rel_text_raw,
                    rel_key,
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
# CLEANING
# -------------------------------------------------------------------

def build_good_relation_set(
    conn: sqlite3.Connection,
    raw_edges_table: str,
    min_rel_total: int,
    min_rel_subj: int,
    min_rel_obj: int,
) -> set[str]:
    rows = conn.execute(
        f"""
        SELECT
            rel_key,
            COUNT(*) AS n_total,
            COUNT(DISTINCT subj_canonical_id) AS n_subj,
            COUNT(DISTINCT obj_canonical_id) AS n_obj
        FROM {raw_edges_table}
        GROUP BY rel_key
        """
    ).fetchall()

    good = set()
    for r in rows:
        if (r["n_total"] >= min_rel_total and
            r["n_subj"]  >= min_rel_subj and
            r["n_obj"]   >= min_rel_obj):
            good.add(r["rel_key"])

    print(f"[INFO] Good relation keys (freq/diversity filter): {len(good)}")
    return good


def apply_automatic_filters(
    conn: sqlite3.Connection,
    raw_edges_table: str,
    clean_edges_table: str,
    min_rel_total: int,
    min_rel_subj: int,
    min_rel_obj: int,
    max_rel_len: int,
    use_taxonomy_filter: bool,
    taxo_edges: Set[Tuple[int, int]],
    ancestor_map: Dict[int, Set[int]],
) -> None:
    init_non_taxonomic_edges_clean_table(conn, clean_edges_table)

    good_relations = build_good_relation_set(conn, raw_edges_table, min_rel_total, min_rel_subj, min_rel_obj)
    rows = conn.execute(f"SELECT * FROM {raw_edges_table}").fetchall()
    cur = conn.cursor()
    inserted = 0

    for r in rows:
        rel_key = r["rel_key"]
        rel_text_raw = r["rel_text_raw"]

        if rel_key not in good_relations:
            continue
        if not is_valid_relation_text(rel_text_raw, max_rel_len):
            continue

        subj_id = int(r["subj_canonical_id"])
        obj_id = int(r["obj_canonical_id"])
        if subj_id == obj_id:
            continue

        if use_taxonomy_filter and is_taxonomy_like_triple(subj_id, rel_key, obj_id, taxo_edges, ancestor_map):
            continue

        if is_generic_span(r["subj_text"]) or is_generic_span(r["obj_text"]):
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
             rel_text_raw,
             rel_key,
             obj_text,
             obj_canonical_id,
             obj_canonical_term,
             method,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["doc_id"],
                r["sentence_id"],
                r["sentence_text"],
                r["subj_text"],
                subj_id,
                r["subj_canonical_term"],
                rel_text_raw,
                rel_key,
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
    p = argparse.ArgumentParser(description="Non-taxonomic relation extraction (OpenIE via spaCy) - cleaned.")

    p.add_argument("--db", required=True, help="Path to SQLite DB.")
    p.add_argument("--spacy_model", default=DEFAULT_SPACY_MODEL, help="spaCy model name.")
    p.add_argument("--stopwords", default=DEFAULT_STOP_WORDS_FILE, help="Path to stopwords file.")

    p.add_argument("--sentence_table", default=DEFAULT_SENTENCE_TABLE, help="Input sentence table.")
    p.add_argument("--cleaned_version", type=int, default=None, help="Filter cleaned_version if column exists.")

    p.add_argument("--term_candidates_table", default=DEFAULT_TERM_CANDIDATES_TABLE, help="term_candidates table.")
    p.add_argument("--term_enrichment_table", default=DEFAULT_TERM_ENRICHMENT_TABLE, help="Base term_enrichment table.")
    p.add_argument("--term_enrichment_ext_table", default=DEFAULT_TERM_ENRICHMENT_EXT_TABLE, help="Extension enrichment table (preferred if exists).")

    p.add_argument("--taxonomy_table", default=DEFAULT_TAXONOMY_TABLE, help="taxonomy is_a table (clean preferred).")
    p.add_argument("--use_taxonomy_filter", action="store_true", help="Drop taxonomy-like triples using taxonomy edges/ancestors.")

    p.add_argument("--raw_edges_table", default=DEFAULT_RAW_EDGES_TABLE, help="Output raw edges table.")
    p.add_argument("--clean_edges_table", default=DEFAULT_CLEAN_EDGES_TABLE, help="Output clean edges table.")
    p.add_argument("--method", default=DEFAULT_METHOD, help="Method name stored in DB.")

    # thresholds as flags (so you can adapt without editing code)
    p.add_argument("--max_rel_len", type=int, default=80, help="Max length of raw relation phrase.")
    p.add_argument("--min_rel_total", type=int, default=3, help="Min total occurrences for rel_key.")
    p.add_argument("--min_rel_subj", type=int, default=2, help="Min distinct subjects for rel_key.")
    p.add_argument("--min_rel_obj", type=int, default=2, help="Min distinct objects for rel_key.")

    p.add_argument("--debug_k", type=int, default=15, help="Print first K kept triples for debugging (0 disables).")

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
            term_enrichment_ext_table=args.term_enrichment_ext_table,
        )

        taxo_edges: Set[Tuple[int, int]] = set()
        parents: Dict[int, Set[int]] = {}
        ancestor_map: Dict[int, Set[int]] = {}

        if args.use_taxonomy_filter:
            taxo_edges, parents = load_taxonomy(conn, args.taxonomy_table)
            ancestor_map = build_ancestor_map(parents) if parents else {}
            print(f"[INFO] Ancestor map built for {len(ancestor_map)} nodes.")
        else:
            print("[INFO] Taxonomy filter OFF.")

        induce_openie_edges(
            conn=conn,
            nlp=nlp,
            id2term=id2term,
            label2id=label2id,
            surface2canonical=surface2canonical,
            sentence_table=args.sentence_table,
            cleaned_version=args.cleaned_version,
            raw_edges_table=args.raw_edges_table,
            method=args.method,
            max_rel_len=args.max_rel_len,
            use_taxonomy_filter=args.use_taxonomy_filter,
            taxo_edges=taxo_edges,
            ancestor_map=ancestor_map,
            debug_k=max(0, int(args.debug_k)),
        )

        apply_automatic_filters(
            conn=conn,
            raw_edges_table=args.raw_edges_table,
            clean_edges_table=args.clean_edges_table,
            min_rel_total=args.min_rel_total,
            min_rel_subj=args.min_rel_subj,
            min_rel_obj=args.min_rel_obj,
            max_rel_len=args.max_rel_len,
            use_taxonomy_filter=args.use_taxonomy_filter,
            taxo_edges=taxo_edges,
            ancestor_map=ancestor_map,
        )

    finally:
        conn.close()

    print("[INFO] Non-taxonomic extraction + cleaning completed.")


if __name__ == "__main__":
    main()
