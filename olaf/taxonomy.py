import sqlite3
import json
import re
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

# ============================================================
# CONFIG
# ============================================================

DB_PATH = r"onto_db/onto_new.db"
CLEANED_VERSION = 1

# Consider only reasonably short terms as children
MIN_TFIDF_CHILD = 10.0
MAX_CHILD_TOKENS = 3

# Subsumption coverage threshold
COVERAGE_THRESHOLD = 0.7
MIN_CHILD_DOCS = 2   # don't trust coverage for very rare terms

# Regex for definitional pattern on lemma string: "Y be a X"
PATTERN_IS_A = re.compile(
    r"\b(?P<hypo>[a-z0-9_]+(?:\s+[a-z0-9_]+){0,2})\s+be\s+a[n]?\s+"
    r"(?P<hyper>[a-z0-9_]+(?:\s+[a-z0-9_]+){0,2})\b"
)

# ---- LLM CONFIG ----
LLM_ENABLED = False           # flip to True when you have a client
LLM_MAX_CALLS = 200           # safety cap on how many edges we ask the LLM about
LLM_MIN_CONFIDENCE = 0.7      # minimal confidence to trust LLM "yes"


# ============================================================
# TABLE INIT (DROP + RECREATE DERIVED TABLES)
# ============================================================

def init_taxonomy_tables(conn: sqlite3.Connection) -> None:
    """
    Drop and recreate term_is_a and concept_taxonomy.

    These tables are fully derived, so it's safe to recreate them
    each time you run the taxonomy phase.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # Drop old versions if they exist (to avoid schema mismatch issues)
    cur.execute("DROP TABLE IF EXISTS term_is_a;")
    cur.execute("DROP TABLE IF EXISTS concept_taxonomy;")

    # term-level is-a (denormalised with names)
    cur.execute("""
        CREATE TABLE term_is_a (
            child_term_id     INTEGER NOT NULL,
            child_term_text   TEXT    NOT NULL,
            parent_term_id    INTEGER NOT NULL,
            parent_term_text  TEXT    NOT NULL,
            source_flags      TEXT    NOT NULL,  -- 'head,pattern,subsumption,llm'
            evidence_count    INTEGER NOT NULL,  -- number of distinct signals
            llm_confidence    REAL,              -- NULL if no LLM / not used
            PRIMARY KEY (child_term_id, parent_term_id),
            FOREIGN KEY (child_term_id)  REFERENCES term_candidates(term_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_term_id) REFERENCES term_candidates(term_id) ON DELETE CASCADE
        );
    """)

    # canonical-level taxonomy (from term_enrichment)
    cur.execute("""
        CREATE TABLE concept_taxonomy (
            child_canonical_id     INTEGER NOT NULL,
            child_canonical_term   TEXT    NOT NULL,
            parent_canonical_id    INTEGER NOT NULL,
            parent_canonical_term  TEXT    NOT NULL,
            source_flags           TEXT    NOT NULL,
            evidence_count         INTEGER NOT NULL,
            PRIMARY KEY (child_canonical_id, parent_canonical_id),
            FOREIGN KEY (child_canonical_id) REFERENCES term_enrichment(canonical_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_canonical_id) REFERENCES term_enrichment(canonical_id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    print("Dropped & recreated term_is_a and concept_taxonomy.")


# ============================================================
# LOAD TERM METADATA
# ============================================================

def load_term_metadata(conn: sqlite3.Connection):
    """
    Load basic term info (text, lemmas, tf_idf, etc.)

    Returns:
      term_meta[term_id] = {
          'text': str,
          'lemma': str,
          'tokens': List[str],
          'length': int,
          'tf_idf': float,
          'freq_total': int,
      }
      lemma_to_ids[lemma] = [term_id, ...]
      best_id_for_lemma[lemma] = term_id with highest tf_idf (or freq_total)
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT term_id, term_text, term_lemma, length_tokens, freq_total,
               COALESCE(tf_idf, 0.0)
        FROM term_candidates
        WHERE length_tokens <= ?
    """, (MAX_CHILD_TOKENS,))

    term_meta: Dict[int, Dict] = {}
    lemma_to_ids: Dict[str, List[int]] = defaultdict(list)
    best_id_for_lemma: Dict[str, int] = {}

    for term_id, text, lemma, length_tokens, freq_total, tf_idf in cur.fetchall():
        term_id = int(term_id)
        text = (text or "").strip()
        lemma = (lemma or "").strip().lower()
        length_tokens = int(length_tokens)
        freq_total = int(freq_total or 0)
        tf_idf = float(tf_idf or 0.0)
        toks = lemma.split()

        term_meta[term_id] = {
            "text": text,
            "lemma": lemma,
            "tokens": toks,
            "length": length_tokens,
            "tf_idf": tf_idf,
            "freq_total": freq_total,
        }

        if lemma:
            lemma_to_ids[lemma].append(term_id)
            # choose best representative for this lemma
            if lemma not in best_id_for_lemma:
                best_id_for_lemma[lemma] = term_id
            else:
                prev_id = best_id_for_lemma[lemma]
                prev_meta = term_meta[prev_id]
                # prefer higher tf_idf, tie-break on freq_total
                if (tf_idf > prev_meta["tf_idf"] or
                    (tf_idf == prev_meta["tf_idf"] and freq_total > prev_meta["freq_total"])):
                    best_id_for_lemma[lemma] = term_id

    print(f"Loaded {len(term_meta)} terms (<= {MAX_CHILD_TOKENS} tokens).")
    return term_meta, lemma_to_ids, best_id_for_lemma


# ============================================================
# HEAD–MODIFIER CANDIDATES
# ============================================================

def build_head_modifier_candidates(
    term_meta: Dict[int, Dict],
    best_id_for_lemma: Dict[str, int]
) -> Set[Tuple[int, int]]:
    """
    For multi-word terms with tf_idf >= MIN_TFIDF_CHILD, propose
    child→parent via head–modifier: last token head lemma.
    Parent is the best term for that head lemma.
    """
    candidates: Set[Tuple[int, int]] = set()

    for term_id, info in term_meta.items():
        length = info["length"]
        tf_idf = info["tf_idf"]
        toks = info["tokens"]

        if length < 2:
            continue
        if tf_idf < MIN_TFIDF_CHILD:
            continue

        head = toks[-1]
        parent_id = best_id_for_lemma.get(head)
        if parent_id is None:
            continue
        if parent_id == term_id:
            continue

        candidates.add((term_id, parent_id))

    print(f"Head–modifier proposed {len(candidates)} candidate child→parent pairs.")
    return candidates


# ============================================================
# DOC SETS FOR SUBSUMPTION
# ============================================================

def load_doc_sets(conn: sqlite3.Connection, term_ids: Set[int]) -> Dict[int, Set[str]]:
    """
    Build doc_id sets per term_id from term_occurrences.
    Only for given term_ids to keep it cheap.
    """
    if not term_ids:
        return {}

    cur = conn.cursor()
    doc_sets: Dict[int, Set[str]] = defaultdict(set)

    term_ids_list = sorted(term_ids)
    CHUNK = 500
    for i in range(0, len(term_ids_list), CHUNK):
        chunk = term_ids_list[i:i + CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        cur.execute(f"""
            SELECT term_id, doc_id
            FROM term_occurrences
            WHERE term_id IN ({placeholders})
        """, chunk)
        for t_id, doc_id in cur.fetchall():
            doc_sets[int(t_id)].add(str(doc_id))

    print(f"Loaded doc sets for {len(doc_sets)} terms.")
    return doc_sets


# ============================================================
# DEFINITIONAL PATTERN EVIDENCE ("Y be a X")
# ============================================================

def collect_pattern_evidence(
    conn: sqlite3.Connection,
    lemma_to_ids: Dict[str, List[int]]
) -> Set[Tuple[int, int]]:
    """
    Scan sentence_lemmatized for 'Y be a X' patterns on lemmas.
    Map Y and X to term_ids via lemma_to_ids, produce child→parent pairs.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT lemmas_json
        FROM sentence_lemmatized
        WHERE cleaned_version = ?
    """, (CLEANED_VERSION,))

    pattern_pairs: Set[Tuple[int, int]] = set()
    count_matches = 0

    for (lemmas_json,) in cur.fetchall():
        lemmas = json.loads(lemmas_json)
        lemma_str = " ".join((l or "").lower() for l in lemmas)

        for m in PATTERN_IS_A.finditer(lemma_str):
            hypo = m.group("hypo").strip()
            hyper = m.group("hyper").strip()
            if not hypo or not hyper:
                continue

            hypo_ids = lemma_to_ids.get(hypo, [])
            hyper_ids = lemma_to_ids.get(hyper, [])
            if not hypo_ids or not hyper_ids:
                continue

            count_matches += 1
            for child_id in hypo_ids:
                for parent_id in hyper_ids:
                    if child_id == parent_id:
                        continue
                    pattern_pairs.add((child_id, parent_id))

    print(f"Found {count_matches} 'Y be a X' pattern matches; "
          f"mapped to {len(pattern_pairs)} distinct child→parent pairs.")
    return pattern_pairs


# ============================================================
# LLM JUDGE (HOOK)
# ============================================================

def call_llm_is_a(
    child_text: str,
    parent_text: str,
    child_lemma: str,
    parent_lemma: str
) -> Optional[Tuple[bool, float]]:
    """
    LLM hook: decide if 'child is-a parent' in HPC scheduling domain.

    RETURN:
      (is_is_a: bool, confidence: float 0..1)
      or None if LLM not configured / error.

    TODO: implement this with your actual client, e.g.:

        import openai, json

        def call_llm_is_a(...):
            prompt = f\"\"\"You are an expert in HPC batch schedulers (Slurm, IBM LSF).

            Decide if the following is a valid is-a (subclass) relation in this domain:

            Child: "{child_text}" (lemma: {child_lemma})
            Parent: "{parent_text}" (lemma: {parent_lemma})

            Answer strictly as JSON:
            {{"decision": "yes" or "no", "confidence": float between 0 and 1}}
            \"\"\"
            ...
    """
    # Placeholder: no LLM integration by default
    return None


# ============================================================
# BUILD term_is_a
# ============================================================

def build_term_is_a(conn: sqlite3.Connection) -> None:
    """
    Main routine:
      1) head–modifier candidates
      2) pattern-based candidates
      3) subsumption coverage
      4) optional LLM validation
      5) decide which child→parent pairs become term_is_a edges
    """
    # Drop + recreate derived tables
    init_taxonomy_tables(conn)

    term_meta, lemma_to_ids, best_id_for_lemma = load_term_metadata(conn)

    # --- 1) head–modifier candidates ---
    head_pairs = build_head_modifier_candidates(term_meta, best_id_for_lemma)

    # --- 2) pattern-based candidates ('Y be a X') ---
    pattern_pairs = collect_pattern_evidence(conn, lemma_to_ids)

    # All candidate pairs that need doc sets for coverage
    all_candidate_pairs: Set[Tuple[int, int]] = set()
    all_candidate_pairs.update(head_pairs)
    all_candidate_pairs.update(pattern_pairs)

    term_ids_for_docs: Set[int] = set()
    for child_id, parent_id in all_candidate_pairs:
        term_ids_for_docs.add(child_id)
        term_ids_for_docs.add(parent_id)

    doc_sets = load_doc_sets(conn, term_ids_for_docs)

    # Evidence map: (child, parent) -> set(sources)
    evidence: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

    for pair in head_pairs:
        evidence[pair].add("head")

    for pair in pattern_pairs:
        evidence[pair].add("pattern")

    # --- 3) subsumption coverage for all candidate pairs ---

    for (child_id, parent_id), srcs in evidence.items():
        docs_child = doc_sets.get(child_id, set())
        docs_parent = doc_sets.get(parent_id, set())
        if len(docs_child) < MIN_CHILD_DOCS:
            continue

        inter = docs_child & docs_parent
        cov = len(inter) / len(docs_child) if docs_child else 0.0

        if cov >= COVERAGE_THRESHOLD:
            srcs.add("subsumption")

    # --- 4) Optional LLM validation on borderline edges ---

    llm_confidence_map: Dict[Tuple[int, int], float] = {}

    if LLM_ENABLED:
        print("Running LLM validation for borderline is-a candidates...")
        llm_calls = 0
        for (child_id, parent_id), srcs in list(evidence.items()):
            if llm_calls >= LLM_MAX_CALLS:
                break

            # Already strong edges don't need LLM
            if "pattern" in srcs or ("head" in srcs and "subsumption" in srcs):
                continue

            child_info = term_meta.get(child_id)
            parent_info = term_meta.get(parent_id)
            if not child_info or not parent_info:
                continue

            result = call_llm_is_a(
                child_text=child_info["text"],
                parent_text=parent_info["text"],
                child_lemma=child_info["lemma"],
                parent_lemma=parent_info["lemma"],
            )
            if result is None:
                print("LLM call returned None; stopping LLM validation early.")
                break

            is_is_a, conf = result
            llm_calls += 1

            if is_is_a and conf >= LLM_MIN_CONFIDENCE:
                srcs.add("llm")
                llm_confidence_map[(child_id, parent_id)] = conf

        print(f"LLM validation finished, total calls: {llm_calls}")

    # --- 5) Decide final edges and write to term_is_a ---

    cur = conn.cursor()
    inserted = 0

    for (child_id, parent_id), srcs in evidence.items():
        # Acceptance rule:
        #  - if we saw a definitional pattern, accept
        #  - OR if we have both head & subsumption
        #  - OR if LLM gave us a strong 'yes'
        accept = False
        if "pattern" in srcs:
            accept = True
        elif "head" in srcs and "subsumption" in srcs:
            accept = True
        elif "llm" in srcs:
            accept = True

        if not accept:
            continue

        child_info = term_meta.get(child_id)
        parent_info = term_meta.get(parent_id)
        if not child_info or not parent_info:
            continue

        flags = ",".join(sorted(srcs))
        evidence_count = len(srcs)
        llm_conf = llm_confidence_map.get((child_id, parent_id))

        cur.execute("""
            INSERT OR REPLACE INTO term_is_a
                (child_term_id, child_term_text,
                 parent_term_id, parent_term_text,
                 source_flags, evidence_count, llm_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            child_id,
            child_info["text"],
            parent_id,
            parent_info["text"],
            flags,
            evidence_count,
            llm_conf,
        ))
        inserted += 1

    conn.commit()
    print(f"Inserted {inserted} term-level is-a edges into term_is_a.")


# ============================================================
# BUILD concept_taxonomy (from term_is_a + term_enrichment)
# ============================================================

def build_concept_taxonomy(conn: sqlite3.Connection) -> None:
    """
    Aggregate term-level is-a edges into canonical concept-level edges
    using term_enrichment.member_term_ids_json.
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='term_enrichment';
        """)
        if not cur.fetchone():
            print("term_enrichment table not found; skipping concept_taxonomy.")
            return
    except sqlite3.Error:
        print("Error checking term_enrichment; skipping concept_taxonomy.")
        return

    # Build term_id -> canonical_id and canonical name mapping
    cur.execute("""
        SELECT canonical_id, canonical_term, member_term_ids_json
        FROM term_enrichment
    """)
    term_to_canon: Dict[int, int] = {}
    canon_names: Dict[int, str] = {}
    for canonical_id, canonical_term, members_json in cur.fetchall():
        canonical_id = int(canonical_id)
        canonical_term = (canonical_term or "").strip()
        canon_names[canonical_id] = canonical_term

        if not members_json:
            continue
        try:
            members = json.loads(members_json)
        except Exception:
            continue
        for mid in members:
            term_to_canon[int(mid)] = canonical_id

    if not term_to_canon:
        print("No term→canonical mappings found; skipping concept_taxonomy.")
        return

    # Load term_is_a edges
    cur.execute("""
        SELECT child_term_id, parent_term_id, source_flags, evidence_count
        FROM term_is_a
    """)
    rows = cur.fetchall()
    if not rows:
        print("No term_is_a edges to aggregate; run build_term_is_a first.")
        return

    # Aggregate per (child_canonical_id, parent_canonical_id)
    concept_edges: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

    for child_tid, parent_tid, flags, _ in rows:
        child_tid = int(child_tid)
        parent_tid = int(parent_tid)

        child_cid = term_to_canon.get(child_tid)
        parent_cid = term_to_canon.get(parent_tid)
        if child_cid is None or parent_cid is None:
            continue
        if child_cid == parent_cid:
            continue  # within same canonical cluster

        for f in (flags or "").split(","):
            f = f.strip()
            if f:
                concept_edges[(child_cid, parent_cid)].add(f)

    # Write to concept_taxonomy
    cur.execute("DELETE FROM concept_taxonomy;")
    inserted = 0
    for (child_cid, parent_cid), srcs in concept_edges.items():
        flags = ",".join(sorted(srcs))
        evidence_count = len(srcs)
        child_term = canon_names.get(child_cid, f"canon_{child_cid}")
        parent_term = canon_names.get(parent_cid, f"canon_{parent_cid}")

        cur.execute("""
            INSERT OR REPLACE INTO concept_taxonomy
                (child_canonical_id, child_canonical_term,
                 parent_canonical_id, parent_canonical_term,
                 source_flags, evidence_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            child_cid,
            child_term,
            parent_cid,
            parent_term,
            flags,
            evidence_count,
        ))
        inserted += 1

    conn.commit()
    print(f"Inserted {inserted} concept-level is-a edges into concept_taxonomy.")


# ============================================================
# MAIN
# ============================================================

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("=== Taxonomy phase: building term-level is-a edges ===")
        build_term_is_a(conn)
        print("=== Aggregating to canonical concepts (concept_taxonomy) ===")
        build_concept_taxonomy(conn)
    finally:
        conn.close()
        print("Done taxonomy is-a building.")

if __name__ == "__main__":
    main()
