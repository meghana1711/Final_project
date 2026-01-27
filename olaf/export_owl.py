import argparse
import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional


def connect(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def iri(s: str) -> str:
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty IRI encountered.")
    if s.startswith("<") and s.endswith(">"):
        return s
    return f"<{s}>"


def ttl_escape_label(s: str) -> str:
    s = (s or "").replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{s}\""


def write_prefixes(f):
    f.write("@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
    f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
    f.write("@prefix owl:  <http://www.w3.org/2002/07/owl#> .\n")
    f.write("@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n")
    f.write("@prefix ex:   <http://example.org/hpc#> .\n\n")


def main():
    ap = argparse.ArgumentParser(description="Export OWL (Turtle) from cleaned taxonomy tables in SQLite.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--classes_table", default="label_to_class_iri")
    ap.add_argument("--edges_table", default="taxonomy_is_a_clean_iri")
    ap.add_argument("--out", default="ontology_clean.owl.ttl")
    ap.add_argument("--ontology_iri", default="http://example.org/hpc/ontology/clean")
    ap.add_argument("--include_edge_metadata", action="store_true",
                    help="Also emit a simple reified edge node with method/confidence/evidence (GraphDB-friendly).")

    args = ap.parse_args()

    con = connect(args.db)
    cur = con.cursor()

    if not table_exists(cur, args.classes_table):
        raise RuntimeError(f"Missing table: {args.classes_table}")
    if not table_exists(cur, args.edges_table):
        raise RuntimeError(f"Missing table: {args.edges_table}")

    # Load classes: label, class_name, iri
    cur.execute(f"SELECT label, class_name, iri FROM {args.classes_table}")
    classes = [(r["label"], r["class_name"], r["iri"]) for r in cur.fetchall()]

    # Load edges: child_iri, parent_iri, labels + metadata
    cur.execute(f"""
      SELECT child_iri, parent_iri, child_label, parent_label, method, confidence, evidence
      FROM {args.edges_table}
    """)
    edges = [(
        r["child_iri"], r["parent_iri"], r["child_label"], r["parent_label"],
        r["method"], r["confidence"], r["evidence"]
    ) for r in cur.fetchall()]

    con.close()

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(args.out, "w", encoding="utf-8") as f:
        write_prefixes(f)

        # Ontology header
        f.write(f"{iri(args.ontology_iri)} a owl:Ontology ;\n")
        f.write(f"  rdfs:label {ttl_escape_label('HPC Scheduler Ontology (Clean Taxonomy)')} ;\n")
        f.write(f"  ex:generatedAt {ttl_escape_label(now)}^^xsd:dateTime .\n\n")

        # Declare classes + rdfs:label
        # (GraphDB loads nicer when classes are explicitly typed)
        f.write("# -------------------------\n")
        f.write("# Classes\n")
        f.write("# -------------------------\n")
        for label, class_name, class_iri in classes:
            f.write(f"{iri(class_iri)} a owl:Class ; rdfs:label {ttl_escape_label(label)} .\n")
        f.write("\n")

        # SubClassOf edges
        f.write("# -------------------------\n")
        f.write("# Taxonomy (SubClassOf)\n")
        f.write("# -------------------------\n")
        edge_id = 0
        for child_iri, parent_iri, child_label, parent_label, method, confidence, evidence in edges:
            f.write(f"{iri(child_iri)} rdfs:subClassOf {iri(parent_iri)} .\n")

            # Optional: simple reification for metadata (no OWL axiom annotations needed)
            if args.include_edge_metadata:
                edge_id += 1
                bn = f"_:edge{edge_id}"
                f.write(f"{bn} a ex:ExtractedTaxonomyEdge ;\n")
                f.write(f"  ex:child {iri(child_iri)} ;\n")
                f.write(f"  ex:parent {iri(parent_iri)} ;\n")
                if method:
                    f.write(f"  ex:method {ttl_escape_label(str(method))} ;\n")
                if confidence is not None:
                    try:
                        c = float(confidence)
                        f.write(f"  ex:confidence \"{c}\"^^xsd:double ;\n")
                    except Exception:
                        pass
                if evidence:
                    f.write(f"  ex:evidence {ttl_escape_label(str(evidence))} ;\n")
                f.write(f"  ex:childLabel {ttl_escape_label(child_label)} ;\n")
                f.write(f"  ex:parentLabel {ttl_escape_label(parent_label)} .\n")

        f.write("\n")

        if args.include_edge_metadata:
            f.write("# -------------------------\n")
            f.write("# Metadata properties (lightweight schema)\n")
            f.write("# -------------------------\n")
            f.write("ex:ExtractedTaxonomyEdge a owl:Class .\n")
            f.write("ex:child a owl:ObjectProperty .\n")
            f.write("ex:parent a owl:ObjectProperty .\n")
            f.write("ex:method a owl:DatatypeProperty .\n")
            f.write("ex:confidence a owl:DatatypeProperty .\n")
            f.write("ex:evidence a owl:DatatypeProperty .\n")
            f.write("ex:childLabel a owl:DatatypeProperty .\n")
            f.write("ex:parentLabel a owl:DatatypeProperty .\n")
            f.write("ex:generatedAt a owl:DatatypeProperty .\n")

    print(f"[OK] Exported {len(classes)} classes and {len(edges)} edges to: {args.out}")


if __name__ == "__main__":
    main()
