# fred_driver.py
# End-to-end: chunks -> FRED -> merged ontology (TTL)

import os, json, time, uuid, pathlib
import requests
from rdflib import Graph
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# === CONFIG ===
FRED_URL = "https://wit.istc.cnr.it/stlab-tools/fred"  # REST entry-point (Swagger lists the API) :contentReference[oaicite:1]{index=1}
ACCEPT = "text/turtle"  # ask FRED for Turtle
OUT_DIR = pathlib.Path("fred")
CHUNK_TTL_DIR = OUT_DIR / "chunks"
MERGED_TTL = OUT_DIR / "ontology_fred_only.ttl"

# FRED options (see Swagger; adjust as you like) :contentReference[oaicite:2]{index=2}
FRED_PARAMS = {
    # minimum set
    "wfd_profile": "general",  # disambiguation profile
    "roles": "true",           # include semantic roles
    "alignToFramester": "true",
    "alpha": "true",           # enables extra inferences/patterns
    # you can add others exposed in the Swagger UI, e.g. "textannotation", "wsd", etc.
}

# === I/O helpers ===
def ensure_dirs():
    CHUNK_TTL_DIR.mkdir(parents=True, exist_ok=True)

def load_chunks(json_path: str):
    """
    Expecting your OLAF-ready preprocessing file. Minimal fields we use:
      - chunk_id (preferred) or sent_id as fallback
      - text  (the content to send to FRED)
    If you don't have chunk_id, we synthesize one.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for rec in data:
        cid = rec.get("chunk_id") or rec.get("sent_id") or f"auto_{uuid.uuid4().hex[:8]}"
        txt = rec.get("text") or rec.get("sentence") or rec.get("content")
        if not txt or not txt.strip():
            continue
        items.append({"chunk_id": cid, "text": txt.strip(), "doc_id": rec.get("doc_id")})
    return items

# === FRED call ===
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def call_fred(text: str, timeout=90) -> str:
    """
    POST text to FRED; return Turtle string.
    Retries with exponential backoff on transient errors.
    """
    headers = {"Accept": ACCEPT}
    data = {"text": text}
    data.update(FRED_PARAMS)
    r = requests.post(FRED_URL, data=data, headers=headers, timeout=timeout)
    # 200 OK with TTL body is expected
    if r.status_code != 200 or not r.text.strip():
        raise RuntimeError(f"FRED error HTTP {r.status_code}: {r.text[:200]}")
    return r.text

def save_ttl(chunk_id: str, ttl: str):
    out = CHUNK_TTL_DIR / f"{chunk_id}.ttl"
    out.write_text(ttl, encoding="utf-8")
    return out

def merge_ttl(folder: pathlib.Path, merged_path: pathlib.Path):
    g = Graph()
    n = 0
    for p in folder.glob("*.ttl"):
        try:
            g.parse(p.as_posix(), format="turtle")
            n += 1
        except Exception as e:
            print(f"[WARN] Failed to parse {p.name}: {e}")
    g.serialize(destination=merged_path.as_posix(), format="turtle")
    return n, len(g)

def print_stats(ttl_path: pathlib.Path):
    g = Graph().parse(ttl_path.as_posix(), format="turtle")
    # Simple stats: counts of classes / properties / triples
    q_classes = """
    SELECT (COUNT(DISTINCT ?c) AS ?n) WHERE {
      { ?c a <http://www.w3.org/2002/07/owl#Class> }
      UNION
      { ?c a <http://www.w3.org/2000/01/rdf-schema#Class> }
    }"""
    q_props = """
    SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
      { ?p a <http://www.w3.org/2002/07/owl#ObjectProperty> }
      UNION
      { ?p a <http://www.w3.org/2002/07/owl#DatatypeProperty> }
      UNION
      { ?p a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> }
    }"""
    n_classes = int(next(g.query(q_classes))[0])
    n_props = int(next(g.query(q_props))[0])
    print("\n=== MERGED ONTOLOGY STATS ===")
    print(f"Triples: {len(g):,}")
    print(f"Classes: {n_classes:,}")
    print(f"Properties: {n_props:,}")

def main(input_json: str):
    ensure_dirs()
    items = load_chunks(input_json)
    print(f"Loaded {len(items)} chunks to process.")
    # Call FRED per chunk (be gentle with remote service; small sleep)
    for rec in tqdm(items, desc="FRED"):
        cid = rec["chunk_id"]
        txt = rec["text"]
        try:
            ttl = call_fred(txt)
            save_ttl(cid, ttl)
        except Exception as e:
            # Save an empty marker to avoid infinite retries on re-run
            (CHUNK_TTL_DIR / f"{cid}.error.txt").write_text(str(e), encoding="utf-8")
        time.sleep(0.25)  # courtesy delay; tune if rate-limited

    n_files, n_triples = merge_ttl(CHUNK_TTL_DIR, MERGED_TTL)
    print(f"\nMerged {n_files} TTL files into: {MERGED_TTL}")
    print_stats(MERGED_TTL)
    print("\nDone.")

if __name__ == "__main__":
    # Example: python fred_driver.py ./data/olaf_chunks.json
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fred_driver.py <chunks.json>")
        sys.exit(1)
    main(sys.argv[1])
