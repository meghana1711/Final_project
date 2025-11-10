# make_chunks.py
import json, uuid

INPUT = "C:/Users/20236193/Final_project/old_pre_processing/sentences_improved.json"      # your sentence-level file
OUTPUT = "olaf_chunks.json"            # FRED input

MAX_SENTS_PER_CHUNK = 4
MAX_CHARS_PER_CHUNK = 900  # keep under ~1k char to be safe

with open(INPUT, "r", encoding="utf-8") as f:
    sents = json.load(f)

chunks = []
curr_doc = None
bucket = []
char_count = 0

def flush(doc_id, bucket, chunks):
    if not bucket: return
    text = " ".join(s["sentence"].strip() for s in bucket if s["sentence"].strip())
    chunk_id = f"{doc_id}_chunk_{uuid.uuid4().hex[:8]}"
    chunks.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": text})

for s in sents:
    doc_id = s["doc_id"]
    if curr_doc is None: curr_doc = doc_id

    # new doc → flush
    if doc_id != curr_doc:
        flush(curr_doc, bucket, chunks)
        curr_doc, bucket, char_count = doc_id, [], 0

    sent = s["sentence"].strip()
    if not sent: 
        continue

    # roll chunk if too big
    if (len(bucket) >= MAX_SENTS_PER_CHUNK) or (char_count + len(sent) > MAX_CHARS_PER_CHUNK):
        flush(curr_doc, bucket, chunks)
        bucket, char_count = [], 0

    bucket.append(s)
    char_count += len(sent) + 1

# flush last
flush(curr_doc, bucket, chunks)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(chunks)} chunks to {OUTPUT}")
