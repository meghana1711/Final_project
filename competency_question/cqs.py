# competency_question/competency_question.py
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------
# Regex + constants
# ---------------------------

DOCID_RE = re.compile(r"\bdoc_[0-9a-fA-F]{2,}(?:_chunk_[0-9]{3,})?\b", re.IGNORECASE)
DOCID_PAREN_RE = re.compile(r"\(\s*doc_[0-9a-fA-F]{2,}[^)]*\)", re.IGNORECASE)

CMD_HINT_RE = re.compile(
    r"\b(lsadmin|badmin|bsub|bjobs|bstop|bresume|bkill|lsrun|"
    r"sacctmgr|sacct|sreport|scontrol|sinfo|sbatch|srun|salloc|scancel)\b",
    re.IGNORECASE,
)
DAEMON_HINT_RE = re.compile(r"\b(lim|res|mbatchd|sbatchd|slurmctld|slurmd|slurmdbd)\b", re.IGNORECASE)
FILE_HINT_RE = re.compile(r"(/etc/[A-Za-z0-9._/-]+|\b[A-Za-z0-9_-]+\.(?:conf|lsf)\b)", re.IGNORECASE)
POLICY_HINT_RE = re.compile(r"\b(must|required|only acceptable|recommended|only)\b", re.IGNORECASE)

HPC_KEYWORDS_RE = re.compile(
    r"\b(slurm|slurmctld|slurmd|slurmdbd|sbatch|srun|salloc|sinfo|scontrol|"
    r"lsf|bsub|bjobs|lsadmin|badmin|queue|partition|node|job|daemon|config|qos|gres|tres|fair[- ]share)\b",
    re.IGNORECASE,
)

WH_STARTERS = ("what", "which", "how", "why", "can", "is", "does", "do", "when", "where")
CQ_TYPES = ["procedure", "constraint", "relationship", "verification"]

BANNED_GENERIC = [
    "key features",
    "overview",
    "introduction",
    "explain",
    "describe",
    "tell me about",
    "in general",
    "what is hpc",
    "what is lsf",
    "what is slurm",
]


# ---------------------------
# Text utils
# ---------------------------

def norm(s: str) -> str:
    s = (s or "").strip().strip(' "\'`')
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def normalize_question(q: str) -> str:
    q = re.sub(r"\s+", " ", (q or "").strip())
    if not q:
        return ""
    if not q.endswith("?"):
        q += "?"
    return q

def clean_question(q: str) -> str:
    q = normalize_question(q)
    q = DOCID_PAREN_RE.sub("", q)
    q = DOCID_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\(\s*\)", "", q)
    q = re.sub(r"\s+\?", "?", q)
    return q.strip()

def looks_generic(q: str) -> bool:
    qn = norm(q)
    return any(p in qn for p in BANNED_GENERIC)

def smart_truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 240:
        return text[:max_chars]
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()

def is_questionish(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.endswith("?"):
        return True
    first = norm(s).split(" ", 1)[0]
    return first in WH_STARTERS

def force_question_mark(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s if s.endswith("?") else (s + "?")

def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


# ---------------------------
# DB helpers
# ---------------------------

def get_table_columns(conn: sqlite3.Connection, table: str) -> List[Tuple[str, str, int]]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    cols = cur.fetchall()
    if not cols:
        raise ValueError(f"Table '{table}' not found or has no columns.")
    return [(c[1], (c[2] or ""), int(c[5] or 0)) for c in cols]

def detect_chunk_and_doc_cols(conn: sqlite3.Connection, table: str) -> Tuple[str, str]:
    cols = get_table_columns(conn, table)
    names = [c[0] for c in cols]
    lower = [n.lower() for n in names]

    pk = [c for c in cols if c[2] == 1]
    chunk_id_col = pk[0][0] if pk else names[0]

    doc_id_col = None
    for cand in ["doc_id", "docid", "document_id", "doc", "document"]:
        if cand in lower:
            doc_id_col = names[lower.index(cand)]
            break
    if doc_id_col is None:
        for i, n in enumerate(lower):
            if "doc" in n and "id" in n:
                doc_id_col = names[i]
                break
    if doc_id_col is None:
        raise ValueError(f"Could not detect doc id column in '{table}'. Columns: {names}")

    return chunk_id_col, doc_id_col

def pick_best_text_col(
    conn: sqlite3.Connection,
    table: str,
    chunk_id_col: str,
    doc_id_col: str,
    override: Optional[str] = None,
    sample_n: int = 120,
) -> str:
    """
    Choose text column by CONTENT, not name.
    """
    if override:
        return override

    cols = get_table_columns(conn, table)
    candidates = []
    for name, ctype, _pk in cols:
        nl = name.lower()
        ctype_u = (ctype or "").upper()
        if name in (chunk_id_col, doc_id_col):
            continue
        if "JSON" in ctype_u or "json" in nl:
            continue
        if not any(t in ctype_u for t in ["TEXT", "CHAR", "CLOB"]):
            continue
        candidates.append(name)

    if not candidates:
        raise ValueError(f"No TEXT-like candidate columns found in {table}.")

    cur = conn.cursor()
    best_col = candidates[0]
    best_score = -1e18

    for col in candidates:
        cur.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT {sample_n};")
        vals = [(r[0] or "") for r in cur.fetchall()]
        if not vals:
            continue

        lens = [len(v) for v in vals]
        avg_len = sum(lens) / max(1, len(lens))

        docid_like = sum(1 for v in vals if DOCID_RE.fullmatch(v.strip()) is not None) / max(1, len(vals))
        very_short = sum(1 for v in vals if len(v.strip()) < 60) / max(1, len(vals))
        hpc_hits = sum(1 for v in vals if HPC_KEYWORDS_RE.search(v) is not None) / max(1, len(vals))

        score = (avg_len) + (400.0 * hpc_hits) - (1200.0 * docid_like) - (200.0 * very_short)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col

def ensure_columns(conn: sqlite3.Connection, table: str, required_cols: Dict[str, str]) -> None:
    """
    Safely migrates existing tables by adding missing columns.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    existing = {row[1] for row in cur.fetchall()}

    for col, coltype in required_cols.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype};")
    conn.commit()

def ensure_out_table(conn: sqlite3.Connection, out_table: str) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            seed_chunk_ids_json TEXT NOT NULL,
            cq_text TEXT NOT NULL,
            source TEXT,
            cq_type TEXT,
            flags_json TEXT,
            anchors_json TEXT,
            model_id TEXT,
            raw_completion TEXT
        );
        """
    )
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{out_table}_doc ON {out_table}(doc_id);")
    conn.commit()

    # Auto-migrate old schema
    ensure_columns(
        conn,
        out_table,
        {
            "doc_id": "TEXT",
            "seed_chunk_ids_json": "TEXT",
            "cq_text": "TEXT",
            "source": "TEXT",
            "cq_type": "TEXT",
            "flags_json": "TEXT",
            "anchors_json": "TEXT",
            "model_id": "TEXT",
            "raw_completion": "TEXT",
        },
    )

def processed_docs(conn: sqlite3.Connection, out_table: str) -> Set[str]:
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT DISTINCT doc_id FROM {out_table};")
        return {str(r[0]) for r in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()


# ---------------------------
# FTS5 (within-document retrieval)
# ---------------------------

def ensure_fts(
    conn: sqlite3.Connection,
    base_table: str,
    fts_table: str,
    chunk_id_col: str,
    doc_id_col: str,
    text_col: str,
    rebuild: bool = False,
) -> None:
    cur = conn.cursor()
    if rebuild:
        cur.execute(f"DROP TABLE IF EXISTS {fts_table};")
        conn.commit()

    cur.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table}
        USING fts5(doc_id UNINDEXED, chunk_id UNINDEXED, chunk_text);
        """
    )
    conn.commit()

    cur.execute(f"SELECT COUNT(*) FROM {fts_table};")
    n = cur.fetchone()[0]
    if n > 0 and not rebuild:
        return

    cur.execute(f"DELETE FROM {fts_table};")
    cur.execute(f"SELECT {doc_id_col}, {chunk_id_col}, {text_col} FROM {base_table} WHERE {text_col} IS NOT NULL;")
    rows = cur.fetchall()
    cur.executemany(
        f"INSERT INTO {fts_table}(doc_id, chunk_id, chunk_text) VALUES(?, ?, ?);",
        [(str(r[0]), str(r[1]), (r[2] or "")) for r in rows],
    )
    conn.commit()

def make_fts_query(text: str, max_terms: int = 12) -> str:
    flags = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]{1,40}", text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9_/.-]{2,50}", text)

    seen = set()
    toks = []
    for t in flags + words:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        toks.append(t)
    toks = toks[:max_terms]

    if not toks:
        return "slurm OR lsf OR job OR queue"

    safe = []
    for t in toks:
        t = re.sub(r'["\']', "", t)
        if any(ch in t for ch in [".", "-", "_", "/", ":"]) or t.startswith("--"):
            safe.append(f'"{t}"')
        else:
            safe.append(t)
    return " OR ".join(safe)

def retrieve_within_doc(
    conn: sqlite3.Connection,
    fts_table: str,
    doc_id: str,
    query_text: str,
    top_k: int,
    exclude_chunk_id: Optional[str] = None,
) -> List[Tuple[str, str]]:
    if top_k <= 0:
        return []
    q = make_fts_query(query_text)
    cur = conn.cursor()

    if exclude_chunk_id:
        cur.execute(
            f"""
            SELECT chunk_id, chunk_text
            FROM {fts_table}
            WHERE doc_id = ?
              AND {fts_table} MATCH ?
              AND chunk_id != ?
            ORDER BY bm25({fts_table})
            LIMIT ?;
            """,
            (doc_id, q, exclude_chunk_id, top_k),
        )
    else:
        cur.execute(
            f"""
            SELECT chunk_id, chunk_text
            FROM {fts_table}
            WHERE doc_id = ?
              AND {fts_table} MATCH ?
            ORDER BY bm25({fts_table})
            LIMIT ?;
            """,
            (doc_id, q, top_k),
        )

    return [(str(r[0]), r[1] or "") for r in cur.fetchall()]


# ---------------------------
# Salience / diversity selection
# ---------------------------

def chunk_salience_score(text: str) -> int:
    t = text or ""
    score = 0
    score += 4 * len(CMD_HINT_RE.findall(t))
    score += 3 * len(DAEMON_HINT_RE.findall(t))
    score += 3 * len(FILE_HINT_RE.findall(t))
    score += 2 * len(POLICY_HINT_RE.findall(t))
    if "procedure" in t.lower() or "run the following" in t.lower():
        score += 3
    if re.search(r"^\s*[%$]", t, flags=re.MULTILINE):
        score += 2
    if len(t.strip()) < 180:
        score -= 2
    return score

def token_set(text: str) -> Set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_/.-]{2,40}", (text or "").lower()))

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0

def select_diverse_top_chunks(chunks: List[Tuple[str, str]], top_m: int, sim_threshold: float = 0.55) -> List[Tuple[str, str]]:
    scored = sorted(chunks, key=lambda x: chunk_salience_score(x[1]), reverse=True)
    selected: List[Tuple[str, str]] = []
    selected_sets: List[Set[str]] = []

    for cid, txt in scored:
        if len(selected) >= top_m:
            break
        s = token_set(txt)
        if not selected:
            selected.append((cid, txt))
            selected_sets.append(s)
            continue
        if all(jaccard(s, ss) < sim_threshold for ss in selected_sets):
            selected.append((cid, txt))
            selected_sets.append(s)

    idx = 0
    while len(selected) < top_m and idx < len(scored):
        cand = scored[idx]
        if cand not in selected:
            selected.append(cand)
        idx += 1

    return selected[:top_m]


# ---------------------------
# Anchors + domain
# ---------------------------

def detect_domain(excerpts: List[str]) -> str:
    t = norm("\n".join(excerpts))
    has_slurm = "slurm" in t
    has_lsf = "lsf" in t
    if has_slurm and not has_lsf:
        return "slurm"
    if has_lsf and not has_slurm:
        return "lsf"
    return "mixed"

def extract_anchors(excerpts: List[str], max_terms: int = 10) -> List[str]:
    """
    Never allow doc_...chunk_... anchors. Prefer commands/config/daemons/files.
    """
    text = "\n".join(excerpts)

    candidates = []
    candidates += re.findall(r"/[A-Za-z0-9._/-]{3,80}", text)
    candidates += re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]{1,40}", text)
    candidates += re.findall(r"\b[A-Za-z]{2,}\.[A-Za-z]{2,}\b", text)
    candidates += re.findall(r"\b[A-Za-z][A-Za-z0-9_/.-]{2,60}\b", text)

    stop = {
        "the", "and", "that", "this", "with", "from", "into", "they", "them", "then", "also",
        "have", "has", "will", "must", "should", "none", "value", "default", "name", "names",
        "use", "used", "using", "example", "examples", "see", "more", "info", "information",
        "command", "commands", "option", "options", "file", "files", "data", "database",
        "procedure", "important", "only", "user", "users", "cluster", "clusters", "system",
        "set", "up", "run", "start", "stop", "check", "status", "jobs", "job",
        "therefore", "towards", "when", "e.g", "eg", "i.e", "ie",
    }

    freq: Dict[str, int] = {}
    first: Dict[str, str] = {}

    for c in candidates:
        c = c.strip()
        if DOCID_RE.fullmatch(c):
            continue
        nc = norm(c)
        if not nc or nc in stop or len(nc) < 3:
            continue
        if nc not in first:
            first[nc] = c

        bonus = 0
        if c.startswith("/"):
            bonus += 3
        if c.startswith("--"):
            bonus += 5
        if "." in c:
            bonus += 2
        if CMD_HINT_RE.search(c):
            bonus += 6
        if DAEMON_HINT_RE.search(c):
            bonus += 5
        if FILE_HINT_RE.search(c):
            bonus += 4

        freq[nc] = freq.get(nc, 0) + 1 + bonus

    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    anchors = [first[nc] for nc, _ in ranked[:max_terms]]

    # Backfill with commands/daemons/files if still weak
    if not anchors:
        anchors.extend(sorted(set(m.group(0) for m in CMD_HINT_RE.finditer(text)), key=str.lower)[:max_terms])
    if not anchors:
        anchors.extend(sorted(set(m.group(0) for m in FILE_HINT_RE.finditer(text)), key=str.lower)[:max_terms])
    if not anchors:
        anchors.extend(sorted(set(m.group(0) for m in DAEMON_HINT_RE.finditer(text)), key=str.lower)[:max_terms])

    # Ensure no doc chunk ids sneak in
    anchors = [a for a in anchors if not DOCID_RE.fullmatch(a.strip())]
    return anchors[:max_terms]


# ---------------------------
# LLM wrapper
# ---------------------------

@dataclass
class LLMConfig:
    model_id: str
    max_new_tokens: int = 320
    do_sample: bool = False
    temperature: float = 0.3
    top_p: float = 0.9

class LocalChatLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    @torch.inference_mode()
    def chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        else:
            prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"
            enc = self.tokenizer(prompt, return_tensors="pt", return_dict=True)

        if torch.cuda.is_available():
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            attention_mask=enc.get("attention_mask", None),
            do_sample=self.cfg.do_sample,
        )
        if self.cfg.do_sample:
            gen_kwargs.update(dict(temperature=self.cfg.temperature, top_p=self.cfg.top_p))

        out = self.model.generate(enc["input_ids"], **gen_kwargs)
        gen_ids = out[0, enc["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


SYSTEM_PROMPT = """\
You are an ontology engineer for HPC schedulers (SLURM and IBM LSF).
Write competency questions (CQs): precise questions that define what an ontology must represent and answer.

Hard rules:
- Use ONLY the provided EXCERPTS.
- Do NOT mention document IDs, chunk IDs, or EXCERPT labels.
- Do NOT invent commands, daemons, file paths, config parameters, flags, or values not present in the EXCERPTS.
- Avoid vague questions ("key features", "overview", "explain", "describe").
- Each CQ must include at least ONE exact technical token copied from the EXCERPTS.
- If unsure, write a more general CQ using ONLY tokens present; never invent details.
"""

def build_prompt(excerpts: List[str], anchors: List[str], n_cqs: int, domain: str) -> str:
    if domain == "slurm":
        domain_rule = "This document is about SLURM only. Do NOT mention IBM LSF."
    elif domain == "lsf":
        domain_rule = "This document is about IBM LSF only. Do NOT mention SLURM."
    else:
        domain_rule = "This document may mention SLURM and IBM LSF. Only mention what appears in the excerpts."

    # Force variety: fixed mix pattern repeated
    quota = [
        "How", "How", "How",
        "Why", "Why",
        "Which", "Which",
        "What happens if", "What happens if",
        "When",
    ]
    # If n_cqs > len(quota), repeat
    want = [quota[i % len(quota)] for i in range(n_cqs)]
    mix_line = ", ".join([f"CQ{i+1} starts with '{want[i]}'" for i in range(n_cqs)])

    anchor_str = ", ".join(anchors[:10]) if anchors else "(none)"

    ex_block = []
    for i, ex in enumerate(excerpts):
        label = chr(65 + i)
        ex_block.append(f"EXCERPT {label}:\n{ex.strip()}")
    ex_block = "\n\n".join(ex_block)

    return f"""\
{domain_rule}

TASK:
Generate exactly {n_cqs} competency questions from the EXCERPTS.

STARTER MIX REQUIREMENT:
{mix_line}

ANCHOR TERMS (use at least one anchor token in EACH question):
{anchor_str}

OUTPUT:
- One question per line
- End each line with '?'
- No numbering required

EXCERPTS:
{ex_block}
"""

def parse_questions(raw: str, max_q: int) -> List[str]:
    lines = [l.strip() for l in (raw or "").splitlines() if l.strip()]
    out: List[str] = []

    for l in lines:
        l2 = re.sub(r"^(CQ\s*\d+\s*[:\)\-]\s*)", "", l, flags=re.IGNORECASE)
        l2 = re.sub(r"^(\d+\s*[\.\)\-]\s*)", "", l2)
        l2 = re.sub(r"^[-•]\s*", "", l2).strip()
        if not l2:
            continue
        if is_questionish(l2):
            out.append(clean_question(force_question_mark(l2)))

    if not out:
        qs = re.findall(r"([A-Z][^?\n]{10,240}\?)", raw or "")
        out = [clean_question(q) for q in qs]

    out = [q for q in out if len(q) >= 12 and not DOCID_RE.search(q)]
    out = dedup_keep_order(out)
    return out[:max_q]


# ---------------------------
# Template fallback (guarantee inserts)
# ---------------------------

def template_fallback(excerpts: List[str], anchors: List[str], n: int, domain: str) -> List[str]:
    text = "\n".join(excerpts)

    cmds = sorted(set(m.group(0) for m in CMD_HINT_RE.finditer(text)), key=str.lower)
    daemons = sorted(set(m.group(0) for m in DAEMON_HINT_RE.finditer(text)), key=str.lower)
    files = sorted(set(m.group(0) for m in FILE_HINT_RE.finditer(text)), key=str.lower)

    qs: List[str] = []

    if domain in ("lsf", "mixed"):
        if ("lsadmin" in [c.lower() for c in cmds] or "badmin" in [c.lower() for c in cmds]) and daemons:
            qs.append("Which daemons are controlled by lsadmin versus badmin in the described configuration?")
        if any(f.lower().endswith("cshrc.lsf") for f in files) or any(f.lower().endswith("profile.lsf") for f in files):
            qs.append("How do you set up the LSF execution environment using cshrc.lsf or profile.lsf?")
        if any("/etc/lsf.sudoers" in f.lower() for f in files):
            qs.append("What is the purpose of /etc/lsf.sudoers for allowing administrators to start and stop LSF daemons?")

    if domain in ("slurm", "mixed"):
        if "sinfo" in [c.lower() for c in cmds]:
            qs.append("How can sinfo be used to check node or partition status as described in the excerpts?")
        if "scontrol" in [c.lower() for c in cmds]:
            qs.append("How can scontrol be used to inspect or modify job/node state as described in the excerpts?")
        if any(c.lower() in ("sbatch", "srun", "salloc") for c in cmds):
            qs.append("Which command (sbatch, srun, or salloc) is used for the described job execution mode in the excerpts?")

    for a in anchors[:10]:
        if DOCID_RE.fullmatch(a.strip()):
            continue
        a_l = a.lower()
        if any(a_l == c.lower() for c in cmds):
            qs.append(f"What does the command {a} do according to the excerpts?")
        elif any(a_l == d.lower() for d in daemons):
            qs.append(f"What role does the daemon {a} play according to the excerpts?")
        elif a.startswith("/"):
            qs.append(f"What is the role of the file {a} according to the excerpts?")
        elif "." in a:
            qs.append(f"What is the role of {a} according to the excerpts?")
        if len(qs) >= n:
            break

    qs = [clean_question(q) for q in qs if q]
    qs = dedup_keep_order(qs)
    return qs[:n]


# ---------------------------
# Insert
# ---------------------------

def insert_rows(
    conn: sqlite3.Connection,
    out_table: str,
    doc_id: str,
    seed_chunk_ids: List[str],
    anchors: List[str],
    model_id: str,
    raw_completion: str,
    questions: List[Tuple[str, str]],
) -> int:
    cur = conn.cursor()
    inserted = 0
    for i, (q, source) in enumerate(questions):
        cq_type = CQ_TYPES[i % len(CQ_TYPES)]
        flags = {
            "generic": looks_generic(q),
            "has_anchor": any(norm(a) in norm(q) for a in anchors) if anchors else False,
        }
        cur.execute(
            f"""
            INSERT INTO {out_table}
            (doc_id, seed_chunk_ids_json, cq_text, source, cq_type, flags_json, anchors_json, model_id, raw_completion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                doc_id,
                json.dumps(seed_chunk_ids, ensure_ascii=False),
                q,
                source,
                cq_type,
                json.dumps(flags, ensure_ascii=False),
                json.dumps(anchors, ensure_ascii=False),
                model_id,
                raw_completion,
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--db", required=True)
    ap.add_argument("--chunk_table", required=True)
    ap.add_argument("--doc_col", default=None)
    ap.add_argument("--text_col", default=None, help="Optional: force text column; otherwise auto-picked by content")

    ap.add_argument("--out_table", default="cq_det")
    ap.add_argument("--fts_table", default="cq_doc_fts")
    ap.add_argument("--rebuild_fts", action="store_true")

    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")

    ap.add_argument("--limit_docs", type=int, default=0, help="0 = all docs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--debug_first", type=int, default=0)

    # doc-level RAG settings
    ap.add_argument("--seed_m", type=int, default=4, help="Important chunks per doc")
    ap.add_argument("--neighbor_k", type=int, default=1, help="Neighbors per seed within doc")
    ap.add_argument("--max_excerpts", type=int, default=10)
    ap.add_argument("--max_chars", type=int, default=800)

    # generation settings
    ap.add_argument("--n_cqs", type=int, default=12)
    ap.add_argument("--min_insert", type=int, default=6, help="Guarantee at least this many CQs per doc (with fallback)")
    ap.add_argument("--max_new_tokens", type=int, default=320)

    # sampling controls (temperature experiments)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling for more diversity")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    # Columns
    chunk_id_col, doc_id_col = detect_chunk_and_doc_cols(conn, args.chunk_table)
    if args.doc_col:
        doc_id_col = args.doc_col

    text_col = pick_best_text_col(conn, args.chunk_table, chunk_id_col, doc_id_col, override=args.text_col)
    print(f"[INFO] Using columns: chunk_id_col={chunk_id_col}, doc_id_col={doc_id_col}, text_col={text_col}")

    # Tables
    ensure_out_table(conn, args.out_table)
    ensure_fts(
        conn,
        args.chunk_table,
        args.fts_table,
        chunk_id_col,
        doc_id_col,
        text_col,
        rebuild=args.rebuild_fts,
    )

    done = processed_docs(conn, args.out_table) if args.resume else set()

    llm = LocalChatLLM(
        LLMConfig(
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )

    # Load chunks grouped by doc
    cur = conn.cursor()
    cur.execute(f"SELECT {doc_id_col}, {chunk_id_col}, {text_col} FROM {args.chunk_table} WHERE {text_col} IS NOT NULL;")
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for d, cid, txt in cur.fetchall():
        groups[str(d)].append((str(cid), txt or ""))

    doc_ids = sorted(groups.keys())
    if args.limit_docs and args.limit_docs > 0:
        doc_ids = doc_ids[: args.limit_docs]

    print(f"[INFO] documents loaded: {len(doc_ids)}")

    inserted_total = 0
    printed = 0

    for di, doc_id in enumerate(doc_ids, start=1):
        if args.resume and doc_id in done:
            continue

        chunks = groups[doc_id]
        if not chunks:
            continue

        seeds = select_diverse_top_chunks(chunks, top_m=args.seed_m)

        selected: List[Tuple[str, str]] = []
        seen = set()

        for seed_cid, seed_txt in seeds:
            if seed_cid not in seen:
                selected.append((seed_cid, seed_txt))
                seen.add(seed_cid)

            neigh = retrieve_within_doc(
                conn, args.fts_table, doc_id, seed_txt, args.neighbor_k, exclude_chunk_id=seed_cid
            )
            for ncid, ntx in neigh:
                if ncid not in seen:
                    selected.append((ncid, ntx))
                    seen.add(ncid)

        selected = selected[: args.max_excerpts]
        seed_chunk_ids = [cid for cid, _ in selected]
        excerpt_texts = [smart_truncate(t, args.max_chars) for _, t in selected]

        # If excerpts are still IDs, skip (wrong column)
        if excerpt_texts and all(DOCID_RE.fullmatch(e.strip()) for e in excerpt_texts if e.strip()):
            print(f"[WARN] doc={doc_id} excerpts look like IDs. Check --text_col / schema.")
            continue

        domain = detect_domain(excerpt_texts)
        anchors = extract_anchors(excerpt_texts, max_terms=10)

        prompt = build_prompt(excerpt_texts, anchors, args.n_cqs, domain)
        completion = llm.chat(SYSTEM_PROMPT, prompt)

        if args.debug_first and printed < args.debug_first:
            printed += 1
            print(f"\n[DEBUG doc={doc_id}] domain={domain} anchors={anchors}")
            print("RAW COMPLETION:\n" + completion + "\n")

        llm_qs = parse_questions(completion, args.n_cqs)
        final: List[Tuple[str, str]] = [(q, "llm") for q in llm_qs]

        # Guarantee minimum inserts
        min_needed = max(1, min(args.min_insert, args.n_cqs))
        if len(final) < min_needed:
            fb = template_fallback(excerpt_texts, anchors, min_needed - len(final), domain)
            final.extend([(q, "template") for q in fb])

        # Dedup
        seen_q = set()
        final2: List[Tuple[str, str]] = []
        for q, src in final:
            q2 = clean_question(q)
            k = q2.lower()
            if not q2 or k in seen_q:
                continue
            seen_q.add(k)
            final2.append((q2, src))

        if args.dry_run:
            print(f"\n=== DOC {di}/{len(doc_ids)} doc_id={doc_id} domain={domain} ===")
            print("anchors:", anchors)
            for q, src in final2:
                print(f"- [{src}] {q}")
            continue

        if final2:
            inserted_total += insert_rows(
                conn,
                args.out_table,
                doc_id,
                seed_chunk_ids,
                anchors,
                args.model,
                completion,
                final2,
            )

        if di % 10 == 0:
            print(f"[INFO] progress {di}/{len(doc_ids)} | inserted_total={inserted_total}")

    print(f"\nDone. Total inserted: {inserted_total}")
    conn.close()


if __name__ == "__main__":
    main()
