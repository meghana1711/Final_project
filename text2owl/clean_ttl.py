from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional

# rdflib needed (already in your container)
from rdflib import Graph  # type: ignore


# ============================================================
# Header handling
# ============================================================

def split_header_and_body(ttl: str) -> Tuple[str, str]:
    """
    Keep @prefix/@base/comment/blank lines as header until first real statement.
    """
    lines = ttl.splitlines(True)
    header: List[str] = []
    body: List[str] = []

    in_header = True
    for ln in lines:
        s = ln.strip()
        if in_header:
            if (not s) or s.startswith("#") or s.startswith("@prefix") or s.startswith("@base"):
                header.append(ln)
                continue
            in_header = False
            body.append(ln)
        else:
            body.append(ln)
    return "".join(header), "".join(body)


# ============================================================
# Block splitting (Text2OWL style)
# ============================================================

def split_blocks(text: str) -> List[str]:
    """
    Split into blocks separated by blank lines.
    """
    lines = text.splitlines(True)
    blocks: List[List[str]] = []
    cur: List[str] = []

    def blank(ln: str) -> bool:
        return ln.strip() == ""

    i = 0
    while i < len(lines):
        ln = lines[i]
        if blank(ln):
            if cur:
                blocks.append(cur)
                cur = []
            while i < len(lines) and blank(lines[i]):
                i += 1
            continue
        cur.append(ln)
        i += 1

    if cur:
        blocks.append(cur)

    return ["".join(b).strip() for b in blocks if "".join(b).strip()]


def first_non_comment_line(block: str) -> Optional[str]:
    for ln in block.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        return ln
    return None


# ============================================================
# Regex patterns (all the errors you’ve hit)
# ============================================================

# Illegal dashed QName anywhere (subject OR object OR domain etc.): hpc:-rconly
ILLEGAL_DASH_QNAME_ANY_RE = re.compile(r"\b[A-Za-z_][\w\-]*:-[A-Za-z0-9_][^\s;,.]*")

# Illegal dashed QName as SUBJECT line (fast drop)
ILLEGAL_SUBJECT_QNAME_RE = re.compile(r"^\s*[A-Za-z_][\w\-]*\s*:\s*-\S+")

# Bytes/markdown artifacts
BAD_LINE_TOKENS = ("b'@prefix", "b'```", "```")

# Injected junk (^b artifacts)
INJECTED_JUNK_RE = re.compile(
    r"'\^b'|\^b'|'\\\^b|\\\^b|\\x27\\?\^b\\x27|\\x27\\?\^b|\^b",
    re.IGNORECASE,
)

# Broken prefix token: hp<junk>c:  -> hpc:
BROKEN_HPC_PREFIX_TOKEN_RE = re.compile(r"\bhp[^\s:]{0,200}c:", re.IGNORECASE)

# Dash-start line = leaked CLI flag statement
DASH_START_LINE_RE = re.compile(r"^\s*-\S+")
# CLI flag injected after semicolon
SEMICOLON_DASH_RE = re.compile(r";\s*-\S+")

# Facet predicates used as normal predicates (noise)
BAD_XSD_FACET_PREDICATES = (
    "xsd:minInclusive",
    "xsd:maxInclusive",
    "xsd:minExclusive",
    "xsd:maxExclusive",
    "xsd:minLength",
    "xsd:maxLength",
    "xsd:pattern",
    "xsd:totalDigits",
    "xsd:fractionDigits",
)

# Typed literal used as predicate line (GraphDB: Illegal predicate value: ... )
# Eg: "-"^^xsd:integer hpc:Job .
LITERAL_PRED_LINE_RE = re.compile(
    r'^\s*(?:"([^"\\]|\\.)*"|\'([^\']|\\.)*\')\s*\^\^\s*(?:xsd:|<http://www\.w3\.org/2001/XMLSchema#)',
    re.IGNORECASE,
)

# Also catch bare numeric literal-as-predicate cases (rare)
NUMERIC_PRED_LINE_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s+\S+", re.IGNORECASE)

# Function-call hallucination:
#   rdfs:domain(hpc:FileLimit) hpc:Job .
DOMAIN_CALL_RE = re.compile(r"^\s*rdfs:domain\(\s*([^)]+)\s*\)\s+(\S+)\s*\.\s*$")
RANGE_CALL_RE  = re.compile(r"^\s*rdfs:range\(\s*([^)]+)\s*\)\s+(\S+)\s*\.\s*$")

# New subject detection
NEW_SUBJECT_RE = re.compile(r"^\s*(?:[A-Za-z_][\w\-]*:[^\s]+|<[^>]+>|_:[A-Za-z][\w\-]*)\s+")
# Indented predicate continuation detection
INDENTED_PRED_LINE_RE = re.compile(r"^\s{1,}\w[\w\-]*:\w")

# Malformed OWL collections: property-value pairs in list
MALFORMED_COLLECTION_RE = re.compile(r"(owl:intersectionOf|owl:unionOf)", re.IGNORECASE)
COLLECTION_HAS_PVPAIR_RE = re.compile(r"\(\s+\w+:\w+\s+\"[^\"]*\"", re.MULTILINE)

# num0/num1/etc (table-index noise) — optional aggressive drop
NUM_PROPERTY_RE = re.compile(r"^\s*[A-Za-z_][\w\-]*:num\d+\b", re.IGNORECASE)


# ============================================================
# Structural checks
# ============================================================

def bracket_balance_ok(text: str) -> bool:
    """
    Heuristic: check [ and ] counts match after stripping string literals.
    Prevents GraphDB: Expected ']', found '.'
    """
    tmp = re.sub(r'"([^"\\]|\\.)*"', '""', text)
    return tmp.count("[") == tmp.count("]")


# ============================================================
# Fixers (drop OR correct)
# ============================================================

def rewrite_domain_range_function_syntax(line: str, stats: Counter) -> Optional[str]:
    """
    Rewrite:
      rdfs:domain(hpc:FileLimit) hpc:Job .
    to:
      hpc:FileLimit rdfs:domain hpc:Job .
    """
    m = DOMAIN_CALL_RE.match(line)
    if m:
        subj = m.group(1).strip()
        dom = m.group(2).strip()
        stats["rewrote_domain_function_syntax"] += 1
        return f"{subj} rdfs:domain {dom} .\n"

    m = RANGE_CALL_RE.match(line)
    if m:
        subj = m.group(1).strip()
        rng = m.group(2).strip()
        stats["rewrote_range_function_syntax"] += 1
        return f"{subj} rdfs:range {rng} .\n"

    return None


def sanitize_lines(lines: List[str], stats: Counter, drop_num_properties: bool) -> List[str]:
    out: List[str] = []

    for raw in lines:
        line = raw

        # 0) rewrite function-call syntax if present
        rewritten = rewrite_domain_range_function_syntax(line.strip(), stats)
        if rewritten is not None:
            out.append(rewritten)
            continue

        # 1) strip injected junk
        stripped = INJECTED_JUNK_RE.sub("", line)
        if stripped != line:
            stats["stripped_injected_junk"] += 1
            line = stripped

        # 2) repair hp<junk>c: -> hpc:
        for _ in range(5):
            repaired = BROKEN_HPC_PREFIX_TOKEN_RE.sub("hpc:", line)
            if repaired == line:
                break
            stats["repaired_broken_hpc_prefix_token"] += 1
            line = repaired

        s = line.strip()

        # 3) drop artifact lines
        if any(tok in line for tok in BAD_LINE_TOKENS):
            stats["dropped_artifact_lines"] += 1
            continue

        # 4) drop any illegal dashed QName occurrences anywhere
        if ILLEGAL_DASH_QNAME_ANY_RE.search(line):
            stats["dropped_illegal_dash_qname_lines"] += 1
            continue

        # 5) drop dash-start lines
        if s and not s.startswith("#") and DASH_START_LINE_RE.match(s):
            stats["dropped_dash_start_lines"] += 1
            continue

        # 6) drop ; -flag fragments
        if s and not s.startswith("#") and SEMICOLON_DASH_RE.search(line):
            stats["dropped_semicolon_dash_lines"] += 1
            continue

        # 7) drop xsd facet predicate lines
        if any(s.startswith(p) for p in BAD_XSD_FACET_PREDICATES):
            stats["dropped_xsd_facet_lines"] += 1
            continue

        # 8) drop literal-as-predicate lines (GraphDB illegal predicate)
        if LITERAL_PRED_LINE_RE.match(s):
            stats["dropped_literal_predicate_lines"] += 1
            continue

        # 9) optional: drop num0/num1 properties (table-index noise)
        if drop_num_properties and NUM_PROPERTY_RE.match(s):
            stats["dropped_numN_property_lines"] += 1
            continue

        # 10) drop obvious malformed numeric predicate lines (rare)
        # Only if it begins with a number AND contains "^^" or looks like predicate position
        if NUMERIC_PRED_LINE_RE.match(s) and "^^" in s:
            stats["dropped_numeric_literal_predicate_lines"] += 1
            continue

        out.append(line)

    return out


def fix_dangling_semicolons(lines: List[str], stats: Counter) -> List[str]:
    """
    If line ends with ';' but next significant line starts a new subject, change ';'->'.'
    """
    out = lines[:]

    def next_sig(i: int) -> Optional[int]:
        for j in range(i, len(out)):
            s = out[j].strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            return j
        return None

    for i in range(len(out)):
        cur_s = out[i].strip()
        if not cur_s or cur_s.startswith("#"):
            continue
        if cur_s.endswith(";"):
            j = next_sig(i + 1)
            if j is None or NEW_SUBJECT_RE.match(out[j]):
                out[i] = re.sub(r";(\s*)$", r".\1", out[i])
                stats["fixed_dangling_semicolon_to_dot"] += 1

    return out


def fix_dot_then_indented_predicate(lines: List[str], stats: Counter) -> List[str]:
    """
    Fix:
      hpc:openhost a owl:ObjectProperty .
          rdfs:domain ...
    to:
      hpc:openhost a owl:ObjectProperty ;
          rdfs:domain ...
    """
    out = lines[:]

    def next_sig(i: int) -> Optional[int]:
        for j in range(i, len(out)):
            s = out[j].strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            return j
        return None

    for i in range(len(out)):
        cur_s = out[i].strip()
        if not cur_s or cur_s.startswith("#"):
            continue
        if cur_s.endswith("."):
            j = next_sig(i + 1)
            if j is None:
                continue
            nxt = out[j]
            if INDENTED_PRED_LINE_RE.match(nxt) and not NEW_SUBJECT_RE.match(nxt):
                out[i] = re.sub(r"\.(\s*)$", r";\1", out[i])
                stats["fixed_dot_to_semicolon_for_indented_predicate"] += 1

    return out


def fix_dots_inside_brackets(text: str, stats: Counter) -> str:
    """
    Fix common restriction syntax bug inside [ ... ]:
      [ a owl:Restriction . owl:onProperty ... . owl:allValuesFrom ... ]
    Replace ". <newline/indent> prefix:" with "; <newline/indent> prefix:" inside brackets.
    """
    out_parts: List[str] = []
    buf: List[str] = []
    depth = 0

    def flush_inside(chunk: str) -> str:
        fixed = re.sub(r"\.\s*(\n\s*)([A-Za-z_][\w\-]*:)", r";\1\2", chunk)
        return fixed

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "[":
            if depth == 0 and buf:
                out_parts.append("".join(buf))
                buf = []
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "]":
            buf.append(ch)
            depth -= 1
            if depth == 0:
                inside = "".join(buf)
                fixed = flush_inside(inside)
                if fixed != inside:
                    stats["fixed_dots_inside_blanknode_to_semicolons"] += 1
                out_parts.append(fixed)
                buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    if buf:
        out_parts.append("".join(buf))

    return "".join(out_parts)


# ============================================================
# Block-level pre-drop (fast)
# ============================================================

def should_drop_block_fast(block: str) -> Tuple[bool, str]:
    ln = first_non_comment_line(block)
    if not ln:
        return False, ""

    # Subject illegal dash QName
    if ILLEGAL_SUBJECT_QNAME_RE.match(ln):
        return True, "illegal_qname_subject_starts_with_dash"

    # Any illegal dashed QName anywhere in the block
    if ILLEGAL_DASH_QNAME_ANY_RE.search(block):
        return True, "contains_illegal_dash_qname"

    if any(tok in block for tok in BAD_LINE_TOKENS):
        return True, "bytes_or_codefence_artifact"

    if MALFORMED_COLLECTION_RE.search(block) and COLLECTION_HAS_PVPAIR_RE.search(block):
        return True, "malformed_owl_collection_with_property_value_pair"

    if not bracket_balance_ok(block):
        return True, "unbalanced_brackets_in_block"

    # If block contains literal-as-predicate line, drop whole block (safer)
    for ln2 in block.splitlines():
        s2 = ln2.strip()
        if not s2 or s2.startswith("#"):
            continue
        if LITERAL_PRED_LINE_RE.match(s2):
            return True, "literal_as_predicate_in_block"

    return False, ""


def try_parse_with_header(header: str, block: str) -> Tuple[bool, str]:
    g = Graph()
    try:
        g.parse(data=(header.rstrip() + "\n\n" + block.strip() + "\n"), format="turtle")
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input TTL")
    ap.add_argument("--out", dest="out", required=True, help="Output TTL (GraphDB-loadable)")
    ap.add_argument("--quarantine", required=True, help="Quarantine blocks that were dropped")
    ap.add_argument("--report", required=True, help="Human-readable report of drops/errors")
    ap.add_argument("--drop-num-properties", action="store_true", help="Drop num0/num1/... properties (table-index noise)")
    ap.add_argument("--max-error-samples", type=int, default=80, help="How many failure samples to include in report")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    quarantine = Path(args.quarantine)
    report = Path(args.report)

    raw = inp.read_text(encoding="utf-8", errors="replace")
    header, body = split_header_and_body(raw)
    blocks = split_blocks(body)

    stats = Counter()
    drop_reasons = Counter()

    kept_blocks: List[str] = []
    quarantined_blocks: List[Tuple[str, str]] = []  # (reason, block)
    error_samples: List[Tuple[int, str, str]] = []  # (block_idx, err, preview)

    for idx, b in enumerate(blocks, start=1):
        # fast pre-drop
        drop, reason = should_drop_block_fast(b)
        if drop:
            drop_reasons[reason] += 1
            quarantined_blocks.append((reason, b))
            continue

        # Apply line sanitation + structural repairs within this block
        lines = b.splitlines(True)
        lines = sanitize_lines(lines, stats, drop_num_properties=args.drop_num_properties)
        lines = fix_dangling_semicolons(lines, stats)
        lines = fix_dot_then_indented_predicate(lines, stats)
        block_fixed = "".join(lines)
        block_fixed = fix_dots_inside_brackets(block_fixed, stats)

        # Parse-guided salvage: keep only blocks that truly parse
        ok, err = try_parse_with_header(header, block_fixed)
        if ok:
            kept_blocks.append(block_fixed.strip())
        else:
            drop_reasons["rdflib_parse_fail"] += 1
            quarantined_blocks.append(("rdflib_parse_fail", block_fixed))
            if len(error_samples) < args.max_error_samples:
                preview = "\n".join(block_fixed.splitlines()[:10])
                error_samples.append((idx, err, preview))

    # Write outputs
    out_text = header.rstrip() + "\n\n" + "\n\n".join(kept_blocks) + "\n"
    out.write_text(out_text, encoding="utf-8")

    quar_text = header.rstrip() + "\n\n"
    for reason, blk in quarantined_blocks:
        quar_text += f"### DROPPED reason={reason}\n{blk.strip()}\n\n"
    quarantine.write_text(quar_text, encoding="utf-8")

    # Report
    rep: List[str] = []
    rep.append(f"Total blocks: {len(blocks)}")
    rep.append(f"Kept blocks : {len(kept_blocks)}")
    rep.append(f"Dropped     : {len(quarantined_blocks)}")
    rep.append("")
    rep.append("Drop reasons:")
    for r, c in drop_reasons.most_common():
        rep.append(f"  - {r}: {c}")
    rep.append("")
    rep.append("Sanitizer stats:")
    for k, v in stats.most_common():
        rep.append(f"  - {k}: {v}")
    rep.append("")
    rep.append("Sample parse failures:")
    for i, err, prev in error_samples:
        rep.append("-" * 80)
        rep.append(f"Block #{i}: {err}")
        rep.append(prev)
    rep.append("")
    report.write_text("\n".join(rep), encoding="utf-8")

    print(f"[OK] GraphDB-loadable TTL -> {out}")
    print(f"[OK] Quarantine TTL       -> {quarantine}")
    print(f"[OK] Report               -> {report}")
    print(f"[INFO] Kept={len(kept_blocks)} Dropped={len(quarantined_blocks)} / {len(blocks)} blocks")


if __name__ == "__main__":
    main()
