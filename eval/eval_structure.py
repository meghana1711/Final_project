"""
basic_eval.py — one basic evaluation for your final ontology tables in SQLite.

What it checks (fast + useful):
1) TAXONOMY structural sanity:
   - self-loops (child == parent)
   - duplicate edges
   - cycles in is-a graph (directed)
   - orphan children (child never appears as parent) [optional signal]

2) NON-TAXONOMY sanity:
   - self-loop triples (s == o)
   - duplicate triples
   - missing/empty evidence

Works with your DB tables (you can pass table/column names).
"""

from __future__ import annotations
import argparse
import sqlite3
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional


def qident(name: str) -> str:
    # very light identifier quoting (assumes trusted inputs)
    return f'"{name}"'


def fetch_edges(conn: sqlite3.Connection, table: str, child_col: str, parent_col: str) -> List[Tuple[str, str]]:
    cur = conn.execute(
        f"SELECT {qident(child_col)}, {qident(parent_col)} FROM {qident(table)};"
    )
    out = []
    for c, p in cur.fetchall():
        if c is None or p is None:
            continue
        out.append((str(c).strip(), str(p).strip()))
    return out


def fetch_triples(conn: sqlite3.Connection, table: str, s_col: str, p_col: str, o_col: str, ev_col: Optional[str]) -> List[Tuple[str, str, str, Optional[str]]]:
    cols = [qident(s_col), qident(p_col), qident(o_col)]
    if ev_col:
        cols.append(qident(ev_col))
    cur = conn.execute(f"SELECT {', '.join(cols)} FROM {qident(table)};")
    out = []
    for row in cur.fetchall():
        s = str(row[0]).strip() if row[0] is not None else ""
        p = str(row[1]).strip() if row[1] is not None else ""
        o = str(row[2]).strip() if row[2] is not None else ""
        ev = None
        if ev_col:
            ev = (str(row[3]).strip() if row[3] is not None else "")
        out.append((s, p, o, ev))
    return out


def count_duplicates(items: List[Tuple]) -> int:
    seen = set()
    dup = 0
    for it in items:
        if it in seen:
            dup += 1
        else:
            seen.add(it)
    return dup


def find_cycles(edges: List[Tuple[str, str]]) -> List[List[str]]:
    """
    Detect cycles in a directed graph using DFS coloring.
    Returns a list of cycles (each cycle is a list of nodes; best-effort).
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    nodes: Set[str] = set()
    for c, p in edges:
        adj[c].append(p)
        nodes.add(c); nodes.add(p)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in nodes}
    parent: Dict[str, Optional[str]] = {n: None for n in nodes}
    cycles: List[List[str]] = []

    def dfs(u: str):
        color[u] = GRAY
        for v in adj.get(u, []):
            if color[v] == WHITE:
                parent[v] = u
                dfs(v)
            elif color[v] == GRAY:
                # Found back-edge u -> v => cycle
                cycle = [v]
                x = u
                while x is not None and x != v:
                    cycle.append(x)
                    x = parent.get(x)
                cycle.append(v)
                cycle.reverse()
                # de-duplicate cycles roughly
                if len(cycle) >= 3 and cycle not in cycles:
                    cycles.append(cycle)
        color[u] = BLACK

    for n in nodes:
        if color[n] == WHITE:
            dfs(n)
    return cycles


def orphan_children(edges: List[Tuple[str, str]]) -> Set[str]:
    children = {c for c, _ in edges}
    parents = {p for _, p in edges}
    # orphan child = never appears as parent (leaf). Not "wrong", but a useful signal.
    return children - parents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)

    # taxonomy table config
    ap.add_argument("--tax-table", default="final_is_a_edges")
    ap.add_argument("--tax-child-col", default="child")
    ap.add_argument("--tax-parent-col", default="parent")

    # triple table config
    ap.add_argument("--triple-table", default="final_non_taxonomy_edges")
    ap.add_argument("--triple-s-col", default="subject")
    ap.add_argument("--triple-p-col", default="predicate")
    ap.add_argument("--triple-o-col", default="object")
    ap.add_argument("--triple-evidence-col", default="evidence")  # set to "" to disable

    args = ap.parse_args()
    conn = sqlite3.connect(args.db)

    # ---- TAXONOMY EVAL ----
    edges = fetch_edges(conn, args.tax_table, args.tax_child_col, args.tax_parent_col)
    self_loops = [(c, p) for (c, p) in edges if c == p]
    dup_edges = count_duplicates(edges)
    cycles = find_cycles([(c, p) for (c, p) in edges if c and p and c != p])
    orphans = orphan_children([(c, p) for (c, p) in edges if c and p])

    print("\n=== BASIC EVALUATION REPORT ===")
    print("\n[TAXONOMY]")
    print(f"Total edges: {len(edges)}")
    print(f"Self-loops (child==parent): {len(self_loops)}")
    print(f"Duplicate edges: {dup_edges}")
    print(f"Cycles found: {len(cycles)}")
    if cycles:
        # show up to 3 cycles
        for i, cyc in enumerate(cycles[:3], 1):
            print(f"  Cycle {i}: {' -> '.join(cyc)}")
    print(f"Leaf/orphan children (never parent): {len(orphans)}")
    # show a few leaves
    if orphans:
        sample = sorted(list(orphans))[:15]
        print("  Sample leaves:", ", ".join(sample))

    # ---- NON-TAX EVAL ----
    ev_col = args.triple_evidence_col if args.triple_evidence_col.strip() else None
    triples = fetch_triples(conn, args.triple_table, args.triple_s_col, args.triple_p_col, args.triple_o_col, ev_col)
    triples_core = [(s, p, o) for (s, p, o, _) in triples]
    dup_triples = count_duplicates(triples_core)
    self_loop_triples = [(s, p, o) for (s, p, o, _) in triples if s == o and s != ""]
    missing_ev = 0
    if ev_col:
        missing_ev = sum(1 for (_, _, _, ev) in triples if ev is None or ev.strip() == "")

    print("\n[NON-TAXONOMY]")
    print(f"Total triples: {len(triples)}")
    print(f"Self-loop triples (s==o): {len(self_loop_triples)}")
    print(f"Duplicate triples: {dup_triples}")
    if ev_col:
        print(f"Missing/empty evidence: {missing_ev}")

    conn.close()


if __name__ == "__main__":
    main()
