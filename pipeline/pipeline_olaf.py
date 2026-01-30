from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


# =============================
# Step model + runner
# =============================

@dataclass
class Step:
    name: str
    module: str
    args: List[str]


def run_step(step: Step, python_exe: str, stop_on_fail: bool) -> int:
    cmd = [python_exe, "-m", step.module] + step.args
    print("\n" + "=" * 90)
    print(f"[STEP] {step.name}")
    print(f"[CMD ] {' '.join(cmd)}")
    print("=" * 90)

    start = time.time()
    p = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    dur = time.time() - start

    if p.returncode == 0:
        print(f"[OK  ] {step.name} ({dur:.1f}s)")
    else:
        print(f"[FAIL] {step.name} exit={p.returncode} ({dur:.1f}s)")
        if stop_on_fail:
            sys.exit(p.returncode)

    return p.returncode


# =============================
# Filtering helpers
# =============================

def step_key(st: Step) -> str:
    return st.module.split(".")[-1]


def apply_only_filter(steps: List[Step], only: str) -> List[Step]:
    if not only:
        return steps
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    if not wanted:
        return steps

    kept: List[Step] = []
    for st in steps:
        if step_key(st) in wanted or st.module in wanted or st.name in wanted:
            kept.append(st)

    if not kept:
        print(f"[WARN] --only matched nothing. Wanted={sorted(wanted)}.")
        print(f"       Available={[step_key(s) for s in steps]}")
    return kept


def apply_range_filter(steps: List[Step], start: str, end: str) -> List[Step]:
    if not start and not end:
        return steps

    def matches(st: Step, token: str) -> bool:
        return token in {st.name, st.module, step_key(st)}

    i0 = 0
    i1 = len(steps) - 1

    if start:
        found = None
        for i, st in enumerate(steps):
            if matches(st, start):
                found = i
                break
        if found is None:
            print(f"[WARN] --from_step '{start}' not found. No range start applied.")
        else:
            i0 = found

    if end:
        found = None
        for i, st in enumerate(steps):
            if matches(st, end):
                found = i
        if found is None:
            print(f"[WARN] --to_step '{end}' not found. No range end applied.")
        else:
            i1 = found

    if i0 > i1:
        print("[WARN] Range invalid (start after end). Returning empty step list.")
        return []

    return steps[i0 : i1 + 1]


def list_steps(steps: List[Step]) -> None:
    print("\n" + "-" * 90)
    print("Available steps (use with --only / --from_step / --to_step):")
    for i, st in enumerate(steps, 1):
        print(f"{i:02d}. key={step_key(st):<28} module={st.module:<35} name={st.name}")
    print("-" * 90 + "\n")


# =============================
# Utilities
# =============================

def remove_db(db_path: str) -> None:
    p = Path(db_path)
    if p.exists():
        print(f"[INFO] Removing DB: {p}")
        p.unlink()
    else:
        print(f"[INFO] DB not found, skip remove: {p}")


def ensure_gpu_if_needed(require_gpu: bool) -> None:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if not has_cuda:
        msg = (
            "[WARN] CUDA GPU not available. LLM steps may be very slow or may OOM on CPU.\n"
            "       If you're on HPC, run inside a GPU allocation (srun/sbatch with --gres=gpu...)."
        )
        print(msg)
        if require_gpu:
            raise SystemExit("[ERROR] --require_gpu set but CUDA is not available. Exiting.")
    else:
        try:
            import torch
            print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("[INFO] CUDA available.")


def choose_next_pipeline(mode: str) -> str:
    if mode != "ask":
        return mode

    print("\n" + "=" * 80)
    print("Choose next pipeline:")
    print("  [1] OLAF (symbolic / hybrid NLP)")
    print("  [2] OLAF + LLM (separate branch)")
    print("  [0] Stop here")
    print("=" * 80)

    while True:
        choice = input("Enter choice [1/2/0]: ").strip()
        if choice == "1":
            return "olaf"
        if choice == "2":
            return "olaf_llm"
        if choice == "0":
            return "stop"
        print("Invalid choice. Please enter 1, 2, or 0.")


# =============================
# PREPROCESSING STEPS
# =============================

def build_preprocess_steps(args: argparse.Namespace) -> List[Step]:
    steps: List[Step] = []

    steps.append(Step(name="Run patterns.py", module="pre_processing.patterns", args=[]))

    steps.append(
        Step(
            name="Ingest .txt files -> raw table",
            module="pre_processing.data_ingest",
            args=[
                "--db", args.db,
                "--input", args.input,
                "--raw_table", args.raw_table,
                "--version", str(args.raw_version),
            ],
        )
    )

    steps.append(
        Step(
            name="Clean raw -> cleaned table",
            module="pre_processing.data_clean",
            args=[
                "--db", args.db,
                "--raw_table", args.raw_table,
                "--cleaned_table", args.cleaned_table,
                "--raw_version", str(args.raw_version),
                "--cleaned_version", str(args.cleaned_version),
                "--min_words", str(args.min_words),
            ],
        )
    )

    steps.append(
        Step(
            name="Sentence segmentation cleaned -> segmented table",
            module="pre_processing.sentence_segment",
            args=[
                "--db", args.db,
                "--cleaned_table", args.cleaned_table,
                "--segmented_table", args.segmented_table,
                "--cleaned_version", str(args.cleaned_version),
                "--spacy_model", args.spacy_model,
                "--max_length", str(args.max_length),
            ],
        )
    )

    lem_args = [
        "--db", args.db,
        "--segmented_table", args.segmented_table,
        "--lemmatized_table", args.lemmatized_table,
        "--cleaned_version", str(args.cleaned_version),
        "--spacy_model", args.spacy_model,
        "--batch_size", str(args.batch_size),
    ]
    if args.keep_pos:
        lem_args.append("--keep_pos")
    if args.remove_stopwords:
        lem_args.append("--remove_stopwords")
    if args.remove_punct:
        lem_args.append("--remove_punct")

    steps.append(
        Step(
            name="Lemmatize segmented -> lemmatized table",
            module="pre_processing.sentence_lemmatize",
            args=lem_args,
        )
    )

    steps.append(
        Step(
            name="Chunk segmented sentences -> contextual chunks",
            module="pre_processing.contextual_chunking",
            args=[
                "--db", args.db,
                "--sentence_table", args.segmented_table,
                "--chunks_table", args.chunks_table,
                "--cleaned_version", str(args.cleaned_version),
                "--min_sentences", str(args.chunk_min_sentences),
                "--max_sentences", str(args.chunk_max_sentences),
                "--min_tokens", str(args.chunk_min_tokens),
                "--max_tokens", str(args.chunk_max_tokens),
                "--overlap_sentences", str(args.chunk_overlap),
            ],
        )
    )

    return steps


# =============================
# OLAF STEPS
# =============================

def build_olaf_steps(args: argparse.Namespace) -> List[Step]:
    steps: List[Step] = []

    # 1) term extraction
    steps.append(
        Step(
            name="OLAF: Term Extraction (TF-IDF)",
            module="olaf.term_extraction_tfidf",
            args=[
                "--db", args.db,
                "--cleaned_version", str(args.cleaned_version),
                "--sentence_table", args.lemmatized_table,
                "--term_candidates_table", args.term_candidates_table,
                "--term_occurrences_table", args.term_occurrences_table,
                "--stopwords", args.stopwords,
                "--max_tfidf_tokens", str(args.max_tfidf_tokens),
            ] + (["--reset_terms"] if args.reset_terms else []),
        )
    )

    # 2) rule-based enrichment
    steps.append(
        Step(
            name="OLAF: Term enrichment (rule-based)",
            module="olaf.term_enrichment",
            args=[
                "--db", args.db,
                "--term_candidates_table", args.term_candidates_table,
                "--term_enrichment_table", args.term_enrichment_table,
                "--min_tf_idf", str(args.min_tf_idf),
            ],
        )
    )

    # 3) LLM enrichment extension -> term_enrichment_exten
    if args.use_llm_enrich:
        ensure_gpu_if_needed(args.require_gpu)
        steps.append(
            Step(
                name="OLAF: Term Enrichment v2 (LLM) -> term_enrichment_exten",
                module="olaf.term_enrichment_extension",
                args=[
                    "--db", args.db,
                    "--src_table", args.term_enrichment_src_table,
                    "--dst_table", args.term_enrichment_ext_table,
                    "--term_occurrences_table", args.term_occurrences_table,
                    "--sentences_table", args.lemmatized_table,
                    "--cleaned_version", str(args.cleaned_version),
                    "--max_terms_llm", str(args.max_terms_llm),
                    "--max_evidence_sents", str(args.max_evidence_sents),
                    "--log_every", str(args.log_every),
                    "--prompt_config", args.term_enrich_prompt_config,
                    "--hf_model", args.hf_model,
                    "--dtype", args.hf_dtype,
                    "--device", args.hf_device,
                    "--batch_size", str(args.hf_batch_size),
                ] + (["--classify_all"] if args.classify_all else []),
            )
        )

    # 4) embeddings
    steps.append(
        Step(
            name="OLAF: Skip-gram embeddings + neighbors",
            module="olaf.embeddings_skipgram",
            args=[
                "--db", args.db,
                "--sentences_table", args.lemmatized_table,
                "--term_candidates_table", args.term_candidates_table,
                "--neighbors_table", args.skipgram_neighbors_table,
                "--cleaned_version", str(args.cleaned_version),
                "--min_tfidf", str(args.skipgram_min_tfidf),
                "--top_k_max", str(args.skipgram_top_k_max),
            ] + (["--train"] if args.skipgram_train else []),
        )
    )

    # 5) taxonomy (NO LLM) => taxonomy_is_a
    # NOTE: your updated olaf.taxonomy has defaults: use_embeddings=True, use_hearst=True, no top_parents cap.
    steps.append(
        Step(
            name="OLAF: Taxonomy extraction (seed + embeddings + Hearst) -> taxonomy_is_a",
            module="olaf.taxonomy",
            args=[
                "--db", args.db,
                "--terms_table", args.term_enrichment_ext_table,
                "--out_table", args.taxonomy_table,             # taxonomy_is_a
                "--term_candidates_table", args.term_candidates_table,
                "--sim_table", args.taxonomy_sim_table,
                "--sent_table", args.taxonomy_sent_table,
                "--sent_col", args.taxonomy_sent_col,
                "--max_sents", str(args.taxonomy_max_sents),
                "--min_children_per_parent", str(args.taxonomy_min_children_per_parent),
                "--top_k_neighbors", str(args.taxonomy_top_k_neighbors),
                "--min_cos", str(args.taxonomy_min_cos),
            ] + (["--require_same_category"] if args.taxonomy_require_same_category else []),
        )
    )

    # 6) taxonomy_extension (LLM) => taxonomy_is_a_final
    if args.use_llm_taxonomy:
        ensure_gpu_if_needed(args.require_gpu)
        steps.append(
            Step(
                name="OLAF: Taxonomy LLM validation -> taxonomy_is_a_final",
                module="olaf.taxonomy_extension",
                args=[
                    "--db", args.db,
                    "--in_taxonomy_table", args.taxonomy_table,        # NOT --in_table
                    "--out_table", args.taxonomy_final_table,
                    "--model", args.llm_taxonomy_model,
                    "--prompt_config", args.taxonomy_prompt_config,
                    "--few_shots_k", str(args.taxonomy_llm_few_shots_k),
                    "--max_rows", str(args.taxonomy_llm_max_rows),
                    "--max_new_tokens", str(args.taxonomy_llm_max_new_tokens),
                    "--sent_table", args.taxonomy_sent_table,
                    "--sent_col", args.taxonomy_sent_col,
                    "--evidence_k", str(args.taxonomy_llm_evidence_k),
                    "--global_parents_k", str(args.taxonomy_llm_global_parents_k),
                    "--debug_print_fail_k", str(args.taxonomy_llm_debug_print_fail_k),
                ],
            )
        )

    # 7) non-taxonomy (OpenIE) => raw + clean
    steps.append(
        Step(
            name="Non-taxonomic extraction (OpenIE spaCy) -> non_taxonomic_edges(_clean)",
            module="olaf.non_taxonomy",
            args=[
                "--db", args.db,
                "--spacy_model", args.non_tax_spacy_model,
                "--stopwords", args.stopwords,
                "--sentence_table", args.non_tax_sentence_table,
                "--cleaned_version", str(args.cleaned_version),
                "--term_candidates_table", args.term_candidates_table,
                "--term_enrichment_table", args.term_enrichment_table,
                "--term_enrichment_ext_table", args.term_enrichment_ext_table,
                "--taxonomy_table", args.taxonomy_final_table,  # use final taxonomy for taxonomy-filter if enabled
                "--use_taxonomy_filter" if args.non_tax_use_taxonomy_filter else "",
                "--raw_edges_table", args.non_tax_raw_table,
                "--clean_edges_table", args.non_tax_clean_table,
                "--method", args.non_tax_method,
                "--max_rel_len", str(args.non_tax_max_rel_len),
                "--min_rel_total", str(args.non_tax_min_rel_total),
                "--min_rel_subj", str(args.non_tax_min_rel_subj),
                "--min_rel_obj", str(args.non_tax_min_rel_obj),
                "--debug_k", str(args.non_tax_debug_k),
            ],
        )
    )
    # remove empty "" tokens from args list (because of conditional flag above)
    steps[-1].args = [a for a in steps[-1].args if a != ""]

    # 8) non_taxonomy_extension (LLM) => non_taxonomic_edges_final (no dropping)
    if args.use_llm_non_taxonomy:
        ensure_gpu_if_needed(args.require_gpu)
        steps.append(
            Step(
                name="LLM validate non-tax edges -> non_taxonomic_edges_llm_binary + accept table",
                module="olaf.non_tax_extension",
                args=[
                    "--db", args.db,
                    "--in_edges_table", args.non_tax_clean_table,
                    "--out_llm_table", args.non_tax_llm_table,
                    "--out_accept_table", args.non_tax_accept_table,
                    "--model", args.llm_non_tax_model,
                    "--device", args.hf_device,  # "auto" or "cuda"
                    "--non_tax_config", args.non_tax_prompt_config,
                    "--batch_size", str(args.non_tax_llm_batch_size),
                    "--max_new_tokens", str(args.non_tax_llm_max_new_tokens),
                    "--temperature", str(args.non_tax_llm_temperature),
                    "--log_every", str(args.non_tax_llm_log_every),
                    "--commit_every", "200",
                ],
            )
        )

    # 9) AXIOM GENERATION (UPDATED olaf.axiom)
    if args.run_axioms:
        # If enabled, auto-pick the best upstream tables unless user overrides explicitly
        if args.auto_pick_axiom_inputs:
            taxonomy_for_axioms = args.taxonomy_final_table if args.use_llm_taxonomy else args.taxonomy_table
            triples_for_axioms = args.non_tax_accept_table if args.use_llm_non_taxonomy else args.non_tax_accept_table
        else:
            taxonomy_for_axioms = args.axiom_taxonomy_table
            triples_for_axioms = args.axiom_triple_table

        ax_out_dir = os.path.join(args.out_dir_root, args.axiom_out_dir)

        steps.append(
            Step(
                name=f"AXIOMS: Generate OWL (.ttl) -> {ax_out_dir}",
                module="olaf.axiom",
                args=[
                    "--db", args.db,
                    "--out_dir", ax_out_dir,

                    "--taxonomy_table", taxonomy_for_axioms,
                    "--triple_table", triples_for_axioms,
                    "--types_table", args.term_enrichment_ext_table,

                    "--tax_child_col", args.axiom_tax_child_col,
                    "--tax_parent_col", args.axiom_tax_parent_col,

                    "--triple_subj_col", args.axiom_triple_subj_col,
                    "--triple_rel_col", args.axiom_triple_rel_col,
                    "--triple_obj_col", args.axiom_triple_obj_col,

                    "--taxonomy_where", args.axiom_taxonomy_where,
                    "--triple_where", args.axiom_triple_where,

                    "--min_support", str(args.axiom_min_support),
                    "--min_purity", str(args.axiom_min_purity),
                    "--evidence_k", str(args.axiom_evidence_k),

                    "--export_owl",
                    "--base_iri", args.axiom_base_iri,
                    "--break_cycles"
                ] + (["--no_hash_iris"] if args.axiom_no_hash_iris else []),
            )
        )
        
    return steps


# =============================
# OLAF-LLM STEPS (separate pipeline)
# =============================

def build_olaf_llm_steps(args: argparse.Namespace) -> List[Step]:
    return [
        Step("OLAF-LLM: Term Enrichment", "olaf_llm.term_enrichment_llm", ["--db", args.db]),
        Step("OLAF-LLM: Relation Induction", "olaf_llm.relations_llm", ["--db", args.db]),
        Step("OLAF-LLM: Axiom Generation", "olaf_llm.axioms_llm", ["--db", args.db]),
    ]


# =============================
# Build one combined plan
# =============================

def build_plan(args: argparse.Namespace) -> List[Step]:
    plan: List[Step] = []
    plan += build_preprocess_steps(args)

    stage = choose_next_pipeline(args.next)
    if stage == "stop":
        return plan
    if stage == "olaf":
        plan += build_olaf_steps(args)
    elif stage == "olaf_llm":
        plan += build_olaf_llm_steps(args)

    return plan


# =============================
# CLI
# =============================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run preprocessing + OLAF pipeline")

    # Core
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--input", required=True, help="Folder containing .txt files")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")

    # Tables (preprocess)
    ap.add_argument("--raw_table", default="raw_documents")
    ap.add_argument("--cleaned_table", default="cleaned_documents")
    ap.add_argument("--segmented_table", default="sentence_segmented")
    ap.add_argument("--lemmatized_table", default="sentence_lemmatized")
    ap.add_argument("--chunks_table", default="contextual_chunk")

    # Versions
    ap.add_argument("--raw_version", type=int, default=1)
    ap.add_argument("--cleaned_version", type=int, default=1)

    # Cleaning params
    ap.add_argument("--min_words", type=int, default=4)

    # spaCy params
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--max_length", type=int, default=5_000_000)

    # Lemmatization params
    ap.add_argument("--keep_pos", dest="keep_pos", action="store_true", default=True)
    ap.add_argument("--no_keep_pos", dest="keep_pos", action="store_false")

    ap.add_argument("--remove_stopwords", dest="remove_stopwords", action="store_true", default=True)
    ap.add_argument("--keep_stopwords", dest="remove_stopwords", action="store_false")

    ap.add_argument("--remove_punct", dest="remove_punct", action="store_true", default=True)
    ap.add_argument("--keep_punct", dest="remove_punct", action="store_false")

    ap.add_argument("--batch_size", type=int, default=2000)

    # Chunking params
    ap.add_argument("--chunk_min_sentences", type=int, default=5)
    ap.add_argument("--chunk_max_sentences", type=int, default=12)
    ap.add_argument("--chunk_min_tokens", type=int, default=400)
    ap.add_argument("--chunk_max_tokens", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=1)

    # Next stage
    ap.add_argument("--next", choices=["ask", "olaf", "olaf_llm", "stop"], default="ask")

    # Term extraction
    ap.add_argument("--term_candidates_table", default="term_candidates")
    ap.add_argument("--term_occurrences_table", default="term_occurrences")
    ap.add_argument("--stopwords", default="stop_word/stop_words.txt")
    ap.add_argument("--max_tfidf_tokens", type=int, default=3)
    ap.add_argument("--reset_terms", action="store_true")

    # Term enrichment
    ap.add_argument("--term_enrichment_table", default="term_enrichment")
    ap.add_argument("--min_tf_idf", type=float, default=5.0)

    # LLM term enrichment extension
    ap.add_argument("--use_llm_enrich", dest="use_llm_enrich", action="store_true", default=True)
    ap.add_argument("--no_llm_enrich", dest="use_llm_enrich", action="store_false")
    ap.add_argument("--require_gpu", action="store_true")

    ap.add_argument("--term_enrichment_src_table", default="term_enrichment")
    ap.add_argument("--term_enrichment_ext_table", default="term_enrichment_exten")
    ap.add_argument("--classify_all", dest="classify_all", action="store_true", default=True)
    ap.add_argument("--hf_batch_size", type=int, default=8)

    ap.add_argument("--hf_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--hf_dtype", default="float16", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--hf_device", default="auto")

    # Prompt configs
    ap.add_argument("--term_enrich_prompt_config", default="prompts/term_enrichment_extension.json")

    # Enrichment knobs
    ap.add_argument("--max_terms_llm", type=int, default=0)   # 0=all
    ap.add_argument("--max_evidence_sents", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=1000)

    # Embeddings
    ap.add_argument("--skipgram_neighbors_table", default="skipgram_neighbors")
    ap.add_argument("--skipgram_min_tfidf", type=float, default=10.0)
    ap.add_argument("--skipgram_top_k_max", type=int, default=20)
    ap.add_argument("--skipgram_train", action="store_true")

    # TAXONOMY (no LLM) -> taxonomy_is_a
    ap.add_argument("--taxonomy_table", default="taxonomy_is_a")
    ap.add_argument("--taxonomy_sim_table", default="skipgram_neighbors")
    ap.add_argument("--taxonomy_sent_table", default="sentence_lemmatized")
    ap.add_argument("--taxonomy_sent_col", default="sentence")
    ap.add_argument("--taxonomy_max_sents", type=int, default=200000)
    ap.add_argument("--taxonomy_min_children_per_parent", type=int, default=2)
    ap.add_argument("--taxonomy_top_k_neighbors", type=int, default=5)
    ap.add_argument("--taxonomy_min_cos", type=float, default=0.75)
    ap.add_argument("--taxonomy_require_same_category", action="store_true", default=True)

    # TAXONOMY LLM extension -> taxonomy_is_a_final
    ap.add_argument("--use_llm_taxonomy", dest="use_llm_taxonomy", action="store_true", default=True)
    ap.add_argument("--no_llm_taxonomy", dest="use_llm_taxonomy", action="store_false")

    ap.add_argument("--taxonomy_final_table", default="taxonomy_is_a_final")
    ap.add_argument("--llm_taxonomy_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--taxonomy_prompt_config", default="prompts/taxonomy_extension.json")
    ap.add_argument("--taxonomy_llm_max_new_tokens", type=int, default=240)
    ap.add_argument("--taxonomy_llm_few_shots_k", type=int, default=6)
    ap.add_argument("--taxonomy_llm_max_rows", type=int, default=0)
    ap.add_argument("--taxonomy_llm_evidence_k", type=int, default=2)
    ap.add_argument("--taxonomy_llm_global_parents_k", type=int, default=12)
    ap.add_argument("--taxonomy_llm_debug_print_fail_k", type=int, default=3)


    # NON-TAXONOMY extraction
    ap.add_argument("--non_tax_sentence_table", default="sentence_segmented")
    ap.add_argument("--non_tax_raw_table", default="non_taxonomic_edges")
    ap.add_argument("--non_tax_clean_table", default="non_taxonomic_edges_clean")
    ap.add_argument("--non_tax_spacy_model", default="en_core_web_sm")
    ap.add_argument("--non_tax_method", default="openie_spacy")
    ap.add_argument("--non_tax_use_taxonomy_filter", action="store_true", default=False)
    ap.add_argument("--non_tax_max_rel_len", type=int, default=80)
    ap.add_argument("--non_tax_min_rel_total", type=int, default=3)
    ap.add_argument("--non_tax_min_rel_subj", type=int, default=2)
    ap.add_argument("--non_tax_min_rel_obj", type=int, default=2)
    ap.add_argument("--non_tax_debug_k", type=int, default=15)

    # NON-TAXONOMY LLM extension -> non_taxonomic_edges_final
    ap.add_argument("--use_llm_non_taxonomy", dest="use_llm_non_taxonomy", action="store_true", default=True)
    ap.add_argument("--no_llm_non_taxonomy", dest="use_llm_non_taxonomy", action="store_false")

    ap.add_argument("--llm_non_tax_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--non_tax_prompt_config", default="prompts/non_tax_extension.json")
    ap.add_argument("--non_tax_final_table", default="non_taxonomic_edges_final")
    ap.add_argument("--non_tax_llm_batch_size", type=int, default=6)
    ap.add_argument("--non_tax_llm_max_new_tokens", type=int, default=220)
    ap.add_argument("--non_tax_llm_temperature", type=float, default=0.0)
    ap.add_argument("--non_tax_llm_log_every", type=int, default=50)
    ap.add_argument("--non_tax_llm_include_sentence", action="store_true", default=True)
    ap.add_argument("--non_tax_dedupe_mode", choices=["none", "soft", "hard"], default="soft")
    ap.add_argument("--non_tax_llm_table", default="non_taxonomic_edges_llm_binary")
    ap.add_argument("--non_tax_accept_table", default="non_taxonomic_edges_accept")


    # Runner controls
    ap.add_argument("--clean_db", action="store_true")
    ap.add_argument("--stop_on_fail", action="store_true")

    # Filters
    ap.add_argument("--only", default="")
    ap.add_argument("--from_step", default="")
    ap.add_argument("--to_step", default="")
    ap.add_argument("--list_steps", action="store_true")

    ap.add_argument("--out_dir_root", default="out_lsf", help="Root output folder for artifacts (axioms, logs, etc.)")
  
    # AXIOMS 
    ap.add_argument("--run_axioms", action="store_true", default=True)
    ap.add_argument("--no_axioms", dest="run_axioms", action="store_false")

    ap.add_argument("--axiom_out_dir", default="axioms", help="Subfolder under --out_dir_root for axiom outputs")
    ap.add_argument("--axiom_base_iri", default="http://example.org/hpc#")
    ap.add_argument("--axiom_no_hash_iris", action="store_true", default=True)

    # Auto-pick: uses taxonomy_final/non_tax_final when LLM steps enabled
    ap.add_argument("--auto_pick_axiom_inputs", action="store_true", default=True)
    ap.add_argument("--no_auto_pick_axiom_inputs", dest="auto_pick_axiom_inputs", action="store_false")

    # Defaults aligned to YOUR DB schema (from your table photo)
    ap.add_argument("--axiom_taxonomy_table", default="taxonomy_exten")
    ap.add_argument("--axiom_triple_table", default="non_taxonomic_edges_accept")

    ap.add_argument("--axiom_tax_child_col", default="child")
    ap.add_argument("--axiom_tax_parent_col", default="llm_best_parent")  

    ap.add_argument("--axiom_triple_subj_col", default="subj_canonical_term")
    ap.add_argument("--axiom_triple_rel_col", default="rel_key")
    ap.add_argument("--axiom_triple_obj_col", default="obj_canonical_term")

    ap.add_argument("--break_cycles", action="store_true",
                    help="Automatically remove cycle-causing taxonomy edges instead of crashing.")

    # Safe filters for your LLM tables
    ap.add_argument(
        "--axiom_taxonomy_where",
        default=(
            "child IS NOT NULL AND TRIM(child) != '' "
            "AND LOWER(child) NOT IN ('none','null','unknown') "
            "AND llm_best_parent IS NOT NULL AND TRIM(llm_best_parent) != '' "
            "AND LOWER(llm_best_parent) NOT IN ('none','null','unknown') "
            "AND (llm_accept = 1 OR LOWER(llm_accept) IN ('true','yes','accept'))"
        ),
    )
    ap.add_argument(
        "--axiom_triple_where",
        default="decision IS NULL OR LOWER(decision) IN ('accept','accepted','yes','true','1')",
    )

    ap.add_argument("--axiom_min_support", type=int, default=2)
    ap.add_argument("--axiom_min_purity", type=float, default=0.55)
    ap.add_argument("--axiom_evidence_k", type=int, default=5)

    return ap.parse_args()


# =============================
# MAIN
# =============================

def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    if args.clean_db:
        remove_db(args.db)

    plan = build_plan(args)

    if args.list_steps:
        list_steps(plan)
        return

    plan = apply_only_filter(plan, args.only)
    plan = apply_range_filter(plan, args.from_step, args.to_step)

    if not plan:
        print("[INFO] No steps selected after filters.")
        return

    failures = 0
    for st in plan:
        rc = run_step(st, python_exe=args.python, stop_on_fail=args.stop_on_fail)
        failures += int(rc != 0)

    print("\n" + "-" * 90)
    if failures == 0:
        print("[DONE] Pipeline finished successfully.")
    else:
        print(f"[DONE] Pipeline finished with {failures} failing step(s).")
    print("-" * 90)


if __name__ == "__main__":
    main()
