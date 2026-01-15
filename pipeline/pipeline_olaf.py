from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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


def build_steps(args: argparse.Namespace) -> List[Step]:
    steps: List[Step] = []

    # 0) PATTERNS (run before anything else)
    steps.append(
        Step(
            name="Run patterns.py",
            module="pre_processing.patterns",
            args=[],  
        )
    )

    # 1) INGEST
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

    # 2) CLEAN
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

    # 3) SEGMENT
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

    # 4) LEMMATIZE
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

    # 5) Contextual Chunking
    steps.append(
        Step(
            name="Chunk lemmatized sentences -> contextual chunks",
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

   
    # Optional: run subset
    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        steps = [s for s in steps if s.module.split(".")[-1] in wanted]

    return steps

def choose_next_pipeline(mode: str) -> str:
    if mode != "ask":
        return mode

    print("\n" + "=" * 80)
    print("Choose next pipeline:")
    print("  [1] OLAF (symbolic / hybrid NLP)")
    print("  [2] OLAF + LLM")
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


def build_olaf_steps(args: argparse.Namespace) -> List[Step]:
    steps: List[Step] = []

    # 1) Term Extraction (TF-IDF)
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

    # 2) Term Enrichment (rule-based)
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

    # 3) Optional: Term Enrichment with LLM (guarded behind --use_llm_enrich)
    if args.use_llm_enrich:
        ensure_gpu_if_needed(args.require_gpu)

        if not args.hf_model:
            raise SystemExit("[ERROR] --use_llm_enrich requires --hf_model to be set.")

        steps.append(
            Step(
                name="OLAF: Term Enrichment (LLM HF Mistral) -> term_enrichment_exten",
                module="olaf.term_enrichment_extension",
                args=[
                    "--db", args.db,
                    "--term_candidates_table", args.term_candidates_table,
                    "--term_occurrences_table", args.term_occurrences_table,
                    "--sentences_table", args.lemmatized_table,
                    "--dst_table", args.term_enrichment_ext_table,
                    "--cleaned_version", str(args.cleaned_version),
                    "--min_tf_idf_keep", str(args.min_tf_idf),
                    "--hardcase_p_low", str(args.hardcase_p_low),
                    "--hardcase_p_high", str(args.hardcase_p_high),
                    "--max_terms_llm", str(args.max_terms_llm),
                    "--max_evidence_sents", str(args.max_evidence_sents),
                    "--hf_model", args.hf_model,
                    "--dtype", args.hf_dtype,
                    "--device", args.hf_device,
                    "--max_new_tokens", str(args.hf_max_new_tokens),
                    "--temperature", str(args.hf_temperature),
                    "--top_p", str(args.hf_top_p),
                ] + (["--fewshot_json", args.fewshot_json] if args.fewshot_json else []),
            )
        )

    # 4) Skip-gram embeddings + neighbors
    steps.append(
        Step(
            name="OLAF: Skip-gram embeddings + neighbors",
            module="olaf.embeddings_skipgram",
            args=[
                "--db", args.db,
                "--sentences_table", args.lemmatized_table,
                "--term_candidates_table", args.term_candidates_table,
                "--neighbors_table", "skipgram_neighbors",
                "--cleaned_version", str(args.cleaned_version),
                "--min_tfidf", "10",
                "--top_k", "10",
                "--train",
            ],
        )
    )

    # 5) Parent terms (taxonomy scaffolding)
    steps.append(
        Step(
            name="OLAF: Parent head candidates (taxonomy scaffolding)",
            module="olaf.parent_terms",
            args=[
                "--db", args.db,
                "--enrichment_table", args.term_enrichment_table,
                "--out_table", "taxonomy_parent_candidates",
                "--max_examples_per_head", "15",
                "--min_head_len", "3",
                "--min_head_freq", "3",
            ],
        )
    )

    # 6) Taxonomy extraction
    steps.append(
        Step(
            name="OLAF: Taxonomy (head-based + typed parents)",
            module="olaf.taxonomy",
            args=[
                "--db", args.db,
                "--enrichment_table", args.term_enrichment_table,
                "--parent_candidates_table", "taxonomy_parent_candidates",
                "--out_table", "taxonomy_is_a",
                "--method", "head_parent_candidates_v2",
                "--clear_out",
                "--add_typed_parents",
            ],
        )
    )

    # Taxonomic part 2, LLM for validation
    if args.use_llm_taxonomy:
        steps.append(
            Step(
                name="LLM validate taxonomy (hard cases only)",
                module="olaf_llm.taxonomy_llm_validate",
                args=[
                    "--db", args.db,
                    "--taxonomy_table", "taxonomy_is_a",
                    "--enrichment_table", args.term_enrichment_table,  # whichever you used
                    "--parent_candidates_table", "taxonomy_parent_candidates",
                    "--chunks_table", args.chunks_table,               # likely contextual_chunk
                    "--out_table", args.taxonomy_validated_table,
                    "--model", args.llm_taxonomy_model,
                    "--evidence_k", "2",
                    "--global_heads_k", "10",
                ],
            )
        )

    # Non-taxonomic extraction
    steps.append(
        Step(
            name="Non-taxonomic extraction (OpenIE spaCy)",
            module="olaf.non_taxonomy",
            args=[
                "--db", args.db,
                "--sentence_table", args.segmented_table,          # or "sentence_segmented"
                "--term_candidates_table", args.term_candidates_table,
                "--term_enrichment_table", args.term_enrichment_table,  # default "term_enrichment"
                "--raw_edges_table", "non_taxonomic_edges",
                "--clean_edges_table", "non_taxonomic_edges_clean",
                "--stopwords", args.stopwords,
                "--spacy_model", args.spacy_model,
                "--method", "openie_spacy",
            ],
        )
    )

    # Non taxonomic LLM
    if args.use_llm_non_taxonomy:
        steps.append(
            Step(
                name="LLM validate + normalize non-tax edges",
                module="olaf.non_taxonomy_llm_extension",
                args=[
                    "--db", args.db,
                    "--in_edges_table", "non_taxonomic_edges_clean",
                    "--out_llm_table", "non_taxonomic_edges_llm",
                    "--model", args.llm_model,                 # reuse your HF model arg
                    "--device", "auto",
                    "--only_hard_cases",
                ],
            )
        )


    return steps


def build_olaf_llm_steps(args) -> List[Step]:
    return [
        Step("OLAF-LLM: Term Enrichment", "olaf_llm.term_enrichment_llm", ["--db", args.db]),
        Step("OLAF-LLM: Relation Induction", "olaf_llm.relations_llm", ["--db", args.db]),
        Step("OLAF-LLM: Axiom Generation", "olaf_llm.axioms_llm", ["--db", args.db]),
    ]

def apply_only_filter(steps, only: str):
    if not only:
        return steps
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    if not wanted:
        return steps

    def key(step):
        # module suffix, e.g. "olaf.term_extraction_tfidf" -> "term_extraction_tfidf"
        return step.module.split(".")[-1]

    kept = [st for st in steps if key(st) in wanted or st.name in wanted]
    if not kept:
        print(f"[WARN] --only matched nothing. Wanted={sorted(wanted)}. Available={[key(s) for s in steps]}")
    return kept


def remove_db(db_path: str) -> None:
    p = Path(db_path)
    if p.exists():
        print(f"[INFO] Removing DB: {p}")
        p.unlink()
    else:
        print(f"[INFO] DB not found, skip remove: {p}")

def ensure_gpu_if_needed(require_gpu: bool) -> None:
    """
    LLM steps often require GPU (HF Mistral).
    This checks CUDA availability and either warns or fails.
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if not has_cuda:
        msg = (
            "[WARN] CUDA GPU not available. HF Mistral step may be very slow or may OOM on CPU.\n"
            "       If you're on HPC, run inside a GPU allocation (srun/sbatch with --gres=gpu...)."
        )
        print(msg)
        if require_gpu:
            raise SystemExit("[ERROR] --require_gpu set but CUDA is not available. Exiting.")
    else:
        # Optional: print the GPU name
        try:
            import torch
            print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("[INFO] CUDA available.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Core
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--input", required=True, help="Folder containing .txt files")

    # Tables
    ap.add_argument("--raw_table", default="raw_documents")
    ap.add_argument("--cleaned_table", default="cleaned_documents")
    ap.add_argument("--segmented_table", default="sentence_segmented")
    ap.add_argument("--lemmatized_table", default="sentence_lemmatized")

    # Versions
    ap.add_argument("--raw_version", type=int, default=1)
    ap.add_argument("--cleaned_version", type=int, default=1)

    # Cleaning params
    ap.add_argument("--min_words", type=int, default=4)

    # spaCy params
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--max_length", type=int, default=1500000)

    # Lemmatization params
    ap.add_argument("--keep_pos", action="store_true")
    ap.add_argument("--remove_stopwords", action="store_true")
    ap.add_argument("--remove_punct", action="store_true")
    ap.add_argument("--batch_size", type=int, default=200)

    # Chunking params
    ap.add_argument("--chunks_table", default="contextual_chunk")
    ap.add_argument("--chunk_min_sentences", type=int, default=5)
    ap.add_argument("--chunk_max_sentences", type=int, default=12)
    ap.add_argument("--chunk_min_tokens", type=int, default=400)
    ap.add_argument("--chunk_max_tokens", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=1)

    # Runner options
    ap.add_argument("--clean_db", action="store_true", help="Delete DB before starting")
    ap.add_argument("--stop_on_fail", action="store_true", help="Stop at first failure")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument(
        "--only",
        default="",
        help="Run only specific steps by module suffix: data_ingest,data_clean,sentence_segment,sentence_lemmatize",
    )

    # Choose a which pipeline to continue with
    ap.add_argument(
        "--next",
        choices=["ask", "olaf", "olaf_llm", "stop"],
        default="ask",
        help="What to run after chunking/term extraction: ask | olaf | olaf_llm | stop",
    )

    #Term extraction with TF_IDF
    ap.add_argument("--term_candidates_table", default="term_candidates")
    ap.add_argument("--term_occurrences_table", default="term_occurrences")
    ap.add_argument("--stopwords", default="stop_word/stop_words.txt")
    ap.add_argument("--max_tfidf_tokens", type=int, default=3)
    ap.add_argument("--reset_terms", action="store_true", help="Clear term tables before term extraction")

    # Term enrichment (part-1)
    ap.add_argument("--term_enrichment_table", default="term_enrichment")
    ap.add_argument("--min_tf_idf", type=float, default=5.0)

    # Term enrichment with LLM (part-2)
    ap.add_argument("--use_llm_enrich", action="store_true",
                help="Run HF Mistral term enrichment extension into term_enrichment_exten")

    ap.add_argument("--require_gpu", action="store_true",
                    help="Fail if CUDA GPU is not available when running LLM steps")

    ap.add_argument("--term_enrichment_ext_table", default="term_enrichment_exten",
                    help="Destination table for LLM-enriched term enrichment output")

    # HF model config
    ap.add_argument("--hf_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--hf_dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--hf_device", default="auto",
                    help="transformers device_map: auto|cuda|cpu")

    ap.add_argument("--hf_max_new_tokens", type=int, default=350)
    ap.add_argument("--hf_temperature", type=float, default=0.0)
    ap.add_argument("--hf_top_p", type=float, default=1.0)

    # few-shot + selection knobs
    ap.add_argument("--fewshot_json", default=None,
                    help="Path to few-shot examples JSON file")
    ap.add_argument("--hardcase_p_low", type=float, default=60.0)
    ap.add_argument("--hardcase_p_high", type=float, default=85.0)
    ap.add_argument("--max_terms_llm", type=int, default=200)
    ap.add_argument("--max_evidence_sents", type=int, default=3)

    # taxonomic LLM validation
    ap.add_argument("--use_llm_taxonomy", action="store_true",
                help="Run LLM validation/rerank for taxonomy hard-cases (GPU recommended)")
    ap.add_argument("--llm_taxonomy_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--taxonomy_validated_table", default="taxonomy_is_a_validated")
    
    # Non-taxonomy (OpenIE) step
    parser.add_argument("--non_tax_sentence_table", default="sentence_segmented")
    parser.add_argument("--non_tax_raw_table", default="non_taxonomic_edges")
    parser.add_argument("--non_tax_clean_table", default="non_taxonomic_edges_clean")
    parser.add_argument("--non_tax_spacy_model", default="en_core_web_sm")
    parser.add_argument("--non_tax_method", default="openie_spacy")

    ap.add_argument("--use_llm_non_taxonomy", action="store_true",
                    help="Run LLM normalization/validation for non-taxonomic edges")
    ap.add_argument("--llm_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--llm_device", default="auto")
    ap.add_argument("--non_tax_llm_out_table", default="non_taxonomic_edges_llm")
    ap.add_argument("--non_tax_llm_in_table", default="non_taxonomic_edges_clean")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure we run from repo root (so python -m works reliably)
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    if args.clean_db:
        remove_db(args.db)

    steps = build_steps(args)
    if not steps:
        print("[INFO] No steps selected.")
        return

    failures = 0
    for step in steps:
        rc = run_step(step, python_exe=args.python, stop_on_fail=args.stop_on_fail)
        failures += int(rc != 0)

    print("\n" + "-" * 90)
    if failures == 0:
        print("[DONE] Preprocessing pipeline finished successfully.")
    else:
        print(f"[DONE] Preprocessing pipeline finished with {failures} failing step(s).")
    print("-" * 90)

    # Decide what to do next (after chunking / term extraction)
    next_stage = choose_next_pipeline(args.next)

    if next_stage == "stop":
        print("\n[INFO] Stopping after preprocessing/chunking/term extraction.")
        return

    if next_stage == "olaf":
        print("\n[INFO] Running OLAF pipeline...")
        for step in build_olaf_steps(args):
            run_step(step, python_exe=args.python, stop_on_fail=args.stop_on_fail)

    elif next_stage == "olaf_llm":
        print("\n[INFO] Running OLAF + LLM pipeline...")
        for step in build_olaf_llm_steps(args):
            run_step(step, python_exe=args.python, stop_on_fail=args.stop_on_fail)
    
    steps = build_steps(args)
    steps = build_olaf_steps(args)
    steps = build_olaf_llm_steps(args)
    
    steps = apply_only_filter(steps, args.only)
    run_steps(steps)

    

if __name__ == "__main__":
    main()
