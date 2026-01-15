import argparse
import re
import sqlite3
from typing import Dict, List

import spacy
from spacy.language import Language

from .db_utils import init_db, utc_now
from . import patterns as pat


class ImprovedSentenceSegmenter:
    def __init__(self, model_name: str, max_length: int):
        self.nlp = spacy.load(model_name)
        self.nlp.max_length = max_length

        # Keep parser for sentence boundaries; disable others for speed.
        disable_pipes = ["ner", "lemmatizer", "textcat"]
        for pipe in disable_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipes(pipe)

        self._add_custom_sentencizer_rules()

    def _add_custom_sentencizer_rules(self):
        abbreviations = pat.ABBREVIATIONS
        non_boundary_patterns = pat.NON_BOUNDARY

        def custom_sentencizer(doc):
            for i, token in enumerate(doc[:-1]):
                if token.text in ".!?":
                    prev_text = doc[max(0, i - 5): i + 1].text.lower()
                    next_token = doc[i + 1]

                    is_abbrev = any(abbrev in prev_text for abbrev in abbreviations)
                    context = doc[max(0, i - 2): min(len(doc), i + 3)].text
                    is_non_boundary = any(
                        re.search(pattern, context, re.IGNORECASE)
                        for pattern in non_boundary_patterns
                    )

                    if not is_abbrev and not is_non_boundary and next_token.is_alpha:
                        next_token.is_sent_start = next_token.text[0].isupper()
                    else:
                        next_token.is_sent_start = False
            return doc

        if not Language.has_factory("custom_sentencizer"):
            Language.component("custom_sentencizer", func=custom_sentencizer)

        if "custom_sentencizer" not in self.nlp.pipe_names:
            try:
                self.nlp.add_pipe("custom_sentencizer", before="parser")
            except ValueError:
                self.nlp.add_pipe("custom_sentencizer", first=True)

    def _is_valid_sentence(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if len(text) < 10:
            if not (text[0].isupper() and text[-1] in ".!?"):
                return False
        if len(text) > 800:
            return False
        for pattern in pat.FRAGMENT_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        if not re.search(r"[a-zA-Z]", text):
            return False
        words = text.split()
        if len(words) == 1 and len(text) < 20:
            if not (text[-1] in ".!?" or len(text) > 5):
                return False
        return True

    def segment_text(self, text: str, doc_id: str) -> List[Dict]:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        doc = self.nlp(text)
        sentences = []
        sent_counter = 0

        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue
            if sentence_text.endswith(":") and len(sentence_text.split()) < 10:
                continue
            if not self._is_valid_sentence(sentence_text):
                continue

            sent_counter += 1
            sentences.append(
                {
                    "doc_id": doc_id,
                    "sent_idx": sent_counter,
                    "sentence": sentence_text,
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                    "length": len(sentence_text),
                }
            )
        return sentences


def segment_cleaned_to_db(
    db_path: str,
    cleaned_table: str,
    segmented_table: str,
    cleaned_version: int,
    spacy_model: str,
    max_length: int,
) -> int:
    init_db(db_path, cleaned_table=cleaned_table, segmented_table=segmented_table)
    seg = ImprovedSentenceSegmenter(model_name=spacy_model, max_length=max_length)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        f"""
        SELECT cd.doc_id, cd.cleaned_text
        FROM {cleaned_table} cd
        WHERE cd.cleaned_version = ?
          AND NOT EXISTS (
              SELECT 1 FROM {segmented_table} s
              WHERE s.doc_id = cd.doc_id
                AND s.cleaned_version = cd.cleaned_version
          )
        """,
        (cleaned_version,),
    )
    rows = cur.fetchall()

    if not rows:
        print(f"No cleaned docs to segment for cleaned_version={cleaned_version}.")
        conn.close()
        return 0

    now = utc_now()
    inserted_total = 0
    print(f"Segmenting {len(rows)} cleaned document(s)...")

    for doc_id, cleaned_text in rows:
        sents = seg.segment_text(cleaned_text, doc_id)
        print(f"  {doc_id}: {len(sents)} sentences")

        for idx, s in enumerate(sents, start=1):
            sentence_id = f"{doc_id}_sent_{idx:05d}"
            cur.execute(
                f"""
                INSERT INTO {segmented_table}
                    (sentence_id, doc_id, sent_idx, sentence, start_char, end_char,
                     length, cleaned_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sentence_id,
                    s["doc_id"],
                    s["sent_idx"],
                    s["sentence"],
                    s["start_char"],
                    s["end_char"],
                    s["length"],
                    cleaned_version,
                    now,
                ),
            )
        inserted_total += len(sents)

    conn.commit()
    conn.close()
    print(f"Inserted {inserted_total} rows into {segmented_table} (cleaned_version={cleaned_version})")
    return inserted_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cleaned_table", default="cleaned_documents")
    ap.add_argument("--segmented_table", default="sentence_segmented")
    ap.add_argument("--cleaned_version", type=int, default=1)
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--max_length", type=int, default=5_000_000)
    args = ap.parse_args()

    segment_cleaned_to_db(
        db_path=args.db,
        cleaned_table=args.cleaned_table,
        segmented_table=args.segmented_table,
        cleaned_version=args.cleaned_version,
        spacy_model=args.spacy_model,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
