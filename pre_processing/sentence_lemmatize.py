import argparse
import json
import sqlite3

import spacy

from .db_utils import init_db, utc_now


class SentenceLemmatizer:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

        # Need tokenizer + tagger + lemmatizer; disable NER & parser
        disable_pipes = ["ner", "parser"]
        for pipe in disable_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipes(pipe)

    def preserve_original_case(self, original_token: str, lemma: str) -> str:
        if not original_token or not lemma:
            return lemma
        if original_token.isupper():
            return lemma.upper()
        if original_token[0].isupper():
            return lemma.capitalize()
        if any(c.isupper() for c in original_token[1:]):
            if len(original_token) == len(lemma):
                out = ""
                for i, ch in enumerate(lemma):
                    out += ch.upper() if i < len(original_token) and original_token[i].isupper() else ch.lower()
                return out
            return lemma.capitalize() if original_token[0].isupper() else lemma
        return lemma

    def lemmatize_sentence(self, sentence: str, keep_pos: bool, remove_stopwords: bool, remove_punct: bool):
        doc = self.nlp(sentence)

        tokens, lemmas, lemmas_with_case, pos_tags = [], [], [], []

        for token in doc:
            if remove_stopwords and token.is_stop:
                continue
            if remove_punct and token.is_punct:
                continue

            tokens.append(token.text)
            lemmas.append(token.lemma_)
            lemmas_with_case.append(self.preserve_original_case(token.text, token.lemma_))
            if keep_pos:
                pos_tags.append(token.pos_)

        result = {
            "tokens": tokens,
            "lemmas": lemmas,
            "lemmas_with_case": lemmas_with_case,
            "lemmatized_text": " ".join(lemmas),
            "lemmatized_text_with_case": " ".join(lemmas_with_case),
        }
        if keep_pos:
            result["pos_tags"] = pos_tags
        return result

    def process_sentences_db(
        self,
        db_path: str,
        segmented_table: str,
        lemmatized_table: str,
        cleaned_version: int,
        keep_pos: bool,
        remove_stopwords: bool,
        remove_punct: bool,
        batch_size: int,
    ) -> int:
        init_db(db_path, segmented_table=segmented_table, lemmatized_table=lemmatized_table)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        cur.execute(
            f"""
            SELECT s.doc_id, s.sent_idx, s.sentence
            FROM {segmented_table} s
            WHERE s.cleaned_version = ?
              AND NOT EXISTS (
                  SELECT 1 FROM {lemmatized_table} l
                  WHERE l.doc_id = s.doc_id
                    AND l.sent_idx = s.sent_idx
                    AND l.cleaned_version = s.cleaned_version
              )
            ORDER BY s.doc_id, s.sent_idx
            """,
            (cleaned_version,),
        )
        rows = cur.fetchall()

        if not rows:
            print(f"No sentences to lemmatize for cleaned_version={cleaned_version}.")
            conn.close()
            return 0

        print(f"Lemmatizing {len(rows)} sentences (cleaned_version={cleaned_version})...")
        now = utc_now()
        total = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            for doc_id, sent_idx, sentence_text in batch:
                lemma_result = self.lemmatize_sentence(
                    sentence_text,
                    keep_pos=keep_pos,
                    remove_stopwords=remove_stopwords,
                    remove_punct=remove_punct,
                )

                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO {lemmatized_table}
                        (doc_id, sent_idx, sentence,
                         tokens_json, lemmas_json, lemmas_with_case_json,
                         lemmatized_text, lemmatized_text_with_case,
                         pos_tags_json, cleaned_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        sent_idx,
                        sentence_text,
                        json.dumps(lemma_result["tokens"], ensure_ascii=False),
                        json.dumps(lemma_result["lemmas"], ensure_ascii=False),
                        json.dumps(lemma_result["lemmas_with_case"], ensure_ascii=False),
                        lemma_result["lemmatized_text"],
                        lemma_result["lemmatized_text_with_case"],
                        json.dumps(lemma_result.get("pos_tags"), ensure_ascii=False) if keep_pos else None,
                        cleaned_version,
                        now,
                    ),
                )
                total += 1

            conn.commit()
            print(f"  Processed {min(i + batch_size, len(rows))}/{len(rows)}...")

        conn.close()
        print(f"Updated {total} rows into {lemmatized_table} (cleaned_version={cleaned_version})")
        return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--segmented_table", default="sentence_segmented")
    ap.add_argument("--lemmatized_table", default="sentence_lemmatized")
    ap.add_argument("--cleaned_version", type=int, default=1)
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--keep_pos", action="store_true")
    ap.add_argument("--remove_stopwords", action="store_true")
    ap.add_argument("--remove_punct", action="store_true")
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()

    lemm = SentenceLemmatizer(model_name=args.spacy_model)
    lemm.process_sentences_db(
        db_path=args.db,
        segmented_table=args.segmented_table,
        lemmatized_table=args.lemmatized_table,
        cleaned_version=args.cleaned_version,
        keep_pos=args.keep_pos,
        remove_stopwords=args.remove_stopwords,
        remove_punct=args.remove_punct,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
