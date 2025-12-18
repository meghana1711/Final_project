import os
import json
from pathlib import Path
from typing import List, Dict
import re

def normalize_whitespace(s: str) -> str:
    """
    Remove weird Unicode line/paragraph separators, non-breaking spaces,
    and collapse funky spacing so VS Code / downstream tools are happy.
    """
    if s is None:
        return ""

    # Turn Unicode line/paragraph separators into normal newlines
    s = s.replace("\u2028", "\n")  # LINE SEPARATOR
    s = s.replace("\u2029", "\n")  # PARAGRAPH SEPARATOR

    # Non-breaking space -> normal space
    s = s.replace("\u00A0", " ")

    # Replace any "odd" whitespace chars (not \n \r \t) with a normal space
    # [^\S\n\r\t] = whitespace that is NOT newline/carriage-return/tab
    s = re.sub(r"[^\S\n\r\t]", " ", s)

    # Collapse runs of spaces/tabs down to a single space
    s = re.sub(r"[ \t]+", " ", s)

    # Strip trailing spaces at end of each line
    s = "\n".join(line.rstrip() for line in s.splitlines())

    return s


def read_data(folder_path: str, output_file: str) -> List[Dict[str, str]]:
    """
    Read all .txt files in folder_path, normalize their content,
    build a list of {doc_id, title, text}, and write that list to output_file.
    """
    documents: List[Dict[str, str]] = []
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        print(f"Warning: No .txt files found in {folder_path}")
        return documents

    print(f"Found {len(txt_files)} .txt files")

    for idx, file_path in enumerate(txt_files, start=1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # normalize BEFORE storing
            cleaned_text = normalize_whitespace(raw_text).strip()

            if not cleaned_text:
                print(f"Skipping empty (after-clean) file: {file_path.name}")
                continue

            doc = {
                "doc_id": f"doc_{idx:04d}",
                "title": file_path.stem,
                "text": cleaned_text,
            }
            documents.append(doc)

            print(f"Processed: {idx:02d} {file_path.name} ({len(cleaned_text)} chars)")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    # extra safety pass: normalize again globally before dump
    for d in documents:
        d["text"] = normalize_whitespace(d["text"])

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(documents)} documents to {output_file}")
    return documents


if __name__ == "__main__":
    # Example usage
    FOLDER_PATH = "C:/Users/20236193/Final_project/data/ibm_lsf/lsf_text"  # Change this to your folder path
    OUTPUT_FILE = "documents_new.json"
    
    # Ingest documents
    docs = read_data(FOLDER_PATH, OUTPUT_FILE)
    
