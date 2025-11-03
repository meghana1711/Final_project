import os
import re
import sys
from typing import List, Tuple, Any
from PyPDF2 import PdfReader

OUTPUT_DIR = "ibm_lsf/lsf_text"

def sanitize_title_for_filename(title: str, index: int) -> str:
    """
    Make a safe filename like '01_IBM_Spectrum_LSF_V10_1_0_documentation'
    """
    safe = re.sub(r"\s+", "_", title.strip())
    safe = re.sub(r"[^A-Za-z0-9_.-]", "", safe)
    safe = f"{index:02d}_{safe}"
    if len(safe) > 80:
        safe = safe[:80]
    return safe or f"{index:02d}_chapter"


def get_pdf_outlines(reader: PdfReader) -> List[Any]:
    """
    Try to fetch the PDF outline / bookmarks from PyPDF2,
    working across versions.
    """
    if hasattr(reader, "outline"):
        return reader.outline
    if hasattr(reader, "outlines"):
        return reader.outlines
    if hasattr(reader, "getOutlines"):
        try:
            return reader.getOutlines()
        except Exception:
            pass
    return []


def flatten_outline_items(reader: PdfReader, outline_list: List[Any], depth_limit=1, _depth=0):
    """
    Walk the outline tree and collect (title, page_index, depth).
    Keep only entries up to depth_limit (so we don't explode into tiny subheads).

    Returns a list of dicts:
    { "title": str, "page_index": int, "depth": int }
    """
    results = []

    for item in outline_list:
        # item might be a nested list (children) or a bookmark/destination
        if isinstance(item, list):
            # recurse into child list
            results.extend(flatten_outline_items(reader, item, depth_limit, _depth + 1))
        else:
            try:
                title = getattr(item, "title", "").strip()
            except Exception:
                title = str(item)

            try:
                page_index = reader.get_destination_page_number(item)
            except Exception:
                # couldn't resolve page number for this outline item
                continue

            if _depth <= depth_limit:
                results.append({
                    "title": title,
                    "page_index": page_index,  # 0-based
                    "depth": _depth,
                })

    return results


def dedupe_same_start_page(entries: List[dict]) -> List[dict]:
    """
    If multiple outline entries claim the SAME starting page,
    we only keep the LAST one for that page.
    """
    # sort by page_index ascending but keep stable order for ties
    sorted_entries = sorted(entries, key=lambda e: e["page_index"])

    deduped = []
    i = 0
    n = len(sorted_entries)
    while i < n:
        same_page_group = [sorted_entries[i]]
        j = i + 1
        while j < n and sorted_entries[j]["page_index"] == sorted_entries[i]["page_index"]:
            same_page_group.append(sorted_entries[j])
            j += 1
        # keep ONLY the last one in this same-page group
        deduped.append(same_page_group[-1])
        i = j

    return deduped


def build_chapter_ranges(toc_entries: List[dict], total_pages: int) -> List[Tuple[str, int, int]]:
    """
    Convert cleaned TOC entries into [(title, start_page, end_page), ...].

    start_page and end_page are 0-based and inclusive.
    """
    # toc_entries should ALREADY be deduped and sorted by page_index ascending.
    toc_entries = sorted(toc_entries, key=lambda e: e["page_index"])

    chapter_ranges = []
    for i, entry in enumerate(toc_entries):
        start_page = entry["page_index"]
        title = entry["title"] or f"Section_{i+1}"

        if i + 1 < len(toc_entries):
            next_start = toc_entries[i + 1]["page_index"]
            end_page = next_start - 1
        else:
            end_page = total_pages - 1  # last one goes to end of PDF

        # Safety: avoid negative
        if end_page < start_page:
            end_page = start_page

        chapter_ranges.append((title, start_page, end_page))

    return chapter_ranges


def extract_text_for_page_range(reader: PdfReader, start_page: int, end_page: int) -> str:
    """
    Pull text from start_page..end_page (inclusive).
    (We are removing the '=== [PAGE X] ===' markers from output.)
    """
    chunks = []
    for p in range(start_page, end_page + 1):
        page_obj = reader.pages[p]
        try:
            text = page_obj.extract_text() or ""
        except Exception as e:
            text = f"[WARN: could not extract text on page {p+1}: {e}]"

        # ORIGINAL (adds page marker lines like '=== [PAGE 47] ==='):
        # chunks.append(f"\n\n=== [PAGE {p+1}] ===\n\n{text}")  # <-- commented out

        # NEW: just add the page text itself, no banner
        chunks.append("\n\n" + text)

    return "".join(chunks).strip()


def write_chapter_files(reader: PdfReader, chapter_ranges: List[Tuple[str, int, int]]):
    """
    Save each chapter range to text/<NN>_<chapter_title>.txt
    We're removing:
      - "# {title}"
      - "Pages X–Y"
    from the top of each file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, (title, start_page, end_page) in enumerate(chapter_ranges, start=1):
        filename_base = sanitize_title_for_filename(title, idx)
        out_path = os.path.join(OUTPUT_DIR, filename_base + ".txt")

        print(f"[info] extracting '{title}' pages {start_page+1}–{end_page+1} -> {out_path}")
        body_text = extract_text_for_page_range(reader, start_page, end_page)

        # ORIGINAL header that added:
        #   "# {title}"
        #   "Pages {start_page+1}–{end_page+1}"
        # header_lines = [
        #     f"# {title}",                            # <-- chapter title at top of file
        #     f"Pages {start_page+1}–{end_page+1}",    # <-- 'Pages 47–47' line
        #     "",
        # ]

        # ORIGINAL combination:
        # full_text = "\n".join(header_lines) + body_text

        # NEW: just use the extracted body text, without header/title/page range
        full_text = body_text

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"[ok] wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_split_by_outline.py <input.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # 1. Get the PDF's outline/bookmarks.
    raw_outline = get_pdf_outlines(reader)
    if not raw_outline:
        print("[error] This PDF has no outline/bookmarks. "
              "You may need to manually define page ranges instead.")
        sys.exit(1)

    # 2. Flatten to (title, page_index, depth) up to depth_limit.
    flat_outline = flatten_outline_items(reader, raw_outline, depth_limit=1)
    if not flat_outline:
        print("[error] Could not flatten outline into entries. "
              "Try raising depth_limit in flatten_outline_items.")
        sys.exit(1)

    # 3. Drop repeated starting pages, keep only the last one.
    cleaned_outline = dedupe_same_start_page(flat_outline)

    # 4. Build (title, start_page, end_page) using that cleaned list.
    chapter_ranges = build_chapter_ranges(cleaned_outline, total_pages)

    # 5. Write each chapter into ./text/
    write_chapter_files(reader, chapter_ranges)

    print("[done]")


if __name__ == "__main__":
    main()

