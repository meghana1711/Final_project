import urllib.request
import re
import urllib.request
from html.parser import HTMLParser

URL = "https://www.shoutmeloud.com/seo-stop-words"
OUTPUT_TXT = "other/stop_words.txt"

def fetch_html(url: str) -> str:
    # Pretend to be a normal web browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="ignore")

class LiTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_li = False
        self.current_text = []
        self.items = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "li":
            self.in_li = True
            self.current_text = []

    def handle_endtag(self, tag):
        if tag.lower() == "li" and self.in_li:
            text = "".join(self.current_text).strip()
            if text:
                self.items.append(text)
            self.in_li = False

    def handle_data(self, data):
        if self.in_li:
            self.current_text.append(data)


def fetch_html(url: str) -> str:
    # Add a User-Agent so the site doesn't block us as a bot
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def extract_stopwords_from_html(html: str):
    parser = LiTextExtractor()
    parser.feed(html)

    words = set()

    for text in parser.items:
        # Only keep single words
        if " " in text:
            continue

        w = text.strip("•-–—:,;()[]{}").strip().lower()
        if not w:
            continue

        # Simple alpha-ish check (allow "don't" style apostrophes)
        if not re.match(r"^[a-zA-Z']+$", w):
            continue

        words.add(w)

    return sorted(words)


def save_to_txt(words, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")


def main():
    print(f"Downloading {URL} ...")
    html = fetch_html(URL)
    print("Extracting stop words...")
    words = extract_stopwords_from_html(html)
    print(f"Found {len(words)} words.")
    print(f"Saving to {OUTPUT_TXT} ...")
    save_to_txt(words, OUTPUT_TXT)
    print("Done. Words are extracted to a text file.")


if __name__ == "__main__":
    main()

