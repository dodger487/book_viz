#!/usr/bin/env python3
"""
Script 2: cached raw API data -> merged metadata -> embedding input -> embeddings.

Reads the raw per-book API responses that Script 1 cached under cache/
(no network calls here -- run scripts/01_fetch_metadata.py first), extracts
and merges each book's description/genre/publication year/author bio, then
builds one embedding-input text per book and embeds it via a pluggable
provider.

Why extraction lives here and not in Script 1: which raw field to prefer
(e.g. Open Library's first_publish_year vs. Google Books' edition-specific
publishedDate) is a judgment call that changes as we learn more about the
data, whereas fetching is slow/rate-limited and shouldn't need to change
alongside it. Since every raw response is already cached, re-running this
script to pick up a merge-logic change costs nothing over the network.

Output:
  data/book_metadata.json      - merged per-book record (for inspection/QA)
  data/embedding_input.json    - the constructed text per book (for inspection)
  data/embeddings_<provider>.npz - slugs[]/embeddings[]/provider/model, one
                                    file per provider so running a different
                                    --provider never overwrites another's
                                    embeddings. Script 3 auto-discovers every
                                    embeddings_*.npz file and lets you switch
                                    between them in the interactive plot.

Usage:
    python scripts/02_generate_embeddings.py                         # local, free, default model
    python scripts/02_generate_embeddings.py --model all-MiniLM-L6-v2  # smaller/faster local model
    python scripts/02_generate_embeddings.py --provider openai --model text-embedding-3-large
    python scripts/02_generate_embeddings.py --provider voyage --model voyage-3-large
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
BOOKS_CSV = DATA_DIR / "books.csv"
OVERRIDES_JSON = DATA_DIR / "overrides.json"
METADATA_JSON = DATA_DIR / "book_metadata.json"
EMBEDDING_INPUT_JSON = DATA_DIR / "embedding_input.json"


def embeddings_path(provider: str) -> Path:
    """Each provider gets its own file (data/embeddings_<provider>.npz) so
    generating one doesn't overwrite another -- Script 3 auto-discovers
    every embeddings_*.npz file and lets you compare them in the UI."""
    return DATA_DIR / f"embeddings_{provider}.npz"

GOOGLE_BOOKS_CACHE = CACHE_DIR / "google_books"
OPEN_LIBRARY_CACHE = CACHE_DIR / "open_library"
WIKIPEDIA_CACHE = CACHE_DIR / "wikipedia"


# --- Metadata extraction (reads Script 1's cache, no network) -----------


def slugify(text: str) -> str:
    """Turn 'Title by Author' into a safe filename stem. Must match Script 1's."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s]+", "_", text)
    return text[:120]


def load_books(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [
        {"title": r["Name"].strip(), "author": r["Author"].strip(), "date_read": r["Date"].strip()}
        for r in rows
    ]


def load_overrides() -> dict:
    if OVERRIDES_JSON.exists():
        with open(OVERRIDES_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def override_key(title: str, author: str) -> str:
    return f"{title}|{author}"


def load_cached(cache_path: Path) -> dict | None:
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def extract_google_books_fields(raw: dict | None) -> dict:
    out = {"description": None, "categories": None, "published_date": None, "match_title": None}
    if not raw:
        return out
    items = raw.get("items") if "items" in raw else ([raw] if "volumeInfo" in raw else None)
    if not items:
        return out
    info = items[0].get("volumeInfo", {})
    out["description"] = info.get("description")
    out["categories"] = info.get("categories")
    out["published_date"] = info.get("publishedDate")
    out["match_title"] = info.get("title")
    return out


def extract_open_library_fields(slug: str) -> dict:
    """Reads both cache/open_library/<slug>.json (search result) and
    <slug>_work.json (work detail, fetched by Script 1 for any search
    match) and merges them -- the work detail has the real description
    and the rich `subjects` list; the search result has first_publish_year."""
    out = {"description": None, "subjects": None, "first_publish_year": None, "match_title": None}
    raw = load_cached(OPEN_LIBRARY_CACHE / f"{slug}.json")
    if not raw:
        return out
    doc = raw["docs"][0] if raw.get("docs") else (raw if "title" in raw else None)
    if not doc:
        return out

    work = load_cached(OPEN_LIBRARY_CACHE / f"{slug}_work.json")
    merged = {**doc, **(work or {})}

    desc = merged.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")
    out["description"] = desc
    # `subjects` (plural) comes from the work-detail endpoint and is much
    # richer than `subject` (singular), which is what search.json returns
    # and is almost always empty.
    out["subjects"] = merged.get("subjects") or merged.get("subject")
    out["first_publish_year"] = merged.get("first_publish_year")
    out["match_title"] = merged.get("title")
    return out


def extract_wikipedia_fields(raw: dict | None) -> dict:
    out = {"bio": None, "matched_page": None}
    if not raw or raw.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
        return out
    out["bio"] = raw.get("extract")
    out["matched_page"] = raw.get("title")
    return out


# Google Books' categories field is a single flat top-level BISAC label for
# essentially every edition in this dataset (verified: 0 of 160 categorized
# books had more than one category, none had a hierarchical/subgenre path).
# For non-fiction that's usually fine ("Business & Economics", "History"),
# but 52% of books just come back "Fiction" with no subgenre. Open Library's
# work-level `subjects` list is noisy (award names, series titles, reading
# levels) but does contain real subgenre signal, so we use it to refine the
# generic "Fiction" bucket specifically, rather than overriding Google
# Books' category everywhere.
GENERIC_GENRE_LABELS = {"fiction", "juvenile fiction", "young adult fiction"}
FICTION_GENRE_KEYWORDS = [
    "science fiction",
    "fantasy",
    "mystery",
    "thriller",
    "romance",
    "historical fiction",
    "horror",
    "literary fiction",
    "true crime",
    "dystopia",
    "satire",
    "poetry",
    "short stories",
    "graphic novel",
    "espionage",
    "detective",
    "crime fiction",
    "noir",
    "space opera",
    "coming of age",
    "family saga",
]


def extract_genre(gb_categories: list | None, ol_subjects: list | None) -> str:
    gb_genre = gb_categories[0] if gb_categories else None
    if gb_genre and gb_genre.lower() not in GENERIC_GENRE_LABELS:
        return gb_genre

    lowered = [s.lower() for s in (ol_subjects or [])]
    tagged = [s.split(":", 1)[1].strip().title() for s in (ol_subjects or []) if s.lower().startswith("genre:")]
    if tagged:
        return tagged[0]
    for kw in FICTION_GENRE_KEYWORDS:
        if any(kw in s for s in lowered):
            return kw.title()

    return gb_genre or "Unknown"


def build_metadata() -> list[dict]:
    books = load_books(BOOKS_CSV)
    overrides = load_overrides()
    records, flagged = [], []

    for book in books:
        title, author = book["title"], book["author"]
        slug = slugify(f"{title}_{author}")
        override = overrides.get(override_key(title, author), {})

        gb = extract_google_books_fields(load_cached(GOOGLE_BOOKS_CACHE / f"{slug}.json"))
        ol = extract_open_library_fields(slug)
        wiki_page = slugify(override["wikipedia_title"]) if "wikipedia_title" in override else slugify(author)
        wiki = extract_wikipedia_fields(load_cached(WIKIPEDIA_CACHE / f"{wiki_page}.json"))

        description = gb["description"] or ol["description"]
        genre = extract_genre(gb["categories"], ol["subjects"])
        # Prefer Open Library's first_publish_year (meant to represent the
        # work's original publication) over Google Books' publishedDate,
        # which reflects whatever specific edition our search matched --
        # often a recent reprint, e.g. Sapiens matched a "Tenth Anniversary
        # Edition" dated 2025 instead of the 2011 original.
        year = ol["first_publish_year"] or gb["published_date"]

        record = {
            "title": title,
            "author": author,
            "date_read": book["date_read"],
            "description": description,
            "genre": genre,
            "categories": gb["categories"],
            "published_year": year,
            "author_bio": wiki["bio"],
            "sources": {
                "google_books_matched_title": gb["match_title"],
                "open_library_matched_title": ol["match_title"],
                "wikipedia_matched_page": wiki["matched_page"],
            },
            "slug": slug,
        }
        records.append(record)

        if not description:
            flagged.append((title, author, "no description found"))
        if not wiki["bio"]:
            flagged.append((title, author, "no author bio found"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} merged records to {METADATA_JSON}")
    if flagged:
        print(f"\n{len(flagged)} issue(s) to review (consider adding to {OVERRIDES_JSON.name}):")
        for title, author, issue in flagged:
            print(f"  - {title} ({author}): {issue}")
        print()

    return records


# --- Embedding input text -------------------------------------------------


def build_text(book: dict) -> str:
    """Combine synopsis + structured tags into one embedding-input string."""
    parts = [f"{book['title']} by {book['author']}."]
    if book.get("published_year"):
        parts.append(f"Published {book['published_year']}.")
    if book.get("genre"):
        parts.append(f"Genre/category: {book['genre']}.")
    if book.get("description"):
        parts.append(book["description"])
    if book.get("author_bio"):
        parts.append(f"About the author: {book['author_bio']}")
    return " ".join(parts)


# --- Embedding providers -----------------------------------------------
# Every provider implements .embed(texts) -> np.ndarray of shape (n, dim).
# Swap providers with --provider; add a new one by subclassing and
# registering it in PROVIDERS below.


class EmbeddingProvider:
    default_model: str

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LocalProvider(EmbeddingProvider):
    """Free, local, no API key. Downloads the model once via sentence-transformers."""

    default_model = "all-mpnet-base-v2"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        )


class OpenAIProvider(EmbeddingProvider):
    default_model = "text-embedding-3-large"

    def __init__(self, model_name: str):
        import os

        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        return np.asarray(vectors)


class VoyageProvider(EmbeddingProvider):
    default_model = "voyage-3-large"

    def __init__(self, model_name: str):
        import os

        import voyageai

        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embed(batch, model=self.model_name, input_type="document")
            vectors.extend(resp.embeddings)
        return np.asarray(vectors)


PROVIDERS = {"local": LocalProvider, "openai": OpenAIProvider, "voyage": VoyageProvider}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS, default="local")
    parser.add_argument("--model", default=None, help="override the provider's default model name")
    args = parser.parse_args()

    print("Extracting/merging metadata from cache...")
    books = build_metadata()

    print(f"Building embedding input text for {len(books)} books...")
    records = [
        {"slug": b["slug"], "title": b["title"], "author": b["author"], "text": build_text(b)}
        for b in books
    ]
    EMBEDDING_INPUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote embedding input text to {EMBEDDING_INPUT_JSON}")

    provider_cls = PROVIDERS[args.provider]
    model_name = args.model or provider_cls.default_model
    provider = provider_cls(model_name)

    print(f"Embedding {len(records)} texts with provider={args.provider}, model={model_name}...")
    embeddings = provider.embed([r["text"] for r in records])
    print(f"Got embeddings with shape {embeddings.shape}")

    slugs = np.array([r["slug"] for r in records])
    out_path = embeddings_path(args.provider)
    np.savez(out_path, slugs=slugs, embeddings=embeddings, provider=args.provider, model=model_name)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
