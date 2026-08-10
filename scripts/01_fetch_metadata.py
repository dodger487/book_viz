#!/usr/bin/env python3
"""
Script 1: book list -> cached raw metadata + combined embedding-input file.

Reads data/books.csv (columns: Date, Type, Name, Author), fetches:
  - Google Books API (primary): description, categories, publishedDate
  - Open Library API (fallback): description, subjects, first_publish_year
  - Wikipedia API: author bio (lead paragraph)

Raw API responses are cached to disk per-book so re-runs are free and
idempotent. A manual overrides file lets you pin the correct edition/
author when auto-matching picks the wrong one.

Output: data/book_metadata.json — one combined record per book, ready
to be turned into embedding-input text in Script 2.

Usage:
    python 01_fetch_metadata.py                # fetch all, skip cached
    python 01_fetch_metadata.py --refresh       # re-fetch everything
    python 01_fetch_metadata.py --limit 5       # test on first 5 rows
"""
import argparse
import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
BOOKS_CSV = DATA_DIR / "books.csv"
OVERRIDES_JSON = DATA_DIR / "overrides.json"
OUTPUT_JSON = DATA_DIR / "book_metadata.json"

GOOGLE_BOOKS_CACHE = CACHE_DIR / "google_books"
OPEN_LIBRARY_CACHE = CACHE_DIR / "open_library"
WIKIPEDIA_CACHE = CACHE_DIR / "wikipedia"

USER_AGENT = "book-embedding-viz/0.1 (personal project; contact: n/a)"
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 0.3  # be polite to free/keyless APIs
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY")


def slugify(text: str) -> str:
    """Turn 'Title by Author' into a safe filename stem."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s]+", "_", text)
    return text[:120]


def load_books(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    books = []
    for r in rows:
        books.append(
            {
                "title": r["Name"].strip(),
                "author": r["Author"].strip(),
                "date_read": r["Date"].strip(),
            }
        )
    return books


def load_overrides() -> dict:
    """
    Manual override format (data/overrides.json), keyed by 'title|author':
    {
      "Siddhartha|Herman Hesse": {
        "google_books_id": "abc123",
        "open_library_key": "/works/OL123W",
        "wikipedia_title": "Herman Hesse"
      }
    }
    Any subset of keys may be given; missing ones fall back to auto-search.
    """
    if OVERRIDES_JSON.exists():
        with open(OVERRIDES_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def override_key(title: str, author: str) -> str:
    return f"{title}|{author}"


MAX_RETRIES = 4


def cached_get(cache_path: Path, url: str, params: dict | None = None, refresh: bool = False) -> dict | None:
    """GET a URL with on-disk JSON caching. Returns parsed JSON or None on failure.

    Retries on 429/5xx with backoff (honoring Retry-After when present) since
    free/keyless APIs under shared-network load return these transiently.
    """
    if cache_path.exists() and not refresh:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                wait = float(resp.headers.get("Retry-After", 0)) or (2**attempt)
                print(f"    ! {resp.status_code}, retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"    ! request failed: {e}")
            return None
        except ValueError:
            print("    ! non-JSON response")
            return None
        break
    else:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return data


def fetch_google_books(title: str, author: str, slug: str, override: dict, refresh: bool) -> dict | None:
    cache_path = GOOGLE_BOOKS_CACHE / f"{slug}.json"
    if "google_books_id" in override:
        url = f"https://www.googleapis.com/books/v1/volumes/{override['google_books_id']}"
        params = {"key": GOOGLE_BOOKS_API_KEY} if GOOGLE_BOOKS_API_KEY else None
        return cached_get(cache_path, url, params=params, refresh=refresh)

    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": f'intitle:"{title}" inauthor:"{author}"', "maxResults": 3}
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY
    data = cached_get(cache_path, url, params=params, refresh=refresh)
    return data


def fetch_open_library(title: str, author: str, slug: str, override: dict, refresh: bool) -> dict | None:
    cache_path = OPEN_LIBRARY_CACHE / f"{slug}.json"
    if "open_library_key" in override:
        key = override["open_library_key"].strip("/")
        url = f"https://openlibrary.org/{key}.json"
        return cached_get(cache_path, url, refresh=refresh)

    url = "https://openlibrary.org/search.json"
    params = {"title": title, "author": author, "limit": 3}
    data = cached_get(cache_path, url, params=params, refresh=refresh)

    # search.json never includes a description, only the /works/OL...W.json
    # detail endpoint does. If we got a match, fetch that too.
    if data and data.get("docs"):
        key = data["docs"][0].get("key")  # e.g. "/works/OL123W"
        if key:
            work_cache_path = OPEN_LIBRARY_CACHE / f"{slug}_work.json"
            work_data = cached_get(work_cache_path, f"https://openlibrary.org{key}.json", refresh=refresh)
            if work_data:
                data = {**data["docs"][0], **work_data}
    return data


def fetch_wikipedia_bio(author: str, override: dict, refresh: bool) -> dict | None:
    # Cache per-author (not per-book) so re-read authors don't trigger repeat
    # lookups against Wikipedia's rate limit.
    page_title = override.get("wikipedia_title", author)
    cache_slug = slugify(page_title) if "wikipedia_title" in override else slugify(author)
    cache_path = WIKIPEDIA_CACHE / f"{cache_slug}.json"
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(page_title)}"
    return cached_get(cache_path, url, refresh=refresh)


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


def extract_open_library_fields(raw: dict | None) -> dict:
    out = {"description": None, "subjects": None, "first_publish_year": None, "match_title": None}
    if not raw:
        return out
    doc = None
    if "docs" in raw and raw["docs"]:
        doc = raw["docs"][0]
    elif "title" in raw:
        doc = raw
    if not doc:
        return out
    desc = doc.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")
    out["description"] = desc
    out["subjects"] = doc.get("subject")
    out["first_publish_year"] = doc.get("first_publish_year")
    out["match_title"] = doc.get("title")
    return out


def extract_wikipedia_fields(raw: dict | None) -> dict:
    out = {"bio": None, "matched_page": None}
    if not raw or raw.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
        return out
    out["bio"] = raw.get("extract")
    out["matched_page"] = raw.get("title")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-fetch even if cached")
    parser.add_argument("--limit", type=int, default=None, help="only process first N books (testing)")
    args = parser.parse_args()

    books = load_books(BOOKS_CSV)
    if args.limit:
        books = books[: args.limit]
    overrides = load_overrides()

    print(f"Processing {len(books)} books...\n")
    records = []
    flagged = []

    for i, book in enumerate(books, 1):
        title, author = book["title"], book["author"]
        slug = slugify(f"{title}_{author}")
        override = overrides.get(override_key(title, author), {})
        print(f"[{i}/{len(books)}] {title} — {author}")

        gb_raw = fetch_google_books(title, author, slug, override, args.refresh)
        ol_raw = fetch_open_library(title, author, slug, override, args.refresh)
        wiki_raw = fetch_wikipedia_bio(author, override, args.refresh)

        gb = extract_google_books_fields(gb_raw)
        ol = extract_open_library_fields(ol_raw)
        wiki = extract_wikipedia_fields(wiki_raw)

        # Prefer Google Books, fall back to Open Library
        description = gb["description"] or ol["description"]
        categories = gb["categories"] or ol["subjects"]
        year = gb["published_date"] or ol["first_publish_year"]

        record = {
            "title": title,
            "author": author,
            "date_read": book["date_read"],
            "description": description,
            "categories": categories,
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
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(records)} records to {OUTPUT_JSON}")
    if flagged:
        print(f"\n{len(flagged)} issue(s) to review (consider adding to {OVERRIDES_JSON.name}):")
        for title, author, issue in flagged:
            print(f"  - {title} ({author}): {issue}")


if __name__ == "__main__":
    main()
