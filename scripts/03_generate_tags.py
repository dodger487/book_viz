#!/usr/bin/env python3
"""
Script 3: book_metadata.json -> LLM-cleaned description + genre + tags.

For each book, sends its title/author/description/genre/bio to an LLM
(OpenAI, gpt-4o-mini by default) and asks for a structured profile: a
marketing-free content summary, a refined genre pick from a fixed
taxonomy, and free-text tone/pacing/themes/setting tags. Explicitly
instructs the model to draw on its own knowledge of well-known books
rather than only rephrasing our (sometimes thin, sometimes
marketing-heavy) fetched description -- see PROMPT below for the exact
wording and the "familiarity" field, which is the model's own signal of
how much it's relying on real knowledge of the book vs. our blurb (useful
for auditing: pull the "low" ones and sanity-check nothing looks
fabricated).

Why this is its own script rather than folded into fetching or embedding:
the prompt/schema here is exactly the kind of thing we'll want to iterate
on, and doing so shouldn't require re-fetching raw API data OR
re-embedding books whose tags didn't change. Every response is cached per
book, so re-runs are free and idempotent, same pattern as Script 1.

Output:
  cache/llm_tags/<slug>.json - raw per-book LLM response (cached)
  data/book_tags.json        - merged list, one record per book

Usage:
    python scripts/03_generate_tags.py --limit 5   # test on a few books first
    python scripts/03_generate_tags.py              # full run (skips cached)
    python scripts/03_generate_tags.py --refresh     # force re-request everything
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
METADATA_JSON = DATA_DIR / "book_metadata.json"
TAGS_JSON = DATA_DIR / "book_tags.json"
LLM_TAGS_CACHE = CACHE_DIR / "llm_tags"

MODEL = "gpt-4o-mini"
MAX_RETRIES = 4

GENRE_TAXONOMY = [
    "Literary Fiction", "Historical Fiction", "Science Fiction", "Fantasy", "Mystery",
    "Thriller", "Espionage/Spy", "Horror", "Romance", "Short Stories", "Graphic Novel",
    "Young Adult Fiction", "Children's Fiction",
    "Biography & Memoir", "History", "Business & Economics", "Politics & Current Affairs",
    "Social Science", "Science & Technology", "Self-Help & Personal Development",
    "True Crime", "Travel", "Food & Cooking", "Religion & Spirituality", "Health & Wellness",
    "Sports & Recreation", "Reference & Education", "Poetry", "Other",
]

PROMPT = f"""You are helping build a structured, content-focused profile of books for a
personal reading-taste visualization. For each book you're given, produce a
JSON object with these fields:

- familiarity: "high", "medium", or "low" -- how confident you are that you
  genuinely know this specific book (not just its author or genre),
  independent of the description provided below.
    "high"   = you have specific, reliable knowledge of this book's actual
               plot/argument/reception.
    "medium" = you recognize the book or its type but aren't fully certain
               of the details.
    "low"    = you don't recognize this specific title/edition and are
               relying almost entirely on the provided description.

- clean_description: A 2-4 sentence, marketing-free summary of what the
  book is actually about (plot, argument, or subject matter) -- no "New
  York Times bestseller," no award callouts, no reviewer blurbs, no "in
  this stunning novel" language. If you recognize this book (familiarity
  high/medium), draw on your own knowledge to make this accurate and
  specific, rather than just rephrasing the description below -- many of
  these are well-known books with far more written about them than our
  fetched blurb captures. If you do NOT recognize this specific book
  (familiarity low), stay close to the provided description and do not
  invent plot details, characters, or specifics you're not confident about.

- genre: Pick exactly ONE label from this fixed list (do not invent new
  ones): {", ".join(GENRE_TAXONOMY)}
  A current genre guess is provided below (it's often too generic, e.g.
  just "Fiction") -- use it as a hint, but override it if a more specific
  label fits better.

- tone_and_style: A short phrase (aim for 3-8 words) covering BOTH
  emotional register and prose style -- e.g. "restrained, formal,
  Victorian diction" or "wry, conversational, fast-paced" or "bleak,
  spare, minimalist prose."

- pacing: A short phrase, e.g. "slow-burn, character-driven" or
  "fast-paced, plot-driven" or "measured, methodical" (for non-fiction,
  describe how the argument/exposition unfolds).

- themes: A short comma-separated list of central themes/ideas, e.g.
  "identity, colonialism, memory."

- setting: A short phrase for time/place, e.g. "contemporary Nigeria and
  the US" or "far-future space colony" -- for non-fiction with no literal
  setting, describe the scope/context instead, e.g. "20th-century American
  economic history."

Rules:
- Keep every field terse. These are embedding inputs, not prose.
- Never fabricate specific plot details, character names, or claims you're
  not confident about -- especially when familiarity is "low."
- Weigh the provided metadata AND your own knowledge of the book according
  to your actual familiarity with this specific title."""

RESPONSE_SCHEMA = {
    "name": "book_tags",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "familiarity": {"type": "string", "enum": ["high", "medium", "low"]},
            "clean_description": {"type": "string"},
            "genre": {"type": "string", "enum": GENRE_TAXONOMY},
            "tone_and_style": {"type": "string"},
            "pacing": {"type": "string"},
            "themes": {"type": "string"},
            "setting": {"type": "string"},
        },
        "required": [
            "familiarity", "clean_description", "genre",
            "tone_and_style", "pacing", "themes", "setting",
        ],
        "additionalProperties": False,
    },
}


def user_message(book: dict) -> str:
    return f"""Title: {book['title']}
Author: {book['author']}
Current genre guess (from Google Books/Open Library, often too generic): {book.get('genre') or 'Unknown'}
Description (as fetched -- may include marketing language): {book.get('description') or '(none found)'}
Author bio (for context only, not to be cleaned): {book.get('author_bio') or '(none found)'}

Produce the JSON profile for this book."""


def call_llm(client, book: dict) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": user_message(book)},
                ],
                response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2**attempt
                print(f"    ! {e!r}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ! failed after {MAX_RETRIES} attempts: {e!r}")
                return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-request even if cached")
    parser.add_argument("--limit", type=int, default=None, help="only process first N books (testing)")
    args = parser.parse_args()

    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    books = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    if args.limit:
        books = books[: args.limit]

    LLM_TAGS_CACHE.mkdir(parents=True, exist_ok=True)
    records = []
    failed = []

    print(f"Generating tags for {len(books)} books via {MODEL}...\n")
    for i, book in enumerate(books, 1):
        slug = book["slug"]
        cache_path = LLM_TAGS_CACHE / f"{slug}.json"
        print(f"[{i}/{len(books)}] {book['title']} — {book['author']}")

        if cache_path.exists() and not args.refresh:
            tags = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            tags = call_llm(client, book)
            if tags is None:
                failed.append((book["title"], book["author"]))
                continue
            cache_path.write_text(json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8")

        records.append({"slug": slug, "title": book["title"], "author": book["author"], **tags})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TAGS_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(records)} tag records to {TAGS_JSON}")
    if failed:
        print(f"\n{len(failed)} book(s) failed after retries:")
        for title, author in failed:
            print(f"  - {title} ({author})")


if __name__ == "__main__":
    main()
