# Book Embedding Visualization

Pipeline: book list → metadata+tags → embeddings → 2D plot.

## Structure

- `data/books.csv` — your book list (Date, Type, Name, Author)
- `data/overrides.json` — manual fixes for mismatched auto-search results
- `data/book_metadata.json` — Script 1 output, combined per-book record
- `cache/google_books/`, `cache/open_library/`, `cache/wikipedia/` — raw
  API responses, one JSON file per book per source. Safe to delete
  individual files to force a re-fetch of just that book+source.
- `scripts/01_fetch_metadata.py` — this script

## Script 1: fetch metadata

```bash
pip install requests
export GOOGLE_BOOKS_API_KEY=your_key_here  # see below; optional but recommended
python scripts/01_fetch_metadata.py --limit 5   # test on a few books first
python scripts/01_fetch_metadata.py              # full run (skips cached)
python scripts/01_fetch_metadata.py --refresh    # force re-fetch everything
```

### Google Books API key

The keyless Google Books quota is shared across every anonymous caller on
the network and gets exhausted fast. Get your own free key:

1. console.cloud.google.com → create/select a project
2. APIs & Services → Library → enable "Books API"
3. APIs & Services → Credentials → Create Credentials → API key

Set it as `GOOGLE_BOOKS_API_KEY` before running.

Run `--limit 5` first and check `data/book_metadata.json` before doing
the full 173-book run — that way you catch any systematic issues (rate
limiting, wrong field names, etc.) early rather than after 173 calls.

At the end it prints a list of books with no description or no author
bio found. For those, look up the correct match yourself (Google Books
volume ID from the URL, Open Library work key like `/works/OL123W`, or
the exact Wikipedia page title) and add an entry to `overrides.json`,
then re-run with `--refresh` for just that concern (or delete the
specific cache file and re-run without `--refresh`, which is cheaper).

## Notes

- Goodreads has no public API access (shut down 2020), hence Google
  Books + Open Library + Wikipedia instead.
- Fuzzy title/author search will occasionally match the wrong edition
  or a same-titled different book — spot-check `book_metadata.json`
  after the full run, especially for short/common titles.
- Next: Script 2 turns `book_metadata.json` into embedding-input text
  (and/or LLM-generated structured tags), Script 3 embeds + reduces +
  plots.
