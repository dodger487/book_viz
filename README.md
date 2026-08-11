# Book Embedding Visualization

Pipeline: book list → metadata+tags → embeddings → 2D plot.

## Structure

- `data/books.csv` — your book list (Date, Type, Name, Author)
- `data/overrides.json` — manual fixes for mismatched auto-search results
- `data/book_metadata.json` — Script 1 output, combined per-book record
- `cache/google_books/`, `cache/open_library/`, `cache/wikipedia/` — raw
  API responses, one JSON file per book per source. Safe to delete
  individual files to force a re-fetch of just that book+source.
- `scripts/01_fetch_metadata.py` — Script 1
- `data/embedding_input.json` — Script 2 intermediate output, the constructed
  text per book before embedding (useful for sanity-checking what actually
  gets embedded)
- `data/embeddings.npz` — Script 2 output: `slugs[]` and `embeddings[]`
  arrays (aligned by index) plus which `provider` produced them
- `scripts/02_generate_embeddings.py` — Script 2
- `output/static_plot.png` — Script 3 static output (plotnine)
- `output/interactive_plot.html` — Script 3 interactive output (plotly);
  open directly in a browser, no server needed
- `scripts/03_visualize.py` — Script 3

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

## Script 2: generate embeddings

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/02_generate_embeddings.py                            # local, free, default model (all-mpnet-base-v2)
python scripts/02_generate_embeddings.py --model all-MiniLM-L6-v2   # smaller/faster local model
python scripts/02_generate_embeddings.py --provider openai --model text-embedding-3-large  # needs OPENAI_API_KEY
python scripts/02_generate_embeddings.py --provider voyage --model voyage-3-large          # needs VOYAGE_API_KEY
```

Combines each book's title/author/year/genre/description/author-bio into
one text blob (see `build_text()`), then embeds all 173 with a pluggable
provider (`EmbeddingProvider` subclasses in the script — add a new one and
register it in `PROVIDERS` to support another embedding API). Local is the
default so this runs free with no API key; swap to OpenAI/Voyage later by
just changing the CLI flags, no code changes needed for existing providers.

`requirements.txt` pins `numpy`/`scipy`/`torch`/`transformers`/
`sentence-transformers` to versions that are mutually compatible in this
environment (the newest available `torch` wheel needs `numpy<2`, which
needs a matching older `scipy`, which needs an older `transformers`/
`sentence-transformers` that doesn't require `torch>=2.5`) — install from
the file rather than picking versions individually, or you'll hit runtime
`numpy`/`torch` ABI errors that don't show up until you actually call
`.encode()`.

## Script 3: visualize

```bash
python scripts/03_visualize.py                    # PCA (default), colored by genre
python scripts/03_visualize.py --method tsne       # t-SNE instead
python scripts/03_visualize.py --color decade      # color by decade read instead of genre
```

Reduces the 768-dim embeddings to 2D via a pluggable `DimReducer` (PCA and
t-SNE via scikit-learn out of the box; a UMAP stub is included but needs
`pip install umap-learn` separately — see the script comment for why it's
not a default dependency) and renders both a static PNG (plotnine) and an
interactive HTML (plotly, hover over a point to see title/author/date
read) to `output/`.

## Notes

- Goodreads has no public API access (shut down 2020), hence Google
  Books + Open Library + Wikipedia instead.
- Fuzzy title/author search will occasionally match the wrong edition
  or a same-titled different book — spot-check `book_metadata.json`
  after the full run, especially for short/common titles.
- See `KNOWN_ISSUES.md` for the current state of data gaps/quirks in
  `book_metadata.json` and the bugs already found and fixed in Script 1.
