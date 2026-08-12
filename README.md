# Book Embedding Visualization

Pipeline: book list → metadata+tags → embeddings → 2D plot.

## Structure

- `data/books.csv` — your book list (Date, Type, Name, Author)
- `data/overrides.json` — manual fixes for mismatched auto-search results
- `cache/google_books/`, `cache/open_library/` (+ `_work.json`),
  `cache/wikipedia/` — raw API responses, one JSON file per book per
  source, written by Script 1. Safe to delete individual files to force
  a re-fetch of just that book+source.
- `scripts/01_fetch_metadata.py` — Script 1: fetches and caches raw API
  responses only. No field extraction/merging happens here — see below.
- `data/book_metadata.json` — Script 2 output, one merged record per book
  (description/genre/year/author bio), built entirely from Script 1's
  cache with no network calls
- `data/embedding_input.json` — Script 2 intermediate output, the constructed
  text per book before embedding (useful for sanity-checking what actually
  gets embedded)
- `data/embeddings.npz` — Script 2 output: `slugs[]` and `embeddings[]`
  arrays (aligned by index) plus which `provider` produced them
- `scripts/02_generate_embeddings.py` — Script 2: cache → merged metadata →
  embedding text → embeddings
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

This script does exactly one thing: fetch and cache raw API responses to
`cache/`. It does not merge/extract fields or write `book_metadata.json` —
that's Script 2's job, deliberately, so that changing how fields are
picked (e.g. which source's genre or publication year to prefer) never
requires re-hitting the network. If nothing needs re-fetching, skip
straight to Script 2.

### Google Books API key

The keyless Google Books quota is shared across every anonymous caller on
the network and gets exhausted fast. Get your own free key:

1. console.cloud.google.com → create/select a project
2. APIs & Services → Library → enable "Books API"
3. APIs & Services → Credentials → Create Credentials → API key

Set it as `GOOGLE_BOOKS_API_KEY` before running.

Run `--limit 5` first before doing the full 173-book run — that way you
catch any systematic issues (rate limiting, wrong field names, etc.) early
rather than after 173 calls.

## Script 2: extract metadata + generate embeddings

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/02_generate_embeddings.py                            # local, free, default model (all-mpnet-base-v2)
python scripts/02_generate_embeddings.py --model all-MiniLM-L6-v2   # smaller/faster local model
python scripts/02_generate_embeddings.py --provider openai --model text-embedding-3-large  # needs OPENAI_API_KEY
python scripts/02_generate_embeddings.py --provider voyage --model voyage-3-large          # needs VOYAGE_API_KEY
```

Two stages, both here:

1. **Extract + merge** (`build_metadata()`): reads Script 1's cache
   directly (no network), picks description/genre/publication year per
   book, and writes `data/book_metadata.json`. Prefers Open Library's
   `first_publish_year` over Google Books' `publishedDate` (which reflects
   whatever specific edition got matched — often a recent reprint, not the
   original). Genre starts from Google Books' category, but Google Books
   only ever gives one flat top-level label (52% of this dataset was just
   "Fiction") — when it's a generic fiction bucket, `extract_genre()`
   looks for a more specific subgenre in Open Library's work-level
   `subjects` list instead. At the end it prints a list of books still
   missing a description or author bio; for those, look up the correct
   match yourself and add an entry to `overrides.json` (see its docstring
   for the field options), then delete the affected cache files (see
   `KNOWN_ISSUES.md`'s "cache staleness trap" note) and re-run.
2. **Embed** (`build_text()` + `EmbeddingProvider`): combines each book's
   title/author/year/genre/description/author-bio into one text blob, then
   embeds all 173 with a pluggable provider (`EmbeddingProvider` subclasses
   in the script — add a new one and register it in `PROVIDERS` to support
   another embedding API). Local is the default so this runs free with no
   API key; swap to OpenAI/Voyage by just changing `--provider`, no code
   changes needed for existing providers. Each provider writes its own
   `data/embeddings_<provider>.npz` (e.g. `embeddings_local.npz`,
   `embeddings_openai.npz`) rather than one shared file, so generating a
   second provider's embeddings never overwrites the first — run both and
   Script 3 lets you compare them side by side.

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
python scripts/03_visualize.py                          # first embeddings_*.npz found, PCA, colored by genre
python scripts/03_visualize.py --source openai --method tsne_p5 --color decade_published
```

Auto-discovers every `data/embeddings_<provider>.npz` Script 2 has
produced — generate embeddings from more than one provider and this script
picks all of them up automatically, no flag needed to "enable" a source.
Reduces each one to 2D via a pluggable `DimReducer` and renders it via
every method in `REDUCERS`: `pca`, and t-SNE at every perplexity in
`TSNE_PERPLEXITIES` (`tsne_p5` through `tsne_p30` — lower perplexity
weights local structure more, which tends to produce tighter, more
separated clusters on a small (173-book) dataset). t-SNE is cheap enough
at this size to just register a bunch of values and compare rather than
guess the "right" one — add/remove values from that one list, nothing
else to edit. A `umap` stub is included too but needs
`pip install umap-learn` separately (skipped automatically if not
installed) — see the script comment for why it's not a default
dependency. **Heads up**: on this machine, `umap-learn`'s dependency
`llvmlite` has no prebuilt wheel for this Python/platform combination and
fails to build from source without a matching system LLVM install (not
present here) — if you want UMAP, either `brew install llvm@15` (version
must match what the installed `llvmlite`/`numba` expects) first, or try a
different Python version where a prebuilt wheel might exist.

Renders:

- `output/static_plot.png` (plotnine) — one fixed source/method/color
  combo, set by `--source`/`--method`/`--color`.
- `output/interactive_plot.html` (plotly) — **every** embedding source ×
  method × color combination is precomputed into the page, with dropdowns
  to switch embedding source, projection method, and color-by live in the
  browser (no re-running the script). `--source`/`--method`/`--color` just
  set what's selected when the page first loads. Color options: genre,
  year read (continuous — the reading list only spans ~11 years, so decade
  buckets would collapse almost everything into 2 colors), decade
  published (books span centuries, so decade buckets make sense here).
  Hover any point for title/author/date read. A fourth dropdown ("On
  hover, show") picks what edges appear when you hover:
  - **Nearest neighbors** (default) — lines to the book's 5 nearest
    neighbors by cosine similarity in the *original* embedding space, not
    the 2D projection — so you'll sometimes see an edge connect two points
    that landed far apart on screen but are still genuinely similar (e.g.
    same topic, different genre).
  - **Reading order (chronological)** — a directed 5-node path centered on
    the hovered book: the 2 books read immediately before it and the 2
    read immediately after, connected as a sequential chain (not a star
    from the hovered book to all 4 — the chain shows the actual reading
    order through that window), independent of the embedding entirely.
    Rendered as arrows (not plain lines) pointing oldest → newest so the
    chronological direction is visible.
  - **No links** — turns edges off.

## Notes

- Goodreads has no public API access (shut down 2020), hence Google
  Books + Open Library + Wikipedia instead.
- Fuzzy title/author search will occasionally match the wrong edition
  or a same-titled different book — spot-check `book_metadata.json`
  after the full run, especially for short/common titles.
- See `KNOWN_ISSUES.md` for the current state of data gaps/quirks in
  `book_metadata.json` and the bugs already found and fixed in Scripts 1-2.
