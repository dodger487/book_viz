# Book Embedding Visualization

Pipeline: book list → metadata+tags → embeddings → 2D plot.

## Structure

The pipeline is five numbered scripts, each doing exactly one job so that
changing one never forces re-running an earlier, slower/costlier one:
fetch (network) → extract (local) → tag (LLM) → embed (local or API) →
visualize.

- `data/books.csv` — your book list (Date, Type, Name, Author)
- `data/overrides.json` — manual fixes for mismatched auto-search results
- `cache/google_books/`, `cache/open_library/` (+ `_work.json`),
  `cache/wikipedia/` — raw API responses, one JSON file per book per
  source, written by Script 1. Safe to delete individual files to force
  a re-fetch of just that book+source.
- `scripts/01_fetch_metadata.py` — Script 1: fetches and caches raw API
  responses only. No field extraction/merging happens here.
- `data/book_metadata.json` — Script 2 output, one merged record per book
  (description/genre/year/author bio), built entirely from Script 1's
  cache with no network calls
- `scripts/02_extract_metadata.py` — Script 2: cache → merged metadata
- `cache/llm_tags/` — raw per-book LLM response, written by Script 3
- `data/book_tags.json` — Script 3 output: one record per book
  (LLM-cleaned description, refined genre, tone/pacing/themes/setting)
- `scripts/03_generate_tags.py` — Script 3: metadata → LLM-cleaned
  description/genre + structured tags
- `data/embedding_input_<variant>.json` — Script 4 intermediate output per
  text variant, the constructed text per book before embedding (useful
  for sanity-checking what actually gets embedded)
- `data/embeddings_<provider>_<variant>.npz` — Script 4 output: one file
  per (provider, text-variant) pair, `slugs[]`/`embeddings[]` arrays plus
  which `provider`/`model`/`text_variant` produced them
- `scripts/04_generate_embeddings.py` — Script 4: metadata (+ tags for v2+)
  → embedding text → embeddings
- `output/static_plot.png` — Script 5 static output (plotnine)
- `output/interactive_plot.html` — Script 5 interactive output (plotly);
  open directly in a browser, no server needed
- `scripts/05_visualize.py` — Script 5

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

## Script 2: extract metadata

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/02_extract_metadata.py
```

Reads Script 1's cache directly (no network), picks description/genre/
publication year per book, and writes `data/book_metadata.json`. Prefers
Open Library's `first_publish_year` over Google Books' `publishedDate`
(which reflects whatever specific edition got matched — often a recent
reprint, not the original). Genre starts from Google Books' category, but
Google Books only ever gives one flat top-level label (52% of this
dataset was just "Fiction") — when it's a generic fiction bucket,
`extract_genre()` looks for a more specific subgenre in Open Library's
work-level `subjects` list instead (Script 3 refines this further with an
LLM). At the end it prints a list of books still missing a description or
author bio; for those, look up the correct match yourself and add an
entry to `overrides.json` (see its docstring for the field options), then
delete the affected cache files (see `KNOWN_ISSUES.md`'s "cache
staleness trap" note) and re-run.

## Script 3: generate tags

```bash
export OPENAI_API_KEY=your_key_here
python scripts/03_generate_tags.py --limit 5   # test on a few books first
python scripts/03_generate_tags.py              # full run (skips cached)
python scripts/03_generate_tags.py --refresh    # force re-request everything
```

For each book, sends title/author/description/genre/bio to `gpt-4o-mini`
and asks for a structured JSON profile: a marketing-free
`clean_description`, a refined `genre` (single pick from a fixed
taxonomy — see `GENRE_TAXONOMY` in the script), and free-text
`tone_and_style`/`pacing`/`themes`/`setting` tags. The prompt (`PROMPT` in
the script) explicitly tells the model to draw on its own knowledge of
well-known books rather than only rephrasing our fetched blurb — many of
these are famous books the model knows far better than our scraped
description — while staying conservative (no invented plot details) for
books it doesn't actually recognize. A `familiarity` field
(`high`/`medium`/`low`) is the model's own signal of which mode it's in
per book, useful for auditing: pull the "low" ones and check nothing
looks fabricated. Every response is cached per book under
`cache/llm_tags/`, so re-runs are free; `--refresh` forces specific books
(or all, with no other cache changes) to redo.

## Script 4: generate embeddings

```bash
python scripts/04_generate_embeddings.py                            # v1 text, local, free, default model
python scripts/04_generate_embeddings.py --text-variant v2          # v2 text, local
python scripts/04_generate_embeddings.py --model all-MiniLM-L6-v2   # smaller/faster local model
python scripts/04_generate_embeddings.py --provider openai --model text-embedding-3-large  # needs OPENAI_API_KEY
python scripts/04_generate_embeddings.py --provider voyage --model voyage-3-large          # needs VOYAGE_API_KEY
```

Combines each book's fields into one text blob via a pluggable "text
variant" (`build_text_v1`/`build_text_v2`, registered in `TEXT_VARIANTS`
— see "Embedding input methodology" below for what each one does), then
embeds all 173 with a pluggable provider (`EmbeddingProvider` subclasses
in the script — add a new one and register it in `PROVIDERS` to support
another embedding API). Local is the default so this runs free with no
API key; swap to OpenAI/Voyage by just changing `--provider`, no code
changes needed for existing providers. Every (provider, text-variant)
pair writes its own `data/embeddings_<provider>_<variant>.npz` (e.g.
`embeddings_local_v1.npz`, `embeddings_openai_v2.npz`) rather than one
shared file, so generating a new provider or trying a new text variant
never overwrites another's output — run as many combinations as you want
and Script 5 lets you compare all of them side by side.

`requirements.txt` pins `numpy`/`scipy`/`torch`/`transformers`/
`sentence-transformers` to versions that are mutually compatible in this
environment (the newest available `torch` wheel needs `numpy<2`, which
needs a matching older `scipy`, which needs an older `transformers`/
`sentence-transformers` that doesn't require `torch>=2.5`) — install from
the file rather than picking versions individually, or you'll hit runtime
`numpy`/`torch` ABI errors that don't show up until you actually call
`.encode()`.

## Embedding input methodology (versions)

What text actually gets embedded matters a lot for cluster quality, and
we're iterating on it — this section tracks what each version does and
why, in plain English. Both versions are available side by side (Script 5
auto-discovers every `data/embeddings_<provider>_<variant>.npz`, selectable
via the "Embedding source" dropdown) — nothing here is retired when a new
version ships. See `data/embeddings_<name>.npz`'s `provider`/`text_variant`
fields and the matching `data/embedding_input_<variant>.json` for exactly
what produced any given file.

### V1

`build_text_v1()` in `scripts/04_generate_embeddings.py` concatenates, per
book, in this order: title, author ("*Title* by *Author*."), publication
year, genre (Script 2's heuristic-extracted single label), the raw
description as fetched from Google Books/Open Library (unedited —
includes whatever marketing language the source had, e.g. "WINNER of the
National Book Award," blurbs, etc.), and the raw Wikipedia author bio.
All of it goes into one string, embedded as a single vector per book.

**Known issues with V1**: title and author, being distinctive/proper-noun
text, appear to get outsized weight from the embedding models relative to
their actual thematic relevance — observed on the OpenAI+t-SNE view,
where books with similar-sounding titles clustered together somewhat
independent of content. Raw descriptions also carry promotional
boilerplate that's noise, not signal (correlates with a book being
popular/awarded, not with what it's *about*). Genre is also often just
the generic "Fiction" bucket even after Script 2's Open Library fallback.
V2 addresses all three.

### V2

`build_text_v2()` drops title and author entirely (the fix for V1's
overweighting issue — see `KNOWN_ISSUES.md`/this file's git history for
that investigation). In their place, it uses Script 3's LLM output
instead of the raw fetched fields: `clean_description` (marketing
language stripped, and for books the model recognizes, informed by the
model's own knowledge rather than only our sometimes-thin scraped blurb —
see Script 3's docs for the "familiarity" safety mechanism against
hallucinating on obscure titles) instead of the raw description, the
LLM-refined `genre` (picked from a fixed taxonomy, meaningfully more
specific than V1's often-generic bucket) instead of the heuristic one,
plus three new fields V1 didn't have at all: `tone_and_style`, `pacing`,
`themes`, `setting`. Publication year and the raw Wikipedia author bio are
still included (bio is biographical/stylistic content, not just a name,
so it wasn't part of the overweighting problem the title/author *label*
caused).

## Script 5: visualize

```bash
python scripts/05_visualize.py                          # defaults: openai_v2, t-SNE(7), colored by genre_llm
python scripts/05_visualize.py --source local_v1 --method tsne_p5 --color decade_published
```

Auto-discovers every `data/embeddings_<provider>_<variant>.npz` Script 4
has produced — generate embeddings from more than one provider and/or text
variant and this script picks all of them up automatically, no flag needed
to "enable" a source; the dropdown label shows provider, model, and
variant (e.g. "OpenAI (text-embedding-3-large) — v2"). If
`data/book_tags.json` exists (Script 3 has been run), an extra "Genre
(LLM-refined)" color-by option appears alongside the heuristic "Genre" —
this is independent of which embedding source is selected, so you can
view V1 embeddings colored by the LLM-refined genre or vice versa.
Reduces each embedding source to 2D via a pluggable `DimReducer` and
renders it via
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
  set what's selected when the page first loads (currently `openai_v2` +
  `tsne_p7` + `genre_llm` — the combo that's looked best so far; change via
  `PREFERRED_DEFAULT_SOURCE`/`--method`/`--color` in `main()`).

  There's no page title over the chart itself — it duplicated the page
  `<h1>` ("A Map of Books"). Plotly's default hover box (which used to
  show `genre_llm=X, x=.., y=..` and cover exactly the nodes/edges it was
  describing) is replaced by a lightweight one showing just the book's
  title at partial opacity (`hoverlabel` in the layout; a custom
  `hovertemplate` per trace, since Plotly Express's own default
  `hovertemplate` otherwise overrides `hoverinfo` and shows the verbose
  box regardless). A **Book details** panel and a searchable **All
  books** list sit in a fixed-width, fixed-height (independently
  scrollable) sidebar to the left of the graph, so both are visible at
  the same time as the plot rather than requiring a click-then-scroll:

  - **Clicking a node** (or a list entry) *selects* it: a hollow ring marks
    it on the plot, its title tooltip stays pinned (via `Plotly.Fx.hover`,
    which — unlike a real hover — doesn't clear on mouseout), its
    edges/links stay drawn even after the mouse moves away, and its
    details stay in the sidebar panel. Clicking the same book again (node,
    list entry, empty canvas, or the "Clear selection" button) deselects
    it. Note: Plotly overlays an invisible drag/zoom rect on top of the
    whole plot area, so a raw click's DOM target is never actually the
    marker — telling a marker click from an empty-canvas click uses a flag
    set by the `plotly_click` event instead (only fires for a real point).
  - **Hovering** a point shows that book's details/edges transiently,
    taking priority over whatever's selected; moving the mouse away falls
    back to the pinned selection (if any), not to nothing.
  - **Clicking a legend entry** highlights that category and dims (never
    hides) every other point, Tableau-style, so relative position is still
    visible — click the same entry again to restore everyone to full
    opacity, or click a different entry to switch the highlight. This
    replaces Plotly's default legend-click behavior, which hides the
    clicked trace entirely.

  A fourth dropdown ("On hover/select, show") picks what edges/links are
  drawn for the active book (hovered, or failing that, selected):
  - **Nearest neighbors** (default) — lines to the book's 5 nearest
    neighbors by cosine similarity in the *original* embedding space, not
    the 2D projection — so you'll sometimes see an edge connect two points
    that landed far apart on screen but are still genuinely similar (e.g.
    same topic, different genre).
  - **Reading order (chronological)** — a directed 5-node path centered on
    the active book: the 2 books read immediately before it and the 2
    read immediately after, connected as a sequential chain (not a star
    from the hovered book to all 4 — the chain shows the actual reading
    order through that window), independent of the embedding entirely.
    Rendered as arrows (not plain lines) pointing oldest → newest so the
    chronological direction is visible.
  - **No links** — turns edges off.

  A small "?" button at the right edge of the controls bar opens a brief
  explainer (embeddings → 2D projection → proximity is a hint, not a
  guarantee) — click elsewhere, click it again, or press Escape to close it.

### Design system: color and type

The palette is a "vintage library" theme — parchment/leather neutrals with
manuscript-green/navy/crimson used as accents — sourced from a "vintage
library" reference palette (Parchment `#f6f2ea`, Leather `#c29b87`, Oak
`#a97954`, Manuscript `#1e423f`, Binding `#532c2e`, Crimson `#34000b`) and
adapted to this page's needs. Chrome tones (`--page-plane`, `--surface`,
`--ink-primary`, `--ink-secondary`, `--ink-muted`, `--border` in the page
CSS) are free to reskin on taste since they carry no data meaning; the
categorical genre palette is not — it's the single biggest visual signal
on the page and still has to pass the dataviz design system's six
colorblind-safety checks, so it was rebuilt (not just recolored) the same
way the previous palette was: candidate warm "book spine" hues generated
in-band (OKLCH lightness 0.43–0.77, chroma ≥ 0.10) against the new
`PLOT_BGCOLOR`, then every ordering brute-forced through the dataviz
skill's `scripts/validate_palette.js` to find one clearing the adjacent-pair
CVD/normal-vision floors — the shipped order (oak → manuscript green → navy
→ olive → plum → crimson → mustard → rose) also has its first 3 slots
(the most common genres) passing the *stricter* all-pairs check scatter
plots need, matching the reference palette's own guarantee. Colors are
still assigned by the job each column does, not picked by eye —
`color_order_and_map()` in the script implements this:

- **Genre / Genre (LLM-refined)** are *nominal* (identity — swapping the
  order wouldn't change the meaning). The first 8 categories by frequency
  get the validated 8-hue set above; categories beyond 8 get additional
  hues generated deterministically (`extended_categorical_palette()` —
  evenly-spaced hues at a fixed phase offset, tuned toward the same deep,
  moderately-saturated character as the validated 8 but not validated
  against the colorblind-safety gates those 8 pass). By default almost
  every distinct value gets shown rather than folding into "Other"
  (`collapse_rare(top_n=...)` — 20 for the heuristic `genre` column, which
  has ~42 raw values and gets capped there for legend legibility and
  render time; effectively uncapped for `genre_llm`, which has ~21). This
  trades strict adjacent-pair colorblind-safety past the 8th category for
  more visible genre variety — acceptable here since same-genre points
  already cluster spatially, so exact color contrast between distant
  points matters less.
- **Decade published / Year read** are *ordinal* (order carries meaning),
  so they get a **diverging** navy↔gray↔crimson ramp in chronological order
  (`ORDINAL_COLOR_COLUMNS`, `interpolate_ordinal_colors()` — navy for
  earliest, warm gray at the midpoint, crimson for latest) rather than a
  single light→dark hue or, worse, a rainbow scale with no ordering signal
  at all (the original bug here). Both stay discrete/click-able legend
  traces, not a continuous colorbar — year read has few enough distinct
  values (~11) that discrete is more legible than continuous.
  `DIVERGING_MIDPOINT` (`#6e6350`) and `NEUTRAL_COLOR` (`#796a56`, used for
  "Other"/"Unknown") were each picked for contrast against `PLOT_BGCOLOR`
  specifically, not eyeballed — a lesson from the previous palette, whose
  midpoint was ~1.03:1 contrast (effectively invisible) against a plot
  background that wasn't the neutral white/off-white surface the original
  value assumed.

Type: **Fraunces** (serif, the page `<h1>` and panel headings, colored with
the palette's Binding tone) paired with **Public Sans** (everything else —
controls, legend, tooltip, axis). Loaded via Google Fonts. Plotly doesn't
inherit the page's CSS font by default, so `PLOT_FONT_FAMILY` is set
explicitly on the Python-built layout *and* re-applied in the JS `update()`
function — otherwise switching dropdowns would reset the chart back to
Plotly's own default font. (There's no separate in-chart title anymore —
it duplicated the page `<h1>`.)

## Notes

- Goodreads has no public API access (shut down 2020), hence Google
  Books + Open Library + Wikipedia instead.
- Fuzzy title/author search will occasionally match the wrong edition
  or a same-titled different book — spot-check `book_metadata.json`
  after the full run, especially for short/common titles.
- See `KNOWN_ISSUES.md` for the current state of data gaps/quirks in
  `book_metadata.json` and the bugs already found and fixed in Scripts 1-2
  (raw fetching and extraction).
