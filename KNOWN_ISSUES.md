# Known data issues (Script 1/2 output)

Last updated after the genre/published-year extraction fixes below.
Regenerate this picture by running `scripts/02_generate_embeddings.py` —
it prints the same flagged list at the end (extraction/merging moved from
Script 1 to Script 2; Script 1 now only fetches and caches raw responses).

## Genre and published-year extraction fixes

Two systemic issues found by spot-checking the data, both in how fields
were picked from the raw cached responses (not missing-data problems, so
not in the flagged list below):

- **Genre was almost always just "Fiction"**: Google Books' `categories`
  field is a single flat top-level BISAC label for every edition in this
  dataset — verified across all 160 categorized books, zero had more than
  one category, zero had a hierarchical/subgenre path. 90 of 173 books
  (52%) came back simply "Fiction." Fix: when Google Books' category is
  generic ("Fiction"/"Juvenile Fiction"/"Young Adult Fiction"), fall back
  to Open Library's work-level `subjects` list (richer, but noisy —
  filtered against a fiction-subgenre keyword list) instead. Result:
  generic "Fiction" dropped from 90 to 24 books, with Science Fiction (24),
  Historical Fiction (8), Mystery (8), Romance (7), Fantasy (7), Thriller
  (5), and others now distinguished. See `extract_genre()` in
  `scripts/02_generate_embeddings.py`.
- **Published year reflected whatever edition got matched, not the
  original publication**: our Google Books search takes the first/
  top-ranked result, and Google's ranking isn't date-aware — it tends to
  surface whatever edition is currently for sale. E.g. *Sapiens* matched a
  "Tenth Anniversary Edition" dated 2025-02-18, not the 2011 original.
  Comparing against Open Library's `first_publish_year` across the whole
  dataset found 57 of 173 books (33%) differed by more than 3 years,
  almost all skewing "too recent" — classics hit hardest (*Pride and
  Prejudice* showed 2017 instead of 1813, *The Picture of Dorian Gray*
  showed 1998 instead of 1890, several Chandler/le Carré novels off by
  25-92 years). Fix: prefer Open Library's `first_publish_year` over
  Google Books' `publishedDate`. (Open Library isn't perfectly reliable
  either — e.g. it gave "Virtual Light" a first-publish year of 1743,
  clearly wrong — but it's right far more often than the edition-specific
  Google Books date was.)

Also note: the `subject` (singular) field on Open Library's search.json
response is essentially always empty; the real subject/genre data lives on
`subjects` (plural) on the work-detail endpoint
(`/works/OL...W.json`, cached as `<slug>_work.json`) — same "wrong field
name" bug pattern as the description issue below.

## Current coverage

- 173/173 books have a record in `data/book_metadata.json`.
- 172/173 have at least one of description/author_bio.
- 1/173 (`The Saint of Mt. Koya`) is missing description but has a bio.
- 0/173 are missing both (`The Mote in God's Eye` and `Lean Analytics`,
  formerly missing both, now each have at least one field — see below).

## Second pass: 8 of 14 flags closed (6 of 12 books fully resolved)

Verified each fix with a live web search for the real author identity/page
before touching `overrides.json` — per the wrong-person lesson below, a
confident non-match was left alone rather than guessed.

| Book | Fix | How |
|---|---|---|
| Democracy for Realists | author bio ✅ | Christopher Achen has no English Wikipedia page (confirmed: search turns up only mentions in other articles, no dedicated page). Larry Bartels, the co-author, does — added `wikipedia_title: "Larry Bartels"`. |
| Getting to Yes | description ✅ | `inauthor:"Roger Fisher and William Ury"` matched nothing (Google Books treats it as one literal name). Added `search_author: "Roger Fisher"`. |
| The Black Monday Murders: Volume 1 | description ✅ | Pinned exact edition via `google_books_id: "FwDuDQAAQBAJ"` (verified: title "The Black Monday Murders Vol. 1", author Jonathan Hickman, real description present). |
| The Mote in God's Eye | both ✅ | Multi-author credit broke both lookups. Added `search_author: "Larry Niven"` + `wikipedia_title: "Larry Niven"` (confirmed real Wikipedia page, co-author of this exact book per its lead paragraph). Note: the Google Books description that comes back is a thin "Science fiction-roman." fragment — every indexed edition of this title has a sparse GB description; Open Library's is fuller but the pipeline prefers GB when non-empty. Left as-is (data-quality nit, not a missing-field flag; not a pipeline bug). |
| Getting It Done: How to Lead When You're Not In Charge | description ✅ | Confirmed via web search that the CSV's 2nd author "John Richardson" is wrong — the real co-author is **Alan Sharp** (Amazon/Google Books/B&N listings all show Fisher & Sharp). Updated `search_author` to `"Alan Sharp"` and added `search_title: "Getting It Done"` (the full title with the curly apostrophe was breaking `intitle:` matching). |
| Lean Analytics | description ✅, bio ✗ | Added `search_author: "Alistair Croll"` — fixes description. Checked both authors' Wikipedia presence directly: neither Alistair Croll nor Benjamin Yoskovitz has an English Wikipedia page (both URLs 404). Bio genuinely unavailable, left unset. |
| Vanishing Treasures | description ✅ | Pinned `google_books_id: "k0_0EAAAQBAJ"` (verified: title "Vanishing Treasures", author Katherine Rundell, full description present — this book's real full title is "Vanishing Treasures: A Bestiary of Extraordinary Endangered Creatures", which likely confused unpinned matching). |

One override was tried and then **reverted**: for `I Deliver Parcels in
Beijing|Hu Anyan`, Hu Anyan has no personal Wikipedia page, but the *book*
does (`en.wikipedia.org/wiki/I_Deliver_Parcels_in_Beijing`). Pointing
`wikipedia_title` at the book's page would have populated `author_bio` with
a book synopsis mislabeled as a bio — caught in review before committing.
Reverted; left as a genuine gap rather than mislabeled content.

## Remaining flagged books (6 flags / 6 books)

| Book | Author (as typed in books.csv) | Missing | Why |
|---|---|---|---|
| Four Futures | Peter Frase | author bio | confirmed no English Wikipedia page (Jacobin editor/DSA member, doesn't have one) |
| The Saint of Mt. Koya | Izumi Kyora | description | real author is Kyōka Izumi (translated 1900 Japanese novella); checked every Google Books/Open Library match for this exact translated title/author — none carry a description, only bare catalog entries. Bio is already fixed (Wikipedia page found via `search_author`/`wikipedia_title` override from the first cleanup pass). |
| Lean Analytics | Alistair Croll and Benjamin Yoskovitz | author bio | confirmed neither co-author has an English Wikipedia page |
| The Lumumba Plot | Stuart A. Reid | author bio | confirmed no dedicated Wikipedia page for this Stuart Reid (CFR senior fellow, ex-Foreign Affairs editor). The `Stuart Reid` disambiguation page's "English journalist" entry is a *different*, older person (b. 1943) — exactly the wrong-person trap from the first pass; correctly left unmatched. |
| We'll Prescribe You a Cat | Syou Ishida | author bio | no English Wikipedia page found (JP author, b. 1975 Kyoto) |
| I Deliver Parcels in Beijing | Hu Anyan | author bio | no personal English Wikipedia page for the author; only the book has one (see reverted-override note above) — using it would mislabel a book synopsis as an author bio, so left unset |

For the three JP/CN authors above (Izumi Kyōka's bio is the exception —
already resolved; Syou Ishida and Hu Anyan remain open), a non-English
Wikipedia page likely exists but the pipeline only queries
`en.wikipedia.org`. Querying `ja.wikipedia.org`/`zh.wikipedia.org` as a
fallback (with machine translation or just raw text) would be a reasonable
future improvement — out of scope here since it requires a script change,
not just an override.

These are considered acceptable gaps for the embedding step — title,
categories, and/or partial text are still available for all of them. Revisit
via `data/overrides.json` (see its `search_title`/`search_author`/
`wikipedia_title`/`google_books_id`/`open_library_key` fields) if any of
these turn out to matter for the clustering results.

## Bugs found and fixed along the way

- **Google Books keyless quota**: shared per-network and was already
  exhausted when first tested. Fixed by adding `GOOGLE_BOOKS_API_KEY` env
  var support — get a free key via Google Cloud Console, restricted to
  "Books API" only (see README).
- **Open Library fallback never returned a description**: the `/search.json`
  endpoint doesn't include one — only the `/works/OL...W.json` detail
  endpoint does. Fixed by fetching the work detail after a search match.
- **Wikipedia 429s**: added retry-with-backoff (honors `Retry-After`) in
  `cached_get`. Cut failures from 88 to 18 on a full run.
- **Redundant Wikipedia lookups**: bios are now cached per-author instead of
  per-book, since several authors repeat across the reading list.
- **Wrong-person match**: "Why Buddhism Is True" was typed as by "Robin
  Wright" in the CSV (should be Robert Wright) — auto-search matched the
  *actress* Robin Wright's Wikipedia page and used her bio. Caught by
  spot-checking, fixed via `search_author`/`wikipedia_title` override. Worth
  remembering as a class of error: auto-matched bios can silently attach the
  wrong real person, not just come back empty.
- Overrides schema extended with `search_title`/`search_author` (corrected
  query terms) alongside the original `google_books_id`/`open_library_key`/
  `wikipedia_title` (exact-ID pins), to fix typos/multi-author CSV entries
  without needing to look up exact IDs for each.
- **Cache staleness trap**: `google_books`/`open_library` cache filenames are
  derived from the *original* CSV title/author slug, not from the resolved
  `search_title`/`search_author`. So editing an override after a cache file
  already exists for that book is a silent no-op until the stale cache file
  is deleted — the script has no way to detect "the query that would
  produce this cache path has changed." Bit us mid-session: a background
  fetch was kicked off before an overrides.json edit had been saved, so the
  run silently used the old queries for several books. Caught by re-checking
  the flagged-issue count against what was expected instead of trusting the
  run blindly. Always delete the affected `cache/google_books/<slug>.json`,
  `cache/open_library/<slug>.json` (+ `<slug>_work.json`), and
  `cache/wikipedia/<wiki-slug>.json` after any overrides.json edit, before
  re-running — see the "Test cycle" note for how to compute the slug.
- **Field mislabeling risk**: tried pointing `wikipedia_title` at a *book's*
  Wikipedia page (for `I Deliver Parcels in Beijing`, whose author Hu Anyan
  has no personal page) to fill in `author_bio`. Reverted on review — the
  page is about the book, not the author, so using it would mislabel a
  synopsis as a bio. This is a different failure mode than the wrong-person
  bio above: not wrong-entity, but wrong-content-type for the field. Worth
  checking for both when reviewing any Wikipedia override.
