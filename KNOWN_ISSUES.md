# Known data issues (Script 1 output)

Last updated after the full 173-book run with `data/overrides.json` applied.
Regenerate this picture by running `scripts/01_fetch_metadata.py` — it
prints the same flagged list at the end.

## Current coverage

- 173/173 books have a record in `data/book_metadata.json`.
- 171/173 have at least one of description/author_bio.
- 2/173 (`The Mote in God's Eye`, `Lean Analytics`) are missing both.

## Remaining flagged books (14 flags / 12 books)

| Book | Author (as typed in books.csv) | Missing | Why |
|---|---|---|---|
| Democracy for Realists | Christopher Achens and Larry Bartels | author bio | multi-author credit; pipeline only fetches one bio/book |
| Getting to Yes | Roger Fisher and William Ury | description | multi-author credit |
| The Black Monday Murders: Volume 1 | Jonathan Hickman | description | no confident Google Books/Open Library match found |
| Four Futures | Peter Frase | author bio | no English Wikipedia page found |
| The Mote in God's Eye | Larry Niven and Jerry Pournelle | both | multi-author credit |
| Getting It Done: How to Lead When You're Not In Charge | John Richardson and Roger Fisher | description | CSV's 2nd author looks wrong (real co-author is likely Alan Sharp) — left uncorrected rather than guess |
| The Saint of Mt. Koya | Izumi Kyora | description | translated/older Japanese title, poor source indexing |
| Lean Analytics | Alistair Croll and Benjamin Yoskovitz | both | multi-author credit |
| The Lumumba Plot | Stuart A. Reid | author bio | "Stuart Reid" is an ambiguous name; couldn't confirm which Wikipedia page (if any) is this author |
| Vanishing Treasures | Katherine Rundell | description | no confident match found |
| We'll Prescribe You a Cat | Syou Ishida | author bio | no English Wikipedia page found (JP author) |
| I Deliver Parcels in Beijing | Hu Anyan | author bio | no English Wikipedia page found (CN author) |

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
