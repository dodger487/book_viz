#!/usr/bin/env python3
"""
Script 0: Google Sheet (published CSV) -> data/books.csv, incrementally.

Fetches the "Publish to web" CSV export of the Books tab from the reading-
tracker spreadsheet (File > Share > Publish to web, choose that one tab,
format CSV -- this exposes only that tab, not the rest of the spreadsheet,
and needs no OAuth/API key) and appends any rows not already present in
data/books.csv, matched by (title, author) case-insensitively. Existing
rows are never modified or reordered, so re-running is always idempotent
and data/overrides.json's exact "title|author" keys for existing books
stay valid.

The CSV URL comes from the BOOKS_SHEET_CSV_URL environment variable (set
as a GitHub Actions secret for the scheduled sync) or --url.

Usage:
    python scripts/00_sync_books.py                # fetch, append new rows
    python scripts/00_sync_books.py --dry-run       # show what would be added, don't write
    python scripts/00_sync_books.py --url <url>     # override the env var
"""
import argparse
import csv
import io
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BOOKS_CSV = ROOT / "data" / "books.csv"
REQUIRED_COLUMNS = {"Date", "Type", "Name", "Author"}


def book_key(title: str, author: str) -> str:
    return f"{title.strip().lower()}|{author.strip().lower()}"


def fetch_sheet_rows(url: str) -> list[dict]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
        raise SystemExit(
            f"Sheet CSV is missing expected columns {sorted(REQUIRED_COLUMNS)}; "
            f"found {reader.fieldnames}. Check the published tab/range."
        )
    return list(reader)


def load_existing() -> tuple[list[str], list[dict]]:
    with open(BOOKS_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("BOOKS_SHEET_CSV_URL"))
    parser.add_argument("--dry-run", action="store_true", help="show new rows, don't write books.csv")
    args = parser.parse_args()

    if not args.url:
        raise SystemExit("No sheet URL given -- set BOOKS_SHEET_CSV_URL or pass --url.")

    fieldnames, existing_rows = load_existing()
    existing_keys = {book_key(r["Name"], r["Author"]) for r in existing_rows}

    sheet_rows = fetch_sheet_rows(args.url)
    new_rows = []
    for r in sheet_rows:
        type_ = (r.get("Type") or "").strip()
        name = (r.get("Name") or "").strip()
        author = (r.get("Author") or "").strip()
        if type_.lower() != "book" or not name or not author:
            continue
        key = book_key(name, author)
        if key in existing_keys:
            continue
        existing_keys.add(key)  # guard against duplicate rows within the sheet itself
        new_rows.append({"Date": (r.get("Date") or "").strip(), "Type": type_, "Name": name, "Author": author})

    if not new_rows:
        print("No new books found.")
        return

    print(f"Found {len(new_rows)} new book(s):")
    for r in new_rows:
        print(f"  - {r['Name']} — {r['Author']} ({r['Date']})")

    if args.dry_run:
        print("\n(dry run -- books.csv not written)")
        return

    # Rewrite the whole file (rather than raw-appending text) so a missing
    # trailing newline on the existing file can never corrupt the last row.
    with open(BOOKS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(new_rows)
    print(f"\nAppended {len(new_rows)} row(s) to {BOOKS_CSV}")


if __name__ == "__main__":
    main()
