#!/usr/bin/env python3
"""
Script 4: book_metadata.json (+ book_tags.json for v2+) -> embedding input -> embeddings.

Builds one embedding-input text per book via a pluggable "text variant"
(see TEXT_VARIANTS below), then embeds it via a pluggable provider. Reads
data/book_metadata.json (Script 2) and, for variants that need it,
data/book_tags.json (Script 3) -- no network calls except the actual
embedding API request.

Versioning: every text variant writes its own
data/embedding_input_<variant>.json and every (provider, variant) pair
writes its own data/embeddings_<provider>_<variant>.npz, so trying a new
variant or provider never overwrites another's output. Script 5
auto-discovers every embeddings_*.npz file and lets you compare all of
them (any provider x any variant) in the interactive plot.

Incremental by default: if data/embeddings_<provider>_<variant>.npz
already exists (and matches the requested model/variant), only books
missing from it are actually sent to the embedding API -- the rest are
carried over from the existing file untouched. This is what makes the
scheduled sync (scripts/00_sync_books.py + CI) cheap: adding one new book
re-embeds one book, not the whole library. Pass --refresh to force a full
re-embed (e.g. after changing what the text variant produces for existing
books).

Output:
  data/embedding_input_<variant>.json      - constructed text per book (for inspection)
  data/embeddings_<provider>_<variant>.npz - slugs[]/embeddings[]/provider/model/variant

Usage:
    python scripts/04_generate_embeddings.py                                    # v1, local, free, default model
    python scripts/04_generate_embeddings.py --text-variant v2                  # v2 text, local
    python scripts/04_generate_embeddings.py --text-variant v2 --provider openai --model text-embedding-3-large
    python scripts/04_generate_embeddings.py --model all-MiniLM-L6-v2           # smaller/faster local model
    python scripts/04_generate_embeddings.py --provider voyage --model voyage-3-large
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
METADATA_JSON = DATA_DIR / "book_metadata.json"
TAGS_JSON = DATA_DIR / "book_tags.json"


def embedding_input_path(variant: str) -> Path:
    return DATA_DIR / f"embedding_input_{variant}.json"


def embeddings_path(provider: str, variant: str) -> Path:
    return DATA_DIR / f"embeddings_{provider}_{variant}.npz"


# --- Embedding input text variants --------------------------------------
# Each variant is build_text(book, tags) -> str, where `tags` is the
# matching data/book_tags.json record (or None for variants that don't
# need it -- see TEXT_VARIANTS_REQUIRE_TAGS). Add a new variant here and
# register it below; nothing else needs to change.


def build_text_v1(book: dict, tags: dict | None) -> str:
    """Original: title+author+year+genre+raw description+raw bio,
    concatenated as fetched. See README's "Embedding input methodology"
    section for the known issues this version has (title/author
    overweighting, unfiltered marketing boilerplate in descriptions)."""
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


def build_text_v2(book: dict, tags: dict) -> str:
    """Drops title and author entirely (the fix for v1's overweighting
    issue). Uses the LLM-cleaned description and LLM-refined genre from
    Script 3 instead of the raw fetched versions, plus the new
    tone/pacing/themes/setting tags. Still includes author_bio -- that's
    biographical/stylistic content, not just a name, and wasn't part of
    the overweighting problem the title/author *label* caused."""
    parts = []
    if book.get("published_year"):
        parts.append(f"Published {book['published_year']}.")
    if tags.get("genre"):
        parts.append(f"Genre: {tags['genre']}.")
    if tags.get("clean_description"):
        parts.append(tags["clean_description"])
    if tags.get("tone_and_style"):
        parts.append(f"Tone and style: {tags['tone_and_style']}.")
    if tags.get("pacing"):
        parts.append(f"Pacing: {tags['pacing']}.")
    if tags.get("themes"):
        parts.append(f"Themes: {tags['themes']}.")
    if tags.get("setting"):
        parts.append(f"Setting: {tags['setting']}.")
    if book.get("author_bio"):
        parts.append(f"About the author: {book['author_bio']}")
    return " ".join(parts)


TEXT_VARIANTS = {"v1": build_text_v1, "v2": build_text_v2}
TEXT_VARIANTS_REQUIRE_TAGS = {"v2"}


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
    parser.add_argument("--text-variant", choices=TEXT_VARIANTS, default="v1")
    parser.add_argument("--refresh", action="store_true", help="re-embed every book, ignoring any cached embeddings")
    args = parser.parse_args()

    if not METADATA_JSON.exists():
        raise SystemExit(f"{METADATA_JSON} not found -- run scripts/02_extract_metadata.py first.")
    books = json.loads(METADATA_JSON.read_text(encoding="utf-8"))

    tags_by_slug = {}
    if args.text_variant in TEXT_VARIANTS_REQUIRE_TAGS:
        if not TAGS_JSON.exists():
            raise SystemExit(f"{TAGS_JSON} not found -- run scripts/03_generate_tags.py first.")
        tags_by_slug = {t["slug"]: t for t in json.loads(TAGS_JSON.read_text(encoding="utf-8"))}
        missing = [b["slug"] for b in books if b["slug"] not in tags_by_slug]
        if missing:
            raise SystemExit(
                f"{len(missing)} book(s) missing tags (e.g. {missing[0]}) -- "
                f"re-run scripts/03_generate_tags.py."
            )

    build_text = TEXT_VARIANTS[args.text_variant]
    print(f"Building embedding input text for {len(books)} books (text variant: {args.text_variant})...")
    records = [
        {
            "slug": b["slug"],
            "title": b["title"],
            "author": b["author"],
            "text": build_text(b, tags_by_slug.get(b["slug"])),
        }
        for b in books
    ]
    input_path = embedding_input_path(args.text_variant)
    input_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote embedding input text to {input_path}")

    provider_cls = PROVIDERS[args.provider]
    model_name = args.model or provider_cls.default_model
    out_path = embeddings_path(args.provider, args.text_variant)

    cached_by_slug = {}
    if out_path.exists() and not args.refresh:
        npz = np.load(out_path, allow_pickle=True)
        if str(npz["model"]) == model_name and str(npz["text_variant"]) == args.text_variant:
            cached_by_slug = dict(zip(npz["slugs"].tolist(), npz["embeddings"]))
        else:
            print(f"  ({out_path.name} was built with a different model/variant -- ignoring cache, full re-embed)")

    to_embed = [r for r in records if r["slug"] not in cached_by_slug]
    print(
        f"Embedding {len(to_embed)} new text(s) (of {len(records)} total, "
        f"{len(records) - len(to_embed)} already cached) with provider={args.provider}, model={model_name}..."
    )
    if to_embed:
        provider = provider_cls(model_name)
        new_vectors = provider.embed([r["text"] for r in to_embed])
        for r, vec in zip(to_embed, new_vectors):
            cached_by_slug[r["slug"]] = vec

    embeddings = np.array([cached_by_slug[r["slug"]] for r in records])
    print(f"Got embeddings with shape {embeddings.shape}")

    slugs = np.array([r["slug"] for r in records])
    np.savez(
        out_path,
        slugs=slugs,
        embeddings=embeddings,
        provider=args.provider,
        model=model_name,
        text_variant=args.text_variant,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
