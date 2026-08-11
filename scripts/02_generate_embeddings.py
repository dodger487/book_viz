#!/usr/bin/env python3
"""
Script 2: book_metadata.json -> embedding input text -> embeddings.

Builds one embedding-input text per book by combining title/author,
publication year, genre/category tags, synopsis, and author bio, then
embeds it via a pluggable provider.

Output:
  data/embedding_input.json  - the constructed text per book (for inspection)
  data/embeddings.npz        - slugs[] and embeddings[] arrays, aligned by index

Usage:
    python scripts/02_generate_embeddings.py                         # local, free, default model
    python scripts/02_generate_embeddings.py --model all-MiniLM-L6-v2  # smaller/faster local model
    python scripts/02_generate_embeddings.py --provider openai --model text-embedding-3-large
    python scripts/02_generate_embeddings.py --provider voyage --model voyage-3-large
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
METADATA_JSON = DATA_DIR / "book_metadata.json"
EMBEDDING_INPUT_JSON = DATA_DIR / "embedding_input.json"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"


def build_text(book: dict) -> str:
    """Combine synopsis + structured tags into one embedding-input string."""
    parts = [f"{book['title']} by {book['author']}."]
    if book.get("published_year"):
        parts.append(f"Published {book['published_year']}.")
    if book.get("categories"):
        parts.append(f"Genre/category: {', '.join(book['categories'])}.")
    if book.get("description"):
        parts.append(book["description"])
    if book.get("author_bio"):
        parts.append(f"About the author: {book['author_bio']}")
    return " ".join(parts)


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
    args = parser.parse_args()

    books = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    print(f"Building embedding input text for {len(books)} books...")

    records = [
        {"slug": b["slug"], "title": b["title"], "author": b["author"], "text": build_text(b)}
        for b in books
    ]
    EMBEDDING_INPUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote embedding input text to {EMBEDDING_INPUT_JSON}")

    provider_cls = PROVIDERS[args.provider]
    provider = provider_cls(args.model or provider_cls.default_model)

    print(f"Embedding {len(records)} texts with provider={args.provider}, model={args.model or provider_cls.default_model}...")
    embeddings = provider.embed([r["text"] for r in records])
    print(f"Got embeddings with shape {embeddings.shape}")

    slugs = np.array([r["slug"] for r in records])
    np.savez(EMBEDDINGS_NPZ, slugs=slugs, embeddings=embeddings, provider=args.provider)
    print(f"Wrote {EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()
