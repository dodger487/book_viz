#!/usr/bin/env python3
"""
Script 3: embeddings -> 2D projection -> static + interactive plots.

Reads data/embeddings.npz + data/book_metadata.json, reduces the embeddings
to 2D via a pluggable DimReducer, then renders:
  - a static plot (plotnine) saved to output/static_plot.png
  - an interactive plot (plotly) with hover tooltips, saved to
    output/interactive_plot.html -- open it directly in a browser

Usage:
    python scripts/03_visualize.py                    # PCA (default), colored by genre
    python scripts/03_visualize.py --method tsne
    python scripts/03_visualize.py --color decade
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
METADATA_JSON = DATA_DIR / "book_metadata.json"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"


# --- Dimensionality reduction -------------------------------------------
# Pluggable the same way Script 2's embedding providers are: implement
# .reduce(X) -> (n, 2) array and register the class in REDUCERS.


class DimReducer:
    def reduce(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PCAReducer(DimReducer):
    def reduce(self, X):
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=0).fit_transform(X)


class TSNEReducer(DimReducer):
    def reduce(self, X):
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(X)


class UMAPReducer(DimReducer):
    """Requires `pip install umap-learn` -- not in requirements.txt by
    default since umap-learn/numba can conflict with this project's
    numpy/scipy/torch pins (see requirements.txt comment)."""

    def reduce(self, X):
        import umap

        return umap.UMAP(n_components=2, random_state=0).fit_transform(X)


REDUCERS = {"pca": PCAReducer, "tsne": TSNEReducer, "umap": UMAPReducer}


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    npz = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    slugs = npz["slugs"]
    embeddings = npz["embeddings"]

    books = {b["slug"]: b for b in json.loads(METADATA_JSON.read_text(encoding="utf-8"))}

    rows = []
    for slug in slugs:
        b = books.get(slug, {})
        year_read = None
        if b.get("date_read"):
            year_read = int(b["date_read"].split("/")[-1])
        genre = (b.get("categories") or ["Unknown"])[0]
        rows.append(
            {
                "slug": slug,
                "title": b.get("title", slug),
                "author": b.get("author", ""),
                "date_read": b.get("date_read", ""),
                "decade_read": f"{(year_read // 10) * 10}s" if year_read else "Unknown",
                "genre": genre,
            }
        )
    return pd.DataFrame(rows), embeddings


def collapse_rare(series: pd.Series, top_n: int = 10) -> pd.Series:
    """Bucket everything outside the top_n most common values into 'Other'
    so the legend stays readable."""
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), "Other")


def make_static_plot(df: pd.DataFrame, color_col: str, method: str) -> Path:
    from plotnine import aes, element_text, geom_point, ggplot, labs, theme, theme_minimal

    p = (
        ggplot(df, aes(x="x", y="y", color=color_col))
        + geom_point(size=2.5, alpha=0.8)
        + theme_minimal()
        + labs(
            title=f"Reading taste ({method.upper()} projection)",
            x="",
            y="",
            color=color_col.replace("_", " ").title(),
        )
        + theme(figure_size=(10, 8), plot_title=element_text(size=14))
    )
    out_path = OUTPUT_DIR / "static_plot.png"
    p.save(out_path, dpi=150, verbose=False)
    return out_path


def make_interactive_plot(df: pd.DataFrame, color_col: str, method: str) -> Path:
    import plotly.express as px

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        hover_name="title",
        hover_data={"author": True, "date_read": True, color_col: True, "x": False, "y": False},
        title=f"Reading taste ({method.upper()} projection)",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")))
    fig.update_layout(legend_title_text=color_col.replace("_", " ").title())
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    out_path = OUTPUT_DIR / "interactive_plot.html"
    fig.write_html(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=REDUCERS, default="pca")
    parser.add_argument("--color", choices=["genre", "decade"], default="genre")
    args = parser.parse_args()

    df, embeddings = load_data()
    print(f"Reducing {embeddings.shape} embeddings to 2D via {args.method}...")
    coords = REDUCERS[args.method]().reduce(embeddings)
    df["x"], df["y"] = coords[:, 0], coords[:, 1]

    color_col = "genre" if args.color == "genre" else "decade_read"
    df[color_col] = collapse_rare(df[color_col])

    OUTPUT_DIR.mkdir(exist_ok=True)
    static_path = make_static_plot(df, color_col, args.method)
    print(f"Wrote {static_path}")
    interactive_path = make_interactive_plot(df, color_col, args.method)
    print(f"Wrote {interactive_path}")


if __name__ == "__main__":
    main()
