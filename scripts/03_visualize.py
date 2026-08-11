#!/usr/bin/env python3
"""
Script 3: embeddings -> 2D projection(s) -> static + interactive plots.

Reads data/embeddings.npz + data/book_metadata.json, reduces the embeddings
to 2D via every available DimReducer, then renders:
  - a static plot (plotnine) for one method/color combo, saved to
    output/static_plot.png
  - an interactive plot (plotly) with dropdowns to switch projection method
    (PCA/t-SNE/...) and color-by (genre/year read/decade published) on
    the fly, plus hover tooltips -- saved to output/interactive_plot.html.
    Fully self-contained (plotly.js embedded), open directly in a browser.
    Year read uses a continuous color scale rather than decade buckets --
    the reading list only spans ~11 years, so decade buckets would collapse
    almost everything into two colors.

The interactive plot always precomputes every method x color combination
so the dropdowns work without re-running the script; --method/--color just
pick what's selected by default when the page loads (and what the static
plot uses).

Usage:
    python scripts/03_visualize.py
    python scripts/03_visualize.py --method tsne --color decade_read
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
METADATA_JSON = DATA_DIR / "book_metadata.json"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"

CATEGORICAL_COLOR_COLUMNS = ["genre", "decade_published"]
CONTINUOUS_COLOR_COLUMNS = ["year_read"]  # numeric -> continuous color scale, not a legend
COLOR_COLUMNS = CATEGORICAL_COLOR_COLUMNS + CONTINUOUS_COLOR_COLUMNS


# --- Dimensionality reduction -------------------------------------------
# Pluggable the same way Script 2's embedding providers are: implement
# .reduce(X) -> (n, 2) array and register the class in REDUCERS. Every
# reducer here is used for the interactive plot's method dropdown, so keep
# this list to reducers that don't need extra setup beyond requirements.txt.


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
    numpy/scipy/torch pins (see requirements.txt comment). Skipped
    automatically if not installed."""

    def reduce(self, X):
        import umap

        return umap.UMAP(n_components=2, random_state=0).fit_transform(X)


REDUCERS = {"pca": PCAReducer, "tsne": TSNEReducer, "umap": UMAPReducer}
REQUIRED_METHODS = {"pca", "tsne"}  # always available via scikit-learn


def extract_year(value) -> int | None:
    if not value:
        return None
    m = re.search(r"(\d{4})", str(value))
    return int(m.group(1)) if m else None


def decade_of(year: int | None) -> str:
    return f"{(year // 10) * 10}s" if year else "Unknown"


def collapse_rare(series: pd.Series, top_n: int = 10) -> pd.Series:
    """Bucket everything outside the top_n most common values into 'Other'
    so the legend stays readable."""
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), "Other")


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    npz = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    slugs = npz["slugs"]
    embeddings = npz["embeddings"]

    books = {b["slug"]: b for b in json.loads(METADATA_JSON.read_text(encoding="utf-8"))}

    rows = []
    for slug in slugs:
        b = books.get(slug, {})
        year_read = extract_year(b.get("date_read"))
        year_published = extract_year(b.get("published_year"))
        genre = (b.get("categories") or ["Unknown"])[0]
        rows.append(
            {
                "slug": slug,
                "title": b.get("title", slug),
                "author": b.get("author", ""),
                "date_read": b.get("date_read", ""),
                "year_read": year_read,
                "decade_published": decade_of(year_published),
                "genre": genre,
            }
        )
    df = pd.DataFrame(rows)
    df["genre"] = collapse_rare(df["genre"])
    return df, embeddings


def compute_reductions(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    coords = {}
    for name, cls in REDUCERS.items():
        try:
            coords[name] = cls().reduce(embeddings)
        except ImportError:
            if name in REQUIRED_METHODS:
                raise
            print(f"  (skipping --method {name}: optional dependency not installed)")
    return coords


def make_static_plot(df: pd.DataFrame, method_coords: dict, method: str, color_col: str) -> Path:
    from plotnine import aes, element_text, geom_point, ggplot, labs, theme, theme_minimal

    d = df.copy()
    d["x"], d["y"] = method_coords[method][:, 0], method_coords[method][:, 1]

    p = (
        ggplot(d, aes(x="x", y="y", color=color_col))
        + geom_point(size=2.5, alpha=0.8)
        + theme_minimal()
        + labs(
            title=f"Reading taste ({method.upper()} projection)",
            x="",
            y="",
            color=color_col.replace("_", " ").title(),
        )
        + theme(figure_size=(13, 7.3), plot_title=element_text(size=14))
    )
    out_path = OUTPUT_DIR / "static_plot.png"
    p.save(out_path, dpi=150, verbose=False)
    return out_path


def build_combo_traces(df: pd.DataFrame, method_coords: dict) -> dict[str, list]:
    """One set of plotly traces per (method, color) combination, keyed
    'method|color'. Built with plotly express so each color category gets
    its own trace (clickable legend), then extracted as plain trace dicts
    so the page can swap between combos client-side with Plotly.react."""
    import plotly.express as px

    combos = {}
    for method, coords in method_coords.items():
        d = df.copy()
        d["x"], d["y"] = coords[:, 0], coords[:, 1]
        for color_col in COLOR_COLUMNS:
            extra_kwargs = {"color_continuous_scale": "Viridis"} if color_col in CONTINUOUS_COLOR_COLUMNS else {}
            fig = px.scatter(
                d,
                x="x",
                y="y",
                color=color_col,
                hover_name="title",
                hover_data={"author": True, "date_read": True, color_col: True, "x": False, "y": False},
                **extra_kwargs,
            )
            fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
            combos[f"{method}|{color_col}"] = fig.to_dict()["data"]
    return combos


def make_interactive_plot(df: pd.DataFrame, method_coords: dict, default_method: str, default_color: str) -> Path:
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.utils

    combos = build_combo_traces(df, method_coords)
    default_key = f"{default_method}|{default_color}"
    default_data = combos.get(default_key, next(iter(combos.values())))

    layout = dict(
        title=f"Reading taste ({default_method.upper()} projection)",
        legend_title_text=default_color.replace("_", " ").title(),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=60),
    )
    fig = go.Figure(data=default_data, layout=layout)
    plot_div = pio.to_html(
        fig, include_plotlyjs=True, full_html=False, div_id="book-plot", config={"responsive": True}
    )

    method_labels = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
    method_options_html = "".join(
        f'<option value="{m}"{" selected" if m == default_method else ""}>{method_labels.get(m, m.upper())}</option>'
        for m in method_coords
    )
    color_labels = {c: c.replace("_", " ").title() for c in COLOR_COLUMNS}
    color_options_html = "".join(
        f'<option value="{c}"{" selected" if c == default_color else ""}>{color_labels[c]}</option>'
        for c in COLOR_COLUMNS
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Reading Taste Visualization</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 16px; }}
  .controls {{ margin-bottom: 12px; }}
  .controls label {{ margin-right: 6px; font-weight: 600; }}
  .controls select {{ margin-right: 24px; padding: 4px 8px; font-size: 14px; }}
  #book-plot {{ width: 100%; max-width: 1300px; aspect-ratio: 16 / 9; margin: 0 auto; }}
</style>
</head>
<body>
  <div class="controls">
    <label for="method-select">Projection method:</label>
    <select id="method-select">{method_options_html}</select>
    <label for="color-select">Color by:</label>
    <select id="color-select">{color_options_html}</select>
  </div>
  {plot_div}
  <script>
    const combos = {json.dumps(combos, cls=plotly.utils.PlotlyJSONEncoder)};
    const methodLabels = {json.dumps(method_labels)};
    const colorLabels = {json.dumps(color_labels)};
    const methodEl = document.getElementById('method-select');
    const colorEl = document.getElementById('color-select');
    const plotDiv = document.getElementById('book-plot');

    function update() {{
      const method = methodEl.value;
      const color = colorEl.value;
      const data = combos[method + '|' + color];
      Plotly.react(plotDiv, data, {{
        title: 'Reading taste (' + (methodLabels[method] || method.toUpperCase()) + ' projection)',
        legend: {{ title: {{ text: colorLabels[color] }} }},
        coloraxis: {{ colorbar: {{ title: {{ text: colorLabels[color] }} }} }},
        xaxis: {{ visible: false }},
        yaxis: {{ visible: false }},
        margin: {{ t: 60 }},
      }});
    }}
    methodEl.addEventListener('change', update);
    colorEl.addEventListener('change', update);
  </script>
</body>
</html>
"""
    out_path = OUTPUT_DIR / "interactive_plot.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(REDUCERS), default="pca", help="default selection on load / static plot method")
    parser.add_argument("--color", choices=COLOR_COLUMNS, default="genre", help="default selection on load / static plot color")
    args = parser.parse_args()

    df, embeddings = load_data()
    print(f"Reducing {embeddings.shape} embeddings to 2D...")
    method_coords = compute_reductions(embeddings)
    if args.method not in method_coords:
        raise SystemExit(f"--method {args.method} unavailable (missing optional dependency)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    static_path = make_static_plot(df, method_coords, args.method, args.color)
    print(f"Wrote {static_path}")
    interactive_path = make_interactive_plot(df, method_coords, args.method, args.color)
    print(f"Wrote {interactive_path}")


if __name__ == "__main__":
    main()
