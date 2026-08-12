#!/usr/bin/env python3
"""
Script 3: embeddings -> 2D projection(s) -> static + interactive plots.

Reads every data/embeddings_<provider>.npz Script 2 has produced (so you
can generate embeddings from multiple providers -- e.g. both `local` and
`openai` -- without one overwriting the other) plus data/book_metadata.json,
reduces each embedding source to 2D via every available DimReducer, then
renders:
  - a static plot (plotnine) for one embedding-source/method/color combo,
    saved to output/static_plot.png
  - an interactive plot (plotly) with dropdowns to switch embedding source,
    projection method (PCA/t-SNE/...), and color-by (genre/year read/decade
    published) on the fly, plus hover tooltips -- saved to
    output/interactive_plot.html. Fully self-contained (plotly.js
    embedded), open directly in a browser. Year read uses a continuous
    color scale rather than decade buckets -- the reading list only spans
    ~11 years, so decade buckets would collapse almost everything into two
    colors. Hovering a point also draws lines to its 5 nearest neighbors
    in the *original* high-dimensional embedding space (see
    compute_neighbors()) -- deliberately not neighbors in the 2D
    projection, since two books can land far apart on screen (different
    genre, different era) while still being genuinely similar in the
    embedding, and that's the more interesting signal.

The interactive plot always precomputes every source x method x color
combination so the dropdowns work without re-running the script;
--source/--method/--color just pick what's selected by default when the
page loads (and what the static plot uses).

Usage:
    python scripts/03_visualize.py
    python scripts/03_visualize.py --source openai --method tsne --color decade_published
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

CATEGORICAL_COLOR_COLUMNS = ["genre", "decade_published"]
CONTINUOUS_COLOR_COLUMNS = ["year_read"]  # numeric -> continuous color scale, not a legend
COLOR_COLUMNS = CATEGORICAL_COLOR_COLUMNS + CONTINUOUS_COLOR_COLUMNS

# Plotly Express's default template colors -- set explicitly (rather than
# relying on the template) so they survive Plotly.react() layout swaps in
# the browser, which otherwise reset to a plain white background.
PLOT_BGCOLOR = "#E5ECF6"
PAPER_BGCOLOR = "white"


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
    """Perplexity roughly controls the local/global tradeoff -- lower
    values weight local structure more, which tends to produce tighter,
    more separated clusters on small datasets. Registered at a few values
    below (REDUCERS) since the right one isn't obvious upfront."""

    def __init__(self, perplexity: int = 30):
        self.perplexity = perplexity

    def reduce(self, X):
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=0, init="pca", perplexity=self.perplexity).fit_transform(X)


class UMAPReducer(DimReducer):
    """Requires `pip install umap-learn` -- not in requirements.txt by
    default since umap-learn/numba can conflict with this project's
    numpy/scipy/torch pins (see requirements.txt comment). Skipped
    automatically if not installed."""

    def reduce(self, X):
        import umap

        return umap.UMAP(n_components=2, random_state=0).fit_transform(X)


# t-SNE is cheap to compute and cheap to store (just x,y per point) at this
# dataset size, so it's fine to register many perplexity values rather than
# guessing the "right" one -- add/remove values here, nothing else to edit.
TSNE_PERPLEXITIES = [5, 6, 7, 8, 9, 10, 15, 30]

REDUCERS = {"pca": PCAReducer}
REDUCERS.update({f"tsne_p{p}": (lambda p=p: TSNEReducer(perplexity=p)) for p in TSNE_PERPLEXITIES})
REDUCERS["umap"] = UMAPReducer

REQUIRED_METHODS = {"pca"} | {f"tsne_p{p}" for p in TSNE_PERPLEXITIES}  # always available via scikit-learn

METHOD_LABELS = {"pca": "PCA"}
METHOD_LABELS.update({f"tsne_p{p}": f"t-SNE (perplexity {p})" for p in TSNE_PERPLEXITIES})
METHOD_LABELS["umap"] = "UMAP"

NUM_NEIGHBORS = 5


def compute_neighbors(embeddings: np.ndarray, slugs: list[str], k: int = NUM_NEIGHBORS) -> dict[str, list[str]]:
    """Each book's k nearest neighbors by cosine similarity in the
    *original* embedding space (not the 2D projection) -- two books can
    project far apart but still be genuinely similar (e.g. same topic,
    different genre), which is exactly what this is for.

    Exact brute-force search, not approximate: at 173 books this is a
    173x173 distance matrix, sub-millisecond regardless, and exact is both
    simpler and strictly more accurate than an ANN index here -- so no
    faiss/hnswlib/annoy dependency (this project has already hit enough
    dependency-pinning pain re-adding torch/numpy/scipy compatibility;
    not worth the risk for zero benefit at this scale). Revisit if the
    reading list ever grows into the tens of thousands of books.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(embeddings)), metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    neighbors = {}
    for i, slug in enumerate(slugs):
        neighbor_idxs = [j for j in indices[i] if j != i][:k]
        neighbors[slug] = [slugs[j] for j in neighbor_idxs]
    return neighbors


def extract_year(value) -> int | None:
    if not value:
        return None
    m = re.search(r"(\d{4})", str(value))
    return int(m.group(1)) if m else None


def decade_of(year: int | None) -> str:
    return f"{(year // 10) * 10}s" if year else "Unknown"


def compute_reading_order(df: pd.DataFrame) -> list[str]:
    """All book slugs sorted by date_read, oldest first. The client builds
    the actual hover window/edges from this (see CHAIN_WINDOW in the JS) --
    independent of embedding source, it's just reading chronology."""
    return df.sort_values("date_read", key=lambda s: pd.to_datetime(s, format="%m/%d/%Y"))["slug"].tolist()


def collapse_rare(series: pd.Series, top_n: int = 10) -> pd.Series:
    """Bucket everything outside the top_n most common values into 'Other'
    so the legend stays readable."""
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), "Other")


def load_metadata() -> pd.DataFrame:
    books = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    rows = []
    for b in books:
        year_read = extract_year(b.get("date_read"))
        year_published = extract_year(b.get("published_year"))
        rows.append(
            {
                "slug": b["slug"],
                "title": b.get("title", b["slug"]),
                "author": b.get("author", ""),
                "date_read": b.get("date_read", ""),
                "year_read": year_read,
                "year_published": year_published,
                "decade_published": decade_of(year_published),
                "genre": b.get("genre") or "Unknown",
            }
        )
    df = pd.DataFrame(rows)
    df["genre"] = collapse_rare(df["genre"])
    return df


PROVIDER_DISPLAY_NAMES = {"local": "Local", "openai": "OpenAI", "voyage": "Voyage AI"}


def load_embedding_sources(canonical_slugs: list[str]) -> dict[str, dict]:
    """Discovers every data/embeddings_<name>.npz file and reindexes each
    to canonical_slugs order (the order load_metadata() returns), so all
    sources line up with the same DataFrame regardless of what order each
    was originally embedded in."""
    sources = {}
    for path in sorted(DATA_DIR.glob("embeddings_*.npz")):
        name = path.stem[len("embeddings_") :]
        npz = np.load(path, allow_pickle=True)
        slug_to_row = {s: i for i, s in enumerate(npz["slugs"])}
        missing = [s for s in canonical_slugs if s not in slug_to_row]
        if missing:
            print(f"  (skipping {path.name}: missing {len(missing)} book(s), e.g. {missing[0]} -- regenerate it)")
            continue
        order = [slug_to_row[s] for s in canonical_slugs]
        provider = str(npz["provider"]) if "provider" in npz.files else name
        model = str(npz["model"]) if "model" in npz.files else ""
        provider_label = PROVIDER_DISPLAY_NAMES.get(provider, provider.title())
        sources[name] = {
            "embeddings": npz["embeddings"][order],
            "label": provider_label + (f" ({model})" if model else ""),
        }
    if not sources:
        raise SystemExit(
            f"No embeddings_*.npz files found in {DATA_DIR}. Run scripts/02_generate_embeddings.py first."
        )
    return sources


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


def make_static_plot(df: pd.DataFrame, all_coords: dict, source: str, method: str, color_col: str, source_label: str) -> Path:
    from plotnine import aes, element_text, geom_point, ggplot, labs, theme, theme_minimal

    d = df.copy()
    coords = all_coords[source][method]
    d["x"], d["y"] = coords[:, 0], coords[:, 1]

    p = (
        ggplot(d, aes(x="x", y="y", color=color_col))
        + geom_point(size=2.5, alpha=0.8)
        + theme_minimal()
        + labs(
            title=f"Reading taste ({source_label}, {METHOD_LABELS.get(method, method.upper())} projection)",
            x="",
            y="",
            color=color_col.replace("_", " ").title(),
        )
        + theme(figure_size=(13, 7.3), plot_title=element_text(size=14))
    )
    out_path = OUTPUT_DIR / "static_plot.png"
    p.save(out_path, dpi=150, verbose=False)
    return out_path


def build_combo_traces(df: pd.DataFrame, all_coords: dict) -> dict[str, dict]:
    """One set of plotly traces per (embedding source, method, color)
    combination, keyed 'source|method|color'. Built with plotly express so
    each color category gets its own trace (clickable legend), then
    extracted as plain trace dicts so the page can swap between combos
    client-side with Plotly.react."""
    import plotly.express as px

    combos = {}
    for source, method_coords in all_coords.items():
        for method, coords in method_coords.items():
            d = df.copy()
            d["x"], d["y"] = coords[:, 0], coords[:, 1]
            for color_col in COLOR_COLUMNS:
                extra_kwargs = {"color_continuous_scale": "Rainbow"} if color_col in CONTINUOUS_COLOR_COLUMNS else {}
                hover_data = {
                    "author": True,
                    "date_read": True,
                    "year_read": True,
                    "year_published": True,
                    "x": False,
                    "y": False,
                }
                hover_data[color_col] = True
                fig = px.scatter(
                    d,
                    x="x",
                    y="y",
                    color=color_col,
                    hover_name="title",
                    hover_data=hover_data,
                    custom_data=["slug"],  # not shown in tooltip; read by the JS hover handler below
                    **extra_kwargs,
                )
                fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
                fig_dict = fig.to_dict()
                # Continuous color (year_read) puts its colorscale/cmin/cmax on
                # layout.coloraxis, not on the trace -- stash it alongside the
                # trace data so the client-side dropdown swap can restore it.
                # Without this, switching to a continuous color-by resets to
                # Plotly's default colorscale instead of the one we picked.
                combos[f"{source}|{method}|{color_col}"] = {
                    "data": fig_dict["data"],
                    "coloraxis": fig_dict["layout"].get("coloraxis", {}),
                }
    return combos


NEIGHBOR_EDGE_COLOR = "rgba(50,50,50,0.6)"
CHAIN_EDGE_COLOR = "rgba(31,119,180,0.7)"

EDGE_TRACE = {
    "type": "scatter",
    "mode": "lines",
    "x": [],
    "y": [],
    "line": {"color": NEIGHBOR_EDGE_COLOR, "width": 1.5},
    "hoverinfo": "skip",
    "showlegend": False,
}


def make_interactive_plot(
    df: pd.DataFrame,
    all_coords: dict,
    all_neighbors: dict,
    reading_order: list[str],
    source_labels: dict,
    default_source: str,
    default_method: str,
    default_color: str,
) -> Path:
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.utils

    combos = build_combo_traces(df, all_coords)
    default_key = f"{default_source}|{default_method}|{default_color}"
    default_combo = combos.get(default_key, next(iter(combos.values())))

    # slug -> [x, y] per (source, method), so the hover handler can look up
    # neighbor coordinates regardless of which color-group trace they're in.
    slugs = df["slug"].tolist()
    coords_lookup = {
        source: {method: dict(zip(slugs, coords.tolist())) for method, coords in method_coords.items()}
        for source, method_coords in all_coords.items()
    }

    def title_for(source, method):
        return f"Reading taste ({source_labels.get(source, source)}, {METHOD_LABELS.get(method, method.upper())} projection)"

    layout = dict(
        title=title_for(default_source, default_method),
        legend_title_text=default_color.replace("_", " ").title(),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=60),
        plot_bgcolor=PLOT_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
    )
    if default_combo["coloraxis"]:
        layout["coloraxis"] = default_combo["coloraxis"]
    fig = go.Figure(data=default_combo["data"] + [EDGE_TRACE], layout=layout)
    plot_div = pio.to_html(
        fig, include_plotlyjs=True, full_html=False, div_id="book-plot", config={"responsive": True}
    )

    source_options_html = "".join(
        f'<option value="{s}"{" selected" if s == default_source else ""}>{source_labels.get(s, s)}</option>'
        for s in all_coords
    )
    any_method_coords = next(iter(all_coords.values()))
    method_options_html = "".join(
        f'<option value="{m}"{" selected" if m == default_method else ""}>{METHOD_LABELS.get(m, m.upper())}</option>'
        for m in any_method_coords
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
    <label for="source-select">Embedding source:</label>
    <select id="source-select">{source_options_html}</select>
    <label for="method-select">Projection method:</label>
    <select id="method-select">{method_options_html}</select>
    <label for="color-select">Color by:</label>
    <select id="color-select">{color_options_html}</select>
    <label for="edges-select">On hover, show:</label>
    <select id="edges-select">
      <option value="off">No links</option>
      <option value="neighbors" selected>Nearest neighbors (by similarity)</option>
      <option value="chain">Reading order (chronological)</option>
    </select>
  </div>
  {plot_div}
  <script>
    const combos = {json.dumps(combos, cls=plotly.utils.PlotlyJSONEncoder)};
    const neighbors = {json.dumps(all_neighbors)};
    const readingOrder = {json.dumps(reading_order)};
    const slugToOrderIndex = {{}};
    readingOrder.forEach((s, i) => {{ slugToOrderIndex[s] = i; }});
    const CHAIN_WINDOW = 2;  // 2 back + 2 forward + the hovered book = 5 nodes, 4 sequential edges
    const coordsLookup = {json.dumps(coords_lookup)};
    const edgeTraceTemplate = {json.dumps(EDGE_TRACE)};
    const neighborEdgeColor = {json.dumps(NEIGHBOR_EDGE_COLOR)};
    const chainEdgeColor = {json.dumps(CHAIN_EDGE_COLOR)};
    const sourceLabels = {json.dumps(source_labels)};
    const methodLabels = {json.dumps(METHOD_LABELS)};
    const colorLabels = {json.dumps(color_labels)};
    const sourceEl = document.getElementById('source-select');
    const methodEl = document.getElementById('method-select');
    const colorEl = document.getElementById('color-select');
    const edgesEl = document.getElementById('edges-select');
    const plotDiv = document.getElementById('book-plot');

    function update() {{
      const source = sourceEl.value;
      const method = methodEl.value;
      const color = colorEl.value;
      const combo = combos[source + '|' + method + '|' + color];
      // combo.coloraxis carries the colorscale/cmin/cmax for continuous
      // color-by columns (e.g. year read) -- without re-applying it here,
      // Plotly.react would fall back to its default colorscale.
      const coloraxis = Object.assign({{}}, combo.coloraxis, {{ colorbar: {{ title: {{ text: colorLabels[color] }} }} }});
      const sourceLabel = sourceLabels[source] || source;
      // Append a fresh (empty) edges trace as the last trace -- combo.data
      // is the shared precomputed array, so copy rather than mutate it, or
      // repeated switches back to the same combo would pile up edge traces.
      const dataWithEdges = combo.data.concat([Object.assign({{}}, edgeTraceTemplate)]);
      Plotly.react(plotDiv, dataWithEdges, {{
        title: 'Reading taste (' + sourceLabel + ', ' + (methodLabels[method] || method.toUpperCase()) + ' projection)',
        legend: {{ title: {{ text: colorLabels[color] }} }},
        coloraxis: coloraxis,
        xaxis: {{ visible: false }},
        yaxis: {{ visible: false }},
        margin: {{ t: 60 }},
        plot_bgcolor: {json.dumps(PLOT_BGCOLOR)},
        paper_bgcolor: {json.dumps(PAPER_BGCOLOR)},
        annotations: [],
      }});
    }}
    sourceEl.addEventListener('change', update);
    methodEl.addEventListener('change', update);
    colorEl.addEventListener('change', update);

    function clearEdges() {{
      const edgeTraceIndex = plotDiv.data.length - 1;
      Plotly.restyle(plotDiv, {{x: [[]], y: [[]]}}, [edgeTraceIndex]);
      Plotly.relayout(plotDiv, {{annotations: []}});
    }}

    function showNeighborEdges(slug, source, method) {{
      const related = (neighbors[source] || {{}})[slug] || [];
      const coordsForMethod = (coordsLookup[source] || {{}})[method] || {{}};
      const origin = coordsForMethod[slug];
      if (!origin || related.length === 0) return;

      const xs = [], ys = [];
      for (const nSlug of related) {{
        const target = coordsForMethod[nSlug];
        if (!target) continue;
        xs.push(origin[0], target[0], null);
        ys.push(origin[1], target[1], null);
      }}
      const edgeTraceIndex = plotDiv.data.length - 1;
      Plotly.restyle(plotDiv, {{x: [xs], y: [ys], 'line.color': [neighborEdgeColor]}}, [edgeTraceIndex]);
    }}

    function showReadingChain(slug, source, method) {{
      // 5-node window centered on the hovered book: 2 read before it, 2
      // read after (clipped at either end of the whole reading history),
      // connected as a single sequential path (4 edges), not a star --
      // that's what makes it a "chain".
      const centerIdx = slugToOrderIndex[slug];
      if (centerIdx === undefined) return;
      const start = Math.max(0, centerIdx - CHAIN_WINDOW);
      const end = Math.min(readingOrder.length - 1, centerIdx + CHAIN_WINDOW);
      const windowSlugs = readingOrder.slice(start, end + 1);

      const coordsForMethod = (coordsLookup[source] || {{}})[method] || {{}};
      const annotations = [];
      for (let i = 0; i < windowSlugs.length - 1; i++) {{
        const earlier = coordsForMethod[windowSlugs[i]];
        const later = coordsForMethod[windowSlugs[i + 1]];
        if (!earlier || !later) continue;
        // Arrow always points from the earlier-read book to the
        // later-read book, so direction = chronological direction.
        annotations.push({{
          x: later[0], y: later[1], xref: 'x', yref: 'y',
          ax: earlier[0], ay: earlier[1], axref: 'x', ayref: 'y',
          showarrow: true, arrowhead: 3, arrowsize: 1.2, arrowwidth: 2,
          arrowcolor: chainEdgeColor, standoff: 7, startstandoff: 7, text: '',
        }});
      }}
      Plotly.relayout(plotDiv, {{annotations: annotations}});
    }}

    // Hover a point -> draw edges to related books, source depending on
    // the "On hover, show" dropdown:
    //   - neighbors: the 5 nearest neighbors by cosine similarity in the
    //     *original* high-dimensional embedding space (computed in Python
    //     via compute_neighbors), NOT neighbors in the current 2D
    //     projection -- some books land far apart on screen but are still
    //     genuinely similar (same topic, different genre/projection quirk).
    //     Undirected, so plain lines.
    //   - chain: a directed 5-node path (2 read before, 2 read after)
    //     through reading order (compute_reading_order), independent of
    //     the embedding entirely. Rendered as arrows (Plotly annotations,
    //     not the plain-line edge trace) pointing oldest -> newest so the
    //     chronological direction is visible.
    //   - off: no edges.
    plotDiv.on('plotly_hover', function (evt) {{
      const mode = edgesEl.value;
      if (mode === 'off') return;

      const point = evt.points[0];
      if (!point.customdata) return;
      const slug = point.customdata[0];
      const source = sourceEl.value;
      const method = methodEl.value;

      if (mode === 'chain') {{
        const edgeTraceIndex = plotDiv.data.length - 1;
        Plotly.restyle(plotDiv, {{x: [[]], y: [[]]}}, [edgeTraceIndex]);
        showReadingChain(slug, source, method);
      }} else {{
        Plotly.relayout(plotDiv, {{annotations: []}});
        showNeighborEdges(slug, source, method);
      }}
    }});
    plotDiv.on('plotly_unhover', clearEdges);
    edgesEl.addEventListener('change', clearEdges);
  </script>
</body>
</html>
"""
    out_path = OUTPUT_DIR / "interactive_plot.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="default embedding source selection (e.g. local, openai); defaults to whichever is found first")
    parser.add_argument("--method", choices=list(REDUCERS), default="pca", help="default selection on load / static plot method")
    parser.add_argument("--color", choices=COLOR_COLUMNS, default="genre", help="default selection on load / static plot color")
    args = parser.parse_args()

    df = load_metadata()
    embedding_sources = load_embedding_sources(df["slug"].tolist())
    source_labels = {name: data["label"] for name, data in embedding_sources.items()}
    print(f"Found embedding sources: {', '.join(f'{n} ({l})' for n, l in source_labels.items())}")

    default_source = args.source or next(iter(embedding_sources))
    if default_source not in embedding_sources:
        raise SystemExit(f"--source {default_source!r} not found. Available: {list(embedding_sources)}")

    slugs = df["slug"].tolist()
    all_coords = {}
    all_neighbors = {}
    for name, data in embedding_sources.items():
        print(f"Reducing {data['embeddings'].shape} embeddings ({name}) to 2D...")
        all_coords[name] = compute_reductions(data["embeddings"])
        all_neighbors[name] = compute_neighbors(data["embeddings"], slugs)
    if args.method not in all_coords[default_source]:
        raise SystemExit(f"--method {args.method} unavailable (missing optional dependency)")

    reading_order = compute_reading_order(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    static_path = make_static_plot(
        df, all_coords, default_source, args.method, args.color, source_labels[default_source]
    )
    print(f"Wrote {static_path}")
    interactive_path = make_interactive_plot(
        df, all_coords, all_neighbors, reading_order, source_labels, default_source, args.method, args.color
    )
    print(f"Wrote {interactive_path}")


if __name__ == "__main__":
    main()
