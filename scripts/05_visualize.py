#!/usr/bin/env python3
"""
Script 5: embeddings -> 2D projection(s) -> static + interactive plots.

Reads every data/embeddings_<provider>_<variant>.npz Script 4 has produced
(so you can generate embeddings from multiple providers AND multiple
embedding-input text variants -- e.g. `local`/`openai` x `v1`/`v2` -- and
none of them overwrite each other) plus data/book_metadata.json and, if
present, data/book_tags.json (Script 3's LLM-refined genre becomes an
extra "genre_llm" color-by option, regardless of which embedding source
is selected), reduces each embedding source to 2D via every available
DimReducer, then renders:
  - a static plot (plotnine) for one embedding-source/method/color combo,
    saved to output/static_plot.png
  - an interactive plot (plotly) with dropdowns to switch embedding source,
    projection method (PCA/t-SNE/...), and color-by (genre/year read/decade
    published) on the fly, plus hover tooltips -- saved to
    output/interactive_plot.html. Fully self-contained (plotly.js
    embedded), open directly in a browser. Color is assigned by the job it
    does (see color_order_and_map()): genre is nominal identity, so it gets
    the fixed-order categorical palette; decade/year read are ordinal (the
    order is the point), so they get a single hue stepped light->dark in
    chronological order, never an arbitrary/rainbow scale. Hovering a
    point also draws lines to its 5 nearest neighbors
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
    python scripts/05_visualize.py
    python scripts/05_visualize.py --source openai_v2 --method tsne_p5 --color decade_published
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
TAGS_JSON = DATA_DIR / "book_tags.json"
HAS_LLM_TAGS = TAGS_JSON.exists()

# Color is assigned by the job it does (see the dataviz design system):
#   - nominal: identity, order doesn't mean anything (genre) -> fixed-order
#     8-hue categorical palette, validated colorblind-safe in adjacent pairs.
#   - ordinal: position in a sequence (decade/year) -> a single hue, light to
#     dark, so the reading order is visible in the color itself. Never a
#     rainbow -- that was the bug with the old "year read" continuous scale,
#     which used every hue with no ordering signal at all.
NOMINAL_COLOR_COLUMNS = ["genre"]
if HAS_LLM_TAGS:
    NOMINAL_COLOR_COLUMNS.append("genre_llm")
ORDINAL_COLOR_COLUMNS = ["decade_published", "year_read"]
COLOR_COLUMNS = NOMINAL_COLOR_COLUMNS + ORDINAL_COLOR_COLUMNS
COLOR_LABEL_OVERRIDES = {"genre_llm": "Genre (LLM-refined)"}


def color_label(color_col: str) -> str:
    return COLOR_LABEL_OVERRIDES.get(color_col, color_col.replace("_", " ").title())


# Validated palette (see dataviz skill's references/palette.md) -- 8 fixed
# hues in fixed order for nominal/identity color, extended below for genre
# columns that have more than 8 real categories (by explicit request: more
# distinct genres on screen, traded off against strict all-pairs
# colorblind-safety past slot 8 -- acceptable here since same-genre points
# already cluster together, so exact neighbor/adjacent contrast matters
# less than overall variety). NEUTRAL_COLOR is muted ink, used for "Other"/
# "Unknown" buckets (never a real palette slot, so those buckets never
# look like a real category or a real position in the order).
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
# Diverging pair (see palette.md "Diverging pair"): blue <-> red poles with
# a neutral gray midpoint. Used for ordinal/ordered color (decade/year) --
# stronger visual contrast than a single-hue light->dark ramp, by request.
DIVERGING_COOL = "#2a78d6"  # blue pole (oldest/earliest)
DIVERGING_MIDPOINT = "#6e6c66"  # neutral gray midpoint -- darkened from the
# documented light-surface value (#f0efec) because that's ~1.03:1 contrast
# against PLOT_BGCOLOR (a light blue tint, not the neutral white/off-white
# surface the reference palette assumes), i.e. effectively invisible.
# #6e6c66 -- a darker step of the same muted-ink family as NEUTRAL_COLOR --
# clears 4.4:1 against PLOT_BGCOLOR.
DIVERGING_WARM = "#e34948"  # red pole (newest/latest)
NEUTRAL_COLOR = "#898781"  # muted ink -- "Other"/"Unknown", never a data value


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(c))) for c in rgb))


def _lerp_hex(a: str, b: str, t: float) -> str:
    ra, rb = _hex_to_rgb(a), _hex_to_rgb(b)
    return _rgb_to_hex(tuple(ra[c] + (rb[c] - ra[c]) * t for c in range(3)))


def interpolate_ordinal_colors(n: int) -> list[str]:
    """n evenly-spaced colors from blue (earliest) through a neutral gray
    midpoint to red (latest) -- the documented diverging pair (palette.md),
    used here (rather than the single-hue sequential ramp the design
    system would default to for ordinal data) for stronger visual
    contrast, by request. 'Equal step count per arm': the midpoint category
    (if n is odd) gets the neutral gray itself."""
    if n <= 1:
        return [DIVERGING_MIDPOINT]
    mid = (n - 1) / 2
    colors = []
    for i in range(n):
        t = i / mid  # 0 at first category, 1 at the midpoint, 2 at the last
        if t <= 1:
            colors.append(_lerp_hex(DIVERGING_COOL, DIVERGING_MIDPOINT, t))
        else:
            colors.append(_lerp_hex(DIVERGING_MIDPOINT, DIVERGING_WARM, t - 1))
    return colors


def extended_categorical_palette(n: int) -> list[str]:
    """The 8 validated hues first (best colorblind-safety, used for the
    most common categories), then additional hues for anything beyond 8,
    evenly spaced around the remaining hue circle (a 0.045 phase offset
    keeps them from landing exactly on a base hue) -- computed, not
    eyeballed, even though they're not from the validated instance file.
    Deterministic and O(n): an earlier rejection-sampling version could
    need more hue-circle room than exists once collision checks pile up
    against every previously placed hue, and would spin (near-)forever."""
    import colorsys

    base = list(CATEGORICAL_PALETTE)
    if n <= len(base):
        return base[:n]
    n_extra = n - len(base)
    extra = []
    for k in range(n_extra):
        h = (0.045 + k / n_extra) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, 0.52, 0.55)
        extra.append(_rgb_to_hex((r * 255, g * 255, b * 255)))
    return base + extra


def nominal_color_map(categories_by_frequency: list[str]) -> dict[str, str]:
    """categories_by_frequency: most-common first. Slot assignment is in
    that fixed order -- never re-derived per book list, so a color keeps
    meaning the same genre across runs as long as its rank is stable."""
    real = [c for c in categories_by_frequency if c != "Other"]
    palette = extended_categorical_palette(len(real))
    cmap = dict(zip(real, palette))
    if "Other" in categories_by_frequency:
        cmap["Other"] = NEUTRAL_COLOR
    return cmap


def ordinal_color_map(categories_chronological: list[str]) -> dict[str, str]:
    """categories_chronological: oldest/earliest first. 'Unknown' (if
    present) gets the neutral color instead of a ramp step -- it has no
    position in the sequence, so it shouldn't look like it does."""
    real = [c for c in categories_chronological if c != "Unknown"]
    cmap = dict(zip(real, interpolate_ordinal_colors(len(real))))
    if "Unknown" in categories_chronological:
        cmap["Unknown"] = NEUTRAL_COLOR
    return cmap

# Plotly Express's default template colors -- set explicitly (rather than
# relying on the template) so they survive Plotly.react() layout swaps in
# the browser, which otherwise reset to a plain white background.
PLOT_BGCOLOR = "#E5ECF6"
PAPER_BGCOLOR = "white"

# Plotly renders its own text (legend, tooltip, axis, title) with its own
# default font, ignoring the page's CSS -- has to be set explicitly or it
# silently mismatches whatever font the surrounding HTML uses. Public Sans
# for everything Plotly draws except the in-chart title, which gets the
# page's serif (Fraunces) to match the <h1> above the plot.
PLOT_FONT_FAMILY = "Public Sans, -apple-system, sans-serif"
PLOT_TITLE_FONT_FAMILY = "Fraunces, Georgia, serif"


# --- Dimensionality reduction -------------------------------------------
# Pluggable the same way Script 4's embedding providers are: implement
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


def collapse_rare(series: pd.Series, top_n: int = 100) -> pd.Series:
    """Bucket everything outside the top_n most common values into 'Other'.
    Defaults to effectively unlimited (100 >> any genre count here) --
    by request, show every real genre rather than folding rare ones into
    'Other', even though that makes the legend longer. Still available
    with an explicit lower top_n if a future color column needs folding."""
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), "Other")


def load_metadata() -> pd.DataFrame:
    books = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    tags_by_slug = {}
    if HAS_LLM_TAGS:
        tags_by_slug = {t["slug"]: t for t in json.loads(TAGS_JSON.read_text(encoding="utf-8"))}

    rows = []
    for b in books:
        year_read = extract_year(b.get("date_read"))
        year_published = extract_year(b.get("published_year"))
        tags = tags_by_slug.get(b["slug"])
        row = {
            "slug": b["slug"],
            "title": b.get("title", b["slug"]),
            "author": b.get("author", ""),
            "date_read": b.get("date_read", ""),
            "year_read": str(year_read) if year_read else "Unknown",
            "year_published": year_published,
            "decade_published": decade_of(year_published),
            "genre": b.get("genre") or "Unknown",
        }
        if HAS_LLM_TAGS:
            row["genre_llm"] = (tags.get("genre") if tags else None) or "Unknown"
            row["clean_description"] = (tags.get("clean_description") if tags else None) or ""
            row["tone_and_style"] = (tags.get("tone_and_style") if tags else None) or ""
            row["pacing"] = (tags.get("pacing") if tags else None) or ""
            row["themes"] = (tags.get("themes") if tags else None) or ""
            row["setting"] = (tags.get("setting") if tags else None) or ""
        rows.append(row)
    df = pd.DataFrame(rows)
    # genre (heuristic) has ~42 distinct values -- capped well below "all"
    # since that many traces noticeably slows figure generation across 144
    # combos and produces a barely-legible legend; genre_llm's natural
    # count (~21) is left effectively uncapped since it's small enough to
    # not cause either problem.
    df["genre"] = collapse_rare(df["genre"], top_n=20)
    if HAS_LLM_TAGS:
        df["genre_llm"] = collapse_rare(df["genre_llm"])
    return df


def _ordinal_sort_key(value: str) -> tuple[bool, int]:
    """Sorts 'Unknown' last, everything else numerically (strips a
    trailing 's' for decade strings like '1920s', handles plain year
    strings like '2015' as-is)."""
    if value == "Unknown":
        return (True, 0)
    return (False, int(value[:-1] if value.endswith("s") else value))


def color_order_and_map(df: pd.DataFrame, color_col: str) -> tuple[list[str], dict[str, str]]:
    """(category_orders list, color_discrete_map dict) for one color
    column -- nominal columns (genre) get the fixed-order categorical
    palette, most-common-first; ordinal columns (decade/year) get the
    light->dark ramp in chronological order. Both stay legend/click-able
    discrete traces rather than a continuous colorbar."""
    if color_col in NOMINAL_COLOR_COLUMNS:
        counts = df[color_col].value_counts()
        order = [c for c in counts.index if c != "Other"]
        if "Other" in counts.index:
            order.append("Other")
        return order, nominal_color_map(order)
    order = sorted(df[color_col].unique(), key=_ordinal_sort_key)
    return order, ordinal_color_map(order)


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
        variant = str(npz["text_variant"]) if "text_variant" in npz.files else None
        provider_label = PROVIDER_DISPLAY_NAMES.get(provider, provider.title())
        label = provider_label + (f" ({model})" if model else "")
        if variant:
            label += f" — {variant}"
        sources[name] = {
            "embeddings": npz["embeddings"][order],
            "label": label,
        }
    if not sources:
        raise SystemExit(
            f"No embeddings_*.npz files found in {DATA_DIR}. Run scripts/04_generate_embeddings.py first."
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
    from plotnine import aes, element_text, geom_point, ggplot, labs, scale_color_manual, theme, theme_minimal

    d = df.copy()
    coords = all_coords[source][method]
    d["x"], d["y"] = coords[:, 0], coords[:, 1]
    order, cmap = color_order_and_map(df, color_col)

    p = (
        ggplot(d, aes(x="x", y="y", color=color_col))
        + geom_point(size=2.5, alpha=0.8)
        + scale_color_manual(values=cmap, limits=order)
        + theme_minimal()
        + labs(
            title=f"Reading taste ({source_label}, {METHOD_LABELS.get(method, method.upper())} projection)",
            x="",
            y="",
            color=color_label(color_col),
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

    # Depends only on df + color_col, not source/method -- compute once
    # per color column rather than 144 times (once per combo), which was
    # slow enough to matter once genre's category count grew.
    order_and_map_by_color = {c: color_order_and_map(df, c) for c in COLOR_COLUMNS}

    combos = {}
    for source, method_coords in all_coords.items():
        for method, coords in method_coords.items():
            d = df.copy()
            d["x"], d["y"] = coords[:, 0], coords[:, 1]
            for color_col in COLOR_COLUMNS:
                order, cmap = order_and_map_by_color[color_col]
                fig = px.scatter(
                    d,
                    x="x",
                    y="y",
                    color=color_col,
                    category_orders={color_col: order},
                    color_discrete_map=cmap,
                    custom_data=["slug", "title"],  # read by the JS hover/click handlers below
                )
                # A lightweight tooltip -- just the title, nothing else --
                # replaces both Plotly's verbose default (genre=X, x=.., y=..)
                # and the old fully-suppressed version that gave no on-canvas
                # feedback at all. Semi-transparent hoverlabel is set in the
                # page layout. Note: a custom hovertemplate always wins over
                # hoverinfo, so hoverinfo is irrelevant here.
                fig.update_traces(
                    marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")),
                    hovertemplate="%{customdata[1]}<extra></extra>",
                )
                combos[f"{source}|{method}|{color_col}"] = {"data": fig.to_dict()["data"]}
    return combos


NEIGHBOR_EDGE_COLOR = "rgba(50,50,50,0.6)"
CHAIN_EDGE_COLOR = "rgba(31,119,180,0.7)"
SELECTION_HALO_COLOR = "#0b0b0b"  # primary ink -- reads against every palette color

EDGE_TRACE = {
    "type": "scatter",
    "mode": "lines",
    "x": [],
    "y": [],
    "line": {"color": NEIGHBOR_EDGE_COLOR, "width": 1.5},
    "hoverinfo": "skip",
    "showlegend": False,
}

# A single large hollow-ring marker drawn on top of whichever book is
# currently selected (clicked, from the plot or the book list) -- "keeps
# it highlighted" independent of hover, since hover is transient and
# selection is meant to persist until deselected.
SELECTION_HALO_TRACE = {
    "type": "scatter",
    "mode": "markers",
    "x": [],
    "y": [],
    "marker": {"size": 22, "color": "rgba(0,0,0,0)", "line": {"width": 3, "color": SELECTION_HALO_COLOR}},
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

    # Book details for the details panel (replaces the floating tooltip)
    # and the searchable list -- keyed by slug so the same lookup backs
    # hover, click, and list-item rendering.
    detail_fields = ["title", "author", "date_read", "year_published", "genre"]
    if HAS_LLM_TAGS:
        detail_fields += ["genre_llm", "clean_description", "tone_and_style", "pacing", "themes", "setting"]
    book_details = {}
    for row in df.to_dict(orient="records"):
        detail = {f: row[f] for f in detail_fields}
        # year_published is a float in the DataFrame (NaN mixed with ints
        # coerces the whole column) -- normalize to a plain int or None so
        # the details panel doesn't render "2015.0".
        yp = detail.get("year_published")
        detail["year_published"] = int(yp) if pd.notna(yp) else None
        book_details[row["slug"]] = detail
    book_order = sorted(slugs, key=lambda s: (book_details[s]["title"].lower(), book_details[s]["author"].lower()))

    # No in-chart title -- it duplicated the page <h1> ("A Map of Books"),
    # which already names the page; source/method are visible in the
    # controls above instead.
    layout = dict(
        legend_title_text=color_label(default_color),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=20),
        plot_bgcolor=PLOT_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
        font=dict(family=PLOT_FONT_FAMILY),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(11,11,11,0.15)",
            font=dict(family=PLOT_FONT_FAMILY, size=12, color="#0b0b0b"),
        ),
    )
    fig = go.Figure(data=default_combo["data"] + [EDGE_TRACE, SELECTION_HALO_TRACE], layout=layout)
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
    color_labels = {c: color_label(c) for c in COLOR_COLUMNS}
    color_options_html = "".join(
        f'<option value="{c}"{" selected" if c == default_color else ""}>{color_labels[c]}</option>'
        for c in COLOR_COLUMNS
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>A Map of Books</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --page-plane: #f9f9f7;
    --surface: #fcfcfb;
    --ink-primary: #0b0b0b;
    --ink-secondary: #52514e;
    --ink-muted: #898781;
    --border: rgba(11,11,11,0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--page-plane);
    color: var(--ink-primary);
    margin: 0;
    padding: 24px 16px 40px;
  }}
  .page {{ max-width: 1500px; margin: 0 auto; }}
  h1 {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 28px;
    letter-spacing: -0.01em;
    margin: 0 0 16px;
  }}
  /* Sidebar (Book details + All books) sits beside the graph, not below
     it, so both are visible at once without clicking then scrolling. */
  .layout {{
    display: flex;
    gap: 20px;
    align-items: flex-start;
  }}
  .sidebar {{
    display: flex;
    flex-direction: column;
    gap: 20px;
    width: 320px;
    flex: 0 0 320px;
  }}
  .main-col {{
    flex: 1 1 auto;
    min-width: 0;
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px 28px;
    margin: 0 0 20px;
    padding: 14px 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
  }}
  .control-group {{
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
  }}
  .control-group label {{
    font-weight: 600;
    font-size: 13px;
    color: var(--ink-secondary);
  }}
  .control-group select {{
    font-family: inherit;
    font-size: 14px;
    color: var(--ink-primary);
    background: var(--page-plane);
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
  }}
  #book-plot {{ width: 100%; aspect-ratio: 16 / 9; }}
  .panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    /* Fixed height, internal scroll -- so the sidebar's total height is
       predictable and both panels stay visible next to the graph. Book
       details gets more room than the list -- descriptions/tone/themes
       run long, while the list is searchable so it doesn't need to show
       many rows at once. */
    display: flex;
    flex-direction: column;
  }}
  #details-panel {{ height: 460px; }}
  #book-list-panel {{ height: 220px; }}
  .panel h2 {{
    flex: 0 0 auto;
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 16px;
    margin: 0 0 10px;
    color: var(--ink-primary);
  }}
  #details-content {{
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
  }}
  #details-content .placeholder {{
    color: var(--ink-muted);
    font-size: 14px;
  }}
  #details-content .book-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 19px;
    margin: 0 0 2px;
  }}
  #details-content .book-author {{
    color: var(--ink-secondary);
    font-size: 14px;
    margin: 0 0 12px;
  }}
  #details-content .book-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px 10px;
    margin: 0 0 12px;
  }}
  #details-content .book-meta span {{
    font-size: 12px;
    color: var(--ink-secondary);
    background: var(--page-plane);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 8px;
  }}
  #details-content dl {{
    margin: 0;
    font-size: 13px;
  }}
  #details-content dt {{
    font-weight: 600;
    color: var(--ink-secondary);
    margin-top: 8px;
  }}
  #details-content dd {{
    margin: 2px 0 0;
    color: var(--ink-primary);
    line-height: 1.45;
  }}
  #details-content .deselect-btn {{
    margin-top: 14px;
    font-family: inherit;
    font-size: 12px;
    color: var(--ink-secondary);
    background: var(--page-plane);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    cursor: pointer;
  }}
  #book-search {{
    flex: 0 0 auto;
    width: 100%;
    font-family: inherit;
    font-size: 14px;
    color: var(--ink-primary);
    background: var(--page-plane);
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 10px;
  }}
  #book-list {{
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    border-top: 1px solid var(--border);
  }}
  .book-list-item {{
    display: block;
    width: 100%;
    text-align: left;
    font-family: inherit;
    font-size: 13px;
    color: var(--ink-primary);
    background: none;
    border: none;
    border-bottom: 1px solid var(--border);
    padding: 8px 4px;
    cursor: pointer;
  }}
  .book-list-item:hover {{ background: var(--page-plane); }}
  .book-list-item.selected {{ background: var(--page-plane); font-weight: 600; }}
  .book-list-item .item-author {{ color: var(--ink-secondary); font-weight: 400; }}
  #book-list .empty {{ color: var(--ink-muted); font-size: 13px; padding: 8px 4px; }}
  @media (max-width: 900px) {{
    .layout {{ flex-direction: column; }}
    .sidebar {{ width: 100%; flex: none; flex-direction: row; flex-wrap: wrap; }}
    .sidebar .panel {{ flex: 1 1 280px; }}
  }}
</style>
</head>
<body>
  <div class="page">
    <h1>A Map of Books</h1>
    <div class="layout">
      <div class="sidebar">
        <div id="details-panel" class="panel">
          <h2>Book Details</h2>
          <div id="details-content"><p class="placeholder">Hover or click a point, or pick a book from the list, to see details here.</p></div>
        </div>
        <div id="book-list-panel" class="panel">
          <h2>All Books</h2>
          <input id="book-search" type="text" placeholder="Search title or author...">
          <div id="book-list"></div>
        </div>
      </div>
      <div class="main-col">
        <div class="controls">
          <div class="control-group">
            <label for="source-select">Embedding source</label>
            <select id="source-select">{source_options_html}</select>
          </div>
          <div class="control-group">
            <label for="method-select">Projection method</label>
            <select id="method-select">{method_options_html}</select>
          </div>
          <div class="control-group">
            <label for="color-select">Color by</label>
            <select id="color-select">{color_options_html}</select>
          </div>
          <div class="control-group">
            <label for="edges-select">On hover/select, show</label>
            <select id="edges-select">
              <option value="off">No links</option>
              <option value="neighbors" selected>Nearest neighbors (by similarity)</option>
              <option value="chain">Reading order (chronological)</option>
            </select>
          </div>
        </div>
        {plot_div}
      </div>
    </div>
  </div>
  <script>
    const combos = {json.dumps(combos, cls=plotly.utils.PlotlyJSONEncoder)};
    const neighbors = {json.dumps(all_neighbors)};
    const readingOrder = {json.dumps(reading_order)};
    const slugToOrderIndex = {{}};
    readingOrder.forEach((s, i) => {{ slugToOrderIndex[s] = i; }});
    const CHAIN_WINDOW = 2;  // 2 back + 2 forward + the hovered book = 5 nodes, 4 sequential edges
    const coordsLookup = {json.dumps(coords_lookup)};
    const edgeTraceTemplate = {json.dumps(EDGE_TRACE)};
    const haloTraceTemplate = {json.dumps(SELECTION_HALO_TRACE)};
    const neighborEdgeColor = {json.dumps(NEIGHBOR_EDGE_COLOR)};
    const chainEdgeColor = {json.dumps(CHAIN_EDGE_COLOR)};
    const sourceLabels = {json.dumps(source_labels)};
    const methodLabels = {json.dumps(METHOD_LABELS)};
    const colorLabels = {json.dumps(color_labels)};
    const bookDetails = {json.dumps(book_details)};
    const bookOrder = {json.dumps(book_order)};
    const BASE_MARKER_OPACITY = 0.85;
    const DIMMED_MARKER_OPACITY = 0.12;
    const sourceEl = document.getElementById('source-select');
    const methodEl = document.getElementById('method-select');
    const colorEl = document.getElementById('color-select');
    const edgesEl = document.getElementById('edges-select');
    const searchEl = document.getElementById('book-search');
    const listEl = document.getElementById('book-list');
    const detailsEl = document.getElementById('details-content');
    const plotDiv = document.getElementById('book-plot');

    // Selection/hover/legend-highlight state. Priority for what's shown:
    //   edges & details panel: hoveredSlug (transient) > selectedSlug
    //     (persistent, from a node or list click) > nothing.
    //   the halo ring: always tracks selectedSlug only, independent of
    //     hover, so a pinned book stays visibly marked while you explore.
    let hoveredSlug = null;
    let selectedSlug = null;
    let highlightedLegendValue = null;

    function escapeHtml(s) {{
      return String(s).replace(/[&<>"']/g, c => ({{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}}[c]));
    }}

    function update() {{
      const source = sourceEl.value;
      const method = methodEl.value;
      const color = colorEl.value;
      const combo = combos[source + '|' + method + '|' + color];
      // Append fresh (empty) edge/halo traces as the last two traces --
      // combo.data is the shared precomputed array, so copy rather than
      // mutate it, or repeated switches back to the same combo would pile
      // up edge traces.
      const dataWithExtras = combo.data.concat([Object.assign({{}}, edgeTraceTemplate), Object.assign({{}}, haloTraceTemplate)]);
      // A color-by change repaints the legend with a different set of
      // categories, so any active legend highlight no longer refers to a
      // real trace name -- reset it rather than leave a stale dim state.
      highlightedLegendValue = null;
      // No in-chart title (see Python side) -- source/method are already
      // visible in the controls above, no need to repeat them here.
      Plotly.react(plotDiv, dataWithExtras, {{
        legend: {{ title: {{ text: colorLabels[color] }} }},
        xaxis: {{ visible: false }},
        yaxis: {{ visible: false }},
        margin: {{ t: 20 }},
        plot_bgcolor: {json.dumps(PLOT_BGCOLOR)},
        paper_bgcolor: {json.dumps(PAPER_BGCOLOR)},
        font: {{ family: {json.dumps(PLOT_FONT_FAMILY)} }},
        hoverlabel: {{
          bgcolor: 'rgba(255,255,255,0.8)',
          bordercolor: 'rgba(11,11,11,0.15)',
          font: {{ family: {json.dumps(PLOT_FONT_FAMILY)}, size: 12, color: '#0b0b0b' }},
        }},
        annotations: [],
      }});
      refreshHalo();
      refreshEdges();
      pinTooltip(selectedSlug);
    }}
    sourceEl.addEventListener('change', update);
    methodEl.addEventListener('change', update);
    colorEl.addEventListener('change', update);

    function clearEdges() {{
      const edgeTraceIndex = plotDiv.data.length - 2;
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
      const edgeTraceIndex = plotDiv.data.length - 2;
      Plotly.restyle(plotDiv, {{x: [xs], y: [ys], 'line.color': [neighborEdgeColor]}}, [edgeTraceIndex]);
    }}

    function showReadingChain(slug, source, method) {{
      // 5-node window centered on the active book: 2 read before it, 2
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

    // The book whose edges/links should currently be drawn: whatever's
    // hovered takes priority (transient exploration); failing that, fall
    // back to whatever's selected (persistent, from a click), so the
    // links stay on screen after the mouse moves away.
    function activeEdgeSlug() {{
      return hoveredSlug || selectedSlug || null;
    }}

    // Source depending on the "On hover/select, show" dropdown:
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
    function refreshEdges() {{
      const slug = activeEdgeSlug();
      const mode = edgesEl.value;
      if (!slug || mode === 'off') {{
        clearEdges();
        return;
      }}
      const source = sourceEl.value;
      const method = methodEl.value;
      if (mode === 'chain') {{
        const edgeTraceIndex = plotDiv.data.length - 2;
        Plotly.restyle(plotDiv, {{x: [[]], y: [[]]}}, [edgeTraceIndex]);
        showReadingChain(slug, source, method);
      }} else {{
        Plotly.relayout(plotDiv, {{annotations: []}});
        showNeighborEdges(slug, source, method);
      }}
    }}
    edgesEl.addEventListener('change', refreshEdges);

    // The hollow-ring halo always tracks selectedSlug (not hover), so a
    // pinned book stays visibly marked while the mouse roams elsewhere.
    function refreshHalo() {{
      const haloTraceIndex = plotDiv.data.length - 1;
      const source = sourceEl.value;
      const method = methodEl.value;
      const coordsForMethod = (coordsLookup[source] || {{}})[method] || {{}};
      const point = selectedSlug ? coordsForMethod[selectedSlug] : null;
      if (point) {{
        Plotly.restyle(plotDiv, {{x: [[point[0]]], y: [[point[1]]]}}, [haloTraceIndex]);
      }} else {{
        Plotly.restyle(plotDiv, {{x: [[]], y: [[]]}}, [haloTraceIndex]);
      }}
    }}

    function renderDetails(slug) {{
      if (!slug || !bookDetails[slug]) {{
        detailsEl.innerHTML = '<p class="placeholder">Hover or click a point, or pick a book from the list, to see details here.</p>';
        return;
      }}
      const b = bookDetails[slug];
      const meta = [];
      if (b.year_published) meta.push('Published ' + b.year_published);
      if (b.date_read) meta.push('Read ' + b.date_read);
      if (b.genre_llm || b.genre) meta.push(b.genre_llm || b.genre);
      let html = '';
      html += '<p class="book-title">' + escapeHtml(b.title) + '</p>';
      html += '<p class="book-author">' + escapeHtml(b.author) + '</p>';
      if (meta.length) {{
        html += '<div class="book-meta">' + meta.map(m => '<span>' + escapeHtml(m) + '</span>').join('') + '</div>';
      }}
      html += '<dl>';
      const longFields = [
        ['Description', b.clean_description],
        ['Tone & style', b.tone_and_style],
        ['Pacing', b.pacing],
        ['Themes', b.themes],
        ['Setting', b.setting],
      ];
      for (const [label, value] of longFields) {{
        if (value) html += '<dt>' + label + '</dt><dd>' + escapeHtml(value) + '</dd>';
      }}
      html += '</dl>';
      if (selectedSlug === slug) {{
        html += '<button type="button" class="deselect-btn" id="deselect-btn">Clear selection</button>';
      }}
      detailsEl.innerHTML = html;
      if (selectedSlug === slug) {{
        document.getElementById('deselect-btn').addEventListener('click', () => selectBook(null));
      }}
    }}

    function renderBookList() {{
      const q = searchEl.value.trim().toLowerCase();
      const filtered = bookOrder.filter(slug => {{
        const b = bookDetails[slug];
        return !q || b.title.toLowerCase().includes(q) || b.author.toLowerCase().includes(q);
      }});
      if (filtered.length === 0) {{
        listEl.innerHTML = '<div class="empty">No books match.</div>';
        return;
      }}
      listEl.innerHTML = filtered.map(slug => {{
        const b = bookDetails[slug];
        const cls = 'book-list-item' + (slug === selectedSlug ? ' selected' : '');
        return '<button type="button" class="' + cls + '" data-slug="' + slug + '">'
          + escapeHtml(b.title) + ' <span class="item-author">— ' + escapeHtml(b.author) + '</span></button>';
      }}).join('');
      listEl.querySelectorAll('.book-list-item').forEach(btn => {{
        btn.addEventListener('click', () => selectBook(btn.dataset.slug));
      }});
    }}
    searchEl.addEventListener('input', renderBookList);

    // Locates a slug's current (curveNumber, pointNumber) so the native
    // tooltip can be triggered programmatically for it (see pinTooltip).
    function pointLocationFor(slug) {{
      for (let i = 0; i < plotDiv.data.length; i++) {{
        const trace = plotDiv.data[i];
        if (!trace.customdata) continue;
        for (let j = 0; j < trace.customdata.length; j++) {{
          if (trace.customdata[j][0] === slug) return {{curveNumber: i, pointNumber: j}};
        }}
      }}
      return null;
    }}

    // Keeps the lightweight title tooltip visible for the selected book
    // even when the mouse isn't over it -- Plotly.Fx.hover can be
    // triggered programmatically and the label stays until explicitly
    // cleared, unlike a real hover which clears on mouseout.
    function pinTooltip(slug) {{
      const loc = slug ? pointLocationFor(slug) : null;
      if (loc) {{
        Plotly.Fx.hover(plotDiv, [loc]);
      }} else if (!hoveredSlug) {{
        Plotly.Fx.unhover(plotDiv);
      }}
    }}

    // Clicking a node or a list entry pins it: the halo marks it, its
    // edges stay drawn even after the mouse moves away, its details stay
    // in the panel, and its tooltip stays pinned. Clicking the same book
    // again (node, list, empty canvas, or the "Clear selection" button)
    // un-pins it.
    function selectBook(slug) {{
      selectedSlug = (slug && slug === selectedSlug) ? null : slug;
      refreshHalo();
      refreshEdges();
      renderDetails(hoveredSlug || selectedSlug);
      renderBookList();
      pinTooltip(selectedSlug);
    }}

    // Plotly overlays an invisible drag/zoom rect on top of the entire
    // plot area, so a native click's event.target is always that rect --
    // never the marker path -- even for a dead-on marker click. DOM
    // target inspection can't tell a marker click from a background
    // click, so use a flag instead: plotly_click only fires (with a real
    // point) when a marker was actually hit, and it fires before the
    // native 'click' event (during Plotly's mouseup handling), so the
    // flag is reliably set by the time the native listener below runs.
    let markerClickHandled = false;
    plotDiv.on('plotly_click', function (evt) {{
      const point = evt.points[0];
      if (!point || !point.customdata) return;
      markerClickHandled = true;
      selectBook(point.customdata[0]);
    }});

    // Clicking anywhere on the plot that ISN'T a marker (empty canvas)
    // clears the current selection -- otherwise the only way to deselect
    // was re-clicking the same node or picking a different one. Excludes
    // the legend and modebar, which have their own click behaviors.
    plotDiv.addEventListener('click', function (evt) {{
      if (markerClickHandled) {{ markerClickHandled = false; return; }}
      if (evt.target.closest('.legend') || evt.target.closest('.modebar')) return;
      if (selectedSlug !== null) selectBook(selectedSlug);
    }});

    plotDiv.on('plotly_hover', function (evt) {{
      const point = evt.points[0];
      if (!point.customdata) return;
      hoveredSlug = point.customdata[0];
      refreshEdges();
      renderDetails(hoveredSlug);
    }});
    plotDiv.on('plotly_unhover', function () {{
      hoveredSlug = null;
      refreshEdges();
      renderDetails(selectedSlug);
      pinTooltip(selectedSlug);
    }});

    // Tableau-style legend click: instead of Plotly's default (hide the
    // clicked series entirely), highlight it and dim -- not hide -- every
    // other series, so relative position is still visible. Clicking the
    // same legend entry again restores everyone to full opacity; clicking
    // a different entry switches the highlight.
    function applyLegendHighlightState() {{
      const indices = [];
      const opacities = [];
      plotDiv.data.forEach((trace, i) => {{
        if (!trace.name) return;  // edge/halo traces have no name
        indices.push(i);
        opacities.push(
          highlightedLegendValue === null || trace.name === highlightedLegendValue
            ? BASE_MARKER_OPACITY
            : DIMMED_MARKER_OPACITY
        );
      }});
      if (indices.length) Plotly.restyle(plotDiv, {{'marker.opacity': opacities}}, indices);
    }}
    function handleLegendClick(evt) {{
      const trace = evt.data[evt.curveNumber];
      highlightedLegendValue = (highlightedLegendValue === trace.name) ? null : trace.name;
      applyLegendHighlightState();
      return false;  // prevent Plotly's default hide-this-trace behavior
    }}
    plotDiv.on('plotly_legendclick', handleLegendClick);
    plotDiv.on('plotly_legenddoubleclick', handleLegendClick);  // same behavior, no isolate-and-hide-rest

    renderBookList();
    renderDetails(null);
  </script>
</body>
</html>
"""
    out_path = OUTPUT_DIR / "interactive_plot.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="default embedding source selection; falls back to openai_v2, then whichever is found first")
    parser.add_argument("--method", choices=list(REDUCERS), default="tsne_p7", help="default selection on load / static plot method")
    default_color_choice = "genre_llm" if "genre_llm" in COLOR_COLUMNS else "genre"
    parser.add_argument("--color", choices=COLOR_COLUMNS, default=default_color_choice, help="default selection on load / static plot color")
    args = parser.parse_args()

    df = load_metadata()
    embedding_sources = load_embedding_sources(df["slug"].tolist())
    source_labels = {name: data["label"] for name, data in embedding_sources.items()}
    print(f"Found embedding sources: {', '.join(f'{n} ({l})' for n, l in source_labels.items())}")

    PREFERRED_DEFAULT_SOURCE = "openai_v2"
    if args.source:
        default_source = args.source
        if default_source not in embedding_sources:
            raise SystemExit(f"--source {default_source!r} not found. Available: {list(embedding_sources)}")
    else:
        default_source = PREFERRED_DEFAULT_SOURCE if PREFERRED_DEFAULT_SOURCE in embedding_sources else next(iter(embedding_sources))

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

    # Also publish as index.html so static hosts (Netlify etc.) serve the
    # viz directly at the site root, with no filename in the URL.
    index_path = OUTPUT_DIR / "index.html"
    index_path.write_text(interactive_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
