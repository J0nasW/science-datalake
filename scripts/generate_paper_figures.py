#!/usr/bin/env python3
"""Generate publication-quality figures for the Science Data Lake paper.

Outputs PDF vector figures to paper/figures/.

Figures:
  1. Cross-source citation Bland-Altman (S2AG vs OA)
  2. UpSet plot of source overlap (5-6 boolean flags)
  3. Temporal coverage ridgeline (5 sources x year)
  4. UMAP of embedding space (topics + ontology terms)
  5. Ontology reach heatmap (domain x ontology)
  6. Vignette composite (2x2 panel)

Usage:
    python scripts/generate_paper_figures.py              # all figures
    python scripts/generate_paper_figures.py --only 1 3   # specific figures
    python scripts/generate_paper_figures.py --skip-umap  # skip GPU figure
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
# Resolve root relative to this script: scripts/ -> parent is repo root
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT / "datalake.duckdb")
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style: Nature Scientific Data guidelines ───────────────────────────────
# Sans-serif, min 6pt, clean lines, no chartjunk
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.minor.width":  0.3,
    "ytick.minor.width":  0.3,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "pdf.fonttype":       42,   # TrueType in PDF (editable text)
    "ps.fonttype":        42,
})

# ── Color palettes ─────────────────────────────────────────────────────────
# Muted, colorblind-safe, print-friendly
DOMAIN_COLORS = {
    "Health Sciences":    "#C62828",
    "Life Sciences":      "#2E7D32",
    "Physical Sciences":  "#1565C0",
    "Social Sciences":    "#E65100",
}

SOURCE_COLORS = {
    "OpenAlex":   "#1565C0",
    "S2AG":       "#2E7D32",
    "SciSciNet":  "#C62828",
    "PWC":        "#7B1FA2",
    "RetWatch":   "#E65100",
    "RoS":        "#00838F",
}

ONTOLOGY_COLORS = {
    "cso":     "#7B1FA2",
    "edam":    "#00897B",
    "unesco":  "#FB8C00",
    "stw":     "#5C6BC0",
    "physh":   "#1E88E5",
    "msc2020": "#D81B60",
    "go":      "#43A047",
    "hpo":     "#F4511E",
    "doid":    "#E53935",
    "mesh":    "#546E7A",
    "ncit":    "#8D6E63",
    "chebi":   "#78909C",
    "agrovoc": "#689F38",
}

# Nature Sci Data: single col ~89mm (3.5"), double col ~183mm (7.2")
SINGLE_COL = 3.54  # inches
DOUBLE_COL = 7.24


def save_fig(fig, name):
    """Save figure as PDF and print confirmation."""
    path = OUT_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Cross-source citation Bland-Altman
# ═══════════════════════════════════════════════════════════════════════════

def fig1_citation_bland_altman(con):
    """Two-panel: log-log density scatter + Bland-Altman (S2AG vs OpenAlex)."""

    df = con.execute("""
        SELECT s2ag_citationcount AS s2ag, oa_cited_by_count AS oa
        FROM xref.unified_papers
        WHERE s2ag_citationcount IS NOT NULL
          AND oa_cited_by_count IS NOT NULL
        USING SAMPLE 500000
    """).df()

    s = np.log10(df["s2ag"].values + 1).astype(np.float32)
    o = np.log10(df["oa"].values + 1).astype(np.float32)
    mean_so = (s + o) / 2.0
    diff_so = s - o  # positive = S2AG higher

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2))

    # ── Panel A: log-log density scatter ──
    hb1 = ax1.hexbin(o, s, gridsize=120, mincnt=1,
                      cmap="inferno", bins="log", linewidths=0.1)
    lim = max(s.max(), o.max()) * 1.05
    ax1.plot([0, lim], [0, lim], color="white", ls="--", lw=0.8, alpha=0.7)
    ax1.set_xlabel("OpenAlex citations (log$_{10}$)")
    ax1.set_ylabel("S2AG citations (log$_{10}$)")
    ax1.set_title("A", loc="left", fontsize=12, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    cb1 = fig.colorbar(hb1, ax=ax1, shrink=0.7, pad=0.02)
    cb1.set_label("Papers (log)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)

    # Annotate correlation
    r = np.corrcoef(s, o)[0, 1]
    ax1.text(0.05, 0.93, f"r = {r:.3f}\nn = {len(df):,}",
             transform=ax1.transAxes, fontsize=7,
             va="top", bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.8, lw=0.5))

    # ── Panel B: Bland-Altman ──
    hb2 = ax2.hexbin(mean_so, diff_so, gridsize=120, mincnt=1,
                      cmap="inferno", bins="log", linewidths=0.1)
    md = np.mean(diff_so)
    sd = np.std(diff_so)
    ax2.axhline(md, color="#1565C0", ls="-", lw=0.8, alpha=0.9)
    ax2.axhline(md + 1.96 * sd, color="#1565C0", ls="--", lw=0.6, alpha=0.7)
    ax2.axhline(md - 1.96 * sd, color="#1565C0", ls="--", lw=0.6, alpha=0.7)
    ax2.axhline(0, color="white", ls=":", lw=0.5, alpha=0.5)

    ax2.set_xlabel("Mean citation count (log$_{10}$)")
    ax2.set_ylabel("Difference: S2AG $-$ OpenAlex (log$_{10}$)")
    ax2.set_title("B", loc="left", fontsize=12, fontweight="bold")
    cb2 = fig.colorbar(hb2, ax=ax2, shrink=0.7, pad=0.02)
    cb2.set_label("Papers (log)", fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    # Annotate limits of agreement
    ax2.text(0.97, 0.95,
             f"Mean diff = {md:.3f}\n"
             f"LoA: [{md - 1.96*sd:.2f}, {md + 1.96*sd:.2f}]",
             transform=ax2.transAxes, fontsize=6.5, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="white", alpha=0.8, lw=0.5))

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "fig1_citation_bland_altman")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: UpSet plot of source overlap
# ═══════════════════════════════════════════════════════════════════════════

def fig2_upset_source_overlap(con):
    """UpSet plot showing multi-source overlap across 5-6 boolean flags."""
    from upsetplot import UpSet

    # Discover which boolean flags exist
    flags_raw = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'xref' AND table_name = 'unified_papers'
          AND column_name LIKE 'has_%'
        ORDER BY column_name
    """).fetchall()
    flag_cols = [f[0] for f in flags_raw]

    # Prettier labels
    label_map = {
        "has_openalex":   "OpenAlex",
        "has_s2ag":       "S2AG",
        "has_sciscinet":  "SciSciNet",
        "has_pwc":        "PWC",
        "has_retraction": "RetWatch",
        "has_patent":     "RoS",
    }

    select_cols = ", ".join(flag_cols)
    df = con.execute(f"""
        SELECT {select_cols}, COUNT(*) AS n
        FROM xref.unified_papers
        GROUP BY {select_cols}
    """).df()

    # Rename columns
    rename = {c: label_map.get(c, c) for c in flag_cols}
    df = df.rename(columns=rename)
    pretty_flags = [label_map.get(c, c) for c in flag_cols]

    # Build MultiIndex Series for upsetplot
    idx = pd.MultiIndex.from_frame(df[pretty_flags].astype(bool))
    data = pd.Series(df["n"].values, index=idx)

    # Source colors for the matrix dots
    facecolor = "#2c2c2c"

    upset = UpSet(
        data,
        subset_size="sum",
        show_percentages=False,
        sort_by="cardinality",
        sort_categories_by="cardinality",
        min_subset_size=10000,
        show_counts=True,
        facecolor=facecolor,
        element_size=32,
        totals_plot_elements=6,
    )

    fig = plt.figure(figsize=(DOUBLE_COL, 5.0))
    axes = upset.plot(fig=fig)

    # Style the intersection size bars
    fmt_millions = ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6
        else f"{x/1e3:.0f}K" if x >= 1e3
        else f"{x:.0f}"
    )
    if "intersections" in axes:
        ax_inter = axes["intersections"]
        ax_inter.set_ylabel("Papers")
        ax_inter.yaxis.set_major_formatter(fmt_millions)
        # Replace raw count labels with human-readable M/K format
        for txt in ax_inter.texts:
            try:
                val = float(txt.get_text().replace(",", ""))
                if val >= 1e6:
                    txt.set_text(f"{val/1e6:.1f}M")
                elif val >= 1e3:
                    txt.set_text(f"{val/1e3:.0f}K")
                txt.set_fontsize(5.5)
                txt.set_color("#333333")
            except ValueError:
                pass
    if "totals" in axes:
        ax_totals = axes["totals"]
        ax_totals.xaxis.set_major_formatter(fmt_millions)
        # Reformat totals bar text annotations to M/K format
        for txt in ax_totals.texts:
            try:
                val = float(txt.get_text().replace(",", ""))
                if val >= 1e6:
                    txt.set_text(f"{val/1e6:.0f}M")
                elif val >= 1e3:
                    txt.set_text(f"{val/1e3:.0f}K")
                txt.set_fontsize(5.5)
                txt.set_color("#333333")
            except ValueError:
                pass

    save_fig(fig, "fig2_upset_source_overlap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Temporal coverage ridgeline
# ═══════════════════════════════════════════════════════════════════════════

def fig3_temporal_ridgeline(con):
    """Ridgeline plot: publication-year distribution per source."""

    df = con.execute("""
        SELECT year,
            SUM(CASE WHEN has_openalex  THEN 1 ELSE 0 END) AS OpenAlex,
            SUM(CASE WHEN has_s2ag      THEN 1 ELSE 0 END) AS S2AG,
            SUM(CASE WHEN has_sciscinet THEN 1 ELSE 0 END) AS SciSciNet,
            SUM(CASE WHEN has_pwc       THEN 1 ELSE 0 END) AS PWC,
            SUM(CASE WHEN has_retraction THEN 1 ELSE 0 END) AS RetWatch
        FROM xref.unified_papers
        WHERE year BETWEEN 1900 AND 2025
        GROUP BY year
        ORDER BY year
    """).df()

    sources = ["OpenAlex", "S2AG", "SciSciNet", "PWC", "RetWatch"]
    colors = [SOURCE_COLORS[s] for s in sources]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))

    spacing = 1.0  # vertical spacing between ridges
    years = df["year"].values

    for i, (src, color) in enumerate(zip(sources, colors)):
        counts = df[src].values.astype(float)
        # Normalize to peak = 1 for shape comparison
        peak = counts.max()
        if peak > 0:
            normed = counts / peak
        else:
            normed = counts
        y_offset = (len(sources) - 1 - i) * spacing
        ax.fill_between(years, y_offset, y_offset + normed * 0.85,
                        color=color, alpha=0.6, lw=0)
        ax.plot(years, y_offset + normed * 0.85,
                color=color, lw=0.8, alpha=0.9)

        # Label on the left
        ax.text(1895, y_offset + 0.35, src, fontsize=8, fontweight="bold",
                color=color, ha="right", va="center")

        # Total count annotation on the right
        total = df[src].sum()
        if total >= 1e6:
            label = f"{total/1e6:.0f}M"
        else:
            label = f"{total/1e3:.0f}K"
        ax.text(2027, y_offset + 0.35, label, fontsize=7,
                color=color, ha="left", va="center")

    ax.set_xlim(1900, 2025)
    ax.set_ylim(-0.15, len(sources) * spacing + 0.2)
    ax.set_xlabel("Publication year")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # Light vertical gridlines at decades
    for decade in range(1900, 2030, 20):
        ax.axvline(decade, color="#cccccc", lw=0.3, zorder=0)

    fig.tight_layout()
    save_fig(fig, "fig3_temporal_ridgeline")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: UMAP of embedding space
# ═══════════════════════════════════════════════════════════════════════════

def fig4_umap_embedding_space(con):
    """UMAP projection of OpenAlex topics colored by domain,
    with ontology terms colored by ontology and margin annotations."""
    import umap
    from matplotlib.legend_handler import HandlerTuple

    # ── Load topics ──
    df_topics = con.execute("""
        SELECT DISTINCT t.id AS topic_id, t.display_name AS name,
               t.domain_display_name AS domain
        FROM openalex.topics t
    """).df()
    print(f"    Topics: {len(df_topics)}")

    # ── Load matched ontology terms (high quality only) ──
    df_terms = con.execute("""
        SELECT DISTINCT ontology_term_label AS name, ontology, domain,
               MAX(similarity) AS best_sim
        FROM xref.topic_ontology_map
        WHERE similarity >= 0.85
        GROUP BY ontology_term_label, ontology, domain
    """).df()
    print(f"    Ontology terms (sim >= 0.85): {len(df_terms)}")

    # ── Encode with BGE-large ──
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("    SKIP: sentence-transformers not available")
        return

    print("    Loading BGE-large-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    topic_texts = df_topics["name"].tolist()
    term_texts = df_terms["name"].tolist()
    all_texts = topic_texts + term_texts

    print(f"    Encoding {len(all_texts)} texts...")
    t0 = time.time()
    embeddings = model.encode(all_texts, batch_size=256, show_progress_bar=False,
                              normalize_embeddings=True)
    print(f"    Encoded in {time.time() - t0:.1f}s")

    # Free GPU memory
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ── UMAP ──
    print("    Running UMAP...")
    n_topics = len(topic_texts)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                        random_state=42, n_jobs=-1)
    coords = reducer.fit_transform(embeddings)

    topic_coords = coords[:n_topics]
    term_coords = coords[n_topics:]

    # ── Layout: scatter plot with margin zones + bottom legend strip ──
    fig = plt.figure(figsize=(DOUBLE_COL + 2.4, 6.4))
    # Main scatter in centre; leave margins for annotations, bottom for legend
    ax = fig.add_axes([0.22, 0.12, 0.56, 0.86])  # [left, bottom, width, height]

    # Plot topics (circles, colored by domain)
    for domain, color in DOMAIN_COLORS.items():
        mask = df_topics["domain"].values == domain
        if mask.sum() == 0:
            continue
        ax.scatter(topic_coords[mask, 0], topic_coords[mask, 1],
                   c=color, s=14, alpha=0.45, edgecolors="none",
                   zorder=3, rasterized=True)

    # ── Plot ontology terms colored by ontology, with × marker ──
    onto_counts = df_terms["ontology"].value_counts()
    top_ontologies = onto_counts.head(8).index.tolist()

    for onto in top_ontologies:
        mask = df_terms["ontology"].values == onto
        color = ONTOLOGY_COLORS.get(onto, "#888888")
        ax.scatter(term_coords[mask, 0], term_coords[mask, 1],
                   c=color, s=18, alpha=0.45, marker="x", linewidths=0.6,
                   zorder=4, rasterized=True)

    other_mask = ~df_terms["ontology"].isin(top_ontologies)
    if other_mask.sum() > 0:
        ax.scatter(term_coords[other_mask.values, 0],
                   term_coords[other_mask.values, 1],
                   c="#AAAAAA", s=12, alpha=0.35, marker="x", linewidths=0.5,
                   zorder=2, rasterized=True)

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # ── Select annotations: 3 per domain (12 total), balanced left/right ──
    df_ann = df_terms.copy()
    df_ann["label_len"] = df_ann["name"].str.len()
    df_ann["ann_score"] = df_ann["best_sim"] - 0.003 * df_ann["label_len"].clip(upper=40)

    # Build fast lookup: term name → UMAP coords
    term_name_to_idx = {}
    for i, name in enumerate(term_texts):
        if name not in term_name_to_idx:
            term_name_to_idx[name] = i
    df_ann["umap_x"] = df_ann["name"].map(
        lambda n: term_coords[term_name_to_idx[n], 0] if n in term_name_to_idx else np.nan)
    df_ann["umap_y"] = df_ann["name"].map(
        lambda n: term_coords[term_name_to_idx[n], 1] if n in term_name_to_idx else np.nan)
    df_ann = df_ann.dropna(subset=["umap_x", "umap_y"])

    x_center = coords[:, 0].mean()

    # Spatial separation threshold
    x_range_data = coords[:, 0].max() - coords[:, 0].min()
    y_range_data = coords[:, 1].max() - coords[:, 1].min()
    min_sep = 0.05 * max(x_range_data, y_range_data)

    def too_close(x, y, placed_list, threshold):
        for px, py, *_ in placed_list:
            if np.sqrt((x - px)**2 + (y - py)**2) < threshold:
                return True
        return False

    all_candidates = []
    used_ontologies_global = set()
    for domain in DOMAIN_COLORS:
        mask = df_ann["domain"].values == domain
        sub = df_ann[mask].sort_values("ann_score", ascending=False)
        picked = 0
        for _, row in sub.iterrows():
            if picked >= 3:
                break
            # After first pick per domain, require a globally new ontology
            if row["ontology"] in used_ontologies_global and picked > 0:
                continue
            if row["label_len"] > 38:
                continue
            x_pt, y_pt = row["umap_x"], row["umap_y"]
            if too_close(x_pt, y_pt, all_candidates, min_sep):
                continue
            entry = (x_pt, y_pt, row["name"], row["ontology"],
                     ONTOLOGY_COLORS.get(row["ontology"], "#888888"), domain)
            all_candidates.append(entry)
            used_ontologies_global.add(row["ontology"])
            picked += 1

    # ── Assign to left / right with strict balancing (max diff = 1) ──
    left_pool = sorted([c for c in all_candidates if c[0] < x_center],
                       key=lambda a: a[0])   # leftmost first
    right_pool = sorted([c for c in all_candidates if c[0] >= x_center],
                        key=lambda a: -a[0])  # rightmost first

    half = len(all_candidates) // 2
    left_annotations = list(left_pool)
    right_annotations = list(right_pool)
    # Move centre-most items from the larger side to the smaller side
    while len(left_annotations) > half and len(right_annotations) < half:
        moved = left_annotations.pop()  # rightmost of the left pool
        right_annotations.append(moved)
    while len(right_annotations) > half and len(left_annotations) < half:
        moved = right_annotations.pop()  # leftmost of the right pool
        left_annotations.append(moved)

    # Sort each side by y for clean vertical distribution
    left_annotations.sort(key=lambda a: a[1])
    right_annotations.sort(key=lambda a: a[1])

    # ── Place annotations on margins with leader lines ──
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_range = xmax - xmin
    y_range = ymax - ymin

    def place_margin_annotations(ann_list, side):
        if not ann_list:
            return
        n = len(ann_list)
        y_pad = y_range * 0.06
        y_slots = np.linspace(ymin + y_pad, ymax - y_pad, n)

        for i, (x_pt, y_pt, label, onto, color, domain) in enumerate(ann_list):
            y_text = y_slots[i]
            if side == "left":
                x_text_data = xmin - x_range * 0.04
                ha = "right"
                rad = "0.12"
            else:
                x_text_data = xmax + x_range * 0.04
                ha = "left"
                rad = "-0.12"

            txt = f"{label} ({onto.upper()})"
            ax.annotate(
                txt, xy=(x_pt, y_pt), xytext=(x_text_data, y_text),
                fontsize=5.8, color="#222222",
                ha=ha, va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color=color, lw=0.8, alpha=0.6,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                          alpha=0.12, edgecolor=color, lw=0.6),
                clip_on=False,
            )
            ax.plot(x_pt, y_pt, "o", color=color, markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=6)

    place_margin_annotations(left_annotations, "left")
    place_margin_annotations(right_annotations, "right")

    # ── Unified legend: two clean rows below the plot ──
    # Row 1: Domains (circles)   Row 2: Ontologies (crosses)
    # Both centred, in one legend with controlled ncol

    domain_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
               markersize=5.5, markeredgewidth=0)
        for c in DOMAIN_COLORS.values()
    ]
    domain_labels = list(DOMAIN_COLORS.keys())

    onto_handles = [
        Line2D([0], [0], marker="x", color=ONTOLOGY_COLORS.get(o, "#888888"),
               markersize=5, linestyle="none", markeredgewidth=1.0)
        for o in top_ontologies
    ]
    onto_labels = [o.upper() for o in top_ontologies]
    if other_mask.sum() > 0:
        onto_handles.append(
            Line2D([0], [0], marker="x", color="#AAAAAA", markersize=5,
                   linestyle="none", markeredgewidth=0.8))
        onto_labels.append("other")

    # Place two separate legends side by side at the bottom
    n_onto = len(onto_handles)
    # Domains legend — left of centre
    leg_dom = fig.legend(
        domain_handles, domain_labels,
        loc="lower left", bbox_to_anchor=(0.12, -0.005),
        ncol=5, fontsize=6, handletextpad=0.15, columnspacing=0.5,
        borderpad=0.3, title=r"$\bf{Topics\ (by\ domain)}$",
        title_fontsize=6.5,
        frameon=True, fancybox=True, framealpha=0.92, edgecolor="#cccccc")

    # Ontologies legend — right of centre
    leg_ont = fig.legend(
        onto_handles, onto_labels,
        loc="lower right", bbox_to_anchor=(0.90, -0.005),
        ncol=min(n_onto, 5), fontsize=5.5, handletextpad=0.15,
        columnspacing=0.4, borderpad=0.3,
        title=r"$\bf{Ontology\ terms\ (\times)}$",
        title_fontsize=6.5,
        frameon=True, fancybox=True, framealpha=0.92, edgecolor="#cccccc")

    fig.savefig(OUT_DIR / "fig4_umap_embedding_space.pdf", format="pdf",
                dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    path = OUT_DIR / "fig4_umap_embedding_space.pdf"
    print(f"  -> {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Ontology reach heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig5_ontology_reach_heatmap(con):
    """Heatmap: mapping count per (domain x ontology), showing which
    ontologies cover which scientific domains."""
    import seaborn as sns

    df = con.execute("""
        SELECT domain, ontology, COUNT(*) AS n
        FROM xref.topic_ontology_map
        WHERE similarity >= 0.65
        GROUP BY domain, ontology
    """).df()

    # Pivot to matrix
    mat = df.pivot_table(index="ontology", columns="domain",
                         values="n", fill_value=0)

    # Order: domains left-to-right by total, ontologies top-to-bottom by total
    domain_order = mat.sum(axis=0).sort_values(ascending=False).index.tolist()
    onto_order = mat.sum(axis=1).sort_values(ascending=False).index.tolist()
    mat = mat.loc[onto_order, domain_order]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.8, 4.2))

    mat = mat.astype(int)
    sns.heatmap(mat, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Mappings", "shrink": 0.8},
                annot_kws={"fontsize": 6.5})

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    save_fig(fig, "fig5_ontology_reach_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Vignette composite (2x2)
# ═══════════════════════════════════════════════════════════════════════════

def fig6_vignette_composite(con):
    """2x2 composite: one highlight per vignette."""

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.8))
    ((axA, axB), (axC, axD)) = axes

    # ── Panel A (V1): Mean disruption with vs without code ──
    df_a = con.execute("""
        SELECT
            has_pwc,
            COUNT(*) AS n,
            AVG(sciscinet_disruption) AS mean_cd5,
            APPROX_QUANTILE(sciscinet_disruption, 0.25) AS q25,
            APPROX_QUANTILE(sciscinet_disruption, 0.50) AS median,
            APPROX_QUANTILE(sciscinet_disruption, 0.75) AS q75
        FROM xref.unified_papers
        WHERE sciscinet_disruption IS NOT NULL
        GROUP BY has_pwc
    """).df()

    row_code = df_a[df_a["has_pwc"] == True].iloc[0]
    row_noco = df_a[df_a["has_pwc"] == False].iloc[0]

    cats = ["Without code", "With code"]
    means = [row_noco["mean_cd5"], row_code["mean_cd5"]]
    medians = [row_noco["median"], row_code["median"]]
    q25s = [row_noco["q25"], row_code["q25"]]
    q75s = [row_noco["q75"], row_code["q75"]]
    ns = [row_noco["n"], row_code["n"]]
    colors_a = ["#999999", SOURCE_COLORS["PWC"]]

    x_pos = [0, 1]
    for i, (cat, m, med, q25, q75, c) in enumerate(
            zip(cats, means, medians, q25s, q75s, colors_a)):
        axA.bar(i, m, color=c, width=0.5, alpha=0.8, zorder=3)
        axA.vlines(i, q25, q75, color="black", lw=1.2, zorder=4)
        axA.scatter(i, med, color="white", s=20, zorder=5,
                    edgecolors="black", linewidths=0.6)
    axA.set_xticks(x_pos)
    axA.set_xticklabels(cats, fontsize=7)
    axA.set_ylabel("Mean CD$_5$ disruption")
    axA.set_title("A  Disruption vs code", loc="left", fontsize=8)
    axA.axhline(0, color="#cccccc", lw=0.5, zorder=1)
    # Sample sizes as x-axis sub-labels (below category names)
    for i, n in enumerate(ns):
        lbl = f"n = {n/1e6:.1f}M" if n >= 1e6 else f"n = {n:,.0f}"
        axA.text(i, -0.18, lbl,
                 ha="center", va="top", fontsize=5.5, color="#888888",
                 transform=axA.get_xaxis_transform())

    # ── Panel B (V2): Retraction-enriched ontology terms ──
    # Deduplicate: same term label can map from multiple ontologies.
    # Aggregate across ontologies per term, pick dominant ontology via window.
    df_b = con.execute("""
        WITH topic_ret AS (
            SELECT wt.topic_id,
                   COUNT(*) AS n,
                   SUM(CASE WHEN u.has_retraction THEN 1 ELSE 0 END) AS n_ret
            FROM xref.unified_papers u
            JOIN openalex.works_topics wt ON wt.work_id = u.openalex_id
            WHERE u.openalex_id IS NOT NULL AND wt.score >= 0.5
            GROUP BY wt.topic_id
        ),
        per_onto AS (
            SELECT LOWER(m.ontology_term_label) AS term,
                   m.ontology, m.domain,
                   SUM(tr.n_ret) AS retracted, SUM(tr.n) AS total
            FROM xref.topic_ontology_map m
            JOIN topic_ret tr ON tr.topic_id = m.topic_id
            WHERE m.similarity >= 0.85
            GROUP BY LOWER(m.ontology_term_label), m.ontology, m.domain
        ),
        per_term AS (
            SELECT term, domain,
                   SUM(retracted) AS retracted, SUM(total) AS total,
                   100.0 * SUM(retracted) / SUM(total) AS ret_rate
            FROM per_onto
            GROUP BY term, domain
            HAVING SUM(total) >= 2000 AND SUM(retracted) >= 20
        ),
        best_onto AS (
            SELECT po.term, po.ontology,
                   ROW_NUMBER() OVER (PARTITION BY po.term ORDER BY po.total DESC) AS rn
            FROM per_onto po
        )
        SELECT pt.term, bo.ontology, pt.domain, pt.retracted, pt.total, pt.ret_rate
        FROM per_term pt
        JOIN best_onto bo ON bo.term = pt.term AND bo.rn = 1
        ORDER BY pt.ret_rate DESC
        LIMIT 10
    """).df()

    overall_rate = con.execute("""
        SELECT 100.0 * SUM(CASE WHEN has_retraction THEN 1 ELSE 0 END)
                     / COUNT(*) FROM xref.unified_papers
    """).fetchone()[0]

    colors_b = [DOMAIN_COLORS.get(d, "#888888") for d in df_b["domain"]]
    y_pos = np.arange(len(df_b))
    bars_b = axB.barh(y_pos, df_b["ret_rate"], color=colors_b,
                      height=0.65, zorder=3)
    # Enrichment ratio annotations — inside bar if bar is wide enough
    max_rate = df_b["ret_rate"].max()
    for i, (_, row) in enumerate(df_b.iterrows()):
        enrich = row["ret_rate"] / overall_rate if overall_rate > 0 else 0
        if row["ret_rate"] > max_rate * 0.35:
            axB.text(row["ret_rate"] - 0.15, i, f"{enrich:.0f}\u00d7",
                     va="center", ha="right", fontsize=6, color="white",
                     fontweight="bold")
        else:
            axB.text(row["ret_rate"] + 0.15, i, f"{enrich:.0f}\u00d7",
                     va="center", ha="left", fontsize=6, color="#444444")
    axB.set_yticks(y_pos)
    # Smart title-case: preserve common acronyms (AI, IoT, DNA, etc.)
    _ACRONYMS = {"ai", "iot", "dna", "rna", "covid", "hiv", "gps", "nlp",
                 "ml", "crispr", "pcr", "mrna", "llm"}
    def smart_title(s, maxlen=25):
        words = s.title().split()
        words = [w.upper() if w.lower() in _ACRONYMS else w for w in words]
        result = " ".join(words)
        if len(result) > maxlen:
            result = result[:maxlen-1].rstrip() + "."
        return result
    axB.set_yticklabels(
        [smart_title(t) for t in df_b["term"]],
        fontsize=6)
    axB.set_xlabel("Retraction rate (%)")
    axB.set_title("B  Retraction hotspots", loc="left", fontsize=8)
    axB.invert_yaxis()
    # Add small padding to the right for labels
    axB.set_xlim(0, max_rate * 1.15)
    # Domain legend — compact
    domain_handles = [Line2D([0], [0], color=c, lw=5, label=d.split()[0])
                      for d, c in DOMAIN_COLORS.items()
                      if d in df_b["domain"].values]
    if domain_handles:
        axB.legend(handles=domain_handles, fontsize=5.5, loc="lower right",
                   handlelength=0.8, borderpad=0.3, labelspacing=0.3)

    # ── Panel C (V3): Patent-cited papers over time ──
    df_c = con.execute("""
        SELECT year,
               SUM(CASE WHEN has_patent THEN 1 ELSE 0 END) AS patent_papers,
               COUNT(*) AS total
        FROM xref.unified_papers
        WHERE year BETWEEN 1960 AND 2023
        GROUP BY year
        ORDER BY year
    """).df()

    axC.fill_between(df_c["year"], df_c["patent_papers"],
                     color=SOURCE_COLORS["RoS"], alpha=0.35, lw=0)
    axC.plot(df_c["year"], df_c["patent_papers"],
             color=SOURCE_COLORS["RoS"], lw=1.0)
    axC.axvline(2017, color="#C62828", ls="--", lw=0.8, alpha=0.7)
    axC.axvspan(2017, 2023, alpha=0.06, color="#C62828")
    # Position cutoff label after data is plotted
    ymax_c = df_c["patent_papers"].max()
    axC.text(2018, ymax_c * 0.92, "RoS\ncutoff",
             fontsize=5.5, color="#C62828", va="top")
    axC.set_xlabel("Publication year")
    axC.set_ylabel("Patent-cited papers")
    axC.set_title("C  Paper-to-patent", loc="left", fontsize=8)
    axC.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    # ── Panel D (V4): Pairwise citation correlation matrix ──
    # Use log-transformed citations for a more meaningful correlation
    df_d = con.execute("""
        SELECT s2ag_citationcount AS s2ag,
               oa_cited_by_count AS oa,
               sciscinet_citation_count AS sciscinet
        FROM xref.unified_papers
        WHERE s2ag_citationcount IS NOT NULL
          AND oa_cited_by_count IS NOT NULL
          AND sciscinet_citation_count IS NOT NULL
        USING SAMPLE 200000
    """).df()

    # Log-transform for Pearson on log scale (same as in Bland-Altman)
    df_log = pd.DataFrame({
        "S2AG": np.log10(df_d["s2ag"].values + 1),
        "OpenAlex": np.log10(df_d["oa"].values + 1),
        "SciSciNet": np.log10(df_d["sciscinet"].values + 1),
    })
    labels_d = list(df_log.columns)
    corr = df_log.corr().values

    # Tighter color range to show differences between 0.93 and 0.99
    im = axD.imshow(corr, cmap="YlOrRd", vmin=0.9, vmax=1.0, aspect="equal")
    for i in range(3):
        for j in range(3):
            axD.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center",
                     fontsize=8, fontweight="bold",
                     color="white" if corr[i, j] > 0.97 else "black")
    axD.set_xticks(range(3))
    axD.set_yticks(range(3))
    axD.set_xticklabels(labels_d, fontsize=7)
    axD.set_yticklabels(labels_d, fontsize=7)
    axD.set_title("D  Citation agreement", loc="left", fontsize=8)
    axD.spines["left"].set_visible(True)
    axD.spines["bottom"].set_visible(True)
    cb = fig.colorbar(im, ax=axD, shrink=0.75, pad=0.04)
    cb.set_label("Pearson r (log scale)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    fig.tight_layout(h_pad=2.0, w_pad=1.8)
    save_fig(fig, "fig6_vignette_composite")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

FIGURES = {
    1: ("Citation Bland-Altman",       fig1_citation_bland_altman),
    2: ("UpSet source overlap",        fig2_upset_source_overlap),
    3: ("Temporal coverage ridgeline", fig3_temporal_ridgeline),
    4: ("UMAP embedding space",        fig4_umap_embedding_space),
    5: ("Ontology reach heatmap",      fig5_ontology_reach_heatmap),
    6: ("Vignette composite",          fig6_vignette_composite),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, nargs="+",
                        help="Only generate specified figures (1-6)")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP figure (requires GPU + model)")
    args = parser.parse_args()

    to_generate = args.only or list(FIGURES.keys())
    if args.skip_umap and 4 in to_generate:
        to_generate.remove(4)

    print(f"Science Data Lake — Paper Figure Generator")
    print(f"Output: {OUT_DIR.relative_to(ROOT)}/")
    print(f"Figures: {to_generate}\n")

    con = duckdb.connect(DB_PATH, read_only=True)

    t_total = time.time()
    for num in to_generate:
        name, func = FIGURES[num]
        print(f"[{num}/6] {name}...")
        t0 = time.time()
        try:
            func(con)
            print(f"       ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  !! FAILED: {e}")
            import traceback
            traceback.print_exc()

    con.close()
    print(f"\nDone. Total: {time.time() - t_total:.1f}s")
    print(f"Figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
