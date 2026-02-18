#!/usr/bin/env python3
"""
Generate coverage statistics and UpSet plot from unified_papers.

Produces:
  - datasets/xref/coverage_stats/upset_data.json
  - datasets/xref/coverage_stats/coverage_summary.json
  - figures/upset_plot.png (UpSet plot — Figure 1 of the paper)
  - figures/coverage_by_year.png

Usage:
    python scripts/generate_coverage_plots.py
    python scripts/generate_coverage_plots.py --stats-only  # Skip plot generation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Resolve data lake root ───────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

UNIFIED_DIR = ROOT / "datasets" / "xref" / "unified_papers"
COVERAGE_DIR = ROOT / "datasets" / "xref" / "coverage_stats"
FIGURES_DIR = ROOT / "figures"


def compute_stats(dry_run=False):
    """Compute coverage statistics from unified_papers Parquet files."""
    import duckdb

    unified_path = UNIFIED_DIR / "*.parquet"
    if not list(UNIFIED_DIR.glob("*.parquet")):
        print("ERROR: unified_papers not materialized yet.")
        print("Run: python scripts/materialize_unified_papers.py")
        return None

    con = duckdb.connect(":memory:")
    con.execute("SET threads=16")

    print("Loading unified_papers...")
    t0 = time.time()

    # Total count
    total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{unified_path}')"
    ).fetchone()[0]
    print(f"Total papers (unique DOIs): {total:,}")

    # Source labels
    sources = {
        "S2AG": "has_s2ag",
        "OpenAlex": "has_openalex",
        "SciSciNet": "has_sciscinet",
        "PWC": "has_pwc",
        "RetWatch": "has_retraction",
        "Patents": "has_patent",
    }

    # Per-source counts
    print("\nPer-source coverage:")
    per_source = {}
    for label, col in sources.items():
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{unified_path}') WHERE {col}"
        ).fetchone()[0]
        per_source[label] = n
        pct = 100.0 * n / total if total > 0 else 0
        print(f"  {label:12s}: {n:>13,} ({pct:5.1f}%)")

    # Full UpSet-style intersection counts
    print("\nComputing all set intersections...")
    cols = list(sources.values())
    col_str = ", ".join(cols)

    upset_rows = con.execute(f"""
        SELECT {col_str}, COUNT(*) AS count
        FROM read_parquet('{unified_path}')
        GROUP BY {col_str}
        ORDER BY count DESC
    """).fetchall()

    # Convert to structured data
    upset_data = []
    for row in upset_rows:
        entry = {"count": row[len(cols)]}
        labels = []
        for i, (label, col) in enumerate(sources.items()):
            entry[col] = bool(row[i])
            if row[i]:
                labels.append(label)
        entry["label"] = " + ".join(labels) if labels else "(none)"
        upset_data.append(entry)

    print(f"  {len(upset_data)} unique combinations")
    print("\nTop 15 intersections:")
    for entry in upset_data[:15]:
        print(f"  {entry['label']:60s}: {entry['count']:>13,}")

    # Year distribution
    print("\nYear distribution (top 20):")
    year_rows = con.execute(f"""
        SELECT year, COUNT(*) AS count,
               SUM(CASE WHEN has_s2ag THEN 1 ELSE 0 END) AS s2ag,
               SUM(CASE WHEN has_openalex THEN 1 ELSE 0 END) AS openalex,
               SUM(CASE WHEN has_sciscinet THEN 1 ELSE 0 END) AS sciscinet
        FROM read_parquet('{unified_path}')
        WHERE year IS NOT NULL AND year BETWEEN 1900 AND 2025
        GROUP BY year
        ORDER BY year DESC
        LIMIT 30
    """).fetchall()
    for r in year_rows[:20]:
        print(f"  {r[0]}: {r[1]:>10,} (S2AG={r[2]:,}, OA={r[3]:,}, SSN={r[4]:,})")

    # Save outputs
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)

    # UpSet data JSON
    json_path = COVERAGE_DIR / "upset_data.json"
    with open(json_path, "w") as f:
        json.dump(upset_data, f, indent=2)
    print(f"\nUpSet data: {json_path}")

    # Year data JSON
    year_data = [
        {"year": r[0], "total": r[1], "s2ag": r[2], "openalex": r[3], "sciscinet": r[4]}
        for r in year_rows
    ]
    year_path = COVERAGE_DIR / "year_distribution.json"
    with open(year_path, "w") as f:
        json.dump(year_data, f, indent=2)

    # Summary JSON
    summary = {
        "total_unique_dois": total,
        "per_source": per_source,
        "n_combinations": len(upset_data),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = COVERAGE_DIR / "coverage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    con.close()
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    return upset_data, year_data, per_source, total


def generate_upset_plot(upset_data, per_source, total):
    """Generate UpSet plot from intersection data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from upsetplot import UpSet
        from pandas import DataFrame
    except ImportError as e:
        print(f"WARNING: Cannot generate plots (missing: {e})")
        print("Install: pip install matplotlib upsetplot pandas")
        return False

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build the membership matrix for upsetplot
    source_names = ["S2AG", "OpenAlex", "SciSciNet", "PWC", "RetWatch", "Patents"]
    col_map = {
        "S2AG": "has_s2ag",
        "OpenAlex": "has_openalex",
        "SciSciNet": "has_sciscinet",
        "PWC": "has_pwc",
        "RetWatch": "has_retraction",
        "Patents": "has_patent",
    }

    # Create DataFrame with multi-index
    rows = []
    counts = []
    for entry in upset_data:
        membership = tuple(entry[col_map[s]] for s in source_names)
        rows.append(membership)
        counts.append(entry["count"])

    import pandas as pd
    index = pd.MultiIndex.from_tuples(rows, names=source_names)
    series = pd.Series(counts, index=index)

    # Generate UpSet plot
    fig = plt.figure(figsize=(20, 10))
    upset = UpSet(
        series,
        sort_by="cardinality",
        show_counts="{:,}",
        show_percentages=False,  # Disable percentages to avoid overlap
        min_subset_size=10000,  # Show intersections >= 10K (includes RetWatch)
        element_size=40,
    )
    axes = upset.plot(fig=fig)
    # Rotate count labels on bars to avoid overlap
    if "intersections" in axes:
        ax_bars = axes["intersections"]
        for txt in ax_bars.texts:
            txt.set_fontsize(8)
            txt.set_rotation(45)
            txt.set_ha("left")
    plt.suptitle(
        f"Science Data Lake: Source Coverage Overlap ({total:,} unique DOIs)",
        fontsize=16, y=1.02,
    )

    plot_path = FIGURES_DIR / "upset_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UpSet plot: {plot_path}")

    # Generate coverage by year plot
    year_path = COVERAGE_DIR / "year_distribution.json"
    if year_path.exists():
        with open(year_path) as f:
            year_data = json.load(f)

        years = [d["year"] for d in sorted(year_data, key=lambda x: x["year"]) if d["year"] and 1950 <= d["year"] <= 2025]
        year_map = {d["year"]: d for d in year_data}

        fig, ax = plt.subplots(figsize=(14, 6))
        for src, color in [("openalex", "#E64A19"), ("s2ag", "#1976D2"), ("sciscinet", "#388E3C")]:
            vals = [year_map.get(y, {}).get(src, 0) for y in years]
            ax.plot(years, vals, label=src.upper() if src != "openalex" else "OpenAlex", linewidth=1.5, color=color)

        ax.set_xlabel("Year")
        ax.set_ylabel("Papers with DOI")
        ax.set_title("Cross-Source Coverage by Publication Year")
        ax.legend()
        ax.grid(True, alpha=0.3)

        year_plot_path = FIGURES_DIR / "coverage_by_year.png"
        fig.savefig(year_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Year plot: {year_plot_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate coverage statistics and UpSet plot"
    )
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute statistics, skip plot generation")
    args = parser.parse_args()

    result = compute_stats()
    if result is None:
        return 1

    upset_data, year_data, per_source, total = result

    if not args.stats_only:
        generate_upset_plot(upset_data, per_source, total)

    return 0


if __name__ == "__main__":
    sys.exit(main())
