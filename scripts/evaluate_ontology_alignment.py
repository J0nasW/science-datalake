#!/usr/bin/env python3
"""
Evaluate ontology alignment quality against a gold-standard annotation set.

Two modes:
  --generate-sample : Produce a stratified 300-pair annotation set from the
                      16,150 existing mappings for manual labelling.
  --compute-metrics : After manual annotation, compute P/R/F1 per stratum,
                      at multiple thresholds, and generate a precision-recall
                      curve figure.

Output:
  paper/evaluation/gold_standard_sample.tsv   — annotation file (generate)
  paper/evaluation/annotation_guidelines.md   — labelling instructions (generate)
  paper/evaluation/alignment_metrics.json     — evaluation results (compute)
  paper/figures/fig_precision_recall.pdf       — PR curve figure (compute)

Usage:
    python scripts/evaluate_ontology_alignment.py --generate-sample
    python scripts/evaluate_ontology_alignment.py --compute-metrics
    python scripts/evaluate_ontology_alignment.py --compute-metrics --baselines paper/evaluation/baseline_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Resolve data lake root ───────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

DB_PATH = ROOT / "datalake.duckdb"
TOPIC_MAP_DIR = ROOT / "datasets" / "xref" / "topic_ontology_map"
EVAL_DIR = ROOT / "paper" / "evaluation"
FIGURES_DIR = ROOT / "paper" / "figures"


def assign_stratum(similarity: float) -> str:
    """Assign a mapping to its quality stratum."""
    if similarity >= 0.95:
        return "exact"
    elif similarity >= 0.85:
        return "high_quality"
    elif similarity >= 0.75:
        return "mid_range"
    else:
        return "borderline"


STRATUM_TARGETS = {
    "exact": 50,
    "high_quality": 100,
    "mid_range": 100,
    "borderline": 50,
}


def generate_sample():
    """Produce a stratified 300-pair annotation set."""
    import duckdb

    print("Loading mappings from Parquet...")
    conn = duckdb.connect(":memory:")
    mappings = conn.execute(f"""
        SELECT topic_id, topic_name, ontology, ontology_term_id,
               ontology_term_label, similarity, match_type
        FROM read_parquet('{TOPIC_MAP_DIR}/*.parquet')
        ORDER BY similarity DESC
    """).fetchall()
    conn.close()

    columns = ["topic_id", "topic_name", "ontology", "ontology_term_id",
               "ontology_term_label", "similarity", "match_type"]

    # Group by stratum
    strata = {"exact": [], "high_quality": [], "mid_range": [], "borderline": []}
    for row in mappings:
        d = dict(zip(columns, row))
        s = assign_stratum(d["similarity"])
        strata[s].append(d)

    print(f"Total mappings: {len(mappings):,}")
    for s, items in strata.items():
        print(f"  {s}: {len(items):,}")

    # Stratified proportional sampling within each stratum across ontologies
    rng = np.random.default_rng(seed=42)
    sample = []
    pair_id = 0

    for stratum, target in STRATUM_TARGETS.items():
        items = strata[stratum]
        if len(items) <= target:
            # Take all if fewer than target
            selected = items
        else:
            # Proportional sampling across ontologies
            by_ontology = {}
            for item in items:
                by_ontology.setdefault(item["ontology"], []).append(item)

            total = len(items)
            selected = []
            remainder_pool = []

            for onto, onto_items in sorted(by_ontology.items()):
                n_select = max(1, round(target * len(onto_items) / total))
                if n_select >= len(onto_items):
                    selected.extend(onto_items)
                else:
                    indices = rng.choice(len(onto_items), size=n_select, replace=False)
                    selected.extend(onto_items[i] for i in indices)
                    remainder_pool.extend(
                        onto_items[i] for i in range(len(onto_items)) if i not in set(indices)
                    )

            # Adjust to hit target exactly
            if len(selected) < target and remainder_pool:
                deficit = target - len(selected)
                extra_indices = rng.choice(
                    len(remainder_pool), size=min(deficit, len(remainder_pool)), replace=False
                )
                selected.extend(remainder_pool[i] for i in extra_indices)
            elif len(selected) > target:
                indices = rng.choice(len(selected), size=target, replace=False)
                selected = [selected[i] for i in sorted(indices)]

        for item in selected:
            pair_id += 1
            sample.append({
                "pair_id": pair_id,
                "topic_name": item["topic_name"],
                "ontology": item["ontology"],
                "ontology_term_label": item["ontology_term_label"],
                "similarity": round(item["similarity"], 4),
                "match_type": item["match_type"],
                "stratum": assign_stratum(item["similarity"]),
                "label": "",  # To be filled by annotator
            })

    # Shuffle to prevent annotator bias from ordered strata
    rng.shuffle(sample)
    # Re-assign pair_ids after shuffle
    for i, s in enumerate(sample):
        s["pair_id"] = i + 1

    # Write TSV
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = EVAL_DIR / "gold_standard_sample.tsv"
    header = ["pair_id", "topic_name", "ontology", "ontology_term_label",
              "similarity", "match_type", "stratum", "label"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for s in sample:
            f.write("\t".join(str(s[h]) for h in header) + "\n")

    print(f"\nWrote {len(sample)} annotation pairs to {tsv_path}")
    for stratum, target in STRATUM_TARGETS.items():
        actual = sum(1 for s in sample if s["stratum"] == stratum)
        print(f"  {stratum}: {actual} (target: {target})")

    # Write annotation guidelines
    guidelines_path = EVAL_DIR / "annotation_guidelines.md"
    with open(guidelines_path, "w") as f:
        f.write(ANNOTATION_GUIDELINES)
    print(f"Wrote annotation guidelines to {guidelines_path}")


ANNOTATION_GUIDELINES = """\
# Ontology Alignment Annotation Guidelines

## Task
For each pair of (OpenAlex topic, ontology term), assign one of three labels:

## Labels

### `correct`
The ontology term is a semantically accurate match for the OpenAlex topic.
This includes:
- Exact or near-exact matches (e.g., "Machine Learning" -> "machine learning")
- Equivalent concepts with different naming conventions (e.g., "Deep Learning" -> "deep neural networks")
- The ontology term covers the same core concept, even if scope differs slightly

### `partial`
There is a meaningful semantic relationship, but the terms are not equivalent:
- Parent-child relationships (e.g., "Convolutional Neural Networks" -> "neural networks")
- Sibling concepts in the same domain (e.g., "Random Forests" -> "decision trees")
- Overlapping but distinct concepts (e.g., "Bioelectronics" -> "Biosensors")

### `incorrect`
The terms are unrelated or the match is spurious:
- No meaningful semantic relationship
- Surface-level string similarity without conceptual overlap
- Homonyms matched incorrectly (e.g., "Mercury" the planet vs "Mercury" the element)

## Guidelines
1. Judge based on semantic meaning, not string similarity
2. Consider the ontology context (e.g., a MeSH term is biomedical, CSO is computer science)
3. When in doubt between `correct` and `partial`, prefer `partial`
4. When in doubt between `partial` and `incorrect`, prefer `partial`
5. Fill in the `label` column in the TSV file with exactly one of: `correct`, `partial`, `incorrect`
"""


def compute_metrics(baselines_path=None):
    """Compute P/R/F1 from annotated gold standard."""
    tsv_path = EVAL_DIR / "gold_standard_sample.tsv"
    if not tsv_path.exists():
        print(f"ERROR: Annotation file not found: {tsv_path}")
        print("Run --generate-sample first, then annotate the TSV.")
        return 1

    # Load annotations
    annotations = []
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                continue
            d = dict(zip(header, fields))
            d["similarity"] = float(d["similarity"])
            annotations.append(d)

    # Check for unannotated pairs
    unannotated = [a for a in annotations if a.get("label", "").strip() == ""]
    if unannotated:
        print(f"ERROR: {len(unannotated)} pairs are not yet annotated.")
        print("Please fill in the 'label' column for all pairs.")
        return 1

    valid_labels = {"correct", "partial", "incorrect"}
    invalid = [a for a in annotations if a["label"].strip().lower() not in valid_labels]
    if invalid:
        print(f"ERROR: {len(invalid)} pairs have invalid labels.")
        print(f"Valid labels: {valid_labels}")
        for a in invalid[:5]:
            print(f"  pair_id={a['pair_id']}: label='{a['label']}'")
        return 1

    # Normalize labels
    for a in annotations:
        a["label"] = a["label"].strip().lower()

    print(f"Loaded {len(annotations)} annotated pairs")

    # ── Per-stratum metrics ──────────────────────────────────────────────────
    strata = ["exact", "high_quality", "mid_range", "borderline"]
    stratum_metrics = {}
    for stratum in strata:
        items = [a for a in annotations if a["stratum"] == stratum]
        if not items:
            continue
        tp = sum(1 for a in items if a["label"] == "correct")
        fp = sum(1 for a in items if a["label"] in ("partial", "incorrect"))
        # For recall: correct mappings that the method found
        # Since we're evaluating the system's output, all items are "predicted positive"
        # Precision = TP / (TP + FP) where FP = partial + incorrect
        precision = tp / len(items) if items else 0
        # Recall requires knowing total true matches — estimated from annotation rate
        stratum_metrics[stratum] = {
            "n": len(items),
            "correct": tp,
            "partial": sum(1 for a in items if a["label"] == "partial"),
            "incorrect": sum(1 for a in items if a["label"] == "incorrect"),
            "precision": round(precision, 4),
        }

    # ── Aggregate metrics ────────────────────────────────────────────────────
    total_tp = sum(1 for a in annotations if a["label"] == "correct")
    total_partial = sum(1 for a in annotations if a["label"] == "partial")
    total_incorrect = sum(1 for a in annotations if a["label"] == "incorrect")
    total_n = len(annotations)
    aggregate_precision = total_tp / total_n if total_n > 0 else 0

    # ── Threshold-based metrics ──────────────────────────────────────────────
    # At each threshold, compute precision on the subset with sim >= threshold
    # and recall as fraction of all correct pairs that pass the threshold
    threshold_metrics = {}
    for thresh in [0.65, 0.75, 0.85, 0.95]:
        above = [a for a in annotations if a["similarity"] >= thresh]
        if not above:
            threshold_metrics[str(thresh)] = {
                "n": 0, "precision": 0, "recall": 0, "f1": 0
            }
            continue
        tp = sum(1 for a in above if a["label"] == "correct")
        precision = tp / len(above) if above else 0
        recall = tp / total_tp if total_tp > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        threshold_metrics[str(thresh)] = {
            "n": len(above),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # ── Precision-recall curve (sweep 0.60-1.0 in 0.01 steps) ───────────────
    pr_curve = []
    for thresh_int in range(60, 101):
        thresh = thresh_int / 100.0
        above = [a for a in annotations if a["similarity"] >= thresh]
        if not above:
            pr_curve.append({"threshold": thresh, "precision": 1.0, "recall": 0.0, "n": 0})
            continue
        tp = sum(1 for a in above if a["label"] == "correct")
        precision = tp / len(above) if above else 0
        recall = tp / total_tp if total_tp > 0 else 0
        pr_curve.append({
            "threshold": round(thresh, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n": len(above),
        })

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Ontology Alignment Evaluation (n={total_n})")
    print(f"{'='*60}")
    print(f"  Correct:   {total_tp:4d} ({100*total_tp/total_n:.1f}%)")
    print(f"  Partial:   {total_partial:4d} ({100*total_partial/total_n:.1f}%)")
    print(f"  Incorrect: {total_incorrect:4d} ({100*total_incorrect/total_n:.1f}%)")
    print(f"  Aggregate precision: {aggregate_precision:.4f}")

    print(f"\n  Per-stratum precision:")
    for stratum in strata:
        if stratum in stratum_metrics:
            m = stratum_metrics[stratum]
            print(f"    {stratum:14s}: {m['precision']:.4f} "
                  f"(n={m['n']}, correct={m['correct']}, "
                  f"partial={m['partial']}, incorrect={m['incorrect']})")

    print(f"\n  Threshold-based P/R/F1:")
    for thresh in ["0.65", "0.75", "0.85", "0.95"]:
        m = threshold_metrics[thresh]
        print(f"    >= {thresh}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (n={m['n']})")

    # ── Generate precision-recall curve figure ───────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

        # BGE-large curve
        recalls = [p["recall"] for p in pr_curve]
        precisions = [p["precision"] for p in pr_curve]
        ax.plot(recalls, precisions, "o-", markersize=2, linewidth=1.5,
                label="BGE-large (ours)", color="#2563eb", zorder=5)

        # Mark key thresholds on the curve
        for thresh_val in [0.65, 0.75, 0.85, 0.95]:
            point = next((p for p in pr_curve if abs(p["threshold"] - thresh_val) < 0.005), None)
            if point and point["n"] > 0:
                ax.annotate(f'{thresh_val}',
                            xy=(point["recall"], point["precision"]),
                            textcoords="offset points", xytext=(8, -4),
                            fontsize=7, color="#2563eb")

        # Load and plot baselines if available
        if baselines_path:
            baselines_file = Path(baselines_path)
            if baselines_file.exists():
                with open(baselines_file) as f:
                    baselines_data = json.load(f)

                colors = {"tfidf": "#dc2626", "bm25": "#16a34a", "jaro_winkler": "#9333ea"}
                labels = {"tfidf": "TF-IDF", "bm25": "BM25", "jaro_winkler": "Jaro-Winkler"}
                markers = {"tfidf": "s", "bm25": "^", "jaro_winkler": "D"}

                for method, mdata in baselines_data.items():
                    if "pr_curve" in mdata:
                        b_recalls = [p["recall"] for p in mdata["pr_curve"]]
                        b_precisions = [p["precision"] for p in mdata["pr_curve"]]
                        ax.plot(b_recalls, b_precisions,
                                f'{markers.get(method, "o")}-',
                                markersize=2, linewidth=1.2,
                                label=labels.get(method, method),
                                color=colors.get(method, "#666666"),
                                alpha=0.8)

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("Ontology Alignment: Precision–Recall", fontsize=12)

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / "fig_precision_recall.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Precision-recall curve saved to {fig_path}")
    except ImportError:
        print("\n  WARNING: matplotlib not available, skipping figure generation")

    # ── Save metrics JSON ────────────────────────────────────────────────────
    results = {
        "n_annotated": total_n,
        "aggregate": {
            "correct": total_tp,
            "partial": total_partial,
            "incorrect": total_incorrect,
            "precision": round(aggregate_precision, 4),
        },
        "per_stratum": stratum_metrics,
        "threshold_metrics": threshold_metrics,
        "pr_curve": pr_curve,
    }

    metrics_path = EVAL_DIR / "alignment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ontology alignment against gold-standard annotations"
    )
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate stratified 300-pair annotation set")
    parser.add_argument("--compute-metrics", action="store_true",
                        help="Compute P/R/F1 from annotated gold standard")
    parser.add_argument("--baselines", type=str, default=None,
                        help="Path to baseline_comparison.json for PR curve overlay")
    args = parser.parse_args()

    if not args.generate_sample and not args.compute_metrics:
        parser.print_help()
        return 1

    if args.generate_sample:
        generate_sample()

    if args.compute_metrics:
        return compute_metrics(baselines_path=args.baselines)

    return 0


if __name__ == "__main__":
    sys.exit(main())
