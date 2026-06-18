"""
Quartile-based complexity analysis for SKU110K.

Computes per-image complexity (bounding box count) and assigns
quartile bins (Q1-Q4) for analysis and error reporting.

Note: We use SKU110K's native train/val/test split.
Quartile bins are for ANALYSIS ONLY, not for re-splitting.

Usage:
    from src.data.splits import compute_complexity_scores, assign_quartile_bins

    scores = compute_complexity_scores(Path("data/processed/SKU110K/labels/train"))
    bins = assign_quartile_bins(scores)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_complexity_scores(label_dir: Path) -> dict[str, int]:
    """Compute per-image complexity as bounding box count.

    Args:
        label_dir: Directory containing YOLO-format .txt label files.

    Returns:
        Dict mapping image stem (filename without extension) to bbox count.
    """
    scores: dict[str, int] = {}
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            count = sum(1 for line in f if line.strip())
        scores[label_file.stem] = count

    logger.info(
        "Computed complexity for %d images in %s (mean=%.1f, median=%d)",
        len(scores),
        label_dir,
        np.mean(list(scores.values())) if scores else 0,
        int(np.median(list(scores.values()))) if scores else 0,
    )
    return scores


def assign_quartile_bins(scores: dict[str, int]) -> dict[str, str]:
    """Assign Q1/Q2/Q3/Q4 complexity bins based on bbox count quartiles.

    Q1 (sparse):  0-25th percentile
    Q2 (medium):  25-50th percentile
    Q3 (dense):   50-75th percentile
    Q4 (very dense): 75-100th percentile

    Args:
        scores: Dict mapping image name to bbox count.

    Returns:
        Dict mapping image name to quartile bin ("Q1", "Q2", "Q3", "Q4").
    """
    if not scores:
        return {}

    values = np.array(list(scores.values()))
    q1, q2, q3 = np.percentile(values, [25, 50, 75])

    bins = {}
    for name, count in scores.items():
        if count <= q1:
            bins[name] = "Q1"
        elif count <= q2:
            bins[name] = "Q2"
        elif count <= q3:
            bins[name] = "Q3"
        else:
            bins[name] = "Q4"

    # Log distribution
    from collections import Counter

    dist = Counter(bins.values())
    logger.info(
        "Quartile distribution: Q1=%d, Q2=%d, Q3=%d, Q4=%d (thresholds: %.0f/%.0f/%.0f)",
        dist.get("Q1", 0),
        dist.get("Q2", 0),
        dist.get("Q3", 0),
        dist.get("Q4", 0),
        q1,
        q2,
        q3,
    )

    return bins


def generate_complexity_report(processed_dir: Path, output_path: Path) -> dict:
    """Generate complexity analysis report for all splits.

    Args:
        processed_dir: Path to processed data directory (contains labels/).
        output_path: Path to save JSON report.

    Returns:
        Report dict with per-split statistics.
    """
    report: dict[str, dict] = {}

    for split in ["train", "val", "test"]:
        label_dir = processed_dir / "labels" / split
        if not label_dir.exists():
            logger.warning("Label directory not found: %s", label_dir)
            continue

        scores = compute_complexity_scores(label_dir)
        bins = assign_quartile_bins(scores)

        values = list(scores.values())
        report[split] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else 0,
            "median": int(np.median(values)) if values else 0,
            "min": int(np.min(values)) if values else 0,
            "max": int(np.max(values)) if values else 0,
            "std": float(np.std(values)) if values else 0,
            "q1_threshold": float(np.percentile(values, 25)) if values else 0,
            "q2_threshold": float(np.percentile(values, 50)) if values else 0,
            "q3_threshold": float(np.percentile(values, 75)) if values else 0,
            "quartile_distribution": {
                q: sum(1 for b in bins.values() if b == q)
                for q in ["Q1", "Q2", "Q3", "Q4"]
            },
        }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Complexity report saved to %s", output_path)
    return report
