"""
Weighted compliance scoring.

Computes a 0-100 compliance score from slot-level evaluation results.
All weights are configurable via configs/compliance.yaml.

Usage:
    from src.compliance.scorer import compute_compliance_score

    score = compute_compliance_score(slot_results, config)
"""

from __future__ import annotations

import logging

from src.compliance.engine import SlotResult
from src.config import ComplianceConfig

logger = logging.getLogger(__name__)


def compute_compliance_score(
    slot_results: list[SlotResult],
    config: ComplianceConfig,
) -> float:
    """Compute weighted compliance score.

    Formula:
        score = (
            w_presence * (correct / total) +
            w_facings * (correct_facings_ratio) +
            w_correctness * (1 - wrong_product_rate)
        ) * 100

    Weights:
        presence: 0.5 (how many slots are correctly filled)
        facings: 0.3 (how many expected products are present)
        correctness: 0.2 (how few wrong products are present)

    Args:
        slot_results: Per-slot evaluation results.
        config: Compliance configuration with weights.

    Returns:
        Compliance score between 0 and 100.
    """
    if not slot_results:
        return 0.0

    total = len(slot_results)
    correct = sum(1 for r in slot_results if r.status == "correct")
    missing = sum(1 for r in slot_results if r.status == "missing")
    wrong = sum(1 for r in slot_results if r.status == "wrong_product")

    # Presence score: fraction of slots that are not missing
    presence_ratio = (total - missing) / total if total > 0 else 0.0

    # Facings score: fraction of slots with correct products
    facings_ratio = correct / total if total > 0 else 0.0

    # Correctness score: complement of wrong product rate
    wrong_rate = wrong / total if total > 0 else 0.0
    correctness_ratio = 1.0 - wrong_rate

    # Get weights
    w_presence = config.weights.get("presence", 0.5)
    w_facings = config.weights.get("facings", 0.3)
    w_correctness = config.weights.get("correctness", 0.2)

    # Normalize weights (in case they don't sum to 1)
    w_total = w_presence + w_facings + w_correctness
    if w_total > 0:
        w_presence /= w_total
        w_facings /= w_total
        w_correctness /= w_total

    # Compute weighted score
    score = (
        w_presence * presence_ratio
        + w_facings * facings_ratio
        + w_correctness * correctness_ratio
    ) * 100.0

    # Clamp to [0, 100]
    score = max(0.0, min(100.0, score))

    logger.debug(
        "Compliance score: %.1f (presence=%.3f, facings=%.3f, correctness=%.3f)",
        score,
        presence_ratio,
        facings_ratio,
        correctness_ratio,
    )

    return round(score, 1)
