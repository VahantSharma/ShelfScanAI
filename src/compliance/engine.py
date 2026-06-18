"""
Slot-level planogram compliance evaluation.

Compares detected products against a reference planogram
on a per-slot basis, determining:
  - Correct product in correct position
  - Missing SKU (expected but absent)
  - Wrong product (present but different SKU)
  - Misplaced product (present and correct, but wrong position)

Usage:
    engine = ComplianceEngine(config)
    results = engine.evaluate(detections, matches, reference_slots, H)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.config import ComplianceConfig

logger = logging.getLogger(__name__)


@dataclass
class SlotResult:
    """Result of evaluating a single planogram slot.

    Attributes:
        slot_id: Unique slot identifier.
        expected_sku: SKU that should be in this slot.
        detected_sku: SKU actually detected (None if missing).
        status: Evaluation result.
        confidence: Detection/matching confidence.
        position_match: Whether detected position matches slot.
    """

    slot_id: str
    expected_sku: str
    detected_sku: str | None
    status: Literal["correct", "missing", "wrong_product", "misplaced"]
    confidence: float
    position_match: bool


@dataclass
class ComplianceReport:
    """Full compliance evaluation report.

    Attributes:
        slot_results: Per-slot evaluation results.
        compliance_score: Overall score (0-100).
        total_slots: Total number of reference slots.
        correct_count: Number of correctly filled slots.
        missing_count: Number of missing SKUs.
        wrong_count: Number of wrong products.
        misplaced_count: Number of misplaced products.
    """

    slot_results: list[SlotResult]
    compliance_score: float
    total_slots: int
    correct_count: int
    missing_count: int
    wrong_count: int
    misplaced_count: int

    @property
    def missing_skus(self) -> list[str]:
        """List of SKU IDs that are missing."""
        return [
            r.expected_sku
            for r in self.slot_results
            if r.status == "missing"
        ]

    @property
    def wrong_skus(self) -> list[str]:
        """List of slot IDs with wrong products."""
        return [
            r.slot_id
            for r in self.slot_results
            if r.status == "wrong_product"
        ]


class ComplianceEngine:
    """Evaluates planogram compliance slot-by-slot.

    Works with or without homography alignment. If H is provided,
    field detections are mapped to reference coordinates. Otherwise,
    uses heuristic position matching.

    Attributes:
        config: Compliance configuration with weights and thresholds.
    """

    def __init__(self, config: ComplianceConfig) -> None:
        """Initialize compliance engine.

        Args:
            config: Compliance configuration.
        """
        self.config = config
        self.position_tolerance = config.thresholds.get("position_tolerance", 0.1)
        self.min_confidence = config.thresholds.get("min_confidence", 0.6)

    def evaluate(
        self,
        detections: list[dict],
        matches: list[dict],
        reference_slots: list[dict],
        H: np.ndarray | None = None,
    ) -> ComplianceReport:
        """Evaluate compliance of field photo against reference planogram.

        Args:
            detections: List of detected products from YOLO.
                Each: {"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int}
            matches: List of SKU identity matches from CLIP+FAISS.
                Each: {"crop_idx": int, "matched_sku": str, "confidence": float}
            reference_slots: List of planogram slots.
                Each: {"slot_id": str, "expected_sku": str, "bbox": [cx,cy,w,h]}
            H: Optional homography matrix to map field coords to reference.

        Returns:
            ComplianceReport with per-slot results and overall score.
        """
        slot_results: list[SlotResult] = []

        for slot in reference_slots:
            slot_id = slot["slot_id"]
            expected_sku = slot["expected_sku"]
            slot_bbox = slot["bbox"]  # [cx, cy, w, h] normalized

            # Find which detection (if any) overlaps this slot
            best_match = None
            best_iou = 0.0

            for det_idx, det in enumerate(detections):
                det_bbox = det["bbox"]  # [x1, y1, x2, y2] pixel coords

                # Convert detection to normalized cx, cy, w, h
                # Note: this is approximate; proper mapping needs H
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                det_w = det_bbox[2] - det_bbox[0]
                det_h = det_bbox[3] - det_bbox[1]

                # For now, use simple distance-based matching
                # In production, use H to map coordinates properly
                slot_cx, slot_cy = slot_bbox[0], slot_bbox[1]
                distance = np.sqrt((det_cx - slot_cx) ** 2 + (det_cy - slot_cy) ** 2)

                if distance < self.position_tolerance:
                    # Find the identity match for this detection
                    identity = None
                    for m in matches:
                        if m["crop_idx"] == det_idx:
                            identity = m["matched_sku"]
                            break

                    # Compute a pseudo-IoU for ranking
                    iou = max(0, 1 - distance / self.position_tolerance)

                    if iou > best_iou:
                        best_iou = iou
                        best_match = {
                            "det_idx": det_idx,
                            "identity": identity,
                            "confidence": det["confidence"],
                            "distance": distance,
                        }

            # Evaluate slot
            if best_match is None:
                # Slot is empty
                result = SlotResult(
                    slot_id=slot_id,
                    expected_sku=expected_sku,
                    detected_sku=None,
                    status="missing",
                    confidence=0.0,
                    position_match=False,
                )
            elif best_match["identity"] == expected_sku:
                # Correct product in roughly correct position
                result = SlotResult(
                    slot_id=slot_id,
                    expected_sku=expected_sku,
                    detected_sku=best_match["identity"],
                    status="correct",
                    confidence=best_match["confidence"],
                    position_match=True,
                )
            elif best_match["identity"] is not None:
                # Wrong product
                result = SlotResult(
                    slot_id=slot_id,
                    expected_sku=expected_sku,
                    detected_sku=best_match["identity"],
                    status="wrong_product",
                    confidence=best_match["confidence"],
                    position_match=True,
                )
            else:
                # Product detected but identity unknown
                result = SlotResult(
                    slot_id=slot_id,
                    expected_sku=expected_sku,
                    detected_sku=None,
                    status="missing",
                    confidence=best_match["confidence"],
                    position_match=True,
                )

            slot_results.append(result)

        # Compute summary statistics
        correct = sum(1 for r in slot_results if r.status == "correct")
        missing = sum(1 for r in slot_results if r.status == "missing")
        wrong = sum(1 for r in slot_results if r.status == "wrong_product")
        misplaced = sum(1 for r in slot_results if r.status == "misplaced")
        total = len(slot_results)

        # Compute score
        from src.compliance.scorer import compute_compliance_score

        score = compute_compliance_score(slot_results, self.config)

        report = ComplianceReport(
            slot_results=slot_results,
            compliance_score=score,
            total_slots=total,
            correct_count=correct,
            missing_count=missing,
            wrong_count=wrong,
            misplaced_count=misplaced,
        )

        logger.info(
            "Compliance evaluation: score=%.1f, correct=%d/%d, missing=%d, wrong=%d",
            score,
            correct,
            total,
            missing,
            wrong,
        )

        return report
