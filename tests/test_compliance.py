"""
Tests for the compliance engine and scorer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.compliance.engine import ComplianceEngine, SlotResult
from src.compliance.scorer import compute_compliance_score
from src.config import ComplianceConfig


@pytest.fixture
def config() -> ComplianceConfig:
    """Default compliance config for testing."""
    return ComplianceConfig(
        weights={"presence": 0.5, "facings": 0.3, "correctness": 0.2},
        thresholds={"min_confidence": 0.6, "position_tolerance": 0.1, "iou_threshold": 0.5},
    )


def test_compute_compliance_score_all_correct(config: ComplianceConfig) -> None:
    """Score should be 100 when all slots are correct."""
    results = [
        SlotResult("S1", "SKU1", "SKU1", "correct", 0.9, True),
        SlotResult("S2", "SKU2", "SKU2", "correct", 0.85, True),
        SlotResult("S3", "SKU3", "SKU3", "correct", 0.88, True),
    ]
    score = compute_compliance_score(results, config)
    assert score == 100.0


def test_compute_compliance_score_all_missing(config: ComplianceConfig) -> None:
    """Score should be 0 when all slots are missing."""
    results = [
        SlotResult("S1", "SKU1", None, "missing", 0.0, False),
        SlotResult("S2", "SKU2", None, "missing", 0.0, False),
    ]
    score = compute_compliance_score(results, config)
    assert score == 0.0


def test_compute_compliance_score_mixed(config: ComplianceConfig) -> None:
    """Score should be between 0 and 100 for mixed results."""
    results = [
        SlotResult("S1", "SKU1", "SKU1", "correct", 0.9, True),
        SlotResult("S2", "SKU2", None, "missing", 0.0, False),
        SlotResult("S3", "SKU3", "SKU4", "wrong_product", 0.7, True),
    ]
    score = compute_compliance_score(results, config)
    assert 0 < score < 100


def test_compute_compliance_score_empty(config: ComplianceConfig) -> None:
    """Score should be 0 for empty results."""
    score = compute_compliance_score([], config)
    assert score == 0.0


def test_compliance_engine_evaluate(
    config: ComplianceConfig,
    sample_detections: list[dict],
    sample_matches: list[dict],
    sample_reference_slots: list[dict],
) -> None:
    """Test ComplianceEngine.evaluate produces valid report."""
    engine = ComplianceEngine(config)
    report = engine.evaluate(
        detections=sample_detections,
        matches=sample_matches,
        reference_slots=sample_reference_slots,
    )

    assert report.total_slots == len(sample_reference_slots)
    assert 0 <= report.compliance_score <= 100
    assert len(report.slot_results) == len(sample_reference_slots)
    assert report.correct_count + report.missing_count + report.wrong_count == report.total_slots


def test_slot_result_status_values() -> None:
    """Test SlotResult accepts only valid status values."""
    valid_statuses = ["correct", "missing", "wrong_product", "misplaced"]
    for status in valid_statuses:
        result = SlotResult("S1", "SKU1", "SKU1" if status != "missing" else None, status, 0.8, True)
        assert result.status == status
