"""
Shared test fixtures for ShelfScan test suite.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client."""
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image (100x100 RGB)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_planogram_path(tmp_path: Path) -> Path:
    """Create a temporary planogram image file."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    path = tmp_path / "planogram.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_field_photo_path(tmp_path: Path) -> Path:
    """Create a temporary field photo image file."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    path = tmp_path / "field_photo.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_detections() -> list[dict]:
    """Sample detection results from YOLO."""
    return [
        {"bbox": [50, 50, 150, 150], "confidence": 0.9, "class_id": 0},
        {"bbox": [150, 50, 250, 150], "confidence": 0.85, "class_id": 0},
    ]


@pytest.fixture
def sample_matches() -> list[dict]:
    """Sample SKU identity matches."""
    return [
        {"crop_idx": 0, "matched_sku": "SKU_001", "confidence": 0.87, "top_k": [("SKU_001", 0.87)]},
        {"crop_idx": 1, "matched_sku": "SKU_002", "confidence": 0.72, "top_k": [("SKU_002", 0.72)]},
    ]


@pytest.fixture
def sample_reference_slots() -> list[dict]:
    """Sample reference planogram slots."""
    return [
        {"slot_id": "R0C0", "expected_sku": "SKU_001", "bbox": [0.1, 0.25, 0.18, 0.3]},
        {"slot_id": "R0C1", "expected_sku": "SKU_002", "bbox": [0.3, 0.25, 0.18, 0.3]},
        {"slot_id": "R0C2", "expected_sku": "SKU_003", "bbox": [0.5, 0.25, 0.18, 0.3]},
    ]
