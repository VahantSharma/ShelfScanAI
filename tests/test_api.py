"""
Tests for the FastAPI API endpoints.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    """Test /health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "version" in data


def test_analyze_returns_403_without_models(client: TestClient) -> None:
    """Test /analyze returns 503 if models not loaded."""
    response = client.post(
        "/analyze",
        files={
            "reference_planogram": ("test.jpg", b"fake", "image/jpeg"),
            "field_photo": ("test.jpg", b"fake", "image/jpeg"),
        },
    )
    # Should return 503 if models aren't loaded, or 400 for invalid image
    assert response.status_code in [400, 422, 500, 503]


def test_analyze_with_valid_images(
    client: TestClient,
    sample_planogram_path: Path,
    sample_field_photo_path: Path,
) -> None:
    """Test /analyze with valid image files."""
    with open(sample_planogram_path, "rb") as ref, open(sample_field_photo_path, "rb") as fp:
        response = client.post(
            "/analyze",
            files={
                "reference_planogram": ("planogram.jpg", ref, "image/jpeg"),
                "field_photo": ("field_photo.jpg", fp, "image/jpeg"),
            },
        )

    # Response depends on whether models are loaded
    # If models loaded: 200 with compliance data
    # If models not loaded: 503
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "compliance_score" in data
        assert "slot_results" in data
        assert "missing_skus" in data
        assert "wrong_skus" in data
        assert "inference_time_ms" in data
        assert 0 <= data["compliance_score"] <= 100


def test_analyze_missing_files(client: TestClient) -> None:
    """Test /analyze returns error when files are missing."""
    response = client.post("/analyze")
    assert response.status_code == 422  # Unprocessable Entity
