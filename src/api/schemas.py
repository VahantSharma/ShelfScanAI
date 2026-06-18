"""
Pydantic request/response models for the ShelfScan API.

All validation uses Pydantic v2 for type safety and serialization.

Usage:
    from src.api.schemas import AnalyzeResponse

    response = AnalyzeResponse(
        compliance_score=85.0,
        slot_results=[...],
        missing_skus=["SKU_003"],
        wrong_skus=[],
        annotated_image_b64="...",
        inference_time_ms=123.4,
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SlotResult(BaseModel):
    """Result of evaluating a single planogram slot."""

    slot_id: str = Field(..., description="Unique slot identifier")
    expected_sku: str = Field(..., description="SKU expected in this slot")
    detected_sku: str | None = Field(None, description="SKU actually detected")
    status: str = Field(
        ...,
        description="Evaluation result: correct, missing, wrong_product, misplaced",
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Detection/matching confidence"
    )


class AnalyzeResponse(BaseModel):
    """Response from the /analyze endpoint."""

    compliance_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall compliance score (0-100)"
    )
    slot_results: list[SlotResult] = Field(
        ..., description="Per-slot evaluation results"
    )
    missing_skus: list[str] = Field(
        default_factory=list, description="List of missing SKU IDs"
    )
    wrong_skus: list[str] = Field(
        default_factory=list, description="List of slot IDs with wrong products"
    )
    annotated_image_b64: str = Field(
        "", description="Base64-encoded annotated overlay image"
    )
    inference_time_ms: float = Field(
        0.0, description="Total inference time in milliseconds"
    )
    alignment_method: str = Field(
        "none", description="Alignment method used (orb/sift/none)"
    )
    alignment_quality: float | None = Field(
        None, description="Homography inlier ratio (None if no alignment)"
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field("healthy", description="Service status")
    models_loaded: bool = Field(False, description="Whether models are loaded")
    version: str = Field("0.1.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error description")
    error_code: str = Field("UNKNOWN", description="Machine-readable error code")
