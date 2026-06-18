"""
FastAPI route definitions for ShelfScan API.

Endpoints:
    POST /analyze  - Full planogram compliance analysis
    GET  /health   - Health check with model status
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import AnalyzeResponse, HealthResponse, SlotResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns service status and whether models are loaded.
    """
    from src.api.main import app

    models_loaded = hasattr(app.state, "detector") and app.state.detector is not None
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_shelf(
    reference_planogram: UploadFile = File(..., description="Reference planogram image"),
    field_photo: UploadFile = File(..., description="Field photo to analyze"),
) -> AnalyzeResponse:
    """Full planogram compliance analysis.

    Accepts a reference planogram and field photo, runs the full pipeline:
    1. Decode uploaded images
    2. Detect products (YOLOv8 ONNX)
    3. Match identities (CLIP + FAISS)
    4. Align planograms (ORB/SIFT homography)
    5. Evaluate compliance
    6. Return structured report + annotated image

    Args:
        reference_planogram: Reference planogram image file.
        field_photo: Field photo image file.

    Returns:
        AnalyzeResponse with compliance score, slot results, and annotated image.

    Raises:
        HTTPException: If image processing fails.
    """
    from src.api.main import app

    start_time = time.perf_counter()

    # Validate models are loaded
    if not hasattr(app.state, "detector") or app.state.detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        # Read and decode images
        ref_bytes = await reference_planogram.read()
        fp_bytes = await field_photo.read()

        ref_image = decode_image(ref_bytes)
        fp_image = decode_image(fp_bytes)

        if ref_image is None:
            raise HTTPException(status_code=400, detail="Invalid reference planogram image")
        if fp_image is None:
            raise HTTPException(status_code=400, detail="Invalid field photo image")

        # Run detection
        detections = app.state.detector.detect(fp_image)
        logger.info("Detected %d products", len(detections))

        # Run identity matching (if we have crops and library)
        matches = []
        if hasattr(app.state, "matcher") and app.state.matcher is not None:
            crops = extract_crops_from_detections(fp_image, detections)
            if crops:
                matches = app.state.matcher.match_detections(crops)
                logger.info("Matched %d identities", len(matches))

        # Run alignment
        H = None
        alignment_quality = None
        alignment_method = "none"
        if hasattr(app.state, "aligner") and app.state.aligner is not None:
            H, alignment_quality = app.state.aligner.estimate_homography(ref_image, fp_image)
            alignment_method = type(app.state.aligner).__name__.replace("Aligner", "").lower()
            if H is not None:
                logger.info("Alignment succeeded: method=%s, quality=%.3f", alignment_method, alignment_quality)

        # Build reference slots (simplified: grid-based)
        reference_slots = build_reference_grid(ref_image.shape[:2], n_cols=5, n_rows=3)

        # Run compliance evaluation
        report = app.state.engine.evaluate(
            detections=detections,
            matches=matches,
            reference_slots=reference_slots,
            H=H,
        )

        # Generate annotated overlay
        annotated = app.state.visualizer.render(report, fp_image, reference_slots)
        annotated_b64 = encode_image_base64(annotated)

        # Build response
        slot_results = [
            SlotResult(
                slot_id=r.slot_id,
                expected_sku=r.expected_sku,
                detected_sku=r.detected_sku,
                status=r.status,
                confidence=r.confidence,
            )
            for r in report.slot_results
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return AnalyzeResponse(
            compliance_score=report.compliance_score,
            slot_results=slot_results,
            missing_skus=report.missing_skus,
            wrong_skus=report.wrong_skus,
            annotated_image_b64=annotated_b64,
            inference_time_ms=round(elapsed_ms, 1),
            alignment_method=alignment_method,
            alignment_quality=alignment_quality,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Decode image bytes to OpenCV array.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        BGR image array, or None if decoding fails.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def encode_image_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string.

    Args:
        image: BGR image array.

    Returns:
        Base64-encoded JPEG string.
    """
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def extract_crops_from_detections(
    image: np.ndarray,
    detections: list[dict],
    padding: float = 0.05,
) -> list[np.ndarray]:
    """Extract image crops for detected products.

    Args:
        image: Full image (BGR).
        detections: List of detection dicts with bbox [x1,y1,x2,y2].
        padding: Relative padding around bbox.

    Returns:
        List of cropped images.
    """
    h, w = image.shape[:2]
    crops = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Add padding
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
            crops.append(crop)

    return crops


def build_reference_grid(
    image_shape: tuple[int, int],
    n_cols: int = 5,
    n_rows: int = 3,
) -> list[dict]:
    """Build a grid of reference slots from image dimensions.

    This is a simplified version. In production, slots would come
    from the planogram specification.

    Args:
        image_shape: (height, width) of reference image.
        n_cols: Number of columns in grid.
        n_rows: Number of rows in grid.

    Returns:
        List of slot dicts with slot_id, expected_sku, bbox.
    """
    h, w = image_shape
    slots = []

    for row in range(n_rows):
        for col in range(n_cols):
            slot_id = f"R{row}C{col}"
            cx = (col + 0.5) / n_cols
            cy = (row + 0.5) / n_rows
            nw = 1.0 / n_cols * 0.9
            nh = 1.0 / n_rows * 0.9

            slots.append({
                "slot_id": slot_id,
                "expected_sku": f"SKU_{slot_id}",
                "bbox": [cx, cy, nw, nh],
            })

    return slots
