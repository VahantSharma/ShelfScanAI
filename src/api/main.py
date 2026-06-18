"""
FastAPI application with lifespan management.

Models are loaded once at startup, not per-request.
All configuration loaded from configs/*.yaml.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

    # Or with Docker:
    docker-compose up
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models at startup, cleanup at shutdown."""
    logger.info("Starting ShelfScan API...")

    try:
        # Load detector (ONNX)
        from src.models.detector import YOLODetector

        onnx_path = Path("artifacts/models/yolov8m.onnx")
        if onnx_path.exists():
            app.state.detector = YOLODetector(onnx_path)
            logger.info("YOLOv8 ONNX detector loaded")
        else:
            logger.warning("ONNX model not found: %s (detector disabled)", onnx_path)
            app.state.detector = None

        # Load embedder + matcher (CLIP + FAISS)
        from src.models.embedder import SKUMatcher

        library_path = Path("data/reference_library")
        index_path = library_path / "index.faiss"
        if index_path.exists():
            app.state.matcher = SKUMatcher(library_path)
            logger.info("SKU matcher loaded from %s", library_path)
        else:
            logger.warning("Reference library not found: %s (matching disabled)", library_path)
            app.state.matcher = None

        # Load aligner
        from src.models.aligner import AlignerFactory

        app.state.aligner = AlignerFactory.create("orb")
        logger.info("ORB aligner loaded")

        # Load compliance engine
        from src.compliance.engine import ComplianceEngine
        from src.compliance.visualizer import ComplianceVisualizer
        from src.config import ComplianceConfig

        config_path = Path("configs/compliance.yaml")
        config = ComplianceConfig.from_yaml(config_path)
        app.state.engine = ComplianceEngine(config)
        app.state.visualizer = ComplianceVisualizer()
        logger.info("Compliance engine loaded")

    except Exception as e:
        logger.error("Failed to load models: %s", e, exc_info=True)
        app.state.detector = None
        app.state.matcher = None
        app.state.aligner = None
        app.state.engine = None
        app.state.visualizer = None

    logger.info("ShelfScan API ready")

    yield

    # Cleanup
    logger.info("Shutting down ShelfScan API...")
    app.state.detector = None
    app.state.matcher = None
    app.state.aligner = None
    app.state.engine = None
    app.state.visualizer = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="ShelfScan API",
        description="Planogram compliance verification from noisy field photos",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Import and include routes
    from src.api.routes import router

    app.include_router(router)

    return app


app = create_app()
