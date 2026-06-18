"""
Compliance result visualization.

Generates annotated overlay images showing:
  - Green boxes: correct product in correct slot
  - Red boxes: missing SKU (expected but absent)
  - Yellow boxes: wrong product
  - White dashed outlines: expected but absent
  - Score text overlay

Usage:
    from src.compliance.visualizer import ComplianceVisualizer

    visualizer = ComplianceVisualizer()
    overlay = visualizer.render(report, field_photo, reference_slots)
    visualizer.save(overlay, "output/annotated.jpg")
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.compliance.engine import ComplianceReport

logger = logging.getLogger(__name__)

# Color palette (BGR for OpenCV)
COLORS = {
    "correct": (0, 200, 0),        # Green
    "missing": (0, 0, 255),        # Red
    "wrong_product": (0, 180, 255), # Yellow
    "misplaced": (255, 100, 0),    # Orange
    "slot_outline": (200, 200, 200), # Light gray
    "text_bg": (40, 40, 40),       # Dark gray
    "text_fg": (255, 255, 255),    # White
}


class ComplianceVisualizer:
    """Generates annotated overlay images from compliance reports.

    Attributes:
        font: OpenCV font for text rendering.
        font_scale: Text size scale.
        thickness: Text/border thickness.
    """

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> None:
        """Initialize visualizer.

        Args:
            font: OpenCV font type.
            font_scale: Text size scale.
            thickness: Line/text thickness.
        """
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def render(
        self,
        report: ComplianceReport,
        field_photo: np.ndarray,
        reference_slots: list[dict],
        show_score: bool = True,
    ) -> np.ndarray:
        """Render annotated overlay image.

        Args:
            report: Compliance evaluation report.
            field_photo: Original field photo (BGR).
            reference_slots: List of planogram slot dicts.
            show_score: Whether to overlay compliance score.

        Returns:
            Annotated image (BGR).
        """
        # Create overlay
        overlay = field_photo.copy()
        h, w = overlay.shape[:2]

        # Draw slot outlines and results
        slot_map = {r.slot_id: r for r in report.slot_results}

        for slot in reference_slots:
            slot_id = slot["slot_id"]
            bbox = slot["bbox"]  # [cx, cy, nw, nh] normalized

            # Convert to pixel coords
            cx = bbox[0] * w
            cy = bbox[1] * h
            nw = bbox[2] * w
            nh = bbox[3] * h
            x1 = int(cx - nw / 2)
            y1 = int(cy - nh / 2)
            x2 = int(cx + nw / 2)
            y2 = int(cy + nh / 2)

            # Get result
            result = slot_map.get(slot_id)
            if result is None:
                continue

            color = COLORS.get(result.status, COLORS["slot_outline"])

            if result.status == "missing":
                # Draw dashed outline (white)
                self._draw_dashed_rect(overlay, (x1, y1), (x2, y2), COLORS["slot_outline"], 2)
                # Red X
                cv2.line(overlay, (x1, y1), (x2, y2), COLORS["missing"], 2)
                cv2.line(overlay, (x2, y1), (x1, y2), COLORS["missing"], 2)
            else:
                # Draw solid box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.thickness)

                # Label
                label = result.status.upper()
                if result.detected_sku:
                    label = f"{result.detected_sku}"
                self._draw_label(overlay, label, (x1, y1 - 5), color)

        # Overlay score
        if show_score:
            self._draw_score(overlay, report.compliance_score)

        return overlay

    def _draw_dashed_rect(
        self,
        image: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
    ) -> None:
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Top
        self._draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness, dash_length)
        # Bottom
        self._draw_dashed_line(image, (x1, y2), (x2, y2), color, thickness, dash_length)
        # Left
        self._draw_dashed_line(image, (x1, y1), (x1, y2), color, thickness, dash_length)
        # Right
        self._draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness, dash_length)

    def _draw_dashed_line(
        self,
        image: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash_length: int,
    ) -> None:
        """Draw a dashed line."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        n_dashes = max(1, int(dist / dash_length))

        for i in range(0, n_dashes, 2):
            t1 = i / n_dashes
            t2 = min((i + 1) / n_dashes, 1.0)
            start = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
            end = (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2))
            cv2.line(image, start, end, color, thickness)

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw text label with background."""
        (tw, th), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        x, y = pos
        y = max(th + 5, y)

        # Background
        cv2.rectangle(image, (x, y - th - 5), (x + tw + 5, y + 5), color, -1)
        # Text
        cv2.putText(image, text, (x + 2, y - 2), self.font, self.font_scale, (255, 255, 255), self.thickness)

    def _draw_score(self, image: np.ndarray, score: float) -> None:
        """Draw compliance score overlay in top-left corner."""
        h, w = image.shape[:2]
        text = f"Compliance: {score:.1f}%"

        # Background rectangle
        (tw, th), _ = cv2.getTextSize(text, self.font, 1.0, 3)
        cv2.rectangle(image, (10, 10), (tw + 20, th + 30), COLORS["text_bg"], -1)

        # Score text
        cv2.putText(
            image, text, (15, th + 20), self.font, 1.0, COLORS["text_fg"], 3
        )

    def save(self, image: np.ndarray, path: str | Path) -> None:
        """Save annotated image to disk.

        Args:
            image: Annotated image (BGR).
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        logger.info("Saved annotated image to %s", path)
