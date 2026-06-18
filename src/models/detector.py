"""
YOLOv8 ONNX inference wrapper.

Provides a lightweight inference interface using ONNX Runtime.
Model is loaded once at startup, not per-request.

Usage:
    detector = YOLODetector("artifacts/models/yolov8m.onnx")
    detections = detector.detect(image)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLOv8 detector with ONNX Runtime backend.

    Loads model once at initialization. All subsequent detect() calls
    reuse the loaded session for fast inference.

    Attributes:
        session: ONNX Runtime inference session.
        input_name: Name of the model's input tensor.
        input_shape: Expected input shape [batch, channels, height, width].
        conf_threshold: Minimum confidence threshold for detections.
        iou_threshold: IoU threshold for Non-Maximum Suppression.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        """Initialize detector with ONNX model.

        Args:
            onnx_path: Path to ONNX model file.
            conf_threshold: Minimum confidence for detections.
            iou_threshold: NMS IoU threshold.
            imgsz: Model input size.
        """
        self.onnx_path = Path(onnx_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        logger.info(
            "YOLODetector loaded: %s (input=%s, shape=%s)",
            self.onnx_path.name,
            self.input_name,
            self.input_shape,
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Args:
            image: Input image in BGR format (OpenCV default).

        Returns:
            Preprocessed tensor [1, 3, H, W] in float32, normalized to [0, 1].
        """
        # Resize
        resized = cv2.resize(image, (self.imgsz, self.imgsz))

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # HWC to CHW, normalize to [0, 1]
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(tensor, axis=0)

    def postprocess(
        self,
        output: np.ndarray,
        orig_shape: tuple[int, int],
    ) -> list[dict[str, float | list[float]]]:
        """Postprocess model output to detection list.

        Args:
            output: Raw model output [1, 84, N] for YOLOv8 (4 bbox + 80 classes).
            orig_shape: Original image shape (height, width).

        Returns:
            List of detection dicts with keys:
                bbox: [x1, y1, x2, y2] in pixel coords
                confidence: Detection confidence
                class_id: Detected class index
        """
        # YOLOv8 output: [1, 84, N] where 84 = 4 (x,y,w,h) + 80 (class scores)
        # Transpose to [N, 84]
        preds = output[0].T  # [N, 84]

        # Extract boxes and scores
        boxes = preds[:, :4]  # [N, 4] - cx, cy, w, h (normalized)
        scores = preds[:, 4:]  # [N, 80] - class scores

        # Get best class per detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2) in original image coords
        img_h, img_w = orig_shape
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * img_h

        # Clip to image bounds
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)

        # Apply NMS
        dets = np.column_stack([x1, y1, x2, y2, confidences])
        indices = cv2.dnn.NMSBoxes(
            [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
             for i in range(len(dets))],
            [float(c) for c in confidences],
            self.conf_threshold,
            self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(
                    {
                        "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                        "confidence": float(confidences[i]),
                        "class_id": int(class_ids[i]),
                    }
                )

        return detections

    def detect(self, image: np.ndarray) -> list[dict[str, float | list[float]]]:
        """Run detection on a single image.

        Args:
            image: Input image in BGR format (OpenCV default).

        Returns:
            List of detection dicts with bbox, confidence, class_id.
        """
        orig_shape = image.shape[:2]
        tensor = self.preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self.postprocess(output, orig_shape)

    def detect_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        """Run detection on a batch of images.

        Args:
            images: List of input images in BGR format.

        Returns:
            List of detection lists, one per image.
        """
        return [self.detect(img) for img in images]
