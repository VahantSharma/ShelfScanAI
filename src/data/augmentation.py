"""
Online augmentation pipeline for retail shelf images.

Augmentations are applied ON-THE-FLY during training inside the Dataset/DataLoader.
No augmented images are ever written to disk. This approach:
  - Saves disk space
  - Is reproducible (seed-based)
  - Allows changing augmentation params without reprocessing

Usage:
    transform = build_field_augmentation(severity="medium")
    augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
"""

from __future__ import annotations

import logging
from typing import Any

import albumentations as A
import cv2

logger = logging.getLogger(__name__)


def build_field_augmentation(severity: str = "medium") -> A.Compose:
    """Build augmentation pipeline simulating field conditions.

    Simulates: phone camera angles, poor lighting, motion blur,
    JPEG compression artifacts, partial occlusion.

    Args:
        severity: One of "light", "medium", "heavy".

    Returns:
        Albumentations Compose pipeline with bbox_params configured for YOLO.

    Raises:
        ValueError: If severity is not one of the allowed values.
    """
    allowed = ("light", "medium", "heavy")
    if severity not in allowed:
        raise ValueError(f"severity must be one of {allowed}, got '{severity}'")

    presets = {
        "light": A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.ImageCompression(quality_lower=50, quality_upper=90, p=0.3),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
        "medium": A.Compose(
            [
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=10, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.GaussNoise(p=0.3),
                A.ImageCompression(quality_lower=40, quality_upper=80, p=0.4),
                A.Perspective(scale=(0.03, 0.1), p=0.3),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
        "heavy": A.Compose(
            [
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.Perspective(scale=(0.05, 0.15), p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
                A.HueSaturationValue(p=0.3),
                A.GaussianBlur(blur_limit=(5, 9), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.4),
                A.GaussNoise(p=0.4),
                A.ImageCompression(quality_lower=25, quality_upper=70, p=0.5),
                A.CoarseDropout(max_holes=6, max_height=24, max_width=24, p=0.3),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
    }

    transform = presets[severity]
    logger.info("Built field augmentation pipeline: severity=%s", severity)
    return transform


def build_inference_preprocessing(imgsz: int = 640) -> A.Compose:
    """Build minimal preprocessing for inference (no augmentation).

    Resizes and normalizes image for model input.

    Args:
        imgsz: Target image size (square).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=imgsz, width=imgsz),
        ]
    )


def build_evaluation_augmentation() -> A.Compose:
    """Build augmentation pipeline for evaluation stress testing.

    Applies heavy augmentation to test robustness. Used in error analysis,
    NOT during training.

    Returns:
        Albumentations Compose pipeline with bbox_params.
    """
    return A.Compose(
        [
            A.Rotate(limit=20, p=0.7),
            A.Perspective(scale=(0.08, 0.2), p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.GaussianBlur(blur_limit=(5, 11), p=0.6),
            A.MotionBlur(blur_limit=9, p=0.5),
            A.GaussNoise(p=0.5),
            A.ImageCompression(quality_lower=20, quality_upper=60, p=0.6),
            A.CoarseDropout(max_holes=10, max_height=32, max_width=32, p=0.4),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
