"""
PyTorch Dataset for SKU110K with online augmentation.

Reads YOLO-format labels and applies augmentations ON-THE-FLY.
No augmented images are ever written to disk.

Usage:
    dataset = SKU110KDataset(
        image_dir=Path("data/processed/SKU110K/images/train"),
        label_dir=Path("data/processed/SKU110K/labels/train"),
        transform=build_field_augmentation("medium"),
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentation import build_field_augmentation

logger = logging.getLogger(__name__)


class SKU110KDataset(Dataset):
    """SKU110K dataset with online augmentation.

    Reads YOLO-format label files and applies augmentation transforms
    on-the-fly during training. Supports optional augmentation for
    training and clean loading for evaluation.

    Attributes:
        image_paths: Sorted list of image file paths.
        label_dir: Directory containing YOLO-format label files.
        transform: Albumentations transform pipeline (applied on-the-fly).
    """

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        transform: "A.Compose | None" = None,
    ) -> None:
        """Initialize dataset.

        Args:
            image_dir: Directory containing .jpg images.
            label_dir: Directory containing .txt YOLO label files.
            transform: Albumentations pipeline with bbox_params.
                If None, no augmentation is applied.
        """
        self.image_paths = sorted(image_dir.glob("*.jpg"))
        self.label_dir = label_dir
        self.transform = transform

        # Filter to images that have corresponding labels
        self.image_paths = [
            p for p in self.image_paths if (label_dir / (p.stem + ".txt")).exists()
        ]

        logger.info(
            "Dataset initialized: %d images in %s",
            len(self.image_paths),
            image_dir,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_labels(self, idx: int) -> tuple[list[list[float]], list[int]]:
        """Load YOLO-format labels for an image.

        Returns:
            Tuple of (bboxes in YOLO format, class labels).
        """
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        bboxes = []
        class_labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        bboxes.append([x_center, y_center, w, h])
                        class_labels.append(cls)

        return bboxes, class_labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load image and labels, apply augmentation if configured.

        Returns:
            Tuple of:
                - image: Tensor [3, H, W], float32, normalized to [0, 1]
                - bboxes: Tensor [N, 4], YOLO format (x_center, y_center, w, h)
                - class_labels: Tensor [N], integer class labels
        """
        img_path = self.image_paths[idx]

        # Load image (OpenCV for augmentation compatibility)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        bboxes, class_labels = self._load_labels(idx)

        # Apply augmentation (on-the-fly, never saved to disk)
        if self.transform is not None and bboxes:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
                image = transformed["image"]
                bboxes = transformed["bboxes"]
                class_labels = transformed["class_labels"]
            except Exception as e:
                logger.warning("Augmentation failed for %s: %s", img_path.name, e)
                # Fall back to no augmentation

        # Convert to tensors
        # Image: HWC -> CHW, uint8 -> float32 [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Bounding boxes
        if bboxes:
            bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
            label_tensor = torch.tensor(class_labels, dtype=torch.long)
        else:
            bbox_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.long)

        return image_tensor, bbox_tensor, label_tensor


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Custom collate function for variable-length bounding box tensors.

    Images are stacked into a batch tensor. Bboxes and labels are
    returned as lists of tensors (variable N per image).
    """
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(bboxes), list(labels)
