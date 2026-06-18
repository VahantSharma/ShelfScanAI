"""
Build reference library from detected product crops.

Extracts crops from SKU110K test set using the trained detector,
embeds each with CLIP, and stores in a FAISS index for identity matching.

IMPORTANT: Reference library is built from exemplar crops, not true
product catalog identities. In a production system, these would come
from the brand's official product database. For this portfolio project,
we use detected crops as representative exemplars.

Usage:
    python scripts/build_reference_library.py \
        --model artifacts/models/yolov8m.onnx \
        --data configs/data.yaml \
        --output data/reference_library \
        --target-skus 80
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import DataConfig
from src.models.detector import YOLODetector
from src.models.embedder import SKUEmbedder

logger = logging.getLogger(__name__)


def extract_crops(
    detector: YOLODetector,
    image_dir: Path,
    label_dir: Path,
    n_skus: int = 80,
    min_area: float = 0.003,
    max_area: float = 0.05,
    min_confidence: float = 0.7,
    seed: int = 42,
) -> list[dict]:
    """Extract diverse product crops from test images.

    Selects crops that represent distinct visual appearances
    (different packaging, brands, sizes) by sampling across
    complexity quartiles.

    Args:
        detector: YOLO detector for finding products.
        image_dir: Directory with test images.
        label_dir: Directory with ground truth labels.
        n_skus: Target number of unique SKU crops.
        min_area: Minimum relative bbox area (filters tiny objects).
        max_area: Maximum relative bbox area (filters shelf-wide regions).
        min_confidence: Minimum detection confidence.
        seed: Random seed for reproducibility.

    Returns:
        List of crop dicts with image, bbox, source info.
    """
    random.seed(seed)
    np.random.seed(seed)

    image_paths = sorted(image_dir.glob("*.jpg"))
    all_crops = []

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        # Load ground truth for reference
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append({
                            "x_center": float(parts[1]),
                            "y_center": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                        })

        # Use ground truth boxes for crop extraction (more reliable than detection)
        for box in gt_boxes:
            area = box["w"] * box["h"]
            if area < min_area or area > max_area:
                continue

            # Convert to pixel coords
            x1 = int((box["x_center"] - box["w"] / 2) * img_w)
            y1 = int((box["y_center"] - box["h"] / 2) * img_h)
            x2 = int((box["x_center"] + box["w"] / 2) * img_w)
            y2 = int((box["y_center"] + box["h"] / 2) * img_h)

            # Clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            all_crops.append({
                "crop": crop,
                "source_image": img_path.name,
                "bbox": [box["x_center"], box["y_center"], box["w"], box["h"]],
                "area": area,
                "aspect_ratio": (x2 - x1) / max(1, y2 - y1),
            })

    logger.info("Extracted %d candidate crops from %d images", len(all_crops), len(image_paths))

    # Diversify: sample across area ranges to get variety
    if len(all_crops) > n_skus:
        # Sort by area for diverse sampling
        all_crops.sort(key=lambda c: c["area"])
        # Sample evenly across the range
        indices = np.linspace(0, len(all_crops) - 1, n_skus, dtype=int)
        selected = [all_crops[i] for i in indices]
    else:
        selected = all_crops

    logger.info("Selected %d crops for reference library", len(selected))
    return selected


def build_library(
    crops: list[dict],
    embedder: SKUEmbedder,
    output_dir: Path,
) -> None:
    """Build FAISS index from extracted crops.

    Args:
        crops: List of crop dicts.
        embedder: CLIP embedder.
        output_dir: Output directory for library files.
    """
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)

    # Embed all crops
    pil_images = [Image.fromarray(cv2.cvtColor(c["crop"], cv2.COLOR_BGR2RGB)) for c in crops]
    embeddings = embedder.embed_batch(pil_images).astype(np.float32)

    # Build FAISS index (Inner Product = cosine similarity for normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(output_dir / "index.faiss"))

    # Build manifest
    manifest = []
    for i, crop in enumerate(crops):
        manifest.append({
            "sku_id": f"SKU_{i:03d}",
            "crop_path": f"crops/sku_{i:03d}.jpg",
            "source_image": crop["source_image"],
            "bbox": crop["bbox"],
            "area": crop["area"],
            "embedding_idx": i,
            "note": "Exemplar crop, not true product catalog identity",
        })

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save crop images for visualization
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    for i, crop in enumerate(crops):
        cv2.imwrite(str(crops_dir / f"sku_{i:03d}.jpg"), crop["crop"])

    # Save raw embeddings for debugging
    np.save(str(output_dir / "embeddings.npy"), embeddings)

    logger.info(
        "Library built: %d SKUs, %d-dimensional embeddings, saved to %s",
        len(crops),
        dimension,
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reference SKU library")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained ONNX model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reference_library"),
        help="Output directory for library",
    )
    parser.add_argument(
        "--target-skus",
        type=int,
        default=80,
        help="Target number of SKU crops",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (auto-detected if not specified)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    data_cfg = DataConfig.from_yaml(args.data)

    # Initialize detector and embedder
    detector = YOLODetector(args.model)
    embedder = SKUEmbedder(device=args.device)

    # Extract crops from test set
    test_image_dir = data_cfg.processed_dir / "images" / "test"
    test_label_dir = data_cfg.processed_dir / "labels" / "test"

    crops = extract_crops(
        detector=detector,
        image_dir=test_image_dir,
        label_dir=test_label_dir,
        n_skus=args.target_skus,
    )

    # Build library
    build_library(crops=crops, embedder=embedder, output_dir=args.output)
