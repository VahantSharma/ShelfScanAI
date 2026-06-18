"""
SKU110K dataset preparation pipeline.

Converts raw SKU110K CSV annotations (absolute pixel coords, single class)
to YOLO format (normalized xywh, class 0). Validates directory structure,
logs statistics, and creates dataset YAML for Ultralytics.

Usage:
    python -m src.data.prepare --config configs/data.yaml
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from src.config import DataConfig

logger = logging.getLogger(__name__)

# SKU110K CSV columns: image_path, x_min, y_min, x_max, y_max, class
EXPECTED_CSV_COLUMNS = ["image_path", "x_min", "y_min", "x_max", "y_max", "class"]

SPLITS = ["train", "val", "test"]


def validate_raw_directory(raw_dir: Path) -> bool:
    """Validate that raw SKU110K directory has expected structure.

    Expected:
        raw_dir/
        ├── images/{train,val,test}/
        └── annotations/{train,val,test}.csv

    Returns:
        True if structure is valid, False otherwise.
    """
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"

    if not images_dir.exists():
        logger.error("Missing images directory: %s", images_dir)
        return False
    if not annotations_dir.exists():
        logger.error("Missing annotations directory: %s", annotations_dir)
        return False

    for split in SPLITS:
        split_img_dir = images_dir / split
        csv_path = annotations_dir / f"{split}.csv"

        if not split_img_dir.exists():
            logger.error("Missing image split directory: %s", split_img_dir)
            return False
        if not csv_path.exists():
            logger.error("Missing annotation CSV: %s", csv_path)
            return False

    logger.info("Raw directory structure validated: %s", raw_dir)
    return True


def parse_sku110k_csv(csv_path: Path) -> list[dict[str, str | float]]:
    """Parse SKU110K CSV annotation file.

    Each row: image_path, x_min, y_min, x_max, y_max, class

    Returns:
        List of annotation dicts.
    """
    annotations = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(
                {
                    "image_path": row["image_path"],
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                }
            )
    return annotations


def convert_csv_to_yolo(
    csv_path: Path, images_dir: Path, output_label_dir: Path
) -> dict[str, int]:
    """Convert SKU110K CSV annotations to YOLO format.

    YOLO format: class x_center y_center width height (all normalized 0-1).

    Returns:
        Dict with statistics: total_images, total_bboxes, empty_labels.
    """
    annotations = parse_sku110k_csv(csv_path)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Group by image
    image_bboxes: dict[str, list[dict]] = {}
    for ann in annotations:
        img_name = Path(ann["image_path"]).name
        if img_name not in image_bboxes:
            image_bboxes[img_name] = []
        image_bboxes[img_name].append(ann)

    stats = {"total_images": 0, "total_bboxes": 0, "empty_labels": 0}

    for img_name, bboxes in image_bboxes.items():
        # Find actual image to get dimensions
        img_path = images_dir / img_name
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            continue

        # Use PIL to get image dimensions
        from PIL import Image

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        label_path = output_label_dir / (Path(img_name).stem + ".txt")
        lines = []
        for bbox in bboxes:
            # Convert absolute pixel coords to normalized xywh
            x_min, y_min = bbox["x_min"], bbox["y_min"]
            x_max, y_max = bbox["x_max"], bbox["y_max"]

            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            # Class 0 for all objects (SKU110K single-class)
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        stats["total_images"] += 1
        stats["total_bboxes"] += len(bboxes)
        if not lines:
            stats["empty_labels"] += 1

    return stats


def copy_images_to_processed(raw_images_dir: Path, processed_images_dir: Path) -> int:
    """Copy (or symlink) images from raw to processed directory.

    Returns:
        Number of images copied.
    """
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in raw_images_dir.glob("*.jpg"):
        dest = processed_images_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        count += 1
    return count


def prepare_dataset(config: DataConfig) -> dict[str, dict[str, int]]:
    """Run full preparation pipeline.

    Converts CSV annotations to YOLO format, copies images, logs statistics.

    Returns:
        Per-split statistics.
    """
    raw_dir = config.raw_dir
    processed_dir = config.processed_dir

    if not validate_raw_directory(raw_dir):
        raise FileNotFoundError(
            f"SKU110K raw directory structure invalid: {raw_dir}\n"
            "Expected: {raw_dir}/images/{{train,val,test}}/ and "
            "{raw_dir}/annotations/{{train,val,test}}.csv"
        )

    all_stats = {}
    for split in SPLITS:
        logger.info("Processing split: %s", split)

        csv_path = raw_dir / "annotations" / f"{split}.csv"
        raw_images_dir = raw_dir / "images" / split
        processed_images_dir = processed_dir / "images" / split
        processed_labels_dir = processed_dir / "labels" / split

        # Convert annotations
        stats = convert_csv_to_yolo(csv_path, raw_images_dir, processed_labels_dir)

        # Copy images
        n_copied = copy_images_to_processed(raw_images_dir, processed_images_dir)
        stats["images_copied"] = n_copied

        all_stats[split] = stats
        logger.info(
            "Split %s: %d images, %d bboxes, %d empty",
            split,
            stats["total_images"],
            stats["total_bboxes"],
            stats["empty_labels"],
        )

    # Log summary
    total_images = sum(s["total_images"] for s in all_stats.values())
    total_bboxes = sum(s["total_bboxes"] for s in all_stats.values())
    logger.info("Dataset preparation complete: %d images, %d bboxes", total_images, total_bboxes)

    return all_stats


def create_dataset_yaml(config: DataConfig) -> Path:
    """Create Ultralytics-compatible dataset YAML.

    Returns:
        Path to created YAML file.
    """
    processed_dir = config.processed_dir
    yaml_path = processed_dir / "dataset.yaml"

    content = f"""# SKU110K dataset - auto-generated by src/data/prepare.py
# Single class: "object" (all retail items)
path: {processed_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ["object"]
"""
    yaml_path.write_text(content)
    logger.info("Created dataset YAML: %s", yaml_path)
    return yaml_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SKU110K dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = DataConfig.from_yaml(args.config)
    prepare_dataset(cfg)
    create_dataset_yaml(cfg)
