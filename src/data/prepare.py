"""
SKU110K dataset preparation pipeline.

Validates a YOLO-formatted dataset, logs statistics, and creates
an Ultralytics-compatible dataset YAML.

Designed for the Kaggle SKU110K dataset which already contains
images/ and labels/ in YOLO format (normalized xywh, class 0).

Usage:
    python -m src.data.prepare --config configs/data.yaml
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import yaml

from src.config import DataConfig

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]


def validate_raw_directory(raw_dir: Path) -> bool:
    """Validate that raw SKU110K directory has expected YOLO structure.

    Expected:
        raw_dir/
        ├── images/{train,val,test}/
        └── labels/{train,val,test}/

    Returns:
        True if structure is valid, False otherwise.
    """
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    if not images_dir.exists():
        logger.error("Missing images directory: %s", images_dir)
        return False
    if not labels_dir.exists():
        logger.error("Missing labels directory: %s", labels_dir)
        return False

    for split in SPLITS:
        split_img_dir = images_dir / split
        split_lbl_dir = labels_dir / split

        if not split_img_dir.exists():
            logger.error("Missing image split directory: %s", split_img_dir)
            return False
        if not split_lbl_dir.exists():
            logger.error("Missing label split directory: %s", split_lbl_dir)
            return False

    logger.info("Raw directory structure validated: %s", raw_dir)
    return True


def validate_label_format(label_path: Path) -> bool:
    """Validate a single YOLO label file.

    Checks:
        - Each line has exactly 5 space-separated fields
        - Class ID is 0
        - x_center, y_center, width, height are all between 0 and 1

    Returns:
        True if format is valid, False otherwise.
    """
    try:
        with open(label_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    logger.error(
                        "Invalid label format at %s line %d: expected 5 fields, got %d",
                        label_path.name,
                        line_num,
                        len(parts),
                    )
                    return False

                cls = int(parts[0])
                if cls != 0:
                    logger.error(
                        "Unexpected class ID %d at %s line %d (expected 0)",
                        cls,
                        label_path.name,
                        line_num,
                    )
                    return False

                for i, val_str in enumerate(parts[1:], 1):
                    val = float(val_str)
                    if not (0.0 <= val <= 1.0):
                        logger.error(
                            "Value %f out of range [0,1] at %s line %d field %d",
                            val,
                            label_path.name,
                            line_num,
                            i + 1,
                        )
                        return False
        return True
    except Exception as e:
        logger.error("Failed to validate %s: %s", label_path, e)
        return False


def log_dataset_statistics(raw_dir: Path) -> dict[str, dict]:
    """Compute and log dataset statistics per split.

    Returns:
        Dict with per-split stats: image_count, label_count, total_bboxes,
        min_objects, max_objects, mean_objects.
    """
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    stats = {}
    for split in SPLITS:
        img_dir = images_dir / split
        lbl_dir = labels_dir / split

        image_count = len(list(img_dir.glob("*.jpg")))
        label_count = len(list(lbl_dir.glob("*.txt")))

        # Count bboxes per image
        bbox_counts = []
        for label_file in lbl_dir.glob("*.txt"):
            with open(label_file) as f:
                count = sum(1 for line in f if line.strip())
            bbox_counts.append(count)

        total_bboxes = sum(bbox_counts)
        min_obj = min(bbox_counts) if bbox_counts else 0
        max_obj = max(bbox_counts) if bbox_counts else 0
        mean_obj = total_bboxes / len(bbox_counts) if bbox_counts else 0

        stats[split] = {
            "image_count": image_count,
            "label_count": label_count,
            "total_bboxes": total_bboxes,
            "min_objects": min_obj,
            "max_objects": max_obj,
            "mean_objects": round(mean_obj, 1),
        }

        logger.info(
            "Split %s: %d images, %d labels, %d bboxes "
            "(min=%d, max=%d, mean=%.1f per image)",
            split,
            image_count,
            label_count,
            total_bboxes,
            min_obj,
            max_obj,
            mean_obj,
        )

    # Log totals
    total_images = sum(s["image_count"] for s in stats.values())
    total_bboxes = sum(s["total_bboxes"] for s in stats.values())
    logger.info(
        "Dataset total: %d images, %d bboxes",
        total_images,
        total_bboxes,
    )

    return stats


def verify_image_label_pairs(raw_dir: Path) -> int:
    """Verify that every image has a corresponding label and vice versa.

    Returns:
        Number of mismatches found (0 = perfect).
    """
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    mismatches = 0

    for split in SPLITS:
        img_stems = {p.stem for p in (images_dir / split).glob("*.jpg")}
        lbl_stems = {p.stem for p in (labels_dir / split).glob("*.txt")}

        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems

        if missing_labels:
            logger.warning(
                "Split %s: %d images without labels (first 5: %s)",
                split,
                len(missing_labels),
                list(missing_labels)[:5],
            )
            mismatches += len(missing_labels)

        if missing_images:
            logger.warning(
                "Split %s: %d labels without images (first 5: %s)",
                split,
                len(missing_images),
                list(missing_images)[:5],
            )
            mismatches += len(missing_images)

        if not missing_labels and not missing_images:
            logger.info("Split %s: all %d images have matching labels", split, len(img_stems))

    return mismatches


def validate_sample_labels(raw_dir: Path, n_samples: int = 20) -> bool:
    """Validate a random sample of label files for format correctness.

    Args:
        raw_dir: Path to raw dataset directory.
        n_samples: Number of label files to validate.

    Returns:
        True if all samples are valid, False otherwise.
    """
    labels_dir = raw_dir / "labels"
    all_label_files = list(labels_dir.rglob("*.txt"))

    if not all_label_files:
        logger.error("No label files found")
        return False

    sample = random.sample(all_label_files, min(n_samples, len(all_label_files)))
    invalid = 0

    for label_path in sample:
        if not validate_label_format(label_path):
            invalid += 1

    if invalid > 0:
        logger.error("Format validation failed: %d/%d samples invalid", invalid, len(sample))
        return False

    logger.info("Format validation passed: %d/%d samples valid", len(sample) - invalid, len(sample))
    return True


def create_dataset_yaml(config: DataConfig) -> Path:
    """Create Ultralytics-compatible dataset YAML.

    The YAML points to the raw directory (which already contains
    images/ and labels/ in YOLO format).

    Returns:
        Path to created YAML file.
    """
    raw_dir = config.raw_dir
    yaml_path = raw_dir / "dataset.yaml"

    content = f"""# SKU110K dataset - auto-generated by src/data/prepare.py
# Single class: "object" (all retail items)
# Labels already in YOLO format (normalized xywh, class 0)
path: {raw_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ["object"]
"""
    yaml_path.write_text(content)
    logger.info("Created dataset YAML: %s", yaml_path)
    return yaml_path


def prepare_dataset(config: DataConfig) -> dict[str, dict]:
    """Run full preparation pipeline.

    Validates directory structure, verifies label format,
    logs statistics, and creates dataset.yaml.

    Returns:
        Per-split statistics.
    """
    raw_dir = config.raw_dir

    # Step 1: Validate directory structure
    if not validate_raw_directory(raw_dir):
        raise FileNotFoundError(
            f"SKU110K directory structure invalid: {raw_dir}\n"
            "Expected: {raw_dir}/images/{{train,val,test}}/ and "
            "{raw_dir}/labels/{{train,val,test}}/"
        )

    # Step 2: Verify image-label pairs
    mismatches = verify_image_label_pairs(raw_dir)
    if mismatches > 0:
        logger.warning("Found %d image-label mismatches (proceeding anyway)", mismatches)

    # Step 3: Validate label format (sample)
    if not validate_sample_labels(raw_dir, n_samples=20):
        logger.warning("Some labels failed format validation (proceeding anyway)")

    # Step 4: Log statistics
    stats = log_dataset_statistics(raw_dir)

    # Step 5: Create dataset.yaml
    create_dataset_yaml(config)

    return stats


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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = DataConfig.from_yaml(args.config)
    prepare_dataset(cfg)
