"""
YOLOv8 evaluation and error analysis script.

Runs inference on test set, computes metrics, and generates
categorized error analysis for understanding failure modes.

Usage:
    python training/eval.py --model artifacts/models/best.pt --data configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import wandb

from src.config import DataConfig

logger = logging.getLogger(__name__)

# Error categories for failure analysis
ERROR_CATEGORIES = {
    "missed_detection": "False Negative - product present but not detected",
    "false_positive": "False Positive - detected but no product there",
    "localization_error": "Localization - detected but IoU < threshold",
}


def evaluate_model(
    model_path: Path,
    dataset_yaml: Path,
    split: str = "test",
) -> dict[str, float]:
    """Run evaluation on test set.

    Args:
        model_path: Path to trained model (.pt or .onnx).
        dataset_yaml: Path to Ultralytics dataset YAML.
        split: Dataset split to evaluate on.

    Returns:
        Dict with evaluation metrics.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(data=str(dataset_yaml), split=split, verbose=True)

    results = {
        "mAP50": float(metrics.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(metrics.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(metrics.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(metrics.results_dict.get("metrics/recall(B)", 0)),
        "fitness": float(metrics.fitness),
    }

    logger.info(
        "Evaluation on %s: mAP50=%.4f, mAP50-95=%.4f, P=%.4f, R=%.4f",
        split,
        results["mAP50"],
        results["mAP50-95"],
        results["precision"],
        results["recall"],
    )

    return results


def analyze_errors(
    model_path: Path,
    test_image_dir: Path,
    test_label_dir: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
) -> dict:
    """Perform error analysis on test set.

    Categorizes failures into:
      1. Missed detections (FN) - especially small/occluded objects
      2. False positives (FP) - shelf edges, packaging confusion
      3. Localization errors - IoU < threshold

    Args:
        model_path: Path to trained model.
        test_image_dir: Directory with test images.
        test_label_dir: Directory with ground truth labels.
        output_path: Path to save error report JSON.
        iou_threshold: Minimum IoU for correct localization.
        conf_threshold: Confidence threshold for detections.

    Returns:
        Error analysis report dict.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    report = {
        "summary": {},
        "categories": {cat: [] for cat in ERROR_CATEGORIES},
        "per_image": [],
    }

    image_paths = sorted(test_image_dir.glob("*.jpg"))
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in image_paths:
        # Load ground truth
        label_path = test_label_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append(
                            {
                                "class": int(parts[0]),
                                "x_center": float(parts[1]),
                                "y_center": float(parts[2]),
                                "w": float(parts[3]),
                                "h": float(parts[4]),
                            }
                        )

        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Convert to YOLO format (normalized xywh)
                img_w, img_h = r.orig_shape[1], r.orig_shape[0]
                x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
                y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
                w = (xyxy[2] - xyxy[0]) / img_w
                h = (xyxy[3] - xyxy[1]) / img_h
                pred_boxes.append(
                    {
                        "class": cls,
                        "x_center": x_center,
                        "y_center": y_center,
                        "w": w,
                        "h": h,
                        "confidence": conf,
                    }
                )

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        image_errors = []

        for pi, pred in enumerate(pred_boxes):
            best_iou = 0.0
            best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                total_tp += 1
            else:
                total_fp += 1
                error = {
                    "type": "false_positive",
                    "image": img_path.name,
                    "prediction": pred,
                    "best_iou": best_iou,
                    "description": ERROR_CATEGORIES["false_positive"],
                }
                report["categories"]["false_positive"].append(error)
                image_errors.append(error)

        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                total_fn += 1
                # Categorize by object size
                area = gt["w"] * gt["h"]
                size_category = "small" if area < 0.005 else "medium" if area < 0.02 else "large"
                error = {
                    "type": "missed_detection",
                    "image": img_path.name,
                    "ground_truth": gt,
                    "size_category": size_category,
                    "description": ERROR_CATEGORIES["missed_detection"],
                }
                report["categories"]["missed_detection"].append(error)
                image_errors.append(error)

        report["per_image"].append(
            {
                "image": img_path.name,
                "n_gt": len(gt_boxes),
                "n_pred": len(pred_boxes),
                "tp": len(matched_pred),
                "fp": len(pred_boxes) - len(matched_pred),
                "fn": len(gt_boxes) - len(matched_gt),
                "errors": image_errors,
            }
        )

    # Summary statistics
    report["summary"] = {
        "total_images": len(image_paths),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        "false_positive_count": total_fp,
        "missed_detection_count": total_fn,
        "missed_by_size": {
            size: sum(
                1
                for e in report["categories"]["missed_detection"]
                if e.get("size_category") == size
            )
            for size in ["small", "medium", "large"]
        },
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Error analysis saved to %s", output_path)
    logger.info(
        "Summary: TP=%d, FP=%d, FN=%d (P=%.3f, R=%.3f)",
        total_tp,
        total_fp,
        total_fn,
        report["summary"]["precision"],
        report["summary"]["recall"],
    )

    return report


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two boxes in YOLO format (normalized xywh).

    Args:
        box_a: First box dict with x_center, y_center, w, h.
        box_b: Second box dict with x_center, y_center, w, h.

    Returns:
        IoU value between 0 and 1.
    """
    # Convert to xyxy
    a_x1 = box_a["x_center"] - box_a["w"] / 2
    a_y1 = box_a["y_center"] - box_a["h"] / 2
    a_x2 = box_a["x_center"] + box_a["w"] / 2
    a_y2 = box_a["y_center"] + box_a["h"] / 2

    b_x1 = box_b["x_center"] - box_b["w"] / 2
    b_y1 = box_b["y_center"] - box_b["h"] / 2
    b_x2 = box_b["x_center"] + box_b["w"] / 2
    b_y2 = box_b["y_center"] + box_b["h"] / 2

    # Intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    a_area = box_a["w"] * box_a["h"]
    b_area = box_b["w"] * box_b["h"]
    union_area = a_area + b_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 and run error analysis")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/SKU110K/dataset.yaml"),
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/error_analysis/error_report.json"),
        help="Path to save error report",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Run evaluation
    metrics = evaluate_model(args.model, args.data, args.split)

    # Save metrics
    metrics_path = Path("artifacts/benchmarks/evaluation_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Run error analysis on test images
    data_cfg = DataConfig.from_yaml(Path("configs/data.yaml"))
    test_image_dir = data_cfg.processed_dir / "images" / args.split
    test_label_dir = data_cfg.processed_dir / "labels" / args.split

    if test_image_dir.exists():
        analyze_errors(
            model_path=args.model,
            test_image_dir=test_image_dir,
            test_label_dir=test_label_dir,
            output_path=args.output,
        )
