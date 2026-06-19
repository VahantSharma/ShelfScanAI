"""
YOLOv8 fine-tuning script with W&B experiment tracking.

Fine-tunes YOLOv8-m on SKU110K for retail product detection.
All hyperparameters loaded from configs/training.yaml.
All metrics logged to Weights & Biases.

Usage:
    python training/train.py --config configs/training.yaml --data configs/data.yaml

For Colab:
    Mount Drive first, then run with data_path pointing to Drive.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import wandb
import yaml

from src.config import TrainingConfig

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Get current git commit hash for experiment tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def setup_wandb(config: TrainingConfig, run_name: str | None = None) -> None:
    """Initialize W&B run with experiment config.

    Args:
        config: Training configuration.
        run_name: Optional run name override.
    """
    wandb.init(
        project="shelfscan",
        config={
            "model": config.model,
            "epochs": config.epochs,
            "imgsz": config.imgsz,
            "batch": config.batch,
            "lr0": config.lr0,
            "lrf": config.lrf,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "warmup_epochs": config.warmup_epochs,
            "patience": config.patience,
            "git_commit": get_git_commit(),
        },
        name=run_name or f"yolov8m_sku110k_{datetime.now():%Y%m%d_%H%M%S}",
    )
    logger.info("W&B run initialized: %s", wandb.run.url)


def train(
    config: TrainingConfig,
    dataset_yaml: Path,
    run_name: str | None = None,
) -> dict[str, float]:
    """Run YOLOv8 training.

    Args:
        config: Training hyperparameters.
        dataset_yaml: Path to Ultralytics dataset YAML.
        run_name: Optional W&B run name override.

    Returns:
        Dict with final metrics (mAP50, mAP50-95, precision, recall).
    """
    from ultralytics import YOLO

    # Initialize W&B
    setup_wandb(config, run_name)

    # Load model
    logger.info("Loading model: %s", config.model)
    model = YOLO(config.model)

    # Train
    logger.info("Starting training for %d epochs", config.epochs)
    results = model.train(
        data=str(dataset_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        lr0=config.lr0,
        lrf=config.lrf,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        patience=config.patience,
        device=config.device,
        workers=config.workers,
        amp=config.amp,
        cache=config.cache,
        project=config.project,
        exist_ok=config.exist_ok,
        verbose=True,
    )

    # Extract metrics
    metrics = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    # Log final metrics to W&B
    wandb.log(metrics)
    wandb.finish()

    logger.info(
        "Training complete: mAP50=%.4f, mAP50-95=%.4f, P=%.4f, R=%.4f",
        metrics["mAP50"],
        metrics["mAP50-95"],
        metrics["precision"],
        metrics["recall"],
    )

    return metrics


def save_experiment_artifacts(
    config: TrainingConfig,
    metrics: dict[str, float],
    run_dir: Path,
) -> None:
    """Save training artifacts for reproducibility.

    Saves:
        - Training config as JSON
        - Final metrics as JSON
        - Best model weights (copied from run dir)
    """
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    runs_dir = artifacts_dir / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = artifacts_dir / "config_snapshot.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Save metrics
    metrics_path = artifacts_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Copy best weights
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        dest = models_dir / "best.pt"
        import shutil

        shutil.copy2(best_weights, dest)
        logger.info("Saved best weights to %s", dest)

    logger.info("Experiment artifacts saved to %s", artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on SKU110K")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/SKU110K/dataset.yaml"),
        help="Path to Ultralytics dataset YAML",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name override",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = TrainingConfig.from_yaml(args.config)

    try:
        metrics = train(config, args.data, args.run_name)

        # Find the run directory
        run_dirs = sorted(Path(config.project).glob("train*"), reverse=True)

        if run_dirs:
            save_experiment_artifacts(config, metrics, run_dirs[0])

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
