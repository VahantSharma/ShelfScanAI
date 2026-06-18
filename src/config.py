"""
Centralized configuration management.

All configuration is loaded from YAML files and validated via dataclasses.
No magic numbers or hardcoded paths anywhere in the codebase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Paths and settings for data management."""

    raw_dir: Path
    processed_dir: Path
    reference_library_dir: Path
    test_pairs_dir: Path

    @classmethod
    def from_yaml(cls, path: Path) -> DataConfig:
        """Load DataConfig from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            raw_dir=Path(raw["raw_dir"]),
            processed_dir=Path(raw["processed_dir"]),
            reference_library_dir=Path(raw["reference_library_dir"]),
            test_pairs_dir=Path(raw["test_pairs_dir"]),
        )


@dataclass
class TrainingConfig:
    """YOLOv8 training hyperparameters."""

    model: str = "yolov8m.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    patience: int = 20
    device: str = "0"
    workers: int = 8
    amp: bool = True
    cache: bool = False
    project: str = "artifacts/runs"
    exist_ok: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        """Load TrainingConfig from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Model architecture and selection choices."""

    detection_model: str = "yolov8m"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    alignment_method: str = "orb"
    onnx_opset: int = 17
    onnx_simplify: bool = True
    onnx_dynamic: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        """Load ModelConfig from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class ComplianceConfig:
    """Compliance scoring weights and thresholds."""

    weights: dict[str, float] = field(
        default_factory=lambda: {"presence": 0.5, "facings": 0.3, "correctness": 0.2}
    )
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "min_confidence": 0.6,
            "position_tolerance": 0.1,
            "iou_threshold": 0.5,
        }
    )

    @classmethod
    def from_yaml(cls, path: Path) -> ComplianceConfig:
        """Load ComplianceConfig from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            weights=raw.get("weights", cls.weights),
            thresholds=raw.get("thresholds", cls.thresholds),
        )


@dataclass
class ExperimentConfig:
    """Experiment registry for reproducibility."""

    run_id: str = ""
    wandb_project: str = "shelfscan"
    wandb_entity: str = ""
    timestamp: str = ""
    git_commit: str = ""
    notes: str = ""

    def save(self, path: Path) -> None:
        """Save experiment config to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        """Load experiment config from JSON."""
        with open(path) as f:
            raw = json.load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def load_all_configs(config_dir: Path) -> dict[str, Any]:
    """Load all configuration files from a directory.

    Returns:
        Dictionary with keys: data, training, model, compliance.
    """
    return {
        "data": DataConfig.from_yaml(config_dir / "data.yaml"),
        "training": TrainingConfig.from_yaml(config_dir / "training.yaml"),
        "model": ModelConfig.from_yaml(config_dir / "model.yaml"),
        "compliance": ComplianceConfig.from_yaml(config_dir / "compliance.yaml"),
    }
