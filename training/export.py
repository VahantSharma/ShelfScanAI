"""
YOLOv8 ONNX export and benchmark script.

Exports trained PyTorch model to ONNX format and benchmarks
inference speed: PyTorch vs ONNX Runtime on CPU.

Usage:
    python training/export.py --model artifacts/models/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: Path,
    output_dir: Path | None = None,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> Path:
    """Export YOLOv8 model to ONNX format.

    Args:
        model_path: Path to trained .pt model.
        output_dir: Directory to save ONNX model. Defaults to same as model.
        imgsz: Input image size.
        opset: ONNX opset version.
        simplify: Whether to simplify the ONNX graph.
        dynamic: Whether to use dynamic input shapes.

    Returns:
        Path to exported ONNX model.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    if output_dir is None:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting to ONNX: opset=%d, simplify=%s, dynamic=%s",
        opset,
        simplify,
        dynamic,
    )

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        nms=False,
    )

    logger.info("ONNX model exported to: %s", onnx_path)
    return Path(onnx_path)


def benchmark_pytorch(
    model_path: Path,
    imgsz: int = 640,
    n_iterations: int = 100,
    warmup: int = 20,
) -> dict[str, float]:
    """Benchmark PyTorch inference speed.

    Args:
        model_path: Path to .pt model.
        imgsz: Input image size.
        n_iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.

    Returns:
        Dict with latency statistics.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # Warmup
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model(dummy, verbose=False)

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        start = time.perf_counter()
        model(dummy, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": 1000.0 / float(np.mean(latencies)),
        "n_iterations": n_iterations,
    }


def benchmark_onnx(
    onnx_path: Path,
    imgsz: int = 640,
    n_iterations: int = 100,
    warmup: int = 20,
) -> dict[str, float]:
    """Benchmark ONNX Runtime inference speed.

    Args:
        onnx_path: Path to .onnx model.
        imgsz: Input image size.
        n_iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.

    Returns:
        Dict with latency statistics.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": 1000.0 / float(np.mean(latencies)),
        "n_iterations": n_iterations,
    }


def run_full_benchmark(
    model_path: Path,
    imgsz: int = 640,
    n_iterations: int = 100,
) -> dict:
    """Run full benchmark: export + PyTorch vs ONNX comparison.

    Args:
        model_path: Path to .pt model.
        imgsz: Input image size.
        n_iterations: Benchmark iterations.

    Returns:
        Complete benchmark results dict.
    """
    logger.info("Starting full benchmark...")

    # Export to ONNX
    onnx_path = export_to_onnx(model_path, imgsz=imgsz)
    logger.info("ONNX export complete: %s", onnx_path)

    # Benchmark PyTorch
    logger.info("Benchmarking PyTorch...")
    pytorch_results = benchmark_pytorch(model_path, imgsz=imgsz, n_iterations=n_iterations)

    # Benchmark ONNX
    logger.info("Benchmarking ONNX Runtime...")
    onnx_results = benchmark_onnx(onnx_path, imgsz=imgsz, n_iterations=n_iterations)

    # Compute speedup
    speedup = pytorch_results["avg_ms"] / onnx_results["avg_ms"] if onnx_results["avg_ms"] > 0 else 0

    results = {
        "pytorch": pytorch_results,
        "onnx": onnx_results,
        "speedup": round(speedup, 2),
        "onnx_path": str(onnx_path),
        "imgsz": imgsz,
    }

    # Log summary
    logger.info("=" * 60)
    logger.info("Benchmark Results (imgsz=%d, %d iterations):", imgsz, n_iterations)
    logger.info(
        "  PyTorch:  %.1f ms avg (%.1f FPS)",
        pytorch_results["avg_ms"],
        pytorch_results["fps"],
    )
    logger.info(
        "  ONNX:     %.1f ms avg (%.1f FPS)",
        onnx_results["avg_ms"],
        onnx_results["fps"],
    )
    logger.info("  Speedup:  %.2fx", speedup)
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX and benchmark")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained .pt model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/benchmark_results.json"),
        help="Path to save benchmark results",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = run_full_benchmark(args.model, imgsz=args.imgsz, n_iterations=args.iterations)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Benchmark results saved to %s", args.output)
