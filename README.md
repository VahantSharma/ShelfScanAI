# ShelfScan

**Planogram Compliance Verification from Noisy Field Photos**

An end-to-end applied computer vision system that takes a reference planogram image and a field photo (blurry, angled, poorly lit) and outputs a structured JSON compliance report with a 0-100 score, per-slot pass/fail, missing SKUs, and an annotated overlay image.

This mirrors the exact production problem at companies like SalesCode AI that serve FMCG brands doing retail shelf audits at scale.

---

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Reference       │     │  Field Photo    │
│  Planogram       │     │  (noisy)        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │  YOLOv8 (ONNX)  │
         │              │  Detection      │
         │              └────────┬────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │  CLIP + FAISS   │
         │              │  Identity Match │
         │              └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  ORB/SIFT       │     │  Matched        │
│  Alignment      │────▶│  Detections     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Compliance     │
            │  Engine         │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  JSON Report    │     │  Annotated      │
│  (0-100 score)  │     │  Overlay Image  │
└─────────────────┘     └─────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 3. Analyze a shelf (curl)
curl -X POST http://localhost:8000/analyze \
  -F "reference_planogram=@planogram.jpg" \
  -F "field_photo=@field.jpg"
```

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Detection | YOLOv8-m (Ultralytics) → ONNX | Fast, exportable, well-documented |
| Embedding/Matching | CLIP ViT-B/32 + FAISS | Zero-shot identity, scalable retrieval |
| Alignment | ORB/SIFT + RANSAC | Simple v1, modular interface for LoFTR v2 |
| Compliance Logic | Custom Python rules engine | Mirrors real production logic |
| Serving | FastAPI + ONNX Runtime | Exactly what the JD asks for |
| Tracking | Weights & Biases | Track runs and compare results cleanly |

## Project Structure

```
shelfscan/
├── src/
│   ├── config.py              # Centralized config dataclasses
│   ├── data/                  # Augmentation, Dataset, preparation
│   ├── models/                # Detector, embedder, aligner
│   ├── compliance/            # Engine, scorer, visualizer
│   └── api/                   # FastAPI routes and schemas
├── training/                  # Train, eval, export scripts
├── tests/                     # pytest test suite
├── configs/                   # YAML configuration files
├── scripts/                   # Data download and library building
├── artifacts/                 # Models, benchmarks, error analysis
├── data/                      # Raw, processed, reference library
└── Dockerfile                 # Production container
```

## Dataset & Augmentation

### Training Data
- **SKU110K**: 11,762 retail shelf images with ~1.73M bounding box annotations
- Single-class detection (all products labeled as "object")
- Augmentations applied **online** during training (not saved to disk)

### Field Condition Simulation
| Augmentation | What it simulates |
|-------------|-------------------|
| Random rotation/perspective | Phone camera angle |
| Brightness/contrast | Poor lighting |
| Gaussian/motion blur | Out-of-focus or moving camera |
| JPEG compression | Phone camera quality |
| Coarse dropout | Partial occlusion |

### Evaluation Data
- Grocery Planogram Control Dataset (Turkey) — real phone photos
- Grocery Dataset (Turkey) — shelf images with planogram IDs
- Wikimedia Commons — diverse shelf photos

## Model Choice Rationale

### Why YOLOv8-m (not nano, not large)?
Medium balances inference speed and accuracy. Nano sacrifices too much accuracy; large is overkill for mobile-edge deployment. Medium shows understanding of the inference tradeoff.

### Why CLIP (not a classifier)?
A classifier requires retraining every time a new SKU is added. CLIP generalizes zero-shot — new SKUs = new embeddings only. Critical for production where SKUs change seasonally.

### Why ORB/SIFT before LoFTR?
Start simple, prove it works, then increase sophistication. ORB/SIFT are faster, easier to debug, and most interviewers appreciate incremental engineering. LoFTR is planned for v2 if classical methods prove insufficient.

### Why ONNX Runtime?
Exactly what the JD asks for: "export-ready for fast deployment." ONNX eliminates Python overhead, runs on any hardware, and typically achieves 1.5-3x speedup over PyTorch on CPU.

## Results

### Detection Metrics (Target: mAP@50 > 0.7)

| Metric | Value |
|--------|-------|
| mAP@50 | *pending training* |
| mAP@50-95 | *pending training* |
| Precision | *pending training* |
| Recall | *pending training* |

### Inference Speed

| Backend | Avg Latency | FPS | Speedup |
|---------|-------------|-----|---------|
| PyTorch (CPU) | *pending* | *pending* | 1.0x |
| ONNX Runtime (CPU) | *pending* | *pending* | *pending* |

### Alignment Comparison

| Method | Inlier Ratio | Success Rate | Latency |
|--------|-------------|--------------|---------|
| ORB + RANSAC | *pending* | *pending* | *pending* |
| SIFT + RANSAC | *pending* | *pending* | *pending* |

## Error Analysis

| # | Failure Case | Root Cause | Impact | What I'd Try Next |
|---|-------------|-----------|--------|-------------------|
| 1 | Heavy occlusion | YOLOv8 NMS too aggressive on overlapping boxes | Missed SKUs | Instance segmentation (SAM) |
| 2 | Similar packaging | CLIP embedding distance < threshold between similar brands | Wrong identity matches | Fine-tune CLIP on SKU images |
| 3 | Extreme angle (>30°) | ORB inlier ratio < 0.3, homography fails | Alignment skipped | LoFTR with indoor checkpoint |
| 4 | Low light (< 50 lux) | ORB feature count drops significantly | Poor keypoint matching | Histogram equalization preprocessing |
| 5 | Small objects (< 30px) | YOLOv8 struggles at small scales | False negatives | Multi-scale inference |

## API Reference

### POST /analyze
Upload a reference planogram and field photo for compliance analysis.

**Request:**
```
Content-Type: multipart/form-data

reference_planogram: (file) Reference planogram image
field_photo: (file) Field photo image
```

**Response:**
```json
{
  "compliance_score": 85.0,
  "slot_results": [
    {
      "slot_id": "R0C0",
      "expected_sku": "SKU_001",
      "detected_sku": "SKU_001",
      "status": "correct",
      "confidence": 0.87
    }
  ],
  "missing_skus": ["SKU_003"],
  "wrong_skus": ["R1C2"],
  "annotated_image_b64": "...",
  "inference_time_ms": 123.4,
  "alignment_method": "orb",
  "alignment_quality": 0.85
}
```

### GET /health
Health check endpoint.

```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "0.1.0"
}
```

## What I'd Do With More Time

1. **Instance segmentation (SAM)** — For accurate facing count per product
2. **LoFTR alignment** — Learning-based matcher for extreme angles
3. **Multi-camera support** — Fuse results from multiple shelf photos
4. **Active learning loop** — Flag uncertain detections for human review
5. **True SKU catalog** — Replace exemplar crops with brand product database
6. **CI/CD pipeline** — GitHub Actions for automated testing and deployment

## License

MIT
