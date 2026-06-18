# Design Decisions

This document records the key technical decisions made during ShelfScan development and the reasoning behind each choice.

---

## Why YOLOv8-m (Medium)?

**Decision:** Use YOLOv8-m, not YOLOv8-n (nano) or YOLOv8-l (large).

**Reasoning:**
- **Nano** is optimized for edge/mobile but sacrifices too much accuracy on dense retail scenes. SKU110K has ~147 objects per image; nano's smaller receptive field misses small products.
- **Large** provides marginally better mAP but at 3x the inference cost. For a portfolio project demonstrating production thinking, the medium model shows you understand the speed-accuracy tradeoff.
- **Medium** hits the sweet spot: mAP50-95 of 50.2 on COCO, ~220ms ONNX CPU inference, 25.9M parameters. This is the model you'd actually deploy.

**Reference:** Ultralytics model cards show YOLOv8-m achieves 50.2 mAP50-95 with 78.9B FLOPs — best accuracy-per-FLOP in the family.

---

## Why CLIP (Not a Classifier)?

**Decision:** Use OpenAI's CLIP ViT-B/32 for SKU identity, not a trained classification head.

**Reasoning:**
- A classifier requires retraining every time a new SKU is added. In retail, SKUs change seasonally, new products launch monthly, and promotional items appear and disappear weekly.
- CLIP generalizes zero-shot: to add a new SKU, you just embed one image and add it to the FAISS index. No retraining required.
- CLIP's 512-dim embeddings capture visual similarity well enough for retail products, which have distinctive packaging.
- The tradeoff is lower accuracy than a fine-tuned classifier on known classes, but vastly better flexibility for production scale.

**Counterargument:** If you had 10,000+ labeled SKU images, a fine-tuned classifier would outperform CLIP on those specific classes. But you'd lose the ability to handle new products without retraining.

---

## Why ORB/SIFT Before LoFTR?

**Decision:** Start with classical feature matching (ORB/SIFT), add LoFTR in v2.

**Reasoning:**
- **Simplicity first:** ORB and SIFT are well-understood, have OpenCV implementations, and are easy to debug. When alignment fails, you can inspect keypoints and matches visually.
- **Speed:** ORB runs in <50ms per pair. LoFTR takes 200-500ms. For a portfolio project, faster iteration matters.
- **Incremental engineering:** Most interviewers appreciate seeing simple → complex progression. Starting with LoFTR signals you reach for the fanciest tool first.
- **Modularity:** The `BaseAligner` interface means swapping in LoFTR later requires zero changes to downstream compliance logic.

**When LoFTR would be needed:** If field photos have >30° perspective distortion, ORB/SIFT fail because the perspective change destroys local feature consistency. LoFTR's transformer attention handles this better.

---

## Why ONNX Runtime (Not TorchServe or Triton)?

**Decision:** Export to ONNX and use ONNX Runtime for inference.

**Reasoning:**
- The JD explicitly mentions "export-ready for fast deployment." ONNX is the industry standard for model portability.
- ONNX Runtime is pip-installable, runs on any hardware (CPU, GPU, edge), and doesn't require a separate serving infrastructure.
- TorchServe and Triton are overkill for a single model serving a few requests per second. ONNX Runtime embedded in FastAPI is exactly the pattern used in production at many companies.
- ONNX eliminates Python's GIL overhead for the inference hot-path. Typical speedup is 1.5-3x over PyTorch on CPU.

---

## Why FAISS (Not Brute-Force Cosine)?

**Decision:** Use Facebook AI Similarity Search for vector retrieval.

**Reasoning:**
- With 50-100 reference SKUs, brute-force cosine search is fine. But the point is to demonstrate you know how this scales.
- FAISS IndexFlatIP with L2-normalized vectors gives exact cosine similarity. For production with 100K+ SKUs, you'd switch to IndexIVFFlat or IndexHNSW for approximate search.
- FAISS is the industry standard for vector retrieval (used at Meta, used in most recommendation systems). Mentioning it in interviews signals production awareness.

---

## Why Online Augmentation (Not Saved to Disk)?

**Decision:** Apply augmentations on-the-fly in the PyTorch Dataset, never save augmented images.

**Reasoning:**
- **Disk space:** SKU110K is ~2GB raw. Saving 5 augmentation variants would be 10GB+.
- **Reproducibility:** With a fixed seed, online augmentation is deterministic. Saved augmentations can get out of sync with code changes.
- **Flexibility:** Want to change augmentation parameters? Just change the config. No reprocessing needed.
- **Standard practice:** This is how virtually all production CV pipelines work.

---

## Why UploadFile (Not Base64)?

**Decision:** Use FastAPI's `UploadFile` for image input, not base64-encoded strings.

**Reasoning:**
- Base64 encoding adds ~33% overhead to image transfer. A 2MB photo becomes 2.67MB of base64 text.
- `UploadFile` uses streaming uploads (spills to disk for large files), avoiding memory pressure.
- This is the standard HTTP pattern for file uploads. Base64 in JSON is a hack.
- `UploadFile` also provides the filename and content-type, which can be useful for validation.

---

## Why Ruff (Not Black + isort + flake8)?

**Decision:** Use Ruff as the single linter/formatter, not the traditional Black + isort + flake8 stack.

**Reasoning:**
- As of 2026, Ruff has fully replaced Black for formatting. Same default style (88-char lines, double quotes), 10-100x faster.
- Running both Black and Ruff causes conflicts and wasted cycles.
- Ruff handles linting (flake8 replacement), import sorting (isort replacement), and formatting (Black replacement) in one binary.
- Fewer dependencies = simpler CI, faster pre-commit hooks.

---

## Why Config Dataclasses (Not Just YAML)?

**Decision:** Wrap YAML configs in typed Python dataclasses.

**Reasoning:**
- YAML has no type safety. A typo in a key silently uses the default value. A string where an int is expected blows up at runtime.
- Dataclasses give you IDE autocompletion, type checking (mypy), and runtime validation.
- The pattern is: load YAML → validate with dataclass → use typed config everywhere.
- This is a small investment that pays off in code quality and developer experience.

---

## Why Exemplar Crops (Not True SKU Catalog)?

**Decision:** Build reference library from detected crops, not a brand's product database.

**Reasoning:**
- This is a portfolio project, not a production system. We don't have access to actual brand product catalogs.
- The important thing is demonstrating the CLIP + FAISS pipeline works. The source of embeddings doesn't change the architecture.
- The code is structured so swapping in a real SKU catalog requires only changing `build_reference_library.py` — the matching pipeline is identical.
- **Documented clearly** in the codebase and README to avoid interviewer confusion.

---

## Decision Log

| Date | Decision | Status |
|------|----------|--------|
| 2026-06 | Use YOLOv8-m over nano/large | Locked |
| 2026-06 | CLIP over classifier for identity | Locked |
| 2026-06 | ORB/SIFT v1, LoFTR v2 | Locked |
| 2026-06 | ONNX Runtime for serving | Locked |
| 2026-06 | FAISS for vector search | Locked |
| 2026-06 | Online augmentation | Locked |
| 2026-06 | UploadFile over base64 | Locked |
| 2026-06 | Ruff over Black+isort | Locked |
| 2026-06 | Config dataclasses | Locked |
| 2026-06 | Exemplar crops for reference library | Locked |
