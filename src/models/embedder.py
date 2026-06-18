"""
CLIP-based SKU embedding and FAISS nearest-neighbor matching.

Why CLIP instead of a classifier:
  A classifier requires retraining every time a new SKU is added.
  CLIP generalizes zero-shot — new SKUs = new embeddings only.
  This is critical for production: SKUs change seasonally,
  new products launch monthly, and retraining is expensive.

Using open-clip-torch (NOT the deprecated openai/CLIP package):
  - Actively maintained
  - Supports 80+ pretrained models
  - Loads original OpenAI weights via pretrained='openai'

FAISS is used for vector search:
  - IndexFlatIP for cosine similarity (embeddings are L2-normalized)
  - Scales to millions of SKUs
  - Much faster than brute-force cosine computation

Usage:
    embedder = SKUEmbedder()
    embedding = embedder.embed_image(image)

    matcher = SKUMatcher(Path("data/reference_library"))
    matches = matcher.match_detections(crops)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SKUEmbedder:
    """CLIP-based image embedder for SKU identity.

    Uses OpenAI's CLIP ViT-B/32 to embed product images into 512-dim vectors.
    Embeddings are L2-normalized for cosine similarity search.

    Attributes:
        model: CLIP model instance.
        preprocess: Image preprocessing pipeline.
        tokenizer: Text tokenizer (for zero-shot classification).
        device: Compute device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        """Initialize CLIP embedder.

        Args:
            model_name: CLIP model architecture.
            pretrained: Pretrained weights to load.
            device: Compute device. Auto-detected if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        logger.info(
            "SKUEmbedder initialized: model=%s, pretrained=%s, device=%s",
            model_name,
            pretrained,
            device,
        )

    @torch.no_grad()
    def embed_image(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Embed a single image into a 512-dim vector.

        Args:
            image: PIL Image or numpy array (HWC, RGB).

        Returns:
            L2-normalized 512-dim embedding as numpy array.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image | np.ndarray]) -> np.ndarray:
        """Embed a batch of images.

        Args:
            images: List of PIL Images or numpy arrays.

        Returns:
            L2-normalized embeddings [N, 512] as numpy array.
        """
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            tensors.append(self.preprocess(img))

        batch = torch.stack(tensors).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text_descriptions: list[str]) -> np.ndarray:
        """Embed text descriptions for zero-shot classification.

        Args:
            text_descriptions: List of text descriptions.

        Returns:
            L2-normalized text embeddings [N, 512] as numpy array.
        """
        tokens = self.tokenizer(text_descriptions).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


class SKUMatcher:
    """Matches detected product crops against a reference library.

    Uses FAISS IndexFlatIP for efficient nearest-neighbor search.
    Cosine similarity is computed via inner product on L2-normalized vectors.

    Attributes:
        index: FAISS index for vector search.
        manifest: SKU metadata per index position.
        embedder: CLIP embedder instance.
    """

    def __init__(self, library_path: Path, device: str | None = None) -> None:
        """Initialize matcher with reference library.

        Args:
            library_path: Path to reference library directory
                (must contain index.faiss and manifest.json).
            device: Compute device for CLIP.

        Raises:
            FileNotFoundError: If library files don't exist.
        """
        self.library_path = library_path
        index_path = library_path / "index.faiss"
        manifest_path = library_path / "manifest.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.index = faiss.read_index(str(index_path))
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.embedder = SKUEmbedder(device=device)

        logger.info(
            "SKUMatcher initialized: %d reference SKUs from %s",
            self.index.ntotal,
            library_path,
        )

    def match_detections(
        self,
        crops: list[np.ndarray | Image.Image],
        top_k: int = 3,
        min_similarity: float = 0.6,
    ) -> list[dict]:
        """Match detected crops against reference library.

        Args:
            crops: List of cropped product images (numpy or PIL).
            top_k: Number of top matches to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of match dicts, one per crop:
                {
                    "crop_idx": int,
                    "matched_sku": str | None,
                    "confidence": float,
                    "top_k": list[tuple[str, float]],
                }
        """
        if not crops:
            return []

        # Embed all crops
        embeddings = self.embed_batch(crops)

        # Search FAISS index
        similarities, indices = self.index.search(
            embeddings.astype(np.float32), min(top_k, self.index.ntotal)
        )

        results = []
        for i in range(len(crops)):
            top_matches = []
            for j in range(min(top_k, len(indices[i]))):
                idx = int(indices[i][j])
                sim = float(similarities[i][j])
                sku_id = self.manifest[idx]["sku_id"]
                top_matches.append((sku_id, sim))

            best_sku = top_matches[0][0] if top_matches else None
            best_sim = top_matches[0][1] if top_matches else 0.0

            if best_sim < min_similarity:
                best_sku = None

            results.append(
                {
                    "crop_idx": i,
                    "matched_sku": best_sku,
                    "confidence": best_sim,
                    "top_k": top_matches,
                }
            )

        return results

    def match_single(
        self,
        crop: np.ndarray | Image.Image,
        top_k: int = 3,
        min_similarity: float = 0.6,
    ) -> dict:
        """Match a single crop against reference library.

        Args:
            crop: Single cropped product image.
            top_k: Number of top matches to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            Match dict with crop_idx, matched_sku, confidence, top_k.
        """
        results = self.match_detections([crop], top_k=top_k, min_similarity=min_similarity)
        return results[0] if results else {
            "crop_idx": 0,
            "matched_sku": None,
            "confidence": 0.0,
            "top_k": [],
        }

    def add_to_library(
        self,
        sku_id: str,
        crop: np.ndarray | Image.Image,
        metadata: dict | None = None,
    ) -> None:
        """Add a new SKU to the reference library.

        Args:
            sku_id: Unique SKU identifier.
            crop: Product image crop.
            metadata: Optional metadata to store.
        """
        embedding = self.embedder.embed_image(crop)
        embedding = embedding.astype(np.float32).reshape(1, -1)

        # Add to FAISS index
        current_idx = self.index.ntotal
        self.index.add(embedding)

        # Add to manifest
        entry = {"sku_id": sku_id, "embedding_idx": current_idx}
        if metadata:
            entry.update(metadata)
        self.manifest.append(entry)

        logger.info(
            "Added SKU '%s' to library (index=%d, total=%d)",
            sku_id,
            current_idx,
            self.index.ntotal,
        )

    def save_library(self) -> None:
        """Save current library state to disk."""
        self.library_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.library_path / "index.faiss"))
        with open(self.library_path / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

        logger.info("Library saved to %s (%d SKUs)", self.library_path, self.index.ntotal)
