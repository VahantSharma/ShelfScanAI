"""
Planogram alignment via feature matching and homography estimation.

Provides a modular interface with ORB and SIFT implementations.
LoFTR will be added in v2 if classical methods prove insufficient.

Why start with ORB/SIFT before LoFTR:
  - Simpler, faster, more interpretable
  - Easier to debug when alignment fails
  - Most interviewers appreciate incremental engineering
  - LoFTR adds complexity; use it only if needed

Both implementations share a common interface so downstream
compliance logic doesn't change when switching methods.

Usage:
    aligner = AlignerFactory.create("orb")
    H, quality = aligner.estimate_homography(reference, field_photo)
    aligned = aligner.warp_to_reference(field_photo, H, reference.shape[:2])
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseAligner(ABC):
    """Abstract base class for planogram alignment.

    All aligners must implement estimate_homography and warp_to_reference.
    """

    @abstractmethod
    def estimate_homography(
        self, reference: np.ndarray, field_photo: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """Estimate homography matrix from feature correspondences.

        Args:
            reference: Reference planogram image (BGR or grayscale).
            field_photo: Field photo to align (BGR or grayscale).

        Returns:
            Tuple of:
                H: 3x3 homography matrix, or None if estimation failed.
                inlier_ratio: Quality metric (0-1), fraction of inlier matches.
        """
        ...

    @abstractmethod
    def warp_to_reference(
        self,
        field_photo: np.ndarray,
        H: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Warp field photo to align with reference coordinate frame.

        Args:
            field_photo: Original field photo.
            H: Homography matrix from estimate_homography.
            target_shape: (height, width) of output image.

        Returns:
            Warped image aligned to reference frame.
        """
        ...

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


class ORBAligner(BaseAligner):
    """v1 baseline: ORB features + RANSAC homography.

    Fast, lightweight, interpretable. Good enough for moderate
    angle shifts (< 30 degrees) and similar lighting conditions.

    Attributes:
        orb: ORB feature detector.
        bf: Brute-force matcher.
        n_features: Number of features to detect.
    """

    def __init__(self, n_features: int = 5000) -> None:
        """Initialize ORB aligner.

        Args:
            n_features: Maximum number of features to detect.
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.n_features = n_features
        logger.info("ORBAligner initialized (n_features=%d)", n_features)

    def estimate_homography(
        self, reference: np.ndarray, field_photo: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """Estimate homography using ORB features + RANSAC.

        Args:
            reference: Reference planogram image.
            field_photo: Field photo to align.

        Returns:
            Tuple of (H matrix or None, inlier_ratio).
        """
        start = time.perf_counter()

        ref_gray = self._to_grayscale(reference)
        fp_gray = self._to_grayscale(field_photo)

        # Detect keypoints and descriptors
        kp_ref, des_ref = self.orb.detectAndCompute(ref_gray, None)
        kp_fp, des_fp = self.orb.detectAndCompute(fp_gray, None)

        if des_ref is None or des_fp is None or len(kp_ref) < 10 or len(kp_fp) < 10:
            logger.warning("ORB: insufficient keypoints (ref=%d, fp=%d)", len(kp_ref), len(kp_fp))
            return None, 0.0

        # Match descriptors
        matches = self.bf.match(des_ref, des_fp)
        matches = sorted(matches, key=lambda m: m.distance)

        # Take top 50% of matches
        n_good = max(10, len(matches) // 2)
        good_matches = matches[:n_good]

        # Extract matched keypoint coordinates
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_fp = np.float32([kp_fp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(pts_fp, pts_ref, cv2.RANSAC, 5.0)

        elapsed_ms = (time.perf_counter() - start) * 1000

        if H is None:
            logger.warning("ORB: homography estimation failed")
            return None, 0.0

        # Compute inlier ratio
        inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0

        logger.info(
            "ORB alignment: %d matches, inlier_ratio=%.3f, %.1fms",
            len(good_matches),
            inlier_ratio,
            elapsed_ms,
        )

        return H, inlier_ratio

    def warp_to_reference(
        self,
        field_photo: np.ndarray,
        H: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Warp field photo using homography.

        Args:
            field_photo: Original field photo.
            H: Homography matrix.
            target_shape: (height, width) of output.

        Returns:
            Warped image.
        """
        h, w = target_shape
        return cv2.warpPerspective(field_photo, H, (w, h))


class SIFTAligner(BaseAligner):
    """v1 alternative: SIFT features + RANSAC homography.

    More accurate than ORB on texture-rich retail scenes,
    but ~10x slower. Better for challenging lighting conditions.

    Attributes:
        sift: SIFT feature detector.
        bf: Brute-force matcher (L2 norm for SIFT).
        n_features: Number of features to detect.
    """

    def __init__(self, n_features: int = 5000) -> None:
        """Initialize SIFT aligner.

        Args:
            n_features: Maximum number of features to detect.
        """
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.n_features = n_features
        logger.info("SIFTAligner initialized (n_features=%d)", n_features)

    def estimate_homography(
        self, reference: np.ndarray, field_photo: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """Estimate homography using SIFT features + RANSAC.

        Uses KNN matching with Lowe's ratio test for better match quality.

        Args:
            reference: Reference planogram image.
            field_photo: Field photo to align.

        Returns:
            Tuple of (H matrix or None, inlier_ratio).
        """
        start = time.perf_counter()

        ref_gray = self._to_grayscale(reference)
        fp_gray = self._to_grayscale(field_photo)

        # Detect keypoints and descriptors
        kp_ref, des_ref = self.sift.detectAndCompute(ref_gray, None)
        kp_fp, des_fp = self.sift.detectAndCompute(fp_gray, None)

        if des_ref is None or des_fp is None or len(kp_ref) < 10 or len(kp_fp) < 10:
            logger.warning("SIFT: insufficient keypoints (ref=%d, fp=%d)", len(kp_ref), len(kp_fp))
            return None, 0.0

        # KNN match (k=2 for Lowe's ratio test)
        knn_matches = self.bf.knnMatch(des_ref, des_fp, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m_pair in knn_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            logger.warning("SIFT: insufficient good matches after ratio test (%d)", len(good_matches))
            return None, 0.0

        # Extract matched keypoint coordinates
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_fp = np.float32([kp_fp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(pts_fp, pts_ref, cv2.RANSAC, 5.0)

        elapsed_ms = (time.perf_counter() - start) * 1000

        if H is None:
            logger.warning("SIFT: homography estimation failed")
            return None, 0.0

        # Compute inlier ratio
        inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0

        logger.info(
            "SIFT alignment: %d matches (after ratio test), inlier_ratio=%.3f, %.1fms",
            len(good_matches),
            inlier_ratio,
            elapsed_ms,
        )

        return H, inlier_ratio

    def warp_to_reference(
        self,
        field_photo: np.ndarray,
        H: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Warp field photo using homography.

        Args:
            field_photo: Original field photo.
            H: Homography matrix.
            target_shape: (height, width) of output.

        Returns:
            Warped image.
        """
        h, w = target_shape
        return cv2.warpPerspective(field_photo, H, (w, h))


class AlignerFactory:
    """Factory to select alignment method from config.

    Supports: "orb", "sift". LoFTR will be added in v2.
    """

    _aligners: dict[str, type[BaseAligner]] = {
        "orb": ORBAligner,
        "sift": SIFTAligner,
    }

    @classmethod
    def create(cls, method: str, **kwargs: int) -> BaseAligner:
        """Create aligner instance.

        Args:
            method: Alignment method name ("orb" or "sift").
            **kwargs: Additional arguments for the aligner.

        Returns:
            BaseAligner instance.

        Raises:
            ValueError: If method is not supported.
        """
        if method not in cls._aligners:
            raise ValueError(
                f"Unsupported alignment method: '{method}'. "
                f"Available: {list(cls._aligners.keys())}"
            )

        aligner = cls._aligners[method](**kwargs)
        logger.info("Created %s aligner", method.upper())
        return aligner

    @classmethod
    def register(cls, name: str, aligner_class: type[BaseAligner]) -> None:
        """Register a new aligner implementation.

        Args:
            name: Method name.
            aligner_class: Aligner class implementing BaseAligner.
        """
        cls._aligners[name] = aligner_class
        logger.info("Registered aligner: %s", name)
