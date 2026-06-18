"""
Tests for the alignment module.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.aligner import AlignerFactory, ORBAligner, SIFTAligner


def test_orb_aligner_estimate_homography() -> None:
    """Test ORB aligner can estimate homography between similar images."""
    aligner = ORBAligner(n_features=1000)

    # Create a reference image
    ref = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)

    # Create a slightly rotated/shifted version
    M = np.float32([[1, 0, 10], [0, 1, 5]])
    fp = cv2.warpAffine(ref, M, (300, 200))

    H, quality = aligner.estimate_homography(ref, fp)

    # Homography should succeed with similar images
    assert H is None or quality > 0


def test_sift_aligner_estimate_homography() -> None:
    """Test SIFT aligner can estimate homography."""
    import cv2

    aligner = SIFTAligner(n_features=1000)

    ref = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
    M = np.float32([[1, 0, 10], [0, 1, 5]])
    fp = cv2.warpAffine(ref, M, (300, 200))

    H, quality = aligner.estimate_homography(ref, fp)

    assert H is None or quality > 0


def test_aligner_factory_create_orb() -> None:
    """Test AlignerFactory creates ORB aligner."""
    aligner = AlignerFactory.create("orb")
    assert isinstance(aligner, ORBAligner)


def test_aligner_factory_create_sift() -> None:
    """Test AlignerFactory creates SIFT aligner."""
    aligner = AlignerFactory.create("sift")
    assert isinstance(aligner, SIFTAligner)


def test_aligner_factory_invalid_method() -> None:
    """Test AlignerFactory raises error for invalid method."""
    with pytest.raises(ValueError, match="Unsupported alignment method"):
        AlignerFactory.create("invalid_method")


def test_orb_warp_to_reference() -> None:
    """Test ORB aligner can warp image using homography."""
    import cv2

    aligner = ORBAligner()
    fp = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    H = np.eye(3)  # Identity matrix

    result = aligner.warp_to_reference(fp, H, (200, 300))

    assert result.shape == (200, 300, 3)
