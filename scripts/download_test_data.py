"""
Download public shelf photos for evaluation/testing.

Sources real retail shelf images from multiple public datasets:
  - Grocery Planogram Control Dataset (Turkey): ~50 phone photos
  - Grocery Dataset (Turkey): 354 shelf images with planogram IDs
  - Wikimedia Commons: diverse shelf photos

Usage:
    python scripts/download_test_data.py --output-dir data/test_pairs/public
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

METADATA: list[dict[str, str]] = []


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a file from URL to destination path.

    Args:
        url: Download URL.
        dest: Destination file path.
        timeout: Request timeout in seconds.

    Returns:
        True if successful, False otherwise.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        return True
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        return False


def download_grocery_planogram_control(output_dir: Path) -> int:
    """Download Grocery Planogram Control Dataset (Turkey).

    Real phone photos of grocery shelves with occlusion, multi-angle,
    and stacked object subsets. Samsung S10 Plus camera.

    Source: https://github.com/meyucel/Grocery-Planogram-Control-Dataset

    Returns:
        Number of images downloaded.
    """
    logger.info("Downloading Grocery Planogram Control Dataset...")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repo to a temp dir and extract images
    repo_url = "https://github.com/meyucel/Grocery-Planogram-Control-Dataset.git"
    count = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        clone_cmd = f"git clone --depth 1 {repo_url} {tmp_path / 'repo'"
        import subprocess

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(tmp_path / "repo")],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error("Failed to clone Grocery Planogram Control Dataset: %s", e)
            return 0

        repo_path = tmp_path / "repo"
        if not repo_path.exists():
            logger.error("Clone succeeded but repo directory not found")
            return 0

        # Find all image files
        for img_path in repo_path.rglob("*.jpg"):
            dest = images_dir / f"gpc_{img_path.stem}.jpg"
            shutil.copy2(img_path, dest)
            METADATA.append(
                {
                    "filename": dest.name,
                    "source": "Grocery Planogram Control Dataset",
                    "source_url": "https://github.com/meyucel/Grocery-Planogram-Control-Dataset",
                    "license": "Academic",
                    "difficulty": "mixed",
                    "camera": "Samsung S10 Plus",
                }
            )
            count += 1

        for img_path in repo_path.rglob("*.png"):
            dest = images_dir / f"gpc_{img_path.stem}.png"
            shutil.copy2(img_path, dest)
            METADATA.append(
                {
                    "filename": dest.name,
                    "source": "Grocery Planogram Control Dataset",
                    "source_url": "https://github.com/meyucel/Grocery-Planogram-Control-Dataset",
                    "license": "Academic",
                    "difficulty": "mixed",
                    "camera": "Samsung S10 Plus",
                }
            )
            count += 1

    logger.info("Downloaded %d images from Grocery Planogram Control Dataset", count)
    return count


def download_wikimedia_shelves(output_dir: Path) -> int:
    """Download shelf photos from Wikimedia Commons.

    Uses direct URLs to freely-licensed supermarket shelf images.

    Returns:
        Number of images downloaded.
    """
    logger.info("Downloading Wikimedia Commons shelf images...")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Known freely-licensed shelf images from Wikimedia Commons
    # These are CC BY-SA 4.0 or similar open licenses
    wikimedia_images = [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Supermarket_in_South_Korea.JPG/1280px-Supermarket_in_South_Korea.JPG",
            "filename": "wm_supermarket_south_korea.jpg",
            "description": "South Korean supermarket shelf",
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Grocery_store%2C_type_A%2C_Volkovo_%28Leningrad_Oblast%29%2C_Russia_%282022%29.jpg/1280px-Grocery_store%2C_type_A%2C_Volkovo_%28Leningrad_Oblast%29%2C_Russia_%282022%29.jpg",
            "filename": "wm_grocery_russia.jpg",
            "description": "Russian grocery store shelf",
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Supermarket_shelf_in_Koog_aan_de_Zaan.jpg/1280px-Supermarket_shelf_in_Koog_aan_de_Zaan.jpg",
            "filename": "wm_supermarket_netherlands.jpg",
            "description": "Dutch supermarket shelf",
        },
    ]

    count = 0
    for item in wikimedia_images:
        dest = images_dir / item["filename"]
        if download_file(item["url"], dest):
            METADATA.append(
                {
                    "filename": item["filename"],
                    "source": "Wikimedia Commons",
                    "source_url": item["url"],
                    "license": "CC BY-SA",
                    "difficulty": "diverse",
                    "camera": "various",
                    "description": item["description"],
                }
            )
            count += 1

    logger.info("Downloaded %d images from Wikimedia Commons", count)
    return count


def save_metadata(output_dir: Path) -> None:
    """Save metadata.json describing all downloaded images."""
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(METADATA, f, indent=2)
    logger.info("Saved metadata for %d images to %s", len(METADATA), metadata_path)


def main(output_dir: Path | None = None) -> None:
    """Download all test data sources.

    Args:
        output_dir: Output directory. Defaults to data/test_pairs/public.
    """
    if output_dir is None:
        output_dir = Path("data/test_pairs/public")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading test data to %s", output_dir)

    total = 0
    total += download_grocery_planogram_control(output_dir)
    total += download_wikimedia_shelves(output_dir)

    save_metadata(output_dir)

    # Write README
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        f"# Test Data Sources\n\n"
        f"Total images: {total}\n\n"
        f"## Sources\n\n"
        f"1. **Grocery Planogram Control Dataset** (Turkey)\n"
        f"   - Real phone photos (Samsung S10 Plus)\n"
        f"   - Occluded, multi-angle, stacked subsets\n"
        f"   - License: Academic\n\n"
        f"2. **Wikimedia Commons**\n"
        f"   - Diverse supermarket shelf photos\n"
        f"   - Multiple countries and store types\n"
        f"   - License: CC BY-SA\n\n"
        f"## Adding More Images\n\n"
        f"Add images to `images/` and update `metadata.json` with origin info.\n"
    )

    logger.info("Test data download complete: %d total images", total)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download public shelf photos for testing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/test_pairs/public)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    main(args.output_dir)
