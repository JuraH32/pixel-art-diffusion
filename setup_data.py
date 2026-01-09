#!/usr/bin/env python3
"""
Downloads the Pixel Art dataset from Kaggle.

Prerequisites:
1. pip install kaggle
2. Create Kaggle API token at https://www.kaggle.com/settings
3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)
   - On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import sys
from pathlib import Path

DATASET = "ebrahimelgazar/pixel-art"
DATA_DIR = Path(__file__).parent / "data" / "pixel_art_dataset"


def check_kaggle_installed():
    """Check if kaggle package is importable."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return True
    except ImportError:
        return False


def check_kaggle_configured():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Also check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def main():
    # Check prerequisites
    if not check_kaggle_installed():
        print("Error: kaggle package not found.")
        print("Install it with: pip install kaggle")
        sys.exit(1)

    if not check_kaggle_configured():
        print("Error: Kaggle API credentials not found.")
        print()
        print("To configure:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' to download kaggle.json")
        print("  3. Place kaggle.json in ~/.kaggle/")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print()
        print("Alternatively, set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        sys.exit(1)

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    sprites_file = DATA_DIR / "sprites.npy"
    labels_file = DATA_DIR / "sprites_labels.npy"

    if sprites_file.exists() and labels_file.exists():
        print(f"Dataset already exists at {DATA_DIR}")
        print(f"  - sprites.npy: {sprites_file.stat().st_size / 1e6:.1f} MB")
        print(f"  - sprites_labels.npy: {labels_file.stat().st_size / 1e6:.1f} MB")
        print("Skipping download. Delete these files to re-download.")
        return

    print(f"Downloading {DATASET} from Kaggle...")
    print(f"Destination: {DATA_DIR}")
    print()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

    # Verify download
    if sprites_file.exists() and labels_file.exists():
        print()
        print("Download complete!")
        print(f"  - sprites.npy: {sprites_file.stat().st_size / 1e6:.1f} MB")
        print(f"  - sprites_labels.npy: {labels_file.stat().st_size / 1e6:.1f} MB")
    else:
        print()
        print("Warning: Expected files not found after download.")
        print(f"Contents of {DATA_DIR}:")
        for f in DATA_DIR.iterdir():
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()