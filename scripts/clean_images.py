# scripts/clean_images.py
"""
Remove all image files (JPG, JPEG, PNG) from the project directory *except* those inside the `sample_data` folder.
This helps keep the repository tidy after preparing the 100‑sample dataset.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "sample_data"

# Define image extensions to target
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Collect files to delete
files_to_delete = []
for path in PROJECT_ROOT.rglob("*.*"):
    if path.suffix.lower() in IMAGE_EXTS:
        # Skip anything inside sample_data
        try:
            path.relative_to(SAMPLE_DIR)
            continue  # Inside sample_data, keep it
        except ValueError:
            files_to_delete.append(path)

if not files_to_delete:
    print("No extra image files found to delete.")
else:
    for f in files_to_delete:
        try:
            f.unlink()
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
    print(f"✅ Cleaned up {len(files_to_delete)} image files outside 'sample_data'.")
