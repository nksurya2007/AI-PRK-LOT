# scripts/archive_data_images.py
"""
Archive all image files from the project's ``data`` directory to a separate folder.

- Scans ``data`` (recursively) for ``.jpg``, ``.jpeg`` and ``.png`` files.
- Moves each file to ``archived_images`` while keeping the relative sub‑directory
  structure (e.g. ``data/patches/train/empty/img.jpg`` → ``archived_images/patches/train/empty/img.jpg``).
- Creates any missing parent directories in the archive location.
- Prints a summary of moved files.

This leaves the original ``data`` directory empty of images, but the files are
still available under ``archived_images`` for future reference.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_ROOT = PROJECT_ROOT / "archived_images"

# Ensure archive root exists
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

moved = 0
for src_path in DATA_DIR.rglob("*.*"):
    if src_path.suffix.lower() in IMAGE_EXTS:
        # Compute relative path inside DATA_DIR and mirror it under ARCHIVE_ROOT
        rel_path = src_path.relative_to(DATA_DIR)
        dst_path = ARCHIVE_ROOT / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        print(f"Moved: {src_path} -> {dst_path}")
        moved += 1

print(f"✅ Archived {moved} image files from 'data' to '{ARCHIVE_ROOT}'.")
