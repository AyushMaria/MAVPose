"""
mavpose/file_validator.py

Validates a flight log file (MAVLink, DataFlash or PX4 ULog) before it is passed to PlotCreator.

Checks performed:
  1. File exists
  2. Not a symlink (path-traversal / symlink attack prevention)
  3. Extension is in VALID_EXTENSIONS
  4. File is not empty
  5. File does not exceed MAX_FILE_SIZE_BYTES
"""

import os

VALID_EXTENSIONS = {".tlog", ".bin", ".log", ".ulg"}

# 200 MB default limit; tests monkeypatch this directly
MAX_FILE_SIZE_BYTES: int = 200 * 1024 * 1024


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_mavlink_file(path: str) -> None:
    """
    Validate *path* as an acceptable MAVLink log file.

    Args:
        path: Absolute or relative path to the file.

    Raises:
        FileNotFoundError: File does not exist.
        FileValidationError: File fails any validation check.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Symlink check — must come before stat() to avoid TOCTOU
    if os.path.islink(path):
        raise FileValidationError(
            f"Symlink not allowed: {path}. Please provide a direct file path."
        )

    # Extension check
    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise FileValidationError(
            f"Invalid file type '{ext}'. "
            f"Accepted types: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    size = os.path.getsize(path)

    # Empty file check
    if size == 0:
        raise FileValidationError(f"File is empty: {path}")

    # Size limit check
    if size > MAX_FILE_SIZE_BYTES:
        limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        actual_mb = size / (1024 * 1024)
        raise FileValidationError(
            f"File size ({actual_mb:.1f} MB) exceeds the {limit_mb:.0f} MB limit: {path}"
        )
