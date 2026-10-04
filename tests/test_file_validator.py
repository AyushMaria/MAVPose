"""
tests/test_file_validator.py

Unit tests for mavpose/file_validator.py.
Uses tmp_path (pytest fixture) to create real temporary files.
"""

import pytest
from mavpose.file_validator import validate_mavlink_file, FileValidationError


def _can_symlink(tmp_path) -> bool:
    """Return True if the current process can create symlinks."""
    try:
        src = tmp_path / "_probe_real.tlog"
        src.write_bytes(b"\x00")
        lnk = tmp_path / "_probe_link.tlog"
        lnk.symlink_to(src)
        lnk.unlink()
        src.unlink()
        return True
    except (OSError, NotImplementedError):
        return False


class TestValidateMavlinkFile:

    def test_valid_tlog_file(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00" * 100)
        validate_mavlink_file(str(f))  # should not raise

    def test_valid_bin_file(self, tmp_path):
        f = tmp_path / "log.bin"
        f.write_bytes(b"\x00" * 100)
        validate_mavlink_file(str(f))  # should not raise

    def test_valid_ulg_file(self, tmp_path):
        f = tmp_path / "log.ulg"
        f.write_bytes(b"\x00" * 100)
        validate_mavlink_file(str(f))  # should not raise

    def test_rejects_wrong_extension(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_bytes(b"import os")
        with pytest.raises(FileValidationError, match="Invalid file type"):
            validate_mavlink_file(str(f))

    def test_rejects_zip_extension(self, tmp_path):
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK\x03\x04")
        with pytest.raises(FileValidationError, match="Invalid file type"):
            validate_mavlink_file(str(f))

    def test_rejects_empty_file(self, tmp_path):
        f = tmp_path / "empty.tlog"
        f.write_bytes(b"")
        with pytest.raises(FileValidationError, match="empty"):
            validate_mavlink_file(str(f))

    def test_rejects_oversized_file(self, tmp_path, monkeypatch):
        f = tmp_path / "big.tlog"
        f.write_bytes(b"\x00" * 100)
        # Patch stat to report a huge size without writing 200 MB to disk
        import mavpose.file_validator as fv
        original_limit = fv.MAX_FILE_SIZE_BYTES
        fv.MAX_FILE_SIZE_BYTES = 50  # temporarily lower the limit
        with pytest.raises(FileValidationError, match="exceeds"):
            validate_mavlink_file(str(f))
        fv.MAX_FILE_SIZE_BYTES = original_limit  # restore

    def test_rejects_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            validate_mavlink_file("/nonexistent/path/flight.tlog")

    def test_rejects_symlink(self, tmp_path):
        if not _can_symlink(tmp_path):
            pytest.skip(
                "Symlink creation requires elevated privileges on Windows. "
                "Run as Administrator or enable Developer Mode to include this test."
            )
        real = tmp_path / "real.tlog"
        real.write_bytes(b"\x00" * 100)
        link = tmp_path / "link.tlog"
        link.symlink_to(real)
        with pytest.raises(FileValidationError, match="Symlink"):
            validate_mavlink_file(str(link))
