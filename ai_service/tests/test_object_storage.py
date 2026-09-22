from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pfi_ai_service.object_storage import (
    FilesystemObjectStorage,
    ObjectStorageError,
    get_object_storage,
    validate_object_key,
)


def test_filesystem_storage_put_bytes_and_get_roundtrip(tmp_path: Path) -> None:
    storage = FilesystemObjectStorage(tmp_path / "root")
    payload = b"hello durable world"

    stored = storage.put_bytes(payload, "runs/run-1/sagittal/overlay.png", "image/png")

    assert stored.provider == "filesystem"
    assert stored.size_bytes == len(payload)
    assert stored.sha256 == hashlib.sha256(payload).hexdigest()
    assert storage.exists("runs/run-1/sagittal/overlay.png")
    with storage.open("runs/run-1/sagittal/overlay.png") as handle:
        assert handle.read() == payload


def test_filesystem_storage_put_file_copies_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"binary-content")
    storage = FilesystemObjectStorage(tmp_path / "root")

    stored = storage.put_file(source, "runs/run-2/sagittal/mask.npy", "application/octet-stream")

    assert stored.local_path is not None
    assert stored.local_path.read_bytes() == b"binary-content"
    assert stored.sha256 == hashlib.sha256(b"binary-content").hexdigest()


def test_filesystem_storage_exists_and_delete(tmp_path: Path) -> None:
    storage = FilesystemObjectStorage(tmp_path / "root")
    storage.put_bytes(b"payload", "runs/run-3/sagittal/input.png", "image/png")

    assert storage.exists("runs/run-3/sagittal/input.png") is True
    storage.delete("runs/run-3/sagittal/input.png")
    assert storage.exists("runs/run-3/sagittal/input.png") is False


def test_filesystem_storage_rejects_object_key_escaping_root(tmp_path: Path) -> None:
    storage = FilesystemObjectStorage(tmp_path / "root")

    with pytest.raises(ObjectStorageError):
        storage.put_bytes(b"payload", "../escape.png", "image/png")


def test_filesystem_storage_rejects_empty_payload(tmp_path: Path) -> None:
    storage = FilesystemObjectStorage(tmp_path / "root")

    with pytest.raises(ObjectStorageError):
        storage.put_bytes(b"", "runs/run-4/sagittal/input.png", "image/png")


@pytest.mark.parametrize(
    "object_key",
    ["", "/absolute", "a//b", "a/../b", "a/./b", "a/..", ".."],
)
def test_validate_object_key_rejects_invalid_keys(object_key: str) -> None:
    with pytest.raises(ObjectStorageError):
        validate_object_key(object_key)


def test_get_object_storage_defaults_to_filesystem(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PFI_STORAGE_PROVIDER", raising=False)
    monkeypatch.setenv("PFI_STORAGE_LOCAL_ROOT", str(tmp_path / "root"))

    storage = get_object_storage()

    assert storage.provider == "filesystem"


def test_get_object_storage_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("PFI_STORAGE_PROVIDER", "not-a-real-provider")

    with pytest.raises(ObjectStorageError):
        get_object_storage()


def test_get_object_storage_fails_closed_when_gcs_bucket_missing(monkeypatch) -> None:
    monkeypatch.setenv("PFI_STORAGE_PROVIDER", "gcs")
    monkeypatch.delenv("PFI_STORAGE_GCS_BUCKET", raising=False)

    with pytest.raises(ObjectStorageError) as exc_info:
        get_object_storage()

    assert "PFI_STORAGE_GCS_BUCKET" in exc_info.value.message
