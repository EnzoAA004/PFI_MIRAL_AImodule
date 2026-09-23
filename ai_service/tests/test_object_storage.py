from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pfi_ai_service.object_storage import (
    FilesystemObjectStorage,
    GoogleCloudStorageObjectStorage,
    ObjectStorageError,
    get_object_storage,
    validate_object_key,
)


class _FakeBlob:
    """Minimal stand-in for google.cloud.storage.Blob, backed by a shared dict.

    The dict plays the role of the actual GCS bucket contents: it survives
    across separate _FakeBucket/_make_gcs_storage instances, the same way a
    real bucket survives across separate container processes/instances.
    A `{key}__size_override` entry lets a test simulate metadata/content
    desync (declared size does not match what actually downloads) without
    needing a real network layer to corrupt.
    """

    def __init__(self, backing: dict[str, bytes], key: str) -> None:
        self._backing = backing
        self._key = key
        self.size: int | None = None

    def exists(self) -> bool:
        return self._key in self._backing

    def reload(self) -> None:
        override = self._backing.get(self._key + "__size_override")
        if override is not None:
            self.size = override
            return
        data = self._backing.get(self._key)
        self.size = len(data) if data is not None else None

    def download_to_filename(self, path: str) -> None:
        Path(path).write_bytes(self._backing[self._key])

    def upload_from_string(self, payload: bytes, content_type: str = "") -> None:
        self._backing[self._key] = bytes(payload)

    def upload_from_filename(self, path: str, content_type: str = "") -> None:
        self._backing[self._key] = Path(path).read_bytes()

    def delete(self) -> None:
        self._backing.pop(self._key, None)


class _FakeBucket:
    def __init__(self, backing: dict[str, bytes]) -> None:
        self._backing = backing

    def blob(self, key: str) -> _FakeBlob:
        return _FakeBlob(self._backing, key)


def _make_gcs_storage(cache_root: Path, backing: dict[str, bytes]) -> GoogleCloudStorageObjectStorage:
    """Build a GoogleCloudStorageObjectStorage without touching real GCS.

    Bypasses __init__ (which requires ADC + a real bucket) and injects a fake
    bucket backed by `backing`, so tests can simulate an object persisted by a
    prior process and read back by a fresh one that only shares the bucket,
    never the disk.
    """
    storage = object.__new__(GoogleCloudStorageObjectStorage)
    storage.bucket_name = "fake-bucket"
    storage.client = None
    storage.bucket = _FakeBucket(backing)
    storage.cache_root = cache_root.resolve()
    storage.cache_root.mkdir(parents=True, exist_ok=True)
    return storage


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



def test_gcs_storage_materializes_absent_object_into_local_cache(tmp_path: Path) -> None:
    backing: dict[str, bytes] = {}
    writer = _make_gcs_storage(tmp_path / "writer-cache", backing)
    payload = b"gcs-materialization-payload"
    writer.put_bytes(payload, "runs/run-a/sagittal/overlay.png", "image/png")

    # Proceso nuevo: mismo bucket (backing), cache local vacio.
    reader = _make_gcs_storage(tmp_path / "reader-cache", backing)
    materialized = reader.local_path("runs/run-a/sagittal/overlay.png")

    assert materialized is not None
    assert materialized.is_file()
    assert materialized.read_bytes() == payload
    assert materialized.is_relative_to(reader.cache_root)


def test_gcs_storage_second_read_reuses_local_cache_without_redownload(tmp_path: Path) -> None:
    backing: dict[str, bytes] = {}
    writer = _make_gcs_storage(tmp_path / "writer-cache", backing)
    payload = b"cached-payload"
    writer.put_bytes(payload, "runs/run-b/sagittal/mask.npy", "application/octet-stream")

    reader = _make_gcs_storage(tmp_path / "reader-cache", backing)
    first = reader.local_path("runs/run-b/sagittal/mask.npy")
    assert first is not None

    # Simula que el objeto ya no esta disponible en el bucket remoto (p.ej.
    # revocado/eliminado entre requests): si la segunda lectura reusa la cache
    # local en vez de volver a bajarlo, sigue funcionando.
    backing.pop("runs/run-b/sagittal/mask.npy")
    second = reader.local_path("runs/run-b/sagittal/mask.npy")

    assert second == first
    assert second.read_bytes() == payload


def test_gcs_storage_materialize_returns_none_for_missing_object(tmp_path: Path) -> None:
    backing: dict[str, bytes] = {}
    storage = _make_gcs_storage(tmp_path / "cache", backing)

    assert storage.local_path("runs/run-c/sagittal/overlay.png") is None


def test_gcs_storage_materialize_rejects_traversal_key(tmp_path: Path) -> None:
    backing: dict[str, bytes] = {}
    storage = _make_gcs_storage(tmp_path / "cache", backing)

    with pytest.raises(ObjectStorageError):
        storage.materialize("../escape.png")


def test_gcs_storage_materialize_rejects_size_mismatch(tmp_path: Path) -> None:
    backing: dict[str, bytes] = {}
    storage = _make_gcs_storage(tmp_path / "cache", backing)
    key = "runs/run-d/sagittal/overlay.png"
    storage.put_bytes(b"original-nine", key, "image/png")

    # Simula metadata GCS desincronizada del contenido real (ej. reload()
    # devuelve un tamano que no coincide con lo que realmente se descarga).
    backing[key + "__size_override"] = 999999

    with pytest.raises(ObjectStorageError) as exc_info:
        storage.materialize(key)

    assert "mismatch" in exc_info.value.message
