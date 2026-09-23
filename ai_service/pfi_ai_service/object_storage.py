from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal


StorageProviderName = Literal["filesystem", "gcs"]


class ObjectStorageError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class StoredObject:
    provider: StorageProviderName
    object_key: str
    content_type: str
    size_bytes: int
    sha256: str
    local_path: Path | None = None


class ObjectStorage:
    provider: StorageProviderName

    def put_file(self, source: Path, object_key: str, content_type: str) -> StoredObject:
        raise NotImplementedError

    def put_bytes(self, payload: bytes, object_key: str, content_type: str) -> StoredObject:
        raise NotImplementedError

    def open(self, object_key: str) -> BinaryIO:
        raise NotImplementedError

    def exists(self, object_key: str) -> bool:
        raise NotImplementedError

    def delete(self, object_key: str) -> None:
        raise NotImplementedError

    def local_path(self, object_key: str) -> Path | None:
        return None

    def list_keys(self, prefix: str) -> list[str]:
        return []


class FilesystemObjectStorage(ObjectStorage):
    provider: StorageProviderName = "filesystem"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source: Path, object_key: str, content_type: str) -> StoredObject:
        source = source.resolve()
        destination = self._path_for_key(object_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            shutil.copy2(source, destination)
        return StoredObject(
            provider=self.provider,
            object_key=object_key,
            content_type=content_type,
            size_bytes=destination.stat().st_size,
            sha256=_sha256_file(destination),
            local_path=destination,
        )

    def put_bytes(self, payload: bytes, object_key: str, content_type: str) -> StoredObject:
        if not payload:
            raise ObjectStorageError("object payload cannot be empty")
        destination = self._path_for_key(object_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return StoredObject(
            provider=self.provider,
            object_key=object_key,
            content_type=content_type,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            local_path=destination,
        )

    def open(self, object_key: str) -> BinaryIO:
        return self._path_for_key(object_key).open("rb")

    def exists(self, object_key: str) -> bool:
        return self._path_for_key(object_key).is_file()

    def delete(self, object_key: str) -> None:
        self._path_for_key(object_key).unlink(missing_ok=True)

    def local_path(self, object_key: str) -> Path | None:
        path = self._path_for_key(object_key)
        return path if path.is_file() else None

    def _path_for_key(self, object_key: str) -> Path:
        validated = validate_object_key(object_key)
        path = (self.root / validated).resolve()
        if not path.is_relative_to(self.root):
            raise ObjectStorageError("object key escapes storage root")
        return path


class GoogleCloudStorageObjectStorage(ObjectStorage):
    provider: StorageProviderName = "gcs"

    def __init__(self, bucket_name: str, cache_root: Path | None = None) -> None:
        if not bucket_name.strip():
            raise ObjectStorageError("PFI_STORAGE_GCS_BUCKET is required when provider=gcs")
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as exc:
            raise ObjectStorageError(
                "google-cloud-storage is required when PFI_STORAGE_PROVIDER=gcs"
            ) from exc
        self.bucket_name = bucket_name.strip()
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        self.cache_root = (cache_root or _default_gcs_cache_root()).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source: Path, object_key: str, content_type: str) -> StoredObject:
        validated = validate_object_key(object_key)
        blob = self.bucket.blob(validated)
        blob.upload_from_filename(str(source), content_type=content_type)
        blob.reload()
        return StoredObject(
            provider=self.provider,
            object_key=validated,
            content_type=content_type,
            size_bytes=int(blob.size or source.stat().st_size),
            sha256=_sha256_file(source),
            local_path=None,
        )

    def put_bytes(self, payload: bytes, object_key: str, content_type: str) -> StoredObject:
        if not payload:
            raise ObjectStorageError("object payload cannot be empty")
        validated = validate_object_key(object_key)
        blob = self.bucket.blob(validated)
        blob.upload_from_string(payload, content_type=content_type)
        blob.reload()
        return StoredObject(
            provider=self.provider,
            object_key=validated,
            content_type=content_type,
            size_bytes=int(blob.size or len(payload)),
            sha256=hashlib.sha256(payload).hexdigest(),
            local_path=None,
        )

    def open(self, object_key: str) -> BinaryIO:
        materialized = self.materialize(object_key)
        if materialized is None:
            raise ObjectStorageError(f"gcs object not found: {object_key}")
        return materialized.open("rb")

    def exists(self, object_key: str) -> bool:
        return self.bucket.blob(validate_object_key(object_key)).exists()

    def delete(self, object_key: str) -> None:
        self.bucket.blob(validate_object_key(object_key)).delete()
        cache_path = self._cache_path_for_key(validate_object_key(object_key))
        cache_path.unlink(missing_ok=True)

    def local_path(self, object_key: str) -> Path | None:
        return self.materialize(object_key)

    def materialize(self, object_key: str) -> Path | None:
        validated = validate_object_key(object_key)
        cache_path = self._cache_path_for_key(validated)
        if cache_path.is_file():
            return cache_path
        blob = self.bucket.blob(validated)
        if not blob.exists():
            return None
        blob.reload()
        expected_size = blob.size
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            blob.download_to_filename(str(temporary))
            actual_size = temporary.stat().st_size
            if expected_size is not None and actual_size != int(expected_size):
                raise ObjectStorageError(
                    f"materialized object size mismatch for {validated}: "
                    f"expected {expected_size}, got {actual_size}"
                )
            temporary.replace(cache_path)
        except ObjectStorageError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise ObjectStorageError(
                f"failed to materialize gcs object {validated}: {exc}"
            ) from exc
        return cache_path

    def _cache_path_for_key(self, object_key: str) -> Path:
        path = (self.cache_root / object_key).resolve()
        if not path.is_relative_to(self.cache_root):
            raise ObjectStorageError("object key escapes cache root")
        return path

    def list_keys(self, prefix: str) -> list[str]:
        normalized = str(prefix or "").strip().replace("\\", "/")
        if not normalized or normalized.startswith("/") or ".." in normalized:
            raise ObjectStorageError("object key prefix is invalid")
        return [blob.name for blob in self.bucket.list_blobs(prefix=normalized)]


def get_object_storage() -> ObjectStorage:
    provider = os.getenv("PFI_STORAGE_PROVIDER", "filesystem").strip().lower()
    if provider in {"", "local", "filesystem"}:
        root = Path(os.getenv("PFI_STORAGE_LOCAL_ROOT", "") or _default_local_root())
        return FilesystemObjectStorage(root)
    if provider == "gcs":
        return GoogleCloudStorageObjectStorage(os.getenv("PFI_STORAGE_GCS_BUCKET", ""))
    raise ObjectStorageError("PFI_STORAGE_PROVIDER must be filesystem or gcs")


def validate_object_key(object_key: str) -> str:
    key = str(object_key or "").strip().replace("\\", "/")
    if not key or key.startswith("/") or "//" in key:
        raise ObjectStorageError("object key is invalid")
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ObjectStorageError("object key is invalid")
    if any(".." in part for part in parts):
        raise ObjectStorageError("object key is invalid")
    return key


def _default_local_root() -> Path:
    from .settings import get_settings

    return get_settings().output_dir / "object_storage"


def _default_gcs_cache_root() -> Path:
    override = os.getenv("PFI_STORAGE_GCS_CACHE_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "pfi-ai-gcs-cache"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
