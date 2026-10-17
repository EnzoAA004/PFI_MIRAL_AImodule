from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from urllib.parse import unquote
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .object_storage import ObjectStorageError, get_object_storage

ALLOWED_ASSET_NAMES = frozenset({"input.png", "mask.npy", "confidence.npy", "overlay.png", "mask-preview.png", "lumbar-3d-mesh.json"})
PUBLIC_BROWSER_ASSET_NAMES = frozenset({"input.png", "overlay.png", "mask-preview.png", "lumbar-3d-mesh.json"})
INTERNAL_RAW_ASSET_NAMES = ALLOWED_ASSET_NAMES - PUBLIC_BROWSER_ASSET_NAMES

# Catalogo de previsualizaciones por corte: `slice-007.png`. Es el unico nombre de
# asset que no puede estar en una lista fija, porque la cantidad de cortes depende
# del estudio. El patron es igual de estricto que la lista: solo digitos, sin
# separadores de ruta ni extension arbitraria, de modo que no amplia la superficie
# de path traversal que la lista ya cerraba.
_SLICE_ASSET_PATTERN = re.compile(r"^slice-\d{3,5}\.(png|raw)$")

# Mascara de una clase de segmentacion: `mask-vertebra_group.png`. Tampoco puede ir
# en una lista fija, porque las clases dependen del modelo. El patron acepta solo
# minusculas, digitos y guion bajo -el formato de los class_names del registro- asi
# que no admite separadores de ruta ni extension arbitraria.
_CLASS_MASK_PATTERN = re.compile(r"^mask-[a-z][a-z0-9_]{0,31}\.png$")


def is_slice_asset_name(asset_name: str) -> bool:
    return bool(_SLICE_ASSET_PATTERN.fullmatch(asset_name))


def is_class_mask_asset_name(asset_name: str) -> bool:
    return bool(_CLASS_MASK_PATTERN.fullmatch(asset_name))


def slice_asset_name(index: int) -> str:
    return f"slice-{int(index):03d}.png"


def slice_pixels_asset_name(index: int) -> str:
    return f"slice-{int(index):03d}.raw"


_VALID_PLANES = {"sagittal", "axial", "workspace"}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


class AssetRegistryError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class AssetRecord:
    run_id: str
    plane: Literal["sagittal", "axial", "workspace"]
    asset_name: str
    path: Path
    size: int
    content_type: str = "application/octet-stream"
    object_key: str | None = None
    sha256: str | None = None
    storage_provider: str = "filesystem"


_ASSET_REGISTRY: dict[tuple[str, str, str], AssetRecord] = {}


def clear_asset_registry() -> None:
    _ASSET_REGISTRY.clear()


def register_run_assets(run_id: str, plane: str, outputs: dict[str, str]) -> dict[str, dict[str, object]]:
    normalized_run_id = validate_run_id(run_id)
    normalized_plane = validate_plane(plane)
    registered: dict[str, dict[str, object]] = {}
    for raw_path in outputs.values():
        path = Path(raw_path)
        asset_name = path.name
        if asset_name not in ALLOWED_ASSET_NAMES and not is_slice_asset_name(asset_name) and not is_class_mask_asset_name(asset_name):
            continue
        if not path.exists() or not path.is_file():
            continue
        stored = persist_asset_file(normalized_run_id, normalized_plane, asset_name, path)
        record = AssetRecord(
            run_id=normalized_run_id,
            plane=normalized_plane,  # type: ignore[arg-type]
            asset_name=asset_name,
            path=stored.local_path or path,
            size=stored.size_bytes,
            content_type=stored.content_type,
            object_key=stored.object_key,
            sha256=stored.sha256,
            storage_provider=stored.provider,
        )
        _ASSET_REGISTRY[(normalized_run_id, normalized_plane, asset_name)] = record
        persist_asset_index(record)
        registered[asset_name] = public_asset_metadata(record)
    return registered


def register_workspace_asset(run_id: str, asset_name: str, path: Path) -> dict[str, object] | None:
    normalized_run_id = validate_run_id(run_id)
    normalized_asset = validate_asset_name(asset_name)
    if not path.exists() or not path.is_file():
        return None
    expected_path = workspace_asset_path(normalized_run_id, normalized_asset)
    if path.resolve() != expected_path:
        raise AssetRegistryError("ruta de asset workspace invalida", status_code=400)
    stored = persist_asset_file(normalized_run_id, "workspace", normalized_asset, expected_path)
    record = AssetRecord(
        run_id=normalized_run_id,
        plane="workspace",
        asset_name=normalized_asset,
        path=stored.local_path or expected_path,
        size=stored.size_bytes,
        content_type=stored.content_type,
        object_key=stored.object_key,
        sha256=stored.sha256,
        storage_provider=stored.provider,
    )
    _ASSET_REGISTRY[(normalized_run_id, "workspace", normalized_asset)] = record
    persist_asset_index(record)
    return public_asset_metadata(record)


def resolve_run_asset(run_id: str, plane: str, asset_name: str) -> AssetRecord:
    normalized_plane = validate_plane(plane)
    normalized_asset = validate_asset_name(asset_name)
    normalized_run_id = validate_run_id(run_id)
    record = _ASSET_REGISTRY.get((normalized_run_id, normalized_plane, normalized_asset))
    if record is None:
        record = rehydrate_run_asset(normalized_run_id, normalized_plane, normalized_asset)
    if record is None:
        record = adopt_legacy_run_asset(normalized_run_id, normalized_plane, normalized_asset)
    if record is None and normalized_plane == "workspace" and normalized_asset == "lumbar-3d-mesh.json":
        record = rehydrate_workspace_asset(normalized_run_id, normalized_asset)
    if record is None:
        raise AssetRegistryError("asset no registrado", status_code=404)
    if not record.path.exists() or not record.path.is_file():
        raise AssetRegistryError("archivo de asset no disponible", status_code=404)
    return record


def rehydrate_workspace_asset(run_id: str, asset_name: str) -> AssetRecord | None:
    normalized_run_id = validate_run_id(run_id)
    path = workspace_asset_path(normalized_run_id, asset_name)
    if not path.exists() or not path.is_file():
        return None
    record = AssetRecord(
        run_id=normalized_run_id,
        plane="workspace",
        asset_name=asset_name,
        path=path,
        size=path.stat().st_size,
        content_type=content_type_for_asset(asset_name),
    )
    _ASSET_REGISTRY[(normalized_run_id, "workspace", asset_name)] = record
    return record


# Una corrida hecha antes de que existiera el indice durable no tiene registro, pero sus archivos siguen en
# `{output_dir}/real_inference/{runId}/{plane}/`. Se adoptan al primer pedido: pasan por el mismo camino de
# persistencia que una corrida nueva (copia al storage, sha256 e indice), asi que despues sobreviven a un reinicio
# sin depender de esta ruta. Un lock evita que los pedidos en paralelo de un visor (decenas de cortes) copien el
# mismo archivo a la vez.
_ADOPT_LOCK = threading.Lock()
_LEGACY_PLANES = ("sagittal", "axial")


def legacy_run_asset_path(run_id: str, plane: str, asset_name: str) -> Path | None:
    if plane not in _LEGACY_PLANES:
        return None
    from .settings import get_settings

    root = (get_settings().output_dir / "real_inference").resolve()
    path = (root / run_id / plane / asset_name).resolve()
    if path.parent.parent.parent != root or not path.is_file():
        return None
    return path


def adopt_legacy_run_asset(run_id: str, plane: str, asset_name: str) -> AssetRecord | None:
    source = legacy_run_asset_path(run_id, plane, asset_name)
    if source is None:
        return None
    # Solo se adopta lo que nunca se registro. Un indice ilegible, o un asset que figura registrado pero cuyo
    # archivo durable falta, siguen fallando de forma explicita (ver AI_DURABLE_ASSET_PERSISTENCE.md): no se
    # tapan con una copia del origen.
    index = asset_index_path(run_id, plane)
    if index.exists():
        payload = _read_local_index_payload(index)
        if payload is None:
            return None
        listed = payload.get("assets")
        if isinstance(listed, dict) and asset_name in listed:
            return None
    key = (run_id, plane, asset_name)
    with _ADOPT_LOCK:
        cached = _ASSET_REGISTRY.get(key)
        if cached is not None:
            return cached
        stored = persist_asset_file(run_id, plane, asset_name, source)
        record = AssetRecord(
            run_id=run_id,
            plane=plane,  # type: ignore[arg-type]
            asset_name=asset_name,
            path=stored.local_path or source,
            size=stored.size_bytes,
            content_type=stored.content_type,
            object_key=stored.object_key,
            sha256=stored.sha256,
            storage_provider=stored.provider,
        )
        _ASSET_REGISTRY[key] = record
        persist_asset_index(record)
    return record


def rehydrate_run_asset(run_id: str, plane: str, asset_name: str) -> AssetRecord | None:
    record = load_asset_index_record(run_id, plane, asset_name)
    if record is None:
        return None
    _ASSET_REGISTRY[(record.run_id, record.plane, record.asset_name)] = record
    return record


def validate_run_id(run_id: str) -> str:
    raw = str(run_id or "").strip()
    decoded = raw
    for _ in range(3):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    lowered_values = {raw.lower(), decoded.lower()}
    forbidden = ("..", "/", "\\", "%2e", "%2f", "%5c")
    if not raw or any(token in value for value in lowered_values for token in forbidden):
        raise AssetRegistryError("runId invalido", status_code=403)
    if raw != decoded or not _RUN_ID_PATTERN.fullmatch(decoded):
        raise AssetRegistryError("runId invalido", status_code=403)
    return decoded


def workspace_asset_root() -> Path:
    from .settings import get_settings

    return (get_settings().output_dir / "multiplanar_3d").resolve()


def workspace_asset_path(run_id: str, asset_name: str) -> Path:
    normalized_run_id = validate_run_id(run_id)
    normalized_asset = validate_asset_name(asset_name)
    root = workspace_asset_root()
    path = (root / normalized_run_id / normalized_asset).resolve()
    if path.parent.parent != root:
        raise AssetRegistryError("ruta de asset workspace invalida", status_code=403)
    return path


def public_asset_metadata(record: AssetRecord) -> dict[str, object]:
    metadata: dict[str, object] = {
        "runId": record.run_id,
        "plane": record.plane,
        "assetName": record.asset_name,
        "size": record.size,
    }
    if record.object_key:
        metadata["objectKey"] = record.object_key
        metadata["storageProvider"] = record.storage_provider
        metadata["contentType"] = record.content_type
        metadata["sha256"] = record.sha256 or ""
    return metadata


def registered_assets_for_run(run_id: str, plane: str) -> dict[str, dict[str, object]]:
    normalized_plane = validate_plane(plane)
    normalized_run_id = validate_run_id(run_id)
    rehydrate_assets_for_run(normalized_run_id, normalized_plane)
    return {
        asset_name: public_asset_metadata(record)
        for (stored_run_id, stored_plane, asset_name), record in sorted(_ASSET_REGISTRY.items())
        if stored_run_id == normalized_run_id and stored_plane == normalized_plane
    }


def validate_plane(plane: str) -> str:
    normalized = str(plane).strip().lower()
    if normalized not in _VALID_PLANES:
        raise AssetRegistryError("plane invalido", status_code=400)
    return normalized


def validate_asset_name(asset_name: str) -> str:
    normalized = str(asset_name).strip()
    if normalized != Path(normalized).name or "/" in normalized or "\\" in normalized or ".." in normalized:
        raise AssetRegistryError("assetName invalido", status_code=403)
    if normalized not in ALLOWED_ASSET_NAMES and not is_slice_asset_name(normalized) and not is_class_mask_asset_name(normalized):
        raise AssetRegistryError("assetName no permitido", status_code=403)
    return normalized


def is_public_browser_asset(asset_name: str) -> bool:
    normalized = validate_asset_name(asset_name)
    return normalized in PUBLIC_BROWSER_ASSET_NAMES or is_slice_asset_name(normalized) or is_class_mask_asset_name(normalized)


def persist_asset_file(run_id: str, plane: str, asset_name: str, path: Path):
    object_key = asset_object_key(run_id, plane, asset_name)
    try:
        return get_object_storage().put_file(path, object_key, content_type_for_asset(asset_name))
    except ObjectStorageError as exc:
        raise AssetRegistryError(exc.message, status_code=500) from exc


def asset_object_key(run_id: str, plane: str, asset_name: str) -> str:
    normalized_run_id = validate_run_id(run_id)
    normalized_plane = validate_plane(plane)
    normalized_asset = validate_asset_name(asset_name)
    return f"runs/{normalized_run_id}/{normalized_plane}/{normalized_asset}"


def asset_index_root() -> Path:
    from .settings import get_settings

    return (get_settings().output_dir / "asset_registry").resolve()


def asset_index_path(run_id: str, plane: str) -> Path:
    normalized_run_id = validate_run_id(run_id)
    normalized_plane = validate_plane(plane)
    root = asset_index_root()
    path = (root / normalized_run_id / f"{normalized_plane}.json").resolve()
    if path.parent.parent != root:
        raise AssetRegistryError("ruta de indice de asset invalida", status_code=403)
    return path


def asset_meta_object_key(run_id: str, plane: str, asset_name: str) -> str:
    """Key for one asset's durable metadata (objectKey/sha256/size/...).

    Deliberately one immutable object per asset rather than one shared,
    repeatedly-overwritten index per run/plane: the AI runtime service
    account only has storage.objectCreator (create) + objectViewer
    (get/list) on the durable bucket, not update/delete, so any object that
    would need overwriting would 403. Each (run_id, plane, asset_name)
    combination is written exactly once in normal operation, which fits
    create-only permissions.
    """
    normalized_run_id = validate_run_id(run_id)
    normalized_plane = validate_plane(plane)
    normalized_asset = validate_asset_name(asset_name)
    return f"runs/{normalized_run_id}/{normalized_plane}/_meta/{normalized_asset}.json"


def _read_local_index_payload(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


_INDEX_LOCK = threading.Lock()


def _merge_entry_into_local_index(path: Path, run_id: str, plane: str, entry: dict) -> None:
    with _INDEX_LOCK:
        _merge_entry_into_local_index_locked(path, run_id, plane, entry)


def _merge_entry_into_local_index_locked(path: Path, run_id: str, plane: str, entry: dict) -> None:
    existing = _read_local_index_payload(path) or {
        "schemaVersion": "pfi.ai.asset-registry.v1",
        "runId": run_id,
        "plane": plane,
        "assets": {},
    }
    assets = existing.get("assets")
    if not isinstance(assets, dict):
        assets = {}
    asset_name = entry.get("assetName")
    if isinstance(asset_name, str):
        assets[asset_name] = entry
    existing["assets"] = assets
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(existing, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def persist_asset_index(record: AssetRecord) -> None:
    entry = {
        "runId": record.run_id,
        "plane": record.plane,
        "assetName": record.asset_name,
        "objectKey": record.object_key,
        "contentType": record.content_type,
        "sizeBytes": record.size,
        "sha256": record.sha256,
        "storageProvider": record.storage_provider,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    path = asset_index_path(record.run_id, record.plane)
    _merge_entry_into_local_index(path, record.run_id, record.plane, entry)
    if not path.exists():
        raise AssetRegistryError("no se pudo persistir indice de asset", status_code=500)

    # El indice local vive en almacenamiento efimero del contenedor (PFI_OUTPUT_DIR),
    # aunque el blob del asset ya sea durable via GCS: sin esta copia adicional, un
    # proceso nuevo no tiene forma de redescubrir el object_key/checksum de un asset
    # ya escrito, aunque el blob siga existiendo. Se sube la metadata de ESTE asset a
    # su propia clave inmutable (nunca se reescribe la misma clave dos veces) para
    # que un proceso fresco pueda reconstruirla sin requerir permiso de sobrescritura.
    if record.storage_provider == "gcs":
        try:
            storage = get_object_storage()
        except ObjectStorageError as exc:
            raise AssetRegistryError(exc.message, status_code=500) from exc
        meta_key = asset_meta_object_key(record.run_id, record.plane, record.asset_name)
        try:
            storage.put_bytes(
                json.dumps(entry, sort_keys=True).encode("utf-8"), meta_key, "application/json"
            )
        except ObjectStorageError as exc:
            raise AssetRegistryError(
                f"no se pudo persistir metadata de asset en almacenamiento durable: {exc.message}",
                status_code=500,
            ) from exc


def load_asset_index_record(run_id: str, plane: str, asset_name: str) -> AssetRecord | None:
    path = asset_index_path(run_id, plane)
    raw = _read_local_index_payload(path)
    assets = raw.get("assets") if isinstance(raw, dict) else None
    item = assets.get(asset_name) if isinstance(assets, dict) else None

    if not isinstance(item, dict):
        # Cache local vacio o asset ausente de el (ej. proceso nuevo tras
        # restart/scale-to-zero): intentar reconstruir SOLO la metadata de
        # este asset desde su copia durable individual en GCS.
        try:
            storage = get_object_storage()
        except ObjectStorageError as exc:
            raise AssetRegistryError(exc.message, status_code=500) from exc
        if storage.provider != "gcs":
            return None
        meta_key = asset_meta_object_key(run_id, plane, asset_name)
        materialized = storage.local_path(meta_key)
        if materialized is None:
            return None
        try:
            item = json.loads(materialized.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(item, dict):
            return None
        _merge_entry_into_local_index(path, run_id, plane, item)

    object_key = item.get("objectKey")
    if not isinstance(object_key, str) or not object_key.strip():
        return None
    try:
        storage = get_object_storage()
    except ObjectStorageError as exc:
        raise AssetRegistryError(exc.message, status_code=500) from exc
    local_path = storage.local_path(object_key)
    if local_path is None or not local_path.is_file():
        return None
    expected_sha256 = str(item.get("sha256", ""))
    if expected_sha256 and _sha256_file(local_path) != expected_sha256:
        raise AssetRegistryError("checksum de asset no coincide", status_code=500)
    return AssetRecord(
        run_id=validate_run_id(str(item.get("runId", run_id))),
        plane=validate_plane(str(item.get("plane", plane))),  # type: ignore[arg-type]
        asset_name=validate_asset_name(str(item.get("assetName", asset_name))),
        path=local_path,
        size=int(item.get("sizeBytes", local_path.stat().st_size) or local_path.stat().st_size),
        content_type=str(item.get("contentType", content_type_for_asset(asset_name))),
        object_key=object_key,
        sha256=expected_sha256,
        storage_provider=str(item.get("storageProvider", "filesystem")),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rehydrate_assets_for_run(run_id: str, plane: str) -> None:
    normalized_run_id = validate_run_id(run_id)
    normalized_plane = validate_plane(plane)
    path = asset_index_path(normalized_run_id, normalized_plane)
    raw = _read_local_index_payload(path)
    asset_names: set[str] = set()
    if raw is not None:
        assets = raw.get("assets")
        if isinstance(assets, dict):
            asset_names.update(name for name in assets if isinstance(name, str))

    try:
        storage = get_object_storage()
    except ObjectStorageError:
        storage = None
    if storage is not None and storage.provider == "gcs":
        # Proceso nuevo sin indice local: listar las claves de metadata
        # individuales de cada asset (list requiere solo objectViewer, ya
        # concedido) en vez de depender de un indice combinado.
        prefix = f"runs/{normalized_run_id}/{normalized_plane}/_meta/"
        try:
            remote_keys = storage.list_keys(prefix)
        except ObjectStorageError:
            remote_keys = []
        for key in remote_keys:
            suffix = key[len(prefix):] if key.startswith(prefix) else ""
            if suffix.endswith(".json"):
                asset_names.add(suffix[: -len(".json")])

    for name in asset_names:
        try:
            record = load_asset_index_record(normalized_run_id, normalized_plane, name)
        except AssetRegistryError:
            continue
        if record is not None:
            _ASSET_REGISTRY[(record.run_id, record.plane, record.asset_name)] = record


def content_type_for_asset(asset_name: str) -> str:
    normalized = validate_asset_name(asset_name)
    if normalized.endswith(".png"):
        return "image/png"
    if normalized.endswith(".json"):
        return "application/json"
    if normalized.endswith(".npy") or normalized.endswith(".raw"):
        return "application/octet-stream"
    return "application/octet-stream"
