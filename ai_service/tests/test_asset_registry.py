from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pfi_ai_service import asset_registry as asset_registry_module
from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import (
    AssetRegistryError,
    clear_asset_registry,
    register_run_assets,
    registered_assets_for_run,
    register_workspace_asset,
    resolve_run_asset,
    workspace_asset_path,
)
from pfi_ai_service.object_storage import GoogleCloudStorageObjectStorage
from pfi_ai_service.real_inference_runtime import clear_model_cache


class _FakeGcsBlob:
    """Same fake used in test_object_storage.py, duplicated to keep each test
    module self-contained (no cross-module test imports)."""

    def __init__(self, backing: dict[str, bytes], key: str) -> None:
        self._backing = backing
        self._key = key
        self.size: int | None = None

    def exists(self) -> bool:
        return self._key in self._backing

    def reload(self) -> None:
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


class _FakeGcsBucket:
    def __init__(self, backing: dict[str, bytes]) -> None:
        self._backing = backing

    def blob(self, key: str) -> _FakeGcsBlob:
        return _FakeGcsBlob(self._backing, key)


def _make_fake_gcs_storage(cache_root: Path, backing: dict[str, bytes]) -> GoogleCloudStorageObjectStorage:
    storage = object.__new__(GoogleCloudStorageObjectStorage)
    storage.bucket_name = "fake-bucket"
    storage.client = None
    storage.bucket = _FakeGcsBucket(backing)
    storage.cache_root = cache_root.resolve()
    storage.cache_root.mkdir(parents=True, exist_ok=True)
    return storage


def _write_via_simulated_process(
    monkeypatch,
    tmp_path: Path,
    process_label: str,
    backing: dict[str, bytes],
    run_id: str,
    plane: str,
    assets: dict[str, bytes],
) -> dict:
    """Simulates one Cloud Run instance: its own PFI_OUTPUT_DIR (ephemeral local
    disk) and its own GCS local materialization cache, sharing only `backing`
    (the durable GCS bucket) with any other simulated process. Mirrors the
    real register_run_assets() write path so the durable-index-upload fix is
    exercised exactly as production does it, not re-implemented ad hoc."""
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / f"outputs-{process_label}"))
    storage = _make_fake_gcs_storage(tmp_path / f"gcs-cache-{process_label}", backing)
    monkeypatch.setattr(asset_registry_module, "get_object_storage", lambda: storage)
    clear_asset_registry()

    outputs: dict[str, str] = {}
    source_dir = tmp_path / f"sources-{process_label}"
    source_dir.mkdir(parents=True, exist_ok=True)
    for asset_name, payload in assets.items():
        # register_run_assets matches on the file basename, not the dict key,
        # so each source file must be literally named like the asset it
        # represents (e.g. overlay.png), not just keyed by it.
        source = source_dir / asset_name
        source.write_bytes(payload)
        outputs[asset_name] = str(source)
    return register_run_assets(run_id, plane, outputs)


def _read_via_simulated_process(
    monkeypatch,
    tmp_path: Path,
    process_label: str,
    backing: dict[str, bytes],
) -> None:
    """Points the module at a brand new PFI_OUTPUT_DIR and a brand new GCS
    local cache, with an empty in-memory registry: nothing this process has
    not just fetched from `backing` is visible to it, simulating a real
    restart/scale-to-zero/fresh replica."""
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / f"outputs-{process_label}"))
    storage = _make_fake_gcs_storage(tmp_path / f"gcs-cache-{process_label}", backing)
    monkeypatch.setattr(asset_registry_module, "get_object_storage", lambda: storage)
    clear_asset_registry()


def run_sagittal_fixture(monkeypatch, tmp_path) -> dict:
    fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    assert fixture.exists()
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_asset_registry()
    clear_model_cache()
    response = TestClient(app).post(
        "/pipeline/run",
        json={
            "caseId": "CASE-AI015-ASSET",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputPath": str(fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai015-assets",
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def assert_public_assets_have_no_paths(assets: dict) -> None:
    text = str(assets)
    assert "outputs/" not in text
    assert "outputs\\" not in text
    assert "ai_service/tests/fixtures" not in text


def test_asset_registry_registers_real_run_assets(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)
    run_id = body["runId"]
    assets = body["assets"]

    # Las tres mascaras por clase de sagittal_spider se suman a los assets fijos: es
    # lo que permite apagar una estructura sin apagar las demas. Se listan y no se
    # derivan del registro, porque son las clases de un artefacto congelado.
    assert set(assets) == {
        "input.png",
        "mask.npy",
        "confidence.npy",
        "overlay.png",
        "mask-preview.png",
        "mask-canal.png",
        "mask-disc_group.png",
        "mask-vertebra_group.png",
    }
    assert_public_assets_have_no_paths(assets)
    for asset_name, metadata in assets.items():
        assert metadata["runId"] == run_id
        assert metadata["plane"] == "sagittal"
        assert metadata["assetName"] == asset_name
        assert metadata["size"] > 0


def test_asset_registry_resolves_allowed_asset(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)
    record = resolve_run_asset(body["runId"], "sagittal", "overlay.png")

    assert record.asset_name == "overlay.png"
    assert record.plane == "sagittal"
    assert record.path.name == "overlay.png"
    assert record.path.exists()
    assert record.size > 0


def test_asset_registry_rehydrates_plane_asset_after_registry_clear(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)
    run_id = body["runId"]
    clear_asset_registry()

    record = resolve_run_asset(run_id, "sagittal", "overlay.png")
    assets = registered_assets_for_run(run_id, "sagittal")

    assert record.asset_name == "overlay.png"
    assert record.path.exists()
    assert record.object_key == f"runs/{run_id}/sagittal/overlay.png"
    assert assets["overlay.png"]["sha256"]
    assert assets["overlay.png"]["storageProvider"] == "filesystem"


@pytest.mark.parametrize("asset_name", ["report.json", "../overlay.png", "sagittal/overlay.png", "overlay.png/evil"])
def test_asset_registry_rejects_disallowed_or_traversal_asset(monkeypatch, tmp_path, asset_name: str) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset(body["runId"], "sagittal", asset_name)

    assert exc_info.value.status_code == 403


def test_asset_registry_rejects_unknown_run_id(monkeypatch, tmp_path) -> None:
    run_sagittal_fixture(monkeypatch, tmp_path)

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset("missing-run", "sagittal", "input.png")

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == "asset no registrado"


def test_workspace_asset_valid_run_id_rehydrates_after_registry_clear(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    run_id = "multi-safe_run-123"
    path = workspace_asset_path(run_id, "lumbar-3d-mesh.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"ok": true}', encoding="utf-8")

    metadata = register_workspace_asset(run_id, "lumbar-3d-mesh.json", path)
    assert metadata is not None
    clear_asset_registry()

    record = resolve_run_asset(run_id, "workspace", "lumbar-3d-mesh.json")
    assert record.path.exists()
    assert record.path.resolve().is_relative_to((tmp_path / "outputs").resolve())


@pytest.mark.parametrize("run_id", ["../evil", "multi/evil", "multi\\evil", "multi%2e%2e", "multi%2fslash", "multi%5cslash", "", "x" * 120])
def test_workspace_asset_rejects_invalid_run_id_without_path_leak(monkeypatch, tmp_path, run_id: str) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset(run_id, "workspace", "lumbar-3d-mesh.json")

    assert exc_info.value.status_code == 403
    assert exc_info.value.message == "runId invalido"
    assert str(tmp_path) not in exc_info.value.message


def test_workspace_asset_path_cannot_escape_multiplanar_3d(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))

    with pytest.raises(AssetRegistryError):
        workspace_asset_path("..%2foutside", "lumbar-3d-mesh.json")


def _asset_index_path(tmp_path: Path, run_id: str, plane: str) -> Path:
    return tmp_path / "outputs" / "asset_registry" / run_id / f"{plane}.json"


def test_asset_registry_rehydrate_fails_on_corrupted_metadata(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)
    run_id = body["runId"]
    clear_asset_registry()
    index_path = _asset_index_path(tmp_path, run_id, "sagittal")
    assert index_path.exists()
    index_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset(run_id, "sagittal", "overlay.png")

    assert exc_info.value.status_code == 404


def test_asset_registry_rehydrate_fails_when_durable_file_missing(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)
    run_id = body["runId"]
    clear_asset_registry()
    object_path = tmp_path / "outputs" / "object_storage" / "runs" / run_id / "sagittal" / "overlay.png"
    assert object_path.exists()
    object_path.unlink()

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset(run_id, "sagittal", "overlay.png")

    assert exc_info.value.status_code == 404


def test_asset_registry_rehydrate_fails_clearly_on_checksum_mismatch(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)
    run_id = body["runId"]
    clear_asset_registry()
    object_path = tmp_path / "outputs" / "object_storage" / "runs" / run_id / "sagittal" / "overlay.png"
    assert object_path.exists()
    object_path.write_bytes(b"corrupted-bytes-do-not-match-checksum")

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset(run_id, "sagittal", "overlay.png")

    assert exc_info.value.status_code == 500
    assert "checksum" in exc_info.value.message


def test_asset_registry_rejects_misconfigured_gcs_provider_on_persist(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv("PFI_STORAGE_PROVIDER", "gcs")
    monkeypatch.delenv("PFI_STORAGE_GCS_BUCKET", raising=False)
    clear_asset_registry()
    clear_model_cache()

    fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    assert fixture.exists()
    with pytest.raises(AssetRegistryError) as exc_info:
        TestClient(app, raise_server_exceptions=True).post(
            "/pipeline/run",
            json={
                "caseId": "CASE-AI015-ASSET-GCS-MISCONFIG",
                "plane": "sagittal",
                "modelKey": "sagittal_spider",
                "inputPath": str(fixture),
                "metadata": {
                    "inferenceMode": "real_baseline",
                    "allowContractFallback": False,
                    "traceId": "trace-ai015-assets-gcs-misconfig",
                },
            },
        )

    assert exc_info.value.status_code == 500
    assert "PFI_STORAGE_GCS_BUCKET" in exc_info.value.message




def test_asset_registry_rehydrates_gcs_backed_asset_in_fresh_process(monkeypatch, tmp_path) -> None:
    backing: dict[str, bytes] = {}
    run_id = "run-fresh-gcs-001"
    plane = "sagittal"
    payload = b"fake-overlay-bytes-for-gcs-rehydration-test"

    registered = _write_via_simulated_process(
        monkeypatch, tmp_path, "writer", backing, run_id, plane, {"overlay.png": payload}
    )
    assert registered["overlay.png"]["storageProvider"] == "gcs"

    # "reinicio real": PFI_OUTPUT_DIR nuevo, cache GCS local nueva, registro en
    # memoria vacio. Solo se comparte `backing` (el bucket durable).
    _read_via_simulated_process(monkeypatch, tmp_path, "reader", backing)

    record = resolve_run_asset(run_id, plane, "overlay.png")

    assert record.path.exists()
    assert record.path.read_bytes() == payload
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.size == len(payload)


def test_asset_registry_rehydrates_multiple_gcs_backed_assets_in_fresh_process(monkeypatch, tmp_path) -> None:
    backing: dict[str, bytes] = {}
    run_id = "run-fresh-gcs-multi-002"
    plane = "sagittal"
    overlay_payload = b"overlay-bytes-multi-asset-test"
    mask_payload = b"\x00\x01\x02mask-npy-bytes-multi-asset-test"

    _write_via_simulated_process(
        monkeypatch,
        tmp_path,
        "writer",
        backing,
        run_id,
        plane,
        {"overlay.png": overlay_payload, "mask.npy": mask_payload},
    )

    _read_via_simulated_process(monkeypatch, tmp_path, "reader", backing)

    overlay_record = resolve_run_asset(run_id, plane, "overlay.png")
    mask_record = resolve_run_asset(run_id, plane, "mask.npy")

    assert overlay_record.path.read_bytes() == overlay_payload
    assert mask_record.path.read_bytes() == mask_payload
    assert overlay_record.sha256 == hashlib.sha256(overlay_payload).hexdigest()
    assert mask_record.sha256 == hashlib.sha256(mask_payload).hexdigest()


def test_asset_registry_second_gcs_read_reuses_materialized_cache(monkeypatch, tmp_path) -> None:
    backing: dict[str, bytes] = {}
    run_id = "run-fresh-gcs-cache-003"
    plane = "sagittal"
    payload = b"cache-reuse-bytes"

    _write_via_simulated_process(
        monkeypatch, tmp_path, "writer", backing, run_id, plane, {"overlay.png": payload}
    )
    _read_via_simulated_process(monkeypatch, tmp_path, "reader", backing)

    first = resolve_run_asset(run_id, plane, "overlay.png")
    # Segunda lectura sin reinicio: debe resolverse desde la cache local /
    # registro en memoria ya poblado, sin error aunque el backing se vacie.
    backing.clear()
    second = resolve_run_asset(run_id, plane, "overlay.png")

    assert first.path == second.path
    assert second.path.read_bytes() == payload


def test_asset_registry_gcs_rehydration_fails_clearly_on_checksum_mismatch(monkeypatch, tmp_path) -> None:
    backing: dict[str, bytes] = {}
    run_id = "run-fresh-gcs-corrupt-004"
    plane = "sagittal"
    payload = b"original-clean-bytes"

    _write_via_simulated_process(
        monkeypatch, tmp_path, "writer", backing, run_id, plane, {"overlay.png": payload}
    )

    # Corrompe el objeto remoto (mismo tamano, contenido distinto) entre la
    # escritura y la lectura del proceso nuevo.
    object_key = f"runs/{run_id}/{plane}/overlay.png"
    corrupted = payload[::-1]
    assert corrupted != payload
    backing[object_key] = corrupted

    _read_via_simulated_process(monkeypatch, tmp_path, "reader", backing)

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset(run_id, plane, "overlay.png")

    assert exc_info.value.status_code == 500
    assert "checksum" in exc_info.value.message


def test_asset_registry_gcs_rehydration_returns_404_when_object_missing_from_bucket(monkeypatch, tmp_path) -> None:
    backing: dict[str, bytes] = {}
    run_id = "run-fresh-gcs-missing-005"
    plane = "sagittal"
    payload = b"will-be-deleted-bytes"

    _write_via_simulated_process(
        monkeypatch, tmp_path, "writer", backing, run_id, plane, {"overlay.png": payload}
    )

    # El indice sigue durable, pero el blob fue borrado del bucket.
    object_key = f"runs/{run_id}/{plane}/overlay.png"
    del backing[object_key]

    _read_via_simulated_process(monkeypatch, tmp_path, "reader", backing)

    with pytest.raises(AssetRegistryError) as exc_info:
        resolve_run_asset(run_id, plane, "overlay.png")

    assert exc_info.value.status_code == 404
