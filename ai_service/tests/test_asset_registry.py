from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import (
    AssetRegistryError,
    clear_asset_registry,
    registered_assets_for_run,
    register_workspace_asset,
    resolve_run_asset,
    workspace_asset_path,
)
from pfi_ai_service.real_inference_runtime import clear_model_cache


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

