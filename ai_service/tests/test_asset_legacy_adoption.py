"""Los assets de una corrida hecha antes del indice durable se recuperan de disco y quedan indexados.

Caso real: las corridas anteriores al indice (y al reinicio del contenedor) tenian sus archivos en
`{output_dir}/real_inference/{runId}/{plane}/` pero ningun registro: el registro vivia solo en memoria y
`GET /assets/...` respondia 404 con los archivos intactos.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import (
    AssetRegistryError,
    clear_asset_registry,
    resolve_run_asset,
)

RUN = "0123456789abcdef"
SLICES = 12


@pytest.fixture()
def legacy_run(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(out))
    clear_asset_registry()
    plane_dir = out / "real_inference" / RUN / "sagittal"
    plane_dir.mkdir(parents=True)
    files = {"input.png": b"\x89PNG-input", "overlay.png": b"\x89PNG-overlay", "mask-preview.png": b"\x89PNG-prev",
             "mask-disc_group.png": b"\x89PNG-disc", "mask.npy": b"npy-interno", "confidence.npy": b"npy-conf"}
    for index in range(SLICES):
        files[f"slice-{index:03d}.png"] = b"\x89PNG-slice-%d" % index
        files[f"slice-{index:03d}.raw"] = (b"\x01\x02" * 64) + bytes([index])
    for name, data in files.items():
        (plane_dir / name).write_bytes(data)
    yield out, plane_dir, files
    clear_asset_registry()


def test_una_corrida_sin_indice_se_recupera_de_disco(legacy_run):
    out, plane_dir, files = legacy_run
    assert not (out / "asset_registry").exists()
    record = resolve_run_asset(RUN, "sagittal", "overlay.png")
    assert record.path.read_bytes() == files["overlay.png"]
    assert record.sha256 == hashlib.sha256(files["overlay.png"]).hexdigest()
    # Quedo indexada para despues del reinicio.
    assert (out / "asset_registry" / RUN / "sagittal.json").is_file()


def test_tras_un_reinicio_se_sirve_desde_el_indice_aunque_el_origen_ya_no_este(legacy_run):
    out, plane_dir, files = legacy_run
    resolve_run_asset(RUN, "sagittal", "slice-003.raw")
    clear_asset_registry()  # reinicio del proceso: se pierde el diccionario
    for path in plane_dir.iterdir():
        path.unlink()
    record = resolve_run_asset(RUN, "sagittal", "slice-003.raw")
    assert record.path.read_bytes() == files["slice-003.raw"]


def test_el_original_no_se_modifica(legacy_run):
    out, plane_dir, files = legacy_run
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in plane_dir.iterdir()}
    for name in ("input.png", "slice-000.raw", "mask-disc_group.png"):
        resolve_run_asset(RUN, "sagittal", name)
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in plane_dir.iterdir()} == before


def test_lo_que_no_existe_sigue_siendo_404(legacy_run):
    with pytest.raises(AssetRegistryError) as missing:
        resolve_run_asset(RUN, "sagittal", "slice-099.raw")
    assert missing.value.status_code == 404
    with pytest.raises(AssetRegistryError) as other_plane:
        resolve_run_asset(RUN, "axial", "overlay.png")
    assert other_plane.value.status_code == 404
    with pytest.raises(AssetRegistryError) as other_run:
        resolve_run_asset("fedcba9876543210", "sagittal", "overlay.png")
    assert other_run.value.status_code == 404


def test_no_se_adopta_nada_fuera_de_la_carpeta_de_corridas(legacy_run):
    out, plane_dir, files = legacy_run
    (out / "secreto.png").write_bytes(b"x")
    for run_id in ("..", "../..", "a/b"):
        with pytest.raises(AssetRegistryError) as bad:
            resolve_run_asset(run_id, "sagittal", "overlay.png")
        assert bad.value.status_code in (403, 404)
    for name in ("../secreto.png", "..\\secreto.png", "secreto.png"):
        with pytest.raises(AssetRegistryError) as bad:
            resolve_run_asset(RUN, "sagittal", name)
        assert bad.value.status_code in (403, 404)


def test_pedidos_en_paralelo_no_pierden_ni_corrompen_assets(legacy_run):
    out, plane_dir, files = legacy_run
    names = [name for name in files if name.startswith("slice-")]
    errors: list[str] = []

    def fetch(name: str) -> None:
        try:
            record = resolve_run_asset(RUN, "sagittal", name)
            if record.path.read_bytes() != files[name]:
                errors.append(f"contenido distinto en {name}")
        except Exception as error:  # noqa: BLE001
            errors.append(f"{name}: {error!r}")

    threads = [threading.Thread(target=fetch, args=(name,)) for name in names for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    # Despues del reinicio estan todos, no solo los ultimos en escribir el indice.
    clear_asset_registry()
    for path in plane_dir.iterdir():
        path.unlink()
    for name in names:
        assert resolve_run_asset(RUN, "sagittal", name).path.read_bytes() == files[name]


def test_el_endpoint_sirve_la_corrida_heredada_y_mantiene_lo_interno_privado(legacy_run):
    out, plane_dir, files = legacy_run
    client = TestClient(app)
    ok = client.get(f"/assets/{RUN}/sagittal/slice-005.raw")
    assert ok.status_code == 200 and ok.content == files["slice-005.raw"]
    assert ok.headers["content-type"].startswith("application/octet-stream")
    assert client.get(f"/assets/{RUN}/sagittal/mask-disc_group.png").status_code == 200
    # mask.npy / confidence.npy no son publicos: se adoptan pero el endpoint los rechaza igual que antes.
    assert client.get(f"/assets/{RUN}/sagittal/mask.npy").status_code == 403
    assert client.get(f"/assets/{RUN}/sagittal/slice-077.raw").status_code == 404


def test_un_indice_corrupto_o_un_asset_registrado_sin_archivo_siguen_fallando_explicitamente(legacy_run):
    out, plane_dir, files = legacy_run
    resolve_run_asset(RUN, "sagittal", "overlay.png")
    clear_asset_registry()
    # Registrado, pero su copia durable ya no esta: no se repone en silencio desde el origen.
    for durable in (out / "object_storage" / "runs" / RUN / "sagittal").glob("overlay.png"):
        durable.unlink()
    with pytest.raises(AssetRegistryError) as gone:
        resolve_run_asset(RUN, "sagittal", "overlay.png")
    assert gone.value.status_code == 404
    # Indice ilegible: tampoco se adopta nada de esa corrida.
    (out / "asset_registry" / RUN / "sagittal.json").write_text("{no es json")
    with pytest.raises(AssetRegistryError) as corrupt:
        resolve_run_asset(RUN, "sagittal", "input.png")
    assert corrupt.value.status_code == 404
