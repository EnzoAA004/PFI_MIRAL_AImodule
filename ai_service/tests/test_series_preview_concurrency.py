"""Los cortes de una serie se publican completos o no se publican.

Causa observada (traceback de uvicorn: `Response content shorter than Content-Length`): varias
peticiones al mismo corte de una serie todavia no rendida entraban a `render_series_previews` a la
vez. Ninguna esperaba a las demas y cada una escribia los PNG directamente en su ruta final, asi que
una peticion que ya habia terminado servia (FileResponse fija el Content-Length al abrir el archivo)
un corte que otra todavia estaba reescribiendo.

Aqui se fuerza esa ventana haciendo lenta la escritura del PNG (mitad de los bytes, pausa, el resto) y
se comprueba que, vuelva quien vuelva, todos los archivos publicados son PNG completos.
"""
import io
import multiprocessing
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pfi_ai_service.real_inference_runtime as runtime

SLICES = 6
SIDE = 48


class _Loaded:
    def __init__(self, array):
        self.array = array


@pytest.fixture()
def previews(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    volume = rng.integers(0, 4000, size=(SLICES, SIDE, SIDE)).astype(np.int16)
    base = tmp_path / "series_preview"
    monkeypatch.setattr(runtime, "series_preview_dir", lambda input_id: base / input_id)
    monkeypatch.setattr(runtime, "load_input", lambda path, plane: _Loaded(volume))
    return base, volume


def _slow_save(delay=0.03, fail_on=None):
    """`Image.save` que tarda: escribe la mitad, espera y escribe el resto."""
    original = Image.Image.save
    state = {"calls": 0}
    lock = threading.Lock()

    def save(self, fp, *args, **kwargs):
        if isinstance(fp, (str, Path)):
            with lock:
                state["calls"] += 1
                call = state["calls"]
            if fail_on is not None and call == fail_on:
                raise OSError("disco lleno (simulado)")
            buffer = io.BytesIO()
            original(self, buffer, "PNG")
            data = buffer.getvalue()
            with open(fp, "wb") as handle:
                handle.write(data[: len(data) // 2])
                handle.flush()
                time.sleep(delay)
                handle.write(data[len(data) // 2:])
            return None
        return original(self, fp, *args, **kwargs)

    return save, state


def _problems(directory: Path, count: int) -> list[str]:
    found = []
    for index in range(count):
        path = directory / runtime.slice_asset_name(index)
        try:
            with Image.open(io.BytesIO(path.read_bytes())) as image:
                image.load()
                if image.size != (SIDE, SIDE):
                    found.append(f"{path.name}: tamano {image.size}")
        except Exception as error:  # archivo ausente, truncado o ilegible
            found.append(f"{path.name}: {type(error).__name__}")
    return found


def test_peticiones_concurrentes_al_mismo_corte_nunca_ven_un_archivo_incompleto(previews, monkeypatch):
    base, _ = previews
    save, _state = _slow_save()
    monkeypatch.setattr(Image.Image, "save", save)
    outcomes: list[tuple[int, list[str]]] = []
    guard = threading.Lock()

    def request(order):
        # Entradas escalonadas: unas peticiones llegan mientras otra todavia esta rindiendo la serie,
        # que es lo que pasa cuando el visor pide varios cortes seguidos de una serie nueva.
        time.sleep(order * 0.05)
        count = runtime.render_series_previews("inp_x", "ignorado")
        # Lo que hace FileResponse despues: abrir el archivo y leerlo entero.
        problems = _problems(base / "inp_x", count)
        with guard:
            outcomes.append((count, problems))

    threads = [threading.Thread(target=request, args=(order,)) for order in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(outcomes) == 8
    assert {count for count, _ in outcomes} == {SLICES}
    assert [problems for _, problems in outcomes if problems] == []


def _child(base_dir, queue, order):
    time.sleep(order * 0.05)
    # Hereda los parches del padre (fork): mismo directorio y misma escritura lenta.
    count = runtime.render_series_previews("inp_x", "ignorado")
    queue.put((count, _problems(Path(base_dir) / "inp_x", count)))


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requiere fork")
def test_procesos_distintos_tambien_se_coordinan(previews, monkeypatch):
    base, _ = previews
    save, _state = _slow_save()
    monkeypatch.setattr(Image.Image, "save", save)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [context.Process(target=_child, args=(str(base), queue, order)) for order in range(4)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=90) for _ in processes]
    for process in processes:
        process.join(timeout=30)

    assert {count for count, _ in results} == {SLICES}
    assert [problems for _, problems in results if problems] == []


def test_un_fallo_a_mitad_no_deja_un_archivo_parcial_reutilizable(previews, monkeypatch):
    base, _ = previews
    save, state = _slow_save(delay=0, fail_on=3)
    monkeypatch.setattr(Image.Image, "save", save)

    with pytest.raises(OSError):
        runtime.render_series_previews("inp_x", "ignorado")

    # Nada publicado: ni carpeta final, ni marcador, ni restos de preparacion.
    assert not (base / "inp_x").exists()
    assert [item.name for item in base.iterdir() if not item.name.endswith(".lock")] == []

    # El pedido siguiente rehace la serie completa.
    assert runtime.render_series_previews("inp_x", "ignorado") == SLICES
    assert _problems(base / "inp_x", SLICES) == []
    assert (base / "inp_x" / "count").read_text() == str(SLICES)


def test_un_render_viejo_cortado_sin_marcador_se_rehace_en_vez_de_servirse(previews, monkeypatch):
    base, _ = previews
    stale = base / "inp_x"
    stale.mkdir(parents=True)
    (stale / runtime.slice_asset_name(0)).write_bytes(b"\x89PNG parcial")  # sin `count`
    monkeypatch.setattr(Image.Image, "save", _slow_save(delay=0)[0])

    assert runtime.render_series_previews("inp_x", "ignorado") == SLICES
    assert _problems(stale, SLICES) == []


def test_una_serie_ya_publicada_no_se_vuelve_a_rendir(previews, monkeypatch):
    base, _ = previews
    save, state = _slow_save(delay=0)
    monkeypatch.setattr(Image.Image, "save", save)
    assert runtime.render_series_previews("inp_x", "ignorado") == SLICES
    first = state["calls"]
    modified = {path.name: path.stat().st_mtime_ns for path in (base / "inp_x").iterdir()}

    assert runtime.render_series_previews("inp_x", "ignorado") == SLICES
    assert state["calls"] == first
    assert {path.name: path.stat().st_mtime_ns for path in (base / "inp_x").iterdir()} == modified


def test_los_pixeles_publicados_son_los_del_corte_normalizado(previews, monkeypatch):
    base, volume = previews
    runtime.render_series_previews("inp_x", "ignorado")
    for index in range(SLICES):
        expected = np.clip(runtime.robust_percentile_normalize(volume[index]) * 255.0, 0, 255).astype(np.uint8)
        with Image.open(base / "inp_x" / runtime.slice_asset_name(index)) as image:
            assert np.array_equal(np.asarray(image), expected)


def test_el_endpoint_responde_cortes_completos_ante_pedidos_concurrentes(previews, monkeypatch):
    """Lo que fallo en produccion: la respuesta HTTP mas corta que su Content-Length."""
    from fastapi.testclient import TestClient

    import pfi_ai_service.api as api

    base, _ = previews
    monkeypatch.setattr(api, "series_preview_dir", lambda input_id: base / input_id)
    monkeypatch.setattr(api, "resolve_registered_input", lambda input_id: type("R", (), {"input_id": input_id, "path": Path("ignorado")})())
    monkeypatch.setattr(Image.Image, "save", _slow_save()[0])
    results: list[tuple[int, int, str | None, list[str]]] = []
    guard = threading.Lock()

    def request(order):
        time.sleep(order * 0.05)
        index = order % SLICES
        response = TestClient(api.app).get(f"/inputs/inp_x/slices/{index}")
        problems = []
        try:
            with Image.open(io.BytesIO(response.content)) as image:
                image.load()
                if image.size != (SIDE, SIDE):
                    problems.append(f"tamano {image.size}")
        except Exception as error:
            problems.append(type(error).__name__)
        with guard:
            results.append((response.status_code, len(response.content), response.headers.get("content-length"), problems))

    threads = [threading.Thread(target=request, args=(order,)) for order in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(results) == 8
    for status, received, declared, problems in results:
        assert status == 200
        assert declared is not None and received == int(declared)
        assert problems == []
