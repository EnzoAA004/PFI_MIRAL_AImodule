"""La secuencia declarada al cargar un archivo viaja hasta la corrida sin confundirse con una deteccion.

Lo que se protege:
- `unknown` es el valor por defecto y no bloquea la carga.
- Una declaracion (`user_declared`) no es una deteccion (`auto_detected`) ni una validacion del modelo.
- `other` exige descripcion; la descripcion no se interpreta nunca como ponderacion.
- Los registros guardados antes de esta version quedan desconocidos, no T2.
"""
import io

import pytest

import pfi_ai_service.input_registry as registry
from pfi_ai_service.input_registry import InputRegistryError, normalize_declared_sequence
from pfi_ai_service.multiplanar_v2_executor import sequence_info_v2, sequence_label, series_v2


@pytest.fixture()
def clean_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "inputs"))
    previous = dict(registry._INPUT_REGISTRY)
    previous_loaded = registry._REGISTRY_LOADED
    registry._INPUT_REGISTRY.clear()
    registry._REGISTRY_LOADED = False
    yield registry
    registry._INPUT_REGISTRY.clear()
    registry._INPUT_REGISTRY.update(previous)
    registry._REGISTRY_LOADED = previous_loaded


def _upload(**kwargs):
    return registry.register_uploaded_input(
        case_id="CASE-1", plane="sagittal", client_filename="vol.mha", stream=io.BytesIO(b"abc"), **kwargs
    )


def _restart(module):
    module._INPUT_REGISTRY.clear()
    module._REGISTRY_LOADED = False


# --- normalizacion -----------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "  ", "unknown", "UNKNOWN"])
def test_sin_declaracion_queda_desconocida_y_sin_procedencia(value):
    assert normalize_declared_sequence(value, None) == ("unknown", "unknown", "")


@pytest.mark.parametrize("value", ["t1", "T1", " t2 "])
def test_t1_y_t2_quedan_declaradas_por_el_usuario(value):
    result = normalize_declared_sequence(value, None)
    assert result[0] in ("t1", "t2") and result[1:] == ("user_declared", "")


def test_otra_exige_descripcion_y_la_conserva():
    assert normalize_declared_sequence("other", "  STIR sagital ") == ("other", "user_declared", "STIR sagital")
    with pytest.raises(InputRegistryError):
        normalize_declared_sequence("other", "   ")
    with pytest.raises(InputRegistryError):
        normalize_declared_sequence("other", None)


@pytest.mark.parametrize("value,text", [("t1", "T1 rapido"), ("t2", "algo"), ("unknown", "texto")])
def test_la_descripcion_no_acompana_a_t1_t2_ni_a_desconocida(value, text):
    # Texto libre junto a una ponderacion: no se acepta, para que nadie lea ahi una segunda ponderacion.
    with pytest.raises(InputRegistryError):
        normalize_declared_sequence(value, text)


@pytest.mark.parametrize("value", ["t3", "flair", "otra", "dixon"])
def test_valores_fuera_de_la_lista_se_rechazan(value):
    with pytest.raises(InputRegistryError) as error:
        normalize_declared_sequence(value, None)
    assert error.value.status_code == 400


def test_descripcion_demasiado_larga_o_con_saltos_de_linea_se_rechaza():
    with pytest.raises(InputRegistryError):
        normalize_declared_sequence("other", "x" * 81)
    with pytest.raises(InputRegistryError):
        normalize_declared_sequence("other", "a\nb")


# --- carga y persistencia ----------------------------------------------------------------------

def test_carga_sin_declaracion_no_se_bloquea_y_queda_desconocida(clean_registry):
    meta = _upload()
    assert meta["sequence"] == {"value": "unknown", "source": "unknown", "description": None}
    record = clean_registry.resolve_registered_input(meta["inputId"])
    assert (record.sequence, record.sequence_source) == ("unknown", "unknown")


def test_carga_declarada_expone_valor_y_procedencia(clean_registry):
    meta = _upload(sequence="t1")
    assert meta["sequence"] == {"value": "t1", "source": "user_declared", "description": None}


def test_una_declaracion_invalida_no_deja_archivo_huerfano(clean_registry, tmp_path):
    with pytest.raises(InputRegistryError):
        _upload(sequence="other", sequence_description="")
    assert clean_registry._INPUT_REGISTRY == {}
    assert not list((tmp_path / "inputs").rglob("*.mha"))


def test_la_secuencia_sobrevive_a_un_reinicio(clean_registry):
    declared = _upload(sequence="other", sequence_description="STIR sagital")
    unknown = _upload()
    _restart(clean_registry)
    first = clean_registry.resolve_registered_input(declared["inputId"])
    second = clean_registry.resolve_registered_input(unknown["inputId"])
    assert (first.sequence, first.sequence_source, first.sequence_description) == ("other", "user_declared", "STIR sagital")
    assert (second.sequence, second.sequence_source) == ("unknown", "unknown")


def test_un_registro_anterior_sin_el_dato_queda_desconocido_no_t2(clean_registry, tmp_path):
    clean_registry.upload_root().mkdir(parents=True, exist_ok=True)
    archivo = tmp_path / "viejo.mha"
    archivo.write_bytes(b"x")
    clean_registry._registry_file().write_text(
        '[{"inputId":"inp_old","caseId":"CASE-1","plane":"sagittal","path":"%s","format":"mha","size":1,'
        '"sourceKey":"upload","analyzable":true,"deidentified":false}]' % str(archivo).replace("\\", "\\\\"),
        encoding="utf-8",
    )
    record = clean_registry.resolve_registered_input("inp_old")
    assert (record.sequence, record.sequence_source) == ("unknown", "unknown")


@pytest.mark.parametrize(
    "stored",
    [
        {"sequence": "t2", "sequenceSource": "unknown"},     # valor sin procedencia
        {"sequence": "unknown", "sequenceSource": "auto_detected"},  # procedencia sin valor
        {"sequence": "flair", "sequenceSource": "user_declared"},    # valor fuera de la lista
    ],
)
def test_un_registro_persistido_incoherente_cae_a_desconocido(stored):
    assert registry._stored_sequence(stored) == {"sequence": "unknown", "sequence_source": "unknown", "sequence_description": ""}


def test_serie_de_estudio_dicom_queda_detectada_y_distinta_de_declarada(clean_registry, monkeypatch, tmp_path):
    # Se evita la de-identificacion real: aqui solo importa lo que se guarda de la ponderacion.
    monkeypatch.setattr(registry, "copy_deidentified", lambda source, destination, remap: destination.write_bytes(b"d"))
    source = tmp_path / "a.dcm"
    source.write_bytes(b"d")
    detected = registry.register_series_files(case_id="CASE-1", plane="sagittal", file_paths=[source], weighting="t2")
    unknown = registry.register_series_files(case_id="CASE-1", plane="sagittal", file_paths=[source], weighting="unknown")
    assert detected["sequence"] == {"value": "t2", "source": "auto_detected", "description": None}
    assert unknown["sequence"] == {"value": "unknown", "source": "unknown", "description": None}


# --- propagacion a la corrida ------------------------------------------------------------------

def _record(sequence, source, description=""):
    return registry.InputRecord(
        input_id="inp_x", case_id="CASE-1", plane="sagittal", path=None, format="mha", size=1, source_key="upload",
        sequence=sequence, sequence_source=source, sequence_description=description,
    )


def test_la_corrida_recibe_la_secuencia_declarada():
    info = sequence_info_v2(_record("t1", "user_declared"))
    assert (info.value, info.source, info.description) == ("t1", "user_declared", None)
    assert sequence_label(info) == "T1"


def test_la_descripcion_de_otra_viaja_solo_para_otra():
    info = sequence_info_v2(_record("other", "user_declared", "STIR"))
    assert (info.value, info.description) == ("other", "STIR")
    assert sequence_label(info) == "STIR"


@pytest.mark.parametrize("record", [None, object(), _record("unknown", "unknown"), _record("t2", "unknown")])
def test_sin_dato_la_corrida_dice_desconocida_y_no_t2(record):
    info = sequence_info_v2(record)
    assert (info.value, info.source) == ("unknown", "unknown")
    assert sequence_label(info) is None


def test_la_serie_de_la_corrida_lleva_la_secuencia_y_no_un_t2_fijo():
    raw = [{"id": "s1", "plane": "sagittal", "sliceCount": 14, "selectedSlice": 7, "status": "real_baseline_ready"}]
    declared = series_v2("sagittal", raw, 7, sequence_info=sequence_info_v2(_record("t1", "user_declared")))[0]
    assert declared.sequence == "T1" and declared.sequenceInfo.source == "user_declared"
    unknown = series_v2("sagittal", raw, 7, sequence_info=sequence_info_v2(None))[0]
    assert unknown.sequence is None and unknown.sequenceInfo.value == "unknown"
