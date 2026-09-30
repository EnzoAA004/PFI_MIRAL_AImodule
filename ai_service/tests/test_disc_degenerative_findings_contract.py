import json

import pytest

from pfi_ai_service.contracts.disc_degenerative_findings import (
    EXPECTED_CHECKPOINT_SHA256,
    SCHEMA_VERSION,
    DiscDegenerativeFindingsContractError,
    level_to_ivd_mapping,
    validate_disc_degenerative_findings_envelope,
)


def valid_envelope() -> dict:
    return {
        "discDegenerativeFindings": {
            "schemaVersion": SCHEMA_VERSION,
            "findings": [
                {
                    "findingId": "opaque-finding",
                    "findingType": "disc_bulging",
                    "anatomy": {"level": "L4-L5", "side": None},
                    "classification": {
                        "kind": "binary",
                        "label": "present",
                        "probabilities": {"absent": 0.2, "present": 0.8},
                    },
                    "evidence": {
                        "deploymentStatus": "supported_internal",
                        "evaluationDataset": "SPIDER_internal_test",
                        "externalValidationAvailable": False,
                    },
                    "evaluation": {"status": "evaluated"},
                    "sourceSeries": [
                        {"role": "sagittal_t1", "available": True, "positions": [10, 11, 12]},
                        {"role": "sagittal_t2", "available": False, "positions": []},
                    ],
                    "localization": {
                        "source": "segmentation_derived_disc_level",
                        "researchOnly": True,
                        "automaticAnatomicalLocalizationValidated": False,
                    },
                    "model": {
                        "modelId": "spider_degenerative_multitask_sagittal_t1_t2_2p5d",
                        "modelSha256": EXPECTED_CHECKPOINT_SHA256,
                    },
                    "review": {"required": True, "status": "pending"},
                    "notClinicalDiagnosis": True,
                }
            ],
        },
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "autonomousDiagnosis": False,
    }


def test_disc_degenerative_contract_accepts_valid_payload() -> None:
    payload = valid_envelope()
    validate_disc_degenerative_findings_envelope(payload)
    text = json.dumps(payload)
    assert "diagnosis" not in text.replace("notClinicalDiagnosis", "")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["discDegenerativeFindings"].update({"schemaVersion": "wrong"}),
        lambda payload: payload.update({"autonomousDiagnosis": True}),
        lambda payload: payload["discDegenerativeFindings"]["findings"][0]["classification"].update({"label": "absent"}),
        lambda payload: payload["discDegenerativeFindings"]["findings"][0]["evidence"].update({"deploymentStatus": "experimental"}),
        lambda payload: payload["discDegenerativeFindings"]["findings"][0].update({"PatientID": "secret"}),
    ],
)
def test_disc_degenerative_contract_rejects_invalid_payloads(mutation) -> None:
    payload = valid_envelope()
    mutation(payload)
    with pytest.raises(DiscDegenerativeFindingsContractError):
        validate_disc_degenerative_findings_envelope(payload)


def test_level_mapping_is_derived_from_spider_ivd_label_order() -> None:
    assert level_to_ivd_mapping("L1-L2").ivd_index == 0
    assert level_to_ivd_mapping("L5-S1").ivd_index == 4
    assert level_to_ivd_mapping("L5-S1").raw_disc_label == 205
    with pytest.raises(ValueError):
        level_to_ivd_mapping("T12-L1")


def region() -> dict:
    return {
        "inputId": "inp-t1",
        "sliceIndex": 11,
        "pointYx": [120.0, 96.0],
        "bboxYx": [108, 132, 84, 108],
        "coordinateSpaceHeight": 256,
        "coordinateSpaceWidth": 200,
    }


def envelope_with_region(**overrides) -> dict:
    payload = valid_envelope()
    series = payload["discDegenerativeFindings"]["findings"][0]["sourceSeries"][0]
    series.update(region())
    series.update(overrides)
    return payload


def test_contract_accepts_the_evaluated_region_of_an_available_series() -> None:
    validate_disc_degenerative_findings_envelope(envelope_with_region())


def test_contract_still_accepts_a_finding_without_any_region() -> None:
    """Una ROI externa no publica geometria: ausente no es invalido."""
    validate_disc_degenerative_findings_envelope(valid_envelope())


@pytest.mark.parametrize(
    "overrides",
    [
        # Media geometria no se puede dibujar: sin espacio de coordenadas el punto
        # no se puede escalar a la imagen que ve el revisor.
        {"coordinateSpaceWidth": None},
        # El centro tiene que ser uno de los cortes que el modelo miro; si no, la
        # geometria y la prediccion vienen de dos preparaciones distintas.
        {"sliceIndex": 40},
        {"bboxYx": [108, 132, 84, 900]},
        {"bboxYx": [132, 108, 84, 108]},
        {"pointYx": [10.0, 96.0]},
        {"pointYx": [120.0, 400.0]},
        {"pointYx": [120.0]},
        {"inputId": "   "},
    ],
)
def test_contract_rejects_a_region_that_could_not_be_drawn(overrides) -> None:
    payload = envelope_with_region()
    series = payload["discDegenerativeFindings"]["findings"][0]["sourceSeries"][0]
    for key, value in overrides.items():
        if value is None:
            series.pop(key)
        else:
            series[key] = value
    with pytest.raises(DiscDegenerativeFindingsContractError):
        validate_disc_degenerative_findings_envelope(payload)


def test_contract_rejects_a_region_on_a_modality_that_was_not_analysed() -> None:
    payload = valid_envelope()
    unavailable = payload["discDegenerativeFindings"]["findings"][0]["sourceSeries"][1]
    unavailable.update(region())
    with pytest.raises(DiscDegenerativeFindingsContractError):
        validate_disc_degenerative_findings_envelope(payload)
