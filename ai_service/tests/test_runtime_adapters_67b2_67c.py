from __future__ import annotations

import math

from pfi_ai_service.absolute_disc_localization_runtime import (
    MODEL_SHA256,
    localize_absolute_disc_levels,
    runtime_status as absolute_runtime_status,
)
from pfi_ai_service.axial_level_pairing_runtime import (
    MAX_PLANE_DISTANCE_MM,
    MIN_MARGIN,
    AxialPlane,
    candidate_geometry,
    pair_disc_to_axial_slice,
    runtime_status as pairing_runtime_status,
)
from pfi_ai_service.product_orchestration import ProductAnalyzeRequest, run_product_analysis
from test_product_orchestration import FakePipelineRunRequest, FakeInputRegistryError


def plane(
    *,
    z: float,
    instance: int,
    physical_index: int | None = None,
    slice_location: float | None = None,
    row_cosines: tuple[float, float, float] = (1.0, 0.0, 0.0),
    column_cosines: tuple[float, float, float] = (0.0, 1.0, 0.0),
    rows: int = 10,
    columns: int = 10,
) -> AxialPlane:
    return AxialPlane(
        seriesId="axial-a",
        physicalIndex=physical_index,
        instanceNumber=instance,
        sliceLocation=slice_location,
        imagePositionPatient=(0.0, 0.0, z),
        rowCosines=row_cosines,
        columnCosines=column_cosines,
        pixelSpacing=(1.0, 1.0),
        rows=rows,
        columns=columns,
    )


def assert_governance(payload: dict) -> None:
    assert payload["validated"] is False
    assert payload["experimental"] is True
    assert payload["reviewRequired"] is True


def test_67b2_localized_valid_output_when_parity_verified() -> None:
    result = localize_absolute_disc_levels(
        {"status": "localized", "level": "L4-L5", "confidence": 0.99},
        parity_verified=True,
    )

    assert result["status"] == "localized"
    assert result["level"] == "L4-L5"
    assert result["modelSha256"] == MODEL_SHA256


def test_67b2_abstention_preserved_when_parity_verified() -> None:
    result = localize_absolute_disc_levels(
        {"status": "abstained", "level": "L4-L5", "confidence": 0.4},
        parity_verified=True,
    )

    assert result["status"] == "abstained"
    assert result["level"] is None
    assert result["abstentionPreserved"] is True


def test_67b2_governance_fields_do_not_become_prediction_features() -> None:
    result = localize_absolute_disc_levels(
        {
            "status": "localized",
            "level": "L5-S1",
            "confidence": 0.91,
            "source": "67B2",
            "parityVerified": True,
        },
        parity_verified=True,
    )

    assert result["status"] == "localized"
    assert result["level"] == "L5-S1"
    assert "parityVerified" not in result


def test_67b2_unavailable_is_not_abstained_without_parity() -> None:
    result = localize_absolute_disc_levels(
        {"status": "localized", "level": "L4-L5", "confidence": 0.99},
        parity_verified=False,
    )

    assert result["status"] == "unavailable"
    assert result["level"] is None
    assert result["abstentionPreserved"] is False
    assert "67B2_RUNTIME_PARITY_NOT_PROVEN" in result["warnings"]


def test_67b2_invalid_level_does_not_fallback_to_best_guess() -> None:
    result = localize_absolute_disc_levels(
        {"status": "localized", "level": "T12-L1", "confidence": 0.99},
        parity_verified=True,
    )

    assert result["status"] == "unavailable"
    assert result["level"] is None
    assert "67B2_FROZEN_PREDICTION_INVALID" in result["warnings"]


def test_67b2_runtime_status_reports_frozen_sha_and_no_product_enablement() -> None:
    status = absolute_runtime_status()

    assert status["modelSha256"] == MODEL_SHA256
    assert status["runtimeAvailable"] is True
    assert status["productEnabled"] is False
    assert status["parityVerified"] is False


def test_67c_pairs_nearest_physical_plane_when_parity_verified() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(5.0, 5.0, 2.2),
        axial_planes=[plane(z=0.0, instance=99), plane(z=2.0, instance=10), plane(z=7.0, instance=1)],
        parity_verified=True,
    )

    assert result["status"] == "paired"
    assert result["instanceNumber"] == 10
    assert math.isclose(result["geometry"]["planeDistanceMm"], 0.2, abs_tol=1e-12)


def test_67c_requires_inside_fov() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(50.0, 5.0, 0.0),
        axial_planes=[plane(z=0.0, instance=1)],
        parity_verified=True,
    )

    assert result["status"] == "rejected"
    assert "67C_NO_VALID_PHYSICAL_PLANE" in result["warnings"]


def test_67c_rejects_planes_beyond_max_distance() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(5.0, 5.0, MAX_PLANE_DISTANCE_MM + 0.01),
        axial_planes=[plane(z=0.0, instance=1)],
        parity_verified=True,
    )

    assert result["status"] == "rejected"


def test_67c_rejects_margin_below_threshold() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(-0.4, 5.0, 0.0),
        axial_planes=[plane(z=0.0, instance=1)],
        parity_verified=True,
    )

    assert result["status"] == "rejected"
    assert result["candidates"][0]["margin"] < MIN_MARGIN


def test_67c_slice_location_does_not_control_pairing() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(5.0, 5.0, 0.0),
        axial_planes=[
            plane(z=20.0, instance=1, slice_location=0.0),
            plane(z=0.0, instance=2, slice_location=999.0),
        ],
        parity_verified=True,
    )

    assert result["status"] == "paired"
    assert result["instanceNumber"] == 2


def test_67c_instance_number_does_not_control_pairing() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(5.0, 5.0, 0.0),
        axial_planes=[plane(z=0.0, instance=99), plane(z=10.0, instance=1)],
        parity_verified=True,
    )

    assert result["status"] == "paired"
    assert result["instanceNumber"] == 99


def test_67c_uses_per_slice_orientation_normal() -> None:
    tilted = plane(
        z=0.0,
        instance=1,
        row_cosines=(1.0, 0.0, 0.0),
        column_cosines=(0.0, 0.0, 1.0),
    )
    flat = plane(z=0.0, instance=2)

    tilted_distance = candidate_geometry((5.0, 1.0, 5.0), tilted)["planeDistanceMm"]
    flat_distance = candidate_geometry((5.0, 1.0, 5.0), flat)["planeDistanceMm"]

    assert math.isclose(abs(tilted_distance), 1.0, abs_tol=1e-12)
    assert math.isclose(abs(flat_distance), 5.0, abs_tol=1e-12)


def test_67c_governance_flags_are_always_reviewable_not_validated() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(5.0, 5.0, 0.0),
        axial_planes=[plane(z=0.0, instance=1)],
        parity_verified=True,
    )

    assert_governance(result)


def test_67c_no_pairing_without_parity_verification() -> None:
    result = pair_disc_to_axial_slice(
        level="L4-L5",
        disc_centroid_patient_xyz=(5.0, 5.0, 0.0),
        axial_planes=[plane(z=0.0, instance=1)],
        parity_verified=False,
    )

    assert result["status"] == "rejected"
    assert_governance(result)
    assert "67C_RUNTIME_PARITY_NOT_PROVEN" in result["warnings"]


def test_67c_runtime_status_reports_not_product_enabled() -> None:
    status = pairing_runtime_status()

    assert status["runtimeAvailable"] is True
    assert status["productEnabled"] is False
    assert status["parityVerified"] is False
    assert status["validated"] is False
    assert status["experimental"] is True
    assert status["reviewRequired"] is True


def test_orchestration_67b2_localized_allows_67c_attempt_but_fails_closed(monkeypatch) -> None:
    install_orchestration_fakes(monkeypatch, localization={"status": "localized", "level": "L4-L5", "source": "67B2", "parityVerified": True, "centroidPatientXYZ": [5.0, 5.0, 0.0]})
    body = run_product_analysis(ProductAnalyzeRequest(caseId="CASE-73", sagittalInputId="sag", axialInputId="ax"))

    assert body["status"] == "partial"
    assert body["discLevels"][0]["localization"]["status"] == "localized"
    assert body["discLevels"][0]["axialPairing"]["status"] == "rejected"
    assert "67C_RUNTIME_PARITY_NOT_PROVEN" in body["discLevels"][0]["axialPairing"]["warnings"]


def test_orchestration_67b2_abstained_skips_67c(monkeypatch) -> None:
    install_orchestration_fakes(monkeypatch, localization={"status": "abstained", "source": "67B2", "parityVerified": True})
    body = run_product_analysis(ProductAnalyzeRequest(caseId="CASE-73", sagittalInputId="sag", axialInputId="ax"))

    assert body["status"] == "partial"
    assert body["discLevels"][0]["localization"]["status"] == "abstained"
    assert body["discLevels"][0]["axialPairing"]["status"] == "not_requested"


def test_orchestration_runtime_unavailable_is_partial(monkeypatch) -> None:
    install_orchestration_fakes(monkeypatch, localization=None)
    body = run_product_analysis(ProductAnalyzeRequest(caseId="CASE-73", sagittalInputId="sag", axialInputId="ax"))

    assert body["status"] == "partial"
    assert body["discLevels"][0]["localization"]["status"] == "unavailable"


def test_orchestration_p10_7_stays_blocked_and_not_diagnostic(monkeypatch) -> None:
    install_orchestration_fakes(monkeypatch, localization={"status": "localized", "level": "L4-L5", "source": "67B2", "parityVerified": True})
    body = run_product_analysis(ProductAnalyzeRequest(caseId="CASE-73", sagittalInputId="sag", axialInputId="ax"))

    findings = body["discLevels"][0]["degenerativeFindings"]
    assert findings["status"] == "blocked_preprocessing_parity"
    assert findings["automaticInferenceEnabled"] is False
    assert body["notClinicalDiagnosis"] is True
    assert body["autonomousDiagnosis"] is False


def install_orchestration_fakes(monkeypatch, *, localization: dict | None) -> None:
    def fake_resolve_input_id(input_id: str, *, case_id: str, plane: str):
        return object()

    def fake_run_pipeline(request):
        payload = {
            "status": "ok",
            "runId": f"run-{request.plane}",
            "inputId": request.input_id,
            "masks": [{"className": "disc"}],
            "measurementValues": [],
        }
        if request.plane == "sagittal" and localization is not None:
            payload["localization"] = localization
        if request.plane == "axial":
            payload["axialPlanes"] = [plane(z=0.0, instance=1).model_dump(mode="json")]
        return payload

    monkeypatch.setattr("pfi_ai_service.product_orchestration.resolve_input_id", fake_resolve_input_id)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.PipelineRunRequest", FakePipelineRunRequest)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.InputRegistryError", FakeInputRegistryError)
