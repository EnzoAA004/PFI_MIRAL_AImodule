from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from pfi_ai_service.product_orchestration import (
    PRODUCT_RELEASE_ID,
    PRODUCT_SCHEMA_VERSION,
    ProductAnalyzeRequest,
    ProductCapabilities,
    product_release_readiness,
    run_product_analysis,
)


class FakePipelineRunRequest:
    def __init__(self, **kwargs):
        self.case_id = kwargs["caseId"]
        self.plane = kwargs["plane"]
        self.model_key = kwargs["modelKey"]
        self.input_id = kwargs.get("inputId")
        self.metadata = kwargs.get("metadata") or {}


class FakeInputRegistryError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def install_fakes(monkeypatch, *, payloads: dict[str, dict[str, Any]] | None = None, fail_planes: set[str] | None = None) -> None:
    payloads = payloads or {}
    fail_planes = fail_planes or set()

    def fake_resolve_input_id(input_id: str, *, case_id: str, plane: str):
        if plane in fail_planes:
            raise FakeInputRegistryError(f"{plane}_missing")
        return object()

    def fake_run_pipeline(request):
        base = {
            "status": "ok",
            "runId": f"run-{request.plane}",
            "inputId": request.input_id,
            "masks": [{"className": "disc"}],
            "measurementValues": [{"className": "disc", "minorAxisMm": 1.5}],
        }
        base.update(payloads.get(request.plane, {}))
        return base

    monkeypatch.setattr("pfi_ai_service.product_orchestration.resolve_input_id", fake_resolve_input_id)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.PipelineRunRequest", FakePipelineRunRequest)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.InputRegistryError", FakeInputRegistryError)


def run_default(monkeypatch, *, payloads: dict[str, dict[str, Any]] | None = None, capabilities: ProductCapabilities | None = None, fail_planes: set[str] | None = None) -> dict[str, Any]:
    install_fakes(monkeypatch, payloads=payloads, fail_planes=fail_planes)
    return run_product_analysis(
        ProductAnalyzeRequest(
            caseId="CASE-PRODUCT-72",
            sagittalInputId="inp_sag",
            axialInputId="inp_ax",
            capabilities=capabilities or ProductCapabilities(),
            metadata={
                "PatientName": "Forbidden",
                "PatientID": "Forbidden",
                "StudyInstanceUID": "1.2.3",
                "SeriesInstanceUID": "1.2.4",
                "SOPInstanceUID": "1.2.5",
                "patientId": "PAT-FORBIDDEN",
            },
        ),
        trace_id="trace-product-72",
    )


def assert_public_payload(value: object) -> None:
    text = str(value)
    forbidden = [
        "C:\\",
        "/tmp",
        "ai_service/tests/fixtures",
        "PatientName",
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "patientId",
        "PAT-",
        "inputPath",
        "sourcePath",
    ]
    assert not any(item in text for item in forbidden), text[:1000]
    assert_no_nonfinite(value)


def assert_no_nonfinite(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            assert_no_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_nonfinite(item)


def test_release_and_governance_flags_are_fixed(monkeypatch) -> None:
    body = run_default(monkeypatch)

    assert body["schemaVersion"] == PRODUCT_SCHEMA_VERSION
    assert body["release"]["releaseId"] == PRODUCT_RELEASE_ID
    assert body["humanReviewRequired"] is True
    assert body["review"]["required"] is True
    assert body["autonomousDiagnosis"] is False
    assert body["notClinicalDiagnosis"] is True


def test_global_status_partial_when_requested_adapter_is_missing(monkeypatch) -> None:
    body = run_default(monkeypatch)

    assert body["status"] == "partial"
    assert "67B2_localization_adapter_pending" in body["warnings"]


def test_67b2_adapter_pending_is_not_reported_as_abstained(monkeypatch) -> None:
    body = run_default(monkeypatch)
    level = body["discLevels"][0]

    assert level["level"] is None
    assert level["localization"]["status"] == "adapter_pending"
    assert level["localization"]["adapterPending"] is True
    assert level["localization"]["abstentionPreserved"] is False


def test_67b2_real_abstention_is_preserved(monkeypatch) -> None:
    body = run_default(monkeypatch, payloads={"sagittal": {"localization": {"status": "abstained"}}})
    level = body["discLevels"][0]

    assert body["status"] == "partial"
    assert level["level"] is None
    assert level["localization"]["status"] == "abstained"
    assert level["localization"]["abstentionPreserved"] is True


def test_67b2_never_fabricates_a_level_without_runtime_output(monkeypatch) -> None:
    body = run_default(monkeypatch)

    assert [item["level"] for item in body["discLevels"]] == [None]


def test_localized_level_only_comes_from_pipeline_localization(monkeypatch) -> None:
    body = run_default(monkeypatch, payloads={"sagittal": {"localization": {"level": "L4-L5"}}})

    assert body["discLevels"][0]["level"] == "L4-L5"
    assert body["discLevels"][0]["localization"]["status"] == "localized"


def test_67c_is_never_validated_and_stays_review_required(monkeypatch) -> None:
    body = run_default(monkeypatch, payloads={"sagittal": {"localization": {"level": "L4-L5"}}})
    pairing = body["discLevels"][0]["axialPairing"]

    assert pairing["status"] == "adapter_pending"
    assert pairing["validated"] is False
    assert pairing["experimental"] is True
    assert pairing["reviewRequired"] is True
    assert pairing["adapterPending"] is True


def test_p10_7_remains_blocked_and_not_automatic(monkeypatch) -> None:
    body = run_default(monkeypatch, capabilities=ProductCapabilities(degenerativeFindingsP10_7=True))
    findings = body["discLevels"][0]["degenerativeFindings"]

    assert body["status"] == "partial"
    assert findings["status"] == "blocked_preprocessing_parity"
    assert findings["automaticInferenceEnabled"] is False


def test_axial_raw_classes_are_not_renamed_anatomically(monkeypatch) -> None:
    body = run_default(monkeypatch, payloads={"axial": {"masks": [{"className": "disc"}]}})

    assert set(body["planes"]["axial"]["segmentation"]["maskClasses"]) == {
        "raw_0",
        "raw_50",
        "raw_100",
        "raw_150",
        "raw_200",
    }


def test_sanitization_removes_patient_uids_paths_nan_and_inf(monkeypatch) -> None:
    body = run_default(
        monkeypatch,
        payloads={
            "sagittal": {
                "measurementValues": [
                    {
                        "className": "disc",
                        "areaMm2": float("nan"),
                        "majorAxisMm": float("inf"),
                        "minorAxisMm": 2.5,
                        "sourcePath": "C:\\private\\study.dcm",
                    }
                ],
            }
        },
    )

    measurement = body["discLevels"][0]["measurements"][0]
    assert measurement["areaMm2"] is None
    assert measurement["majorAxisMm"] is None
    assert_public_payload(body)


def test_capability_failure_is_partial_fail_closed_and_not_diagnostic(monkeypatch) -> None:
    body = run_default(monkeypatch, fail_planes={"axial"})

    assert body["status"] == "partial"
    assert body["planes"]["axial"]["status"] == "unavailable"
    assert body["notClinicalDiagnosis"] is True
    assert body["autonomousDiagnosis"] is False
    assert "clinicalDiagnosis" not in body
    assert "diagnosticConclusion" not in body
    assert "diagnosticLabel" not in body


def test_no_useful_plane_result_is_failed_and_not_diagnostic(monkeypatch) -> None:
    body = run_default(monkeypatch, fail_planes={"sagittal", "axial"})

    assert body["status"] == "failed"
    assert body["notClinicalDiagnosis"] is True
    assert body["autonomousDiagnosis"] is False


def test_completed_requires_no_requested_pending_blocked_or_experimental_capability(monkeypatch) -> None:
    body = run_default(
        monkeypatch,
        capabilities=ProductCapabilities(
            absoluteDiscLocalization67B2=False,
            axialPairing67C=False,
            degenerativeFindingsP10_7=False,
        ),
    )

    assert body["status"] == "completed"
    assert body["discLevels"][0]["localization"]["status"] == "not_requested"
    assert body["discLevels"][0]["axialPairing"]["status"] == "not_requested"


def test_product_request_rejects_local_paths() -> None:
    with pytest.raises(ValidationError):
        ProductAnalyzeRequest.model_validate({
            "caseId": "CASE-BAD",
            "inputPath": "C:\\private\\study",
        })


def test_product_release_readiness_matrix_distinguishes_states() -> None:
    product = product_release_readiness()

    localization = product["capabilities"]["absoluteDiscLocalization67B2"]
    pairing = product["capabilities"]["axialPairing67C"]
    p10 = product["capabilities"]["pathologyP10_7"]

    assert product["releaseId"] == PRODUCT_RELEASE_ID
    assert localization["artifactAvailable"] is True
    assert localization["runtimeAvailable"] is False
    assert localization["validated"] is True
    assert localization["adapterPending"] is True
    assert pairing["experimental"] is True
    assert pairing["validated"] is False
    assert pairing["adapterPending"] is True
    assert p10["blocked"] is True
    assert p10["automaticInferenceEnabled"] is False
