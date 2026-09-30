from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from pfi_ai_service.product_orchestration import (
    ProductAnalyzeRequest,
    product_release_readiness,
    run_product_analysis,
)
from test_product_orchestration import FakeInputRegistryError, FakePipelineRunRequest


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "artifacts/post_e50/runtime_parity/74_evidence_inventory.json"
SAMPLE_PATH = ROOT / "artifacts/post_e50/runtime_parity/74_parity_sample_manifest.json"
SUMMARY_67B2_PATH = ROOT / "metrics/post_e50/74/74_67B2_parity_summary.json"
SUMMARY_67C_PATH = ROOT / "metrics/post_e50/74/74_67C_parity_summary.json"
PARITY_67B2_CSV = ROOT / "results/post_e50/74/74_67B2_parity.csv"
PARITY_67C_CSV = ROOT / "results/post_e50/74/74_67C_parity.csv"
FREEZE_PATH = ROOT / "docs/contracts/post-e50-runtime-freeze-v1.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_74_inventory_exists_and_is_evidence_blocked() -> None:
    inventory = load_json(INVENTORY_PATH)

    assert inventory["schemaVersion"] == "pfi.post-e50.runtime-parity-evidence-inventory.v1"
    assert inventory["overallStatus"] == "EVIDENCE_BLOCKED"
    assert inventory["seed"] == 20260825


def test_74_inventory_records_no_forbidden_dataset_access() -> None:
    policy = load_json(INVENTORY_PATH)["policy"]

    assert policy["trainingPerformed"] is False
    assert policy["checkpointsModified"] is False
    assert policy["validationAccessed"] is False
    assert policy["internalTestAccessed"] is False
    assert policy["officialTestAccessed"] is False


def test_74_inventory_has_no_eligible_artifacts() -> None:
    inventory = load_json(INVENTORY_PATH)

    assert inventory["eligibleCounts"] == {"67B2": 0, "67C": 0}
    assert all(item["eligibleForParity"] is False for item in inventory["artifacts"])


def test_74_inventory_artifacts_have_sha_size_and_reason() -> None:
    for item in load_json(INVENTORY_PATH)["artifacts"]:
        assert item["path"]
        assert item["sizeBytes"] >= 0
        assert len(item["sha256"]) == 64
        assert item["reason"]


def test_74_inventory_paths_are_relative_and_public() -> None:
    text = json.dumps(load_json(INVENTORY_PATH), sort_keys=True)

    assert "C:\\" not in text
    assert "/content/drive" not in text
    assert "PatientName" not in text
    assert "PatientID" not in text


def test_74_sample_manifest_is_empty_when_evidence_blocked() -> None:
    sample = load_json(SAMPLE_PATH)

    assert sample["status"] == "EVIDENCE_BLOCKED"
    assert sample["sampleCounts"] == {"67B2": 0, "67C": 0}
    assert sample["samples"] == {"67B2": [], "67C": []}


def test_74_67b2_summary_is_not_proven() -> None:
    summary = load_json(SUMMARY_67B2_PATH)

    assert summary["status"] == "67B2_RUNTIME_PARITY_NOT_PROVEN"
    assert summary["sampleSize"] == 0
    assert summary["eligibleArtifactCount"] == 0
    assert all(value == "NOT_RUN" for value in summary["gates"].values())


def test_74_67c_summary_is_not_proven_and_not_validated() -> None:
    summary = load_json(SUMMARY_67C_PATH)

    assert summary["status"] == "67C_RUNTIME_PARITY_NOT_PROVEN"
    assert summary["sampleSize"] == 0
    assert summary["eligibleArtifactCount"] == 0
    assert summary["productEnablement"]["validated"] is False
    assert summary["productEnablement"]["experimental"] is True
    assert summary["productEnablement"]["reviewRequired"] is True


def test_74_parity_csvs_have_headers_only() -> None:
    for path in (PARITY_67B2_CSV, PARITY_67C_CSV):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 1


def test_74_freeze_contract_status_and_release() -> None:
    freeze = load_json(FREEZE_PATH)

    assert freeze["schemaVersion"] == "pfi.post-e50.runtime-freeze.v1"
    assert freeze["releaseId"] == "post-e50-ai-product-runtime-freeze-v1"
    assert freeze["runtimeFreezeStatus"] == "EVIDENCE_BLOCKED"


def test_74_freeze_keeps_67b2_product_disabled() -> None:
    capability = load_json(FREEZE_PATH)["capabilities"]["absoluteDiscLocalization67B2"]

    assert capability["runtimeAvailable"] is True
    assert capability["parityVerified"] is False
    assert capability["productEnabled"] is False
    assert capability["validated"] is True
    assert capability["abstentionSupported"] is True
    assert capability["blocker"] == "67B2_RUNTIME_PARITY_NOT_PROVEN"


def test_74_freeze_keeps_67c_experimental_and_disabled() -> None:
    capability = load_json(FREEZE_PATH)["capabilities"]["axialPairing67C"]

    assert capability["runtimeAvailable"] is True
    assert capability["parityVerified"] is False
    assert capability["productEnabled"] is False
    assert capability["validated"] is False
    assert capability["experimental"] is True
    assert capability["reviewRequired"] is True
    assert capability["blocker"] == "67C_RUNTIME_PARITY_NOT_PROVEN"


def test_74_freeze_keeps_p10_7_blocked() -> None:
    capability = load_json(FREEZE_PATH)["capabilities"]["P10_7"]

    assert capability["runtimeCheckpointVerified"] is True
    assert capability["automaticInferenceEnabled"] is False
    assert capability["productEnabled"] is False
    assert capability["blocker"] == "preprocessing_parity_not_proven"


def test_74_product_readiness_matches_evidence_blocked_freeze() -> None:
    readiness = product_release_readiness()["capabilities"]

    assert readiness["absoluteDiscLocalization67B2"]["parityVerified"] is False
    assert readiness["absoluteDiscLocalization67B2"]["productEnabled"] is False
    assert readiness["axialPairing67C"]["parityVerified"] is False
    assert readiness["axialPairing67C"]["productEnabled"] is False
    assert readiness["axialPairing67C"]["validated"] is False


def test_74_product_analyze_example_is_partial_and_sanitized(monkeypatch) -> None:
    install_fakes(monkeypatch)
    body = run_product_analysis(
        ProductAnalyzeRequest(
            caseId="CASE-74",
            sagittalInputId="sag",
            axialInputId="ax",
            metadata={
                "PatientName": "Forbidden",
                "PatientID": "Forbidden",
                "StudyInstanceUID": "1.2.3",
            },
        )
    )

    assert body["status"] == "partial"
    assert body["discLevels"][0]["localization"]["status"] == "unavailable"
    assert "67B2_RUNTIME_PARITY_NOT_PROVEN" in body["warnings"]
    assert body["humanReviewRequired"] is True
    assert body["autonomousDiagnosis"] is False
    assert body["notClinicalDiagnosis"] is True
    assert_public_payload(body)


def assert_public_payload(value: object) -> None:
    text = str(value)
    assert "PatientName" not in text
    assert "PatientID" not in text
    assert "StudyInstanceUID" not in text
    assert "C:\\" not in text
    assert "/tmp" not in text
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


def install_fakes(monkeypatch) -> None:
    def fake_resolve_input_id(input_id: str, *, case_id: str, plane: str):
        return object()

    def fake_run_pipeline(request):
        return {
            "status": "ok",
            "runId": f"run-{request.plane}",
            "inputId": request.input_id,
            "masks": [{"className": "disc"}],
            "measurementValues": [{"className": "disc", "areaMm2": float("nan")}],
        }

    monkeypatch.setattr("pfi_ai_service.product_orchestration.resolve_input_id", fake_resolve_input_id)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.PipelineRunRequest", FakePipelineRunRequest)
    monkeypatch.setattr("pfi_ai_service.product_orchestration.InputRegistryError", FakeInputRegistryError)
