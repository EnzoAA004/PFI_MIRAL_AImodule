from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .absolute_disc_localization_runtime import (
    localize_absolute_disc_levels,
    runtime_status as absolute_disc_runtime_status,
)
from .axial_level_pairing_runtime import (
    pair_disc_to_axial_slice,
    runtime_status as axial_pairing_runtime_status,
)
from .security import SENSITIVE_KEYS, sanitize_public_text

PRODUCT_SCHEMA_VERSION = "pfi.ai-product-result.v1"
PRODUCT_RELEASE_ID = "post-e50-ai-product-freeze-v1"
P10_7_CHECKPOINT_SHA256 = "16eccff327e6794b127fe372ecd03ea619a0f69d939b84ae1aa2e904191c6293"
AUTONOMOUS_DIAGNOSIS = False
InputRegistryError = None
PipelineRunRequest = None
resolve_input_id = None
run_pipeline = None


class ProductCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sagittalSegmentation: bool = True
    axialSegmentation: bool = True
    absoluteDiscLocalization67B2: bool = True
    axialPairing67C: bool = True
    sagittalMorphometry68: bool = True
    axialMorphometry68: bool = True
    degenerativeFindingsP10_7: bool = False


class ProductAnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_by_name=True, extra="forbid")

    case_id: str = Field(..., alias="caseId", min_length=1)
    sagittal_input_id: str | None = Field(default=None, alias="sagittalInputId")
    axial_input_id: str | None = Field(default=None, alias="axialInputId")
    input_ids: dict[str, str] | None = Field(default=None, alias="inputIds")
    capabilities: ProductCapabilities = Field(default_factory=ProductCapabilities)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def merge_input_ids(self) -> "ProductAnalyzeRequest":
        if self.input_ids:
            sagittal = self.input_ids.get("sagittal") or self.input_ids.get("sagittalInputId")
            axial = self.input_ids.get("axial") or self.input_ids.get("axialInputId")
            if sagittal and self.sagittal_input_id and sagittal != self.sagittal_input_id:
                raise ValueError("sagittal_input_conflict")
            if axial and self.axial_input_id and axial != self.axial_input_id:
                raise ValueError("axial_input_conflict")
            self.sagittal_input_id = self.sagittal_input_id or sagittal
            self.axial_input_id = self.axial_input_id or axial
        if not self.sagittal_input_id and not self.axial_input_id:
            raise ValueError("at_least_one_registered_input_required")
        return self


def product_release_readiness() -> dict[str, Any]:
    absolute_status = absolute_disc_runtime_status()
    pairing_status = axial_pairing_runtime_status()
    return {
        "releaseId": PRODUCT_RELEASE_ID,
        "schemaVersion": PRODUCT_SCHEMA_VERSION,
        "orchestrationReady": True,
        "automaticPathologyInference": False,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        "autonomousDiagnosis": AUTONOMOUS_DIAGNOSIS,
        "capabilities": {
            "sagittalSegmentation": capability_state(True, "validated", runtime_available=True),
            "axialSegmentationV2": capability_state(True, "validated", runtime_available=True),
            "absoluteDiscLocalization67B2": capability_state(
                bool(absolute_status["artifactAvailable"]),
                str(absolute_status["adapterStatus"]),
                runtime_available=bool(absolute_status["runtimeAvailable"]),
                product_enabled=bool(absolute_status["productEnabled"]),
                parity_verified=bool(absolute_status["parityVerified"]),
                abstention_supported=bool(absolute_status["abstentionSupported"]),
                blocker="67B2_RUNTIME_PARITY_NOT_PROVEN",
            ),
            "axialPairing67C": capability_state(
                bool(pairing_status["artifactAvailable"]),
                str(pairing_status["adapterStatus"]),
                validated=False,
                runtime_available=bool(pairing_status["runtimeAvailable"]),
                product_enabled=bool(pairing_status["productEnabled"]),
                parity_verified=bool(pairing_status["parityVerified"]),
                experimental=True,
                review_required=True,
                blocker="67C_RUNTIME_PARITY_NOT_PROVEN",
            ),
            "sagittalMorphometry68": capability_state(True, "technical_qc_pass_descriptive", runtime_available=True),
            "axialMorphometry68": capability_state(True, "technical_qc_pass_raw_classes", runtime_available=True),
            "pathologyP10_7": {
                "artifactAvailable": True,
                "runtimeAvailable": False,
                "validated": False,
                "enabled": False,
                "blocked": True,
                "adapterPending": False,
                "status": "blocked_preprocessing_parity",
                "automaticInferenceEnabled": False,
                "checkpointSha256": P10_7_CHECKPOINT_SHA256,
                "supportedInternalTasks": [
                    "upper_endplate_change",
                    "lower_endplate_change",
                    "disc_narrowing",
                    "disc_bulging",
                ],
                "experimentalTasks": ["pfirrmann_grade"],
                "notProductSupportedTasks": [
                    "modic_change",
                    "spondylolisthesis",
                    "disc_herniation",
                ],
            },
        },
    }


def capability_state(
    artifact_available: bool,
    status: str,
    *,
    validated: bool = True,
    runtime_available: bool,
    product_enabled: bool = True,
    parity_verified: bool | None = None,
    abstention_supported: bool | None = None,
    experimental: bool = False,
    adapter_pending: bool = False,
    review_required: bool = True,
    blocker: str | None = None,
) -> dict[str, Any]:
    payload = {
        "artifactAvailable": artifact_available,
        "runtimeAvailable": runtime_available,
        "productEnabled": product_enabled,
        "validated": validated,
        "experimental": experimental,
        "blocked": bool(blocker and not adapter_pending),
        "adapterPending": adapter_pending,
        "status": status,
        "reviewRequired": review_required,
    }
    if parity_verified is not None:
        payload["parityVerified"] = parity_verified
    if abstention_supported is not None:
        payload["abstentionSupported"] = abstention_supported
    if blocker:
        payload["blocker"] = blocker
    return payload


def run_product_analysis(request: ProductAnalyzeRequest, *, trace_id: str | None = None) -> dict[str, Any]:
    input_error, pipeline_request, resolve_input, pipeline_runner = runtime_bindings()
    warnings: list[str] = []
    planes: dict[str, Any] = {"sagittal": None, "axial": None}
    failures: list[str] = []

    for plane, input_id, model_key in (
        ("sagittal", request.sagittal_input_id, "sagittal_spider"),
        ("axial", request.axial_input_id, "axial_t2_alkafri"),
    ):
        if not input_id:
            warnings.append(f"{plane}_input_not_requested")
            continue
        try:
            resolve_input(input_id, case_id=request.case_id, plane=plane)
            plane_request = pipeline_request(
                caseId=request.case_id,
                plane=plane,
                modelKey=model_key,
                inputId=input_id,
                metadata={
                    **request.metadata,
                    "traceId": trace_id,
                    "correlationId": trace_id,
                    "inferenceMode": "contract",
                    "productReleaseId": PRODUCT_RELEASE_ID,
                },
            )
            planes[plane] = pipeline_runner(plane_request)
        except input_error as exc:
            failures.append(f"{plane}_input_unavailable")
            warnings.append(f"{plane}_input_unavailable:{exc.message}")
            planes[plane] = unavailable_plane(plane, input_id, exc.message)
        except Exception as exc:
            failures.append(f"{plane}_pipeline_unavailable")
            warnings.append(f"{plane}_pipeline_unavailable:{type(exc).__name__}")
            planes[plane] = unavailable_plane(plane, input_id, "pipeline_unavailable")

    demo_case_id = request.case_id if is_demo_fixture_request(request.metadata) else None
    disc_levels = build_disc_levels(planes, request.capabilities, warnings, demo_case_id=demo_case_id)
    has_available_plane = any(
        isinstance(plane, dict) and plane.get("status") != "unavailable"
        for plane in planes.values()
        if plane is not None
    )
    status = product_status(
        has_available_plane=has_available_plane,
        failures=failures,
        disc_levels=disc_levels,
        capabilities=request.capabilities,
    )
    result = {
        "schemaVersion": PRODUCT_SCHEMA_VERSION,
        "runId": product_run_id(request),
        "traceId": trace_id,
        "caseId": request.case_id,
        "release": {"releaseId": PRODUCT_RELEASE_ID},
        "status": status,
        "planes": {
            "sagittal": product_plane_payload(planes.get("sagittal"), "sagittal"),
            "axial": product_plane_payload(planes.get("axial"), "axial"),
        },
        "discLevels": disc_levels,
        "warnings": warnings,
        "review": {
            "required": HUMAN_REVIEW_REQUIRED,
            "status": "pending",
            "professionalReviewRequired": HUMAN_REVIEW_REQUIRED,
        },
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        "autonomousDiagnosis": AUTONOMOUS_DIAGNOSIS,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
    }
    return sanitize_product_payload(result)


def sanitize_product_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            normalized_key = text_key.lower()
            if text_key in SENSITIVE_KEYS or normalized_key in {
                "patientname",
                "patientid",
                "patient_id",
                "studyinstanceuid",
                "seriesinstanceuid",
                "sopinstanceuid",
            }:
                continue
            sanitized[text_key] = sanitize_product_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_product_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_product_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_public_text(value)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
    return value


def runtime_bindings():
    global InputRegistryError, PipelineRunRequest, resolve_input_id, run_pipeline
    if InputRegistryError is None or resolve_input_id is None:
        from .input_registry import InputRegistryError as registry_error, resolve_input_id as resolver

        InputRegistryError = registry_error
        resolve_input_id = resolver
    if PipelineRunRequest is None or run_pipeline is None:
        from .pipeline import PipelineRunRequest as request_model, run_pipeline as pipeline_runner

        PipelineRunRequest = request_model
        run_pipeline = pipeline_runner
    return InputRegistryError, PipelineRunRequest, resolve_input_id, run_pipeline


def product_run_id(request: ProductAnalyzeRequest) -> str:
    raw = "|".join([
        PRODUCT_RELEASE_ID,
        request.case_id,
        request.sagittal_input_id or "",
        request.axial_input_id or "",
    ])
    return "product-" + sha256(raw.encode("utf-8")).hexdigest()[:16]


def product_status(
    *,
    has_available_plane: bool,
    failures: list[str],
    disc_levels: list[dict[str, Any]],
    capabilities: ProductCapabilities,
) -> str:
    if not has_available_plane:
        return "failed"
    if failures:
        return "partial"
    if capabilities.absoluteDiscLocalization67B2:
        if any(level.get("localization", {}).get("status") in {"adapter_pending", "abstained", "unavailable", "demo_fixture"} for level in disc_levels):
            return "partial"
    if capabilities.axialPairing67C:
        if any(
            level.get("axialPairing", {}).get("status") in {"adapter_pending", "experimental", "unavailable", "rejected"}
            or (
                level.get("axialPairing", {}).get("status") != "not_requested"
                and level.get("axialPairing", {}).get("experimental") is True
            )
            for level in disc_levels
        ):
            return "partial"
    if capabilities.degenerativeFindingsP10_7:
        if any(level.get("degenerativeFindings", {}).get("status") == "blocked_preprocessing_parity" for level in disc_levels):
            return "partial"
    return "completed"


def unavailable_plane(plane: str, input_id: str | None, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "plane": plane,
        "inputId": input_id,
        "reason": reason,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def product_plane_payload(plane_payload: Any, plane: str) -> dict[str, Any]:
    if not isinstance(plane_payload, dict):
        return {"status": "not_requested", "runtimeAvailable": False, "reviewRequired": HUMAN_REVIEW_REQUIRED}
    if plane_payload.get("status") == "unavailable":
        return {
            "status": "unavailable",
            "runtimeAvailable": False,
            "inputId": plane_payload.get("inputId"),
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
        }
    masks = plane_payload.get("masks") if isinstance(plane_payload.get("masks"), list) else []
    measurements = plane_payload.get("measurementValues") if isinstance(plane_payload.get("measurementValues"), list) else []
    return {
        "status": "available",
        "runtimeAvailable": True,
        "inputId": plane_payload.get("inputId"),
        "runId": plane_payload.get("runId"),
        "segmentation": {
            "status": "completed",
            "validated": True,
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
            "maskClasses": mask_classes(masks, plane),
        },
        "morphometry": {
            "status": "available",
            "validated": True,
            "descriptiveOnly": True,
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
            "measurementCount": len(measurements),
        },
    }


def mask_classes(masks: list[Any], plane: str) -> list[str]:
    classes = [str(item.get("className")) for item in masks if isinstance(item, dict) and item.get("className")]
    if plane == "axial" and not any(item.startswith("raw_") for item in classes):
        return ["raw_0", "raw_50", "raw_100", "raw_150", "raw_200"]
    return classes


def build_disc_levels(
    planes: dict[str, Any],
    capabilities: ProductCapabilities,
    warnings: list[str],
    *,
    demo_case_id: str | None = None,
) -> list[dict[str, Any]]:
    sagittal = planes.get("sagittal") if isinstance(planes.get("sagittal"), dict) else {}
    axial = planes.get("axial") if isinstance(planes.get("axial"), dict) else {}
    localization = localization_result(sagittal, capabilities)
    localization_status = str(localization.get("status"))
    if localization_status == "unavailable":
        warnings.extend(str(item) for item in localization.get("warnings", []))
        if demo_case_id is not None:
            # 67B2 genuinely has no proven parity artifact (see
            # absolute_disc_localization_runtime.runtime_status) -- this never overrides a
            # real localized/abstained result, only the "nothing to show" case, and only when
            # the caller explicitly opted in via metadata.mode=="demo".
            localization = demo_fixture_localization(demo_case_id)
            localization_status = str(localization.get("status"))
            warnings.append(DEMO_FIXTURE_WARNING)
    elif localization_status == "abstained":
        warnings.append("67B2_localization_abstained")
    levels = [localization.get("level")]
    return [
        disc_level_payload(level, localization, sagittal, axial, capabilities, demo_case_id=demo_case_id)
        for level in levels
    ]


DEMO_FIXTURE_SOURCE = "DEMO_FIXTURE"
DEMO_FIXTURE_LEVEL = "L4-L5"
DEMO_FIXTURE_WARNING = "DEMO_FIXTURE_LOCALIZATION_NOT_REAL_INFERENCE"


def is_demo_fixture_request(metadata: dict[str, Any]) -> bool:
    return str((metadata or {}).get("mode", "")).strip().lower() == "demo"


def demo_fixture_major_axis_mm(case_id: str) -> float:
    """Deterministic, non-clinical value. Mirrors demoMajorAxisMm in
    docker/post-e50-mocks/mock-ai.mjs so the post-e50 mock stack and this module's demo
    path agree on the same seeded values for the same caseId convention."""
    normalized = (case_id or "").upper()
    if "DEMO-LUMBAR-2026-01" in normalized:
        return 11.9
    if "DEMO-LUMBAR-2026-02" in normalized:
        return 12.3
    if "DEMO-LUMBAR-2026-03" in normalized:
        return 13.1
    return 12.3


def demo_fixture_localization(case_id: str) -> dict[str, Any]:
    """Never a substitute for 67B2: status is "demo_fixture" (never "localized") and source
    is DEMO_FIXTURE_SOURCE (never "67B2"), so a consumer that only checks status/source
    cannot mistake this for real inference."""
    return {
        "status": "demo_fixture",
        "level": DEMO_FIXTURE_LEVEL,
        "source": DEMO_FIXTURE_SOURCE,
        "validated": False,
        "abstentionPreserved": False,
        "confidence": None,
        "modelSha256": None,
        "centroidPatientXYZ": None,
        "warnings": [DEMO_FIXTURE_WARNING],
    }


def demo_fixture_measurement(case_id: str, level: str | None) -> dict[str, Any]:
    major_axis_mm = demo_fixture_major_axis_mm(case_id)
    return {
        "plane": "sagittal",
        "level": level,
        "structure": "disc",
        "areaMm2": major_axis_mm,
        "majorAxisMm": major_axis_mm,
        "minorAxisMm": 8.4,
        "reviewRequired": HUMAN_REVIEW_REQUIRED,
    }


def localization_result(sagittal: dict[str, Any], capabilities: ProductCapabilities) -> dict[str, Any]:
    if not capabilities.absoluteDiscLocalization67B2:
        return {
            "status": "not_requested",
            "level": None,
            "source": "67B2",
            "validated": True,
            "abstentionPreserved": False,
            "warnings": [],
        }
    localization = sagittal.get("localization") if isinstance(sagittal.get("localization"), dict) else {}
    if localization.get("source") == "67B2" and localization.get("parityVerified") is True:
        return localize_absolute_disc_levels(localization, parity_verified=True)
    return localize_absolute_disc_levels()


def disc_level_payload(
    level: str | None,
    localization: dict[str, Any],
    sagittal: dict[str, Any],
    axial: dict[str, Any],
    capabilities: ProductCapabilities,
    *,
    demo_case_id: str | None = None,
) -> dict[str, Any]:
    localization_status = str(localization.get("status"))
    pairing = pairing_result(level, localization, axial, capabilities)
    pairing_status = str(pairing.get("status"))
    measurements = measurements_for_level(level, sagittal, axial)
    if demo_case_id is not None and localization.get("source") == DEMO_FIXTURE_SOURCE:
        measurements = [*measurements, demo_fixture_measurement(demo_case_id, level)]
    return {
        "level": level,
        "localization": {
            "status": localization_status,
            "validated": bool(localization.get("validated", True)),
            "source": str(localization.get("source") or "67B2"),
            "abstentionPreserved": localization_status == "abstained",
            "adapterPending": False,
            "confidence": localization.get("confidence"),
            "modelSha256": localization.get("modelSha256"),
            "warnings": localization.get("warnings", []),
        },
        "axialPairing": {
            "status": pairing_status,
            "validated": False,
            "experimental": True,
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
            "adapterPending": False,
            "source": "67C",
            "seriesId": pairing.get("seriesId"),
            "instanceNumber": pairing.get("instanceNumber"),
            "geometry": pairing.get("geometry"),
            "warnings": pairing.get("warnings", []),
        },
        "measurements": measurements,
        "degenerativeFindings": {
            "status": "blocked_preprocessing_parity",
            "automaticInferenceEnabled": False,
            "checkpointSha256": P10_7_CHECKPOINT_SHA256,
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
        },
    }


def pairing_result(
    level: str | None,
    localization: dict[str, Any],
    axial: dict[str, Any],
    capabilities: ProductCapabilities,
) -> dict[str, Any]:
    if not capabilities.axialPairing67C:
        return {"status": "not_requested", "warnings": []}
    if localization.get("status") != "localized" or not level:
        return {"status": "not_requested", "warnings": ["67C_NOT_REQUESTED_WITHOUT_67B2_LOCALIZED_LEVEL"]}
    point = localization.get("centroidPatientXYZ")
    planes = axial.get("axialPlanes") if isinstance(axial.get("axialPlanes"), list) else []
    if point is None or not planes:
        return {
            "status": "unavailable",
            "source": "67C",
            "validated": False,
            "experimental": True,
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
            "warnings": ["67C_RUNTIME_INPUT_UNAVAILABLE"],
        }
    return pair_disc_to_axial_slice(
        level=level,
        disc_centroid_patient_xyz=point,
        axial_planes=planes,
        parity_verified=False,
    )


def measurements_for_level(level: str | None, sagittal: dict[str, Any], axial: dict[str, Any]) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for plane_name, plane_payload in (("sagittal", sagittal), ("axial", axial)):
        values = plane_payload.get("measurementValues") if isinstance(plane_payload.get("measurementValues"), list) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            measurements.append({
                "plane": plane_name,
                "level": level,
                "structure": measurement_structure(item, plane_name),
                "areaMm2": finite_or_none(item.get("areaMm2") or item.get("area_mm2")),
                "majorAxisMm": finite_or_none(item.get("majorAxisMm") or item.get("major_axis_mm")),
                "minorAxisMm": finite_or_none(item.get("minorAxisMm") or item.get("minor_axis_mm")),
                "reviewRequired": HUMAN_REVIEW_REQUIRED,
            })
    return measurements


def measurement_structure(item: dict[str, Any], plane: str) -> str:
    for key in ("className", "structure", "maskClass", "label"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if plane == "axial":
        return "raw_unmapped"
    return "unmapped_structure"


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number
