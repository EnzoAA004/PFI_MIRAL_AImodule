from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
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
                True,
                "adapter_pending",
                runtime_available=False,
                adapter_pending=True,
                blocker="product_adapter_pending",
            ),
            "axialPairing67C": capability_state(
                True,
                "adapter_pending",
                validated=False,
                runtime_available=False,
                experimental=True,
                adapter_pending=True,
                review_required=True,
                blocker="product_adapter_pending",
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
    experimental: bool = False,
    adapter_pending: bool = False,
    review_required: bool = True,
    blocker: str | None = None,
) -> dict[str, Any]:
    payload = {
        "artifactAvailable": artifact_available,
        "runtimeAvailable": runtime_available,
        "validated": validated,
        "experimental": experimental,
        "blocked": bool(blocker and not adapter_pending),
        "adapterPending": adapter_pending,
        "status": status,
        "reviewRequired": review_required,
    }
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

    disc_levels = build_disc_levels(planes, request.capabilities, warnings)
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
        if any(level.get("localization", {}).get("status") in {"adapter_pending", "abstained"} for level in disc_levels):
            return "partial"
    if capabilities.axialPairing67C:
        if any(level.get("axialPairing", {}).get("status") in {"adapter_pending", "experimental", "unavailable"} for level in disc_levels):
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


def build_disc_levels(planes: dict[str, Any], capabilities: ProductCapabilities, warnings: list[str]) -> list[dict[str, Any]]:
    sagittal = planes.get("sagittal") if isinstance(planes.get("sagittal"), dict) else {}
    axial = planes.get("axial") if isinstance(planes.get("axial"), dict) else {}
    localization_status, levels = localization_result(sagittal, capabilities)
    if localization_status == "adapter_pending":
        warnings.append("67B2_localization_adapter_pending")
    elif localization_status == "abstained":
        warnings.append("67B2_localization_abstained")
    return [
        disc_level_payload(level, localization_status, sagittal, axial, capabilities)
        for level in levels
    ]


def localization_result(sagittal: dict[str, Any], capabilities: ProductCapabilities) -> tuple[str, list[str | None]]:
    if not capabilities.absoluteDiscLocalization67B2:
        return "not_requested", [None]
    localization = sagittal.get("localization") if isinstance(sagittal.get("localization"), dict) else {}
    if localization.get("status") == "abstained":
        return "abstained", [None]
    value = localization.get("level")
    if isinstance(value, str) and value.strip():
        return "localized", [value.strip()]
    values = localization.get("levels")
    if isinstance(values, list):
        levels = [str(item).strip() for item in values if str(item).strip()]
        if levels:
            return "localized", levels
    return "adapter_pending", [None]


def disc_level_payload(
    level: str | None,
    localization_status: str,
    sagittal: dict[str, Any],
    axial: dict[str, Any],
    capabilities: ProductCapabilities,
) -> dict[str, Any]:
    pairing_status = "not_requested"
    if capabilities.axialPairing67C:
        pairing_status = "adapter_pending"
    return {
        "level": level,
        "localization": {
            "status": localization_status,
            "validated": True,
            "source": "67B2",
            "abstentionPreserved": localization_status == "abstained",
            "adapterPending": localization_status == "adapter_pending",
        },
        "axialPairing": {
            "status": pairing_status,
            "validated": False,
            "experimental": True,
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
            "adapterPending": pairing_status == "adapter_pending",
            "source": "67C",
        },
        "measurements": measurements_for_level(level, sagittal, axial),
        "degenerativeFindings": {
            "status": "blocked_preprocessing_parity",
            "automaticInferenceEnabled": False,
            "checkpointSha256": P10_7_CHECKPOINT_SHA256,
            "reviewRequired": HUMAN_REVIEW_REQUIRED,
        },
    }


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
