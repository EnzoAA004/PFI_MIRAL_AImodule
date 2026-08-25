from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


SOURCE = "67B2"
MODEL_SHA256 = "bb9e9d812972305cbc3dc87b96b95f78b7c9de4afc8ac4e6f071b35ae6fffd2"
LOCKED_PREDICTIONS_SHA256 = "baae269fe8f0e7bc7725a44ebeb1431600d7ce23d56b7dd82c05a0cbcc0405"
SUMMARY_SHA256 = "2b74c8a5b0d8ee562f6a774e136abf95225074d32c32c7646a5ca3f0fd6e8dcb"
SPINE_LEVELS = {"L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"}


class Frozen67B2Localization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    level: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    centroidPatientXYZ: tuple[float, float, float] | None = None
    warnings: list[str] = Field(default_factory=list)


def runtime_status() -> dict[str, Any]:
    return {
        "source": SOURCE,
        "artifactAvailable": False,
        "runtimeAvailable": True,
        "productEnabled": False,
        "validated": True,
        "abstentionSupported": True,
        "parityVerified": False,
        "adapterStatus": "parity_not_proven",
        "modelSha256": MODEL_SHA256,
        "lockedPredictionsSha256": LOCKED_PREDICTIONS_SHA256,
        "summarySha256": SUMMARY_SHA256,
    }


def localize_absolute_disc_levels(
    frozen_prediction: Frozen67B2Localization | dict[str, Any] | None = None,
    *,
    parity_verified: bool = False,
) -> dict[str, Any]:
    if not parity_verified:
        return unavailable("67B2_RUNTIME_PARITY_NOT_PROVEN")
    if frozen_prediction is None:
        return unavailable("67B2_FROZEN_PREDICTION_UNAVAILABLE")
    if isinstance(frozen_prediction, dict):
        allowed_keys = {"status", "level", "confidence", "centroidPatientXYZ", "warnings"}
        frozen_prediction = {
            key: value
            for key, value in frozen_prediction.items()
            if key in allowed_keys
        }
    prediction = (
        frozen_prediction
        if isinstance(frozen_prediction, Frozen67B2Localization)
        else Frozen67B2Localization.model_validate(frozen_prediction)
    )
    normalized = prediction.status.strip().lower()
    if normalized in {"abstained", "abstain"}:
        return envelope(
            status="abstained",
            level=None,
            confidence=prediction.confidence,
            abstention_preserved=True,
            centroid_patient_xyz=prediction.centroidPatientXYZ,
            warnings=prediction.warnings,
        )
    if normalized == "localized" and prediction.level in SPINE_LEVELS:
        return envelope(
            status="localized",
            level=prediction.level,
            confidence=prediction.confidence,
            abstention_preserved=False,
            centroid_patient_xyz=prediction.centroidPatientXYZ,
            warnings=prediction.warnings,
        )
    return unavailable("67B2_FROZEN_PREDICTION_INVALID", warnings=prediction.warnings)


def unavailable(reason: str, *, warnings: list[str] | None = None) -> dict[str, Any]:
    return envelope(
        status="unavailable",
        level=None,
        confidence=None,
        abstention_preserved=False,
        centroid_patient_xyz=None,
        warnings=[reason, *(warnings or [])],
    )


def envelope(
    *,
    status: str,
    level: str | None,
    confidence: float | None,
    abstention_preserved: bool,
    centroid_patient_xyz: tuple[float, float, float] | None,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "level": level,
        "confidence": confidence,
        "source": SOURCE,
        "validated": True,
        "abstentionPreserved": abstention_preserved,
        "modelSha256": MODEL_SHA256,
        "lockedPredictionsSha256": LOCKED_PREDICTIONS_SHA256,
        "summarySha256": SUMMARY_SHA256,
        "centroidPatientXYZ": list(centroid_patient_xyz) if centroid_patient_xyz is not None else None,
        "warnings": list(warnings),
    }
