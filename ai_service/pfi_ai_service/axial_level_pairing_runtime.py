from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_policy import HUMAN_REVIEW_REQUIRED


SOURCE = "67C"
METHOD = "BASELINE_A_NEAREST_PLANE_PLUS_FOV"
SUMMARY_SHA256 = "4568440b9a2637996a1fc5722de0609e81c95f37f17273b7bcf0b596d5dc32bd"
ARTIFACT_PREDICTIONS_SHA256 = "9aa3778f1cbb24e6308346566dc1b95cfbe3486185ffc065a1dc42bf120c3a07"
REQUIRE_INSIDE_FOV = True
MIN_MARGIN = 0.26162020362301863
MAX_PLANE_DISTANCE_MM = 23.818094078633493


class AxialPlane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seriesId: str
    physicalIndex: int | None = None
    instanceNumber: int | None = None
    sliceLocation: float | None = None
    imagePositionPatient: tuple[float, float, float]
    rowCosines: tuple[float, float, float]
    columnCosines: tuple[float, float, float]
    pixelSpacing: tuple[float, float]
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)

    @field_validator("pixelSpacing")
    @classmethod
    def positive_spacing(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("pixel_spacing_must_be_positive")
        return value


def runtime_status() -> dict[str, Any]:
    return {
        "source": SOURCE,
        "method": METHOD,
        "artifactAvailable": False,
        "runtimeAvailable": True,
        "productEnabled": False,
        "validated": False,
        "experimental": True,
        "reviewRequired": HUMAN_REVIEW_REQUIRED,
        "parityVerified": False,
        "adapterStatus": "parity_not_proven",
        "summarySha256": SUMMARY_SHA256,
        "artifactPredictionsSha256": ARTIFACT_PREDICTIONS_SHA256,
        "requireInsideFov": REQUIRE_INSIDE_FOV,
        "minMargin": MIN_MARGIN,
        "maxPlaneDistanceMm": MAX_PLANE_DISTANCE_MM,
    }


def pair_disc_to_axial_slice(
    *,
    level: str,
    disc_centroid_patient_xyz: tuple[float, float, float] | list[float],
    axial_planes: list[AxialPlane | dict[str, Any]],
    parity_verified: bool = False,
) -> dict[str, Any]:
    if not parity_verified:
        return rejected(level=level, warnings=["67C_RUNTIME_PARITY_NOT_PROVEN"])
    planes = [plane if isinstance(plane, AxialPlane) else AxialPlane.model_validate(plane) for plane in axial_planes]
    if not planes:
        return unavailable(level=level, warnings=["67C_AXIAL_PLANES_UNAVAILABLE"])
    point = vector3(disc_centroid_patient_xyz)
    candidates = [candidate_geometry(point, plane) for plane in planes]
    valid = [
        item for item in candidates
        if item["insideFov"]
        and item["margin"] >= MIN_MARGIN
        and abs(item["planeDistanceMm"]) <= MAX_PLANE_DISTANCE_MM
    ]
    if not valid:
        return rejected(level=level, warnings=["67C_NO_VALID_PHYSICAL_PLANE"], candidates=candidates)
    best = min(valid, key=lambda item: abs(float(item["planeDistanceMm"])))
    return {
        "status": "paired",
        "source": SOURCE,
        "method": METHOD,
        "validated": False,
        "experimental": True,
        "reviewRequired": HUMAN_REVIEW_REQUIRED,
        "level": level,
        "seriesId": best["seriesId"],
        "instanceNumber": best["instanceNumber"],
        "physicalIndex": best["physicalIndex"],
        "geometry": {
            "planeDistanceMm": best["planeDistanceMm"],
            "insideFov": best["insideFov"],
            "margin": best["margin"],
            "rowFloat": best["rowFloat"],
            "colFloat": best["colFloat"],
        },
        "summarySha256": SUMMARY_SHA256,
        "artifactPredictionsSha256": ARTIFACT_PREDICTIONS_SHA256,
        "warnings": [],
    }


def candidate_geometry(point: tuple[float, float, float], plane: AxialPlane) -> dict[str, Any]:
    row = normalize(plane.rowCosines)
    col = normalize(plane.columnCosines)
    normal = normalize(cross(row, col))
    origin = vector3(plane.imagePositionPatient)
    delta = subtract(point, origin)
    row_mm = dot(delta, row)
    col_mm = dot(delta, col)
    row_float = row_mm / float(plane.pixelSpacing[0])
    col_float = col_mm / float(plane.pixelSpacing[1])
    distance = dot(delta, normal)
    inside_fov = (-0.5 <= row_float <= plane.rows - 0.5) and (-0.5 <= col_float <= plane.columns - 0.5)
    margin = min(
        row_float + 0.5,
        plane.rows - 0.5 - row_float,
        col_float + 0.5,
        plane.columns - 0.5 - col_float,
    )
    return {
        "seriesId": plane.seriesId,
        "instanceNumber": plane.instanceNumber,
        "physicalIndex": plane.physicalIndex,
        "planeDistanceMm": float(distance),
        "insideFov": bool(inside_fov),
        "margin": float(margin),
        "rowFloat": float(row_float),
        "colFloat": float(col_float),
    }


def unavailable(*, level: str, warnings: list[str]) -> dict[str, Any]:
    return base_failure("unavailable", level=level, warnings=warnings)


def rejected(*, level: str, warnings: list[str], candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = base_failure("rejected", level=level, warnings=warnings)
    if candidates is not None:
        payload["candidates"] = candidates
    return payload


def base_failure(status: str, *, level: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "source": SOURCE,
        "method": METHOD,
        "validated": False,
        "experimental": True,
        "reviewRequired": HUMAN_REVIEW_REQUIRED,
        "level": level,
        "seriesId": None,
        "instanceNumber": None,
        "geometry": None,
        "summarySha256": SUMMARY_SHA256,
        "artifactPredictionsSha256": ARTIFACT_PREDICTIONS_SHA256,
        "warnings": list(warnings),
    }


def vector3(value: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("point_must_have_three_coordinates")
    return float(value[0]), float(value[1]), float(value[2])


def subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(dot(value, value))
    if norm <= 0:
        raise ValueError("orientation_vector_zero")
    return value[0] / norm, value[1] / norm, value[2] / norm
