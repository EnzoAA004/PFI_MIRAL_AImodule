"""EXT-03: request/response contract tests for POST /degenerative-findings/subarticular/explain.

Mocks at the same boundary `test_subarticular_runtime_api.py` mocks the prediction service at for
its own contract-shape tests -- the real, non-mocked Grad-CAM computation itself is proven
separately in test_gradcam.py against the actual frozen checkpoint (this file never duplicates
that fake-timm-backbone setup, since the fake backbone has no `bn2` attribute for Grad-CAM to
hook).
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pfi_ai_service import api as api_module
from pfi_ai_service import gradcam
from pfi_ai_service.api import app
from pfi_ai_service.subarticular_frozen_classifier import SubarticularClassifierError


def _fake_result(target_class: str = "moderate") -> gradcam.GradCamResult:
    return gradcam.GradCamResult(
        modelId="rsna_subarticular_axial_t2_2p5d",
        modelVersion="sha256:fake",
        task="subarticular_stenosis_left_right",
        targetClass=target_class,
        layer=gradcam.TARGET_LAYER,
        method="gradcam",
        heatmap=np.zeros((224, 224), dtype=np.float32) + 0.5,
        generatedAt="2026-09-06T12:00:00Z",
        experimental=True,
        disclaimer=gradcam.DISCLAIMER,
        layerDisclaimer=gradcam.LAYER_DISCLAIMER,
    )


def _request_body(**overrides) -> dict:
    body = {"inputId": "inp_fake", "instanceNumber": 9, "x": 32.0, "y": 31.0, "side": "left", "level": "L4-L5"}
    body.update(overrides)
    return body


def test_explain_returns_metadata_and_a_base64_png_heatmap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "explain_subarticular_from_registered_roi", lambda **kwargs: _fake_result())

    response = TestClient(app).post("/degenerative-findings/subarticular/explain", json=_request_body())

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "gradcam"
    assert body["experimental"] is True
    assert body["layer"] == "backbone.bn2"
    assert "disclaimer" in body and "layerDisclaimer" in body
    assert isinstance(body["heatmapPng"], str) and len(body["heatmapPng"]) > 0
    # A real base64 PNG signature ("iVBORw0KGgo..."), not a raw array dumped as text.
    import base64
    decoded = base64.b64decode(body["heatmapPng"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_explain_never_inlines_the_raw_heatmap_array(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "explain_subarticular_from_registered_roi", lambda **kwargs: _fake_result())
    response = TestClient(app).post("/degenerative-findings/subarticular/explain", json=_request_body())
    body = response.json()
    assert "heatmap" not in body  # only the encoded PNG field, never the numpy array itself


def test_explain_propagates_classifier_errors_as_public_error_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**kwargs):
        raise SubarticularClassifierError("subarticular_checkpoint_not_found")

    monkeypatch.setattr(api_module, "explain_subarticular_from_registered_roi", _raise)
    response = TestClient(app).post("/degenerative-findings/subarticular/explain", json=_request_body())

    assert response.status_code >= 400
    body = response.json()
    assert body["status"] == "error"
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    assert "traceId" in body


def test_explain_rejects_unknown_request_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "explain_subarticular_from_registered_roi", lambda **kwargs: _fake_result())
    response = TestClient(app).post("/degenerative-findings/subarticular/explain", json=_request_body(extraField="not_allowed"))
    assert response.status_code == 422


def test_explain_forwards_the_optional_target_class(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_result(target_class=kwargs.get("target_class") or "moderate")

    monkeypatch.setattr(api_module, "explain_subarticular_from_registered_roi", _capture)
    response = TestClient(app).post("/degenerative-findings/subarticular/explain", json=_request_body(targetClass="severe"))

    assert response.status_code == 200
    assert captured["target_class"] == "severe"
    assert response.json()["targetClass"] == "severe"
