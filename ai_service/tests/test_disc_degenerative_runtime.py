import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient
from torch import nn

from pfi_ai_service.api import app
from pfi_ai_service.disc_degenerative_runtime import (
    DiscDegenerativeClassifier,
    DiscDegenerativeInvalidRequest,
    PreparedDiscLevel,
    get_disc_degenerative_runtime_status,
)


class FakeDiscModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, t1, t2, availability, ivd_index):
        return {
            "pfirrmann_grade": torch.tensor([[0.0, 0.1, 3.0, 0.2, 0.1]], device=t1.device),
            "modic_change": torch.tensor([[3.0, 0.1, 0.1, 0.1]], device=t1.device),
            "upper_endplate_change": torch.tensor([[2.0]], device=t1.device),
            "lower_endplate_change": torch.tensor([[2.0]], device=t1.device),
            "spondylolisthesis": torch.tensor([[-2.0]], device=t1.device),
            "disc_herniation": torch.tensor([[-2.0]], device=t1.device),
            "disc_narrowing": torch.tensor([[2.0]], device=t1.device),
            "disc_bulging": torch.tensor([[2.0]], device=t1.device),
        }


def classifier_with_fake_model() -> DiscDegenerativeClassifier:
    classifier = DiscDegenerativeClassifier.__new__(DiscDegenerativeClassifier)
    classifier.model = FakeDiscModel().eval()
    classifier.metadata = {}
    return classifier


def crop(value: float = 0.5) -> np.ndarray:
    return np.full((3, 224, 224), value, dtype=np.float32)


@pytest.mark.parametrize(
    "sample",
    [
        PreparedDiscLevel(level="L4-L5", t1=crop(), t2=crop(), t1_positions=(10, 11, 12), t2_positions=(9, 10, 11)),
        PreparedDiscLevel(level="L4-L5", t1=crop(), t2=None, t1_positions=(10, 11, 12)),
        PreparedDiscLevel(level="L4-L5", t1=None, t2=crop(), t2_positions=(9, 10, 11)),
    ],
)
def test_preprocessed_prediction_contract_for_dual_and_single_modality(sample: PreparedDiscLevel) -> None:
    envelope = classifier_with_fake_model().predict_preprocessed([sample])
    findings = envelope["discDegenerativeFindings"]["findings"]
    assert len(findings) == 8
    assert envelope["humanReviewRequired"] is True
    assert envelope["autonomousDiagnosis"] is False
    for finding in findings:
        probabilities = finding["classification"]["probabilities"]
        assert finding["review"] == {"required": True, "status": "pending"}
        assert finding["classification"]["label"] == max(probabilities, key=probabilities.get)
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in probabilities.values())


def test_preprocessed_prediction_rejects_absent_modalities() -> None:
    with pytest.raises(DiscDegenerativeInvalidRequest):
        classifier_with_fake_model().predict_preprocessed([
            PreparedDiscLevel(level="L4-L5", t1=None, t2=None)
        ])


def test_runtime_status_is_safe_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFI_P10_7_CHECKPOINT_PATH", raising=False)
    status = get_disc_degenerative_runtime_status(verify_hash=True)
    assert status["status"] == "not_configured"
    assert status["preprocessingParityValidated"] is False
    assert "path" not in status


def test_endpoint_rejects_productive_request_until_preprocessing_parity_is_validated() -> None:
    response = TestClient(app).post(
        "/degenerative-findings/disc-multitask/predict",
        json={
            "caseId": "case-1",
            "levels": [
                {
                    "level": "L4-L5",
                    "sourceSeries": [
                        {"role": "sagittal_t1", "available": True, "positions": [1, 2, 3]}
                    ],
                }
            ],
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "DISC_DEGENERATIVE_PREPROCESSING_NOT_AVAILABLE"


def test_findings_publish_the_evaluated_region_of_each_available_series() -> None:
    """T1 y T2 se localizan por separado: cada rol lleva su corte y su region.

    Publicarlas juntas obligaria a elegir una, y el revisor terminaria mirando el
    corte de la otra secuencia creyendo que es el que se clasifico.
    """
    t1_region = {
        "inputId": "inp-t1",
        "sliceIndex": 11,
        "pointYx": [120.0, 96.0],
        "bboxYx": [108, 132, 84, 108],
        "coordinateSpaceHeight": 256,
        "coordinateSpaceWidth": 200,
    }
    sample = PreparedDiscLevel(
        level="L4-L5",
        t1=crop(),
        t2=None,
        t1_positions=(10, 11, 12),
        t1_region=t1_region,
        localization_source="segmentation_derived_disc_level",
    )

    envelope = classifier_with_fake_model().predict_preprocessed([sample])
    sources = {
        item["role"]: item
        for item in envelope["discDegenerativeFindings"]["findings"][0]["sourceSeries"]
    }

    assert sources["sagittal_t1"]["positions"] == [10, 11, 12]
    assert sources["sagittal_t1"]["sliceIndex"] == 11
    assert sources["sagittal_t1"]["pointYx"] == [120.0, 96.0]
    # La modalidad que no entro no inventa geometria.
    assert sources["sagittal_t2"]["available"] is False
    assert "pointYx" not in sources["sagittal_t2"]


def test_findings_omit_the_region_when_the_roi_did_not_come_from_segmentation() -> None:
    sample = PreparedDiscLevel(level="L4-L5", t1=crop(), t2=None, t1_positions=(10, 11, 12))

    envelope = classifier_with_fake_model().predict_preprocessed([sample])
    source = envelope["discDegenerativeFindings"]["findings"][0]["sourceSeries"][0]

    assert source["positions"] == [10, 11, 12]
    assert "pointYx" not in source and "bboxYx" not in source
