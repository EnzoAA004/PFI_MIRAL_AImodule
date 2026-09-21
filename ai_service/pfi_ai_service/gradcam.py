"""EXT-03: Grad-CAM explainability for the frozen RSNA subarticular classifier.

Read-only with respect to the classifier itself: this module never modifies
`SubarticularFrozenClassifier`'s weights, checkpoint, or preprocessing — it only registers
temporary forward/backward hooks on the already-loaded, already-frozen model to read out
activations and gradients, and always removes them afterward. No retraining, no online/active
learning, no checkpoint change.

The heatmap explains the *visual branch* only (the EfficientNet-B0 backbone's last spatial
feature map, `backbone.bn2`, before global average pooling) — the model's final decision also
depends on the side/level embeddings concatenated after pooling, which have no spatial location
and therefore cannot be visualized this way. Every result and disclaimer generated here must make
this explicit (see `LAYER_DISCLAIMER`).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .subarticular_frozen_classifier import (
    CLASS_NAMES,
    LEVEL_TO_INDEX,
    MODEL_ID,
    SIDE_TO_INDEX,
    SubarticularClassifierError,
    SubarticularFrozenClassifier,
    SubarticularRoi,
    _validate_side_level,
    build_preprocessed_from_roi,
    preprocess_prepared_2p5d,
)

METHOD = "gradcam"
TARGET_LAYER = "backbone.bn2"
LAYER_DISCLAIMER = (
    "Este mapa de activación explica únicamente la rama visual de la predicción "
    "(la salida espacial de la última capa convolucional antes del pooling global). "
    "La predicción final también depende de los embeddings de lado y nivel, que no tienen "
    "ubicación espacial y por lo tanto no están representados en este mapa."
)
DISCLAIMER = (
    "Mapa de activación experimental asociado a la predicción del clasificador. "
    "No representa una explicación causal ni señala una zona de patología; "
    "no constituye diagnóstico clínico autónomo."
)


class GradCamError(SubarticularClassifierError):
    pass


@dataclass(frozen=True)
class GradCamResult:
    modelId: str
    modelVersion: str
    task: str
    targetClass: str
    layer: str
    method: str
    heatmap: np.ndarray  # (imageSize, imageSize) float32, normalized 0..1
    generatedAt: str
    experimental: bool
    disclaimer: str
    layerDisclaimer: str

    def to_metadata(self) -> dict:
        """Everything except the heatmap array itself — safe to serialize directly."""
        return {
            "modelId": self.modelId,
            "modelVersion": self.modelVersion,
            "task": self.task,
            "targetClass": self.targetClass,
            "layer": self.layer,
            "method": self.method,
            "generatedAt": self.generatedAt,
            "experimental": self.experimental,
            "disclaimer": self.disclaimer,
            "layerDisclaimer": self.layerDisclaimer,
            "heatmapShape": list(self.heatmap.shape),
        }


def _compute_gradcam(
    classifier: SubarticularFrozenClassifier,
    sample: np.ndarray,
    *,
    side: str,
    level: str,
    target_class: str | None,
) -> GradCamResult:
    model = classifier._require_model()  # noqa: SLF001 - reuses the classifier's own lazy-load guard, never re-implements it
    device = classifier._device or torch.device("cpu")  # noqa: SLF001
    side, level = _validate_side_level(side, level)

    image = preprocess_prepared_2p5d(sample, classifier.config).to(device)
    # Grad-CAM needs a gradient graph even though the frozen model's own parameters carry
    # requires_grad=False (set once at load, never touched here) -- making the INPUT the graph's
    # leaf is enough to get gradients back at an intermediate activation without ever computing a
    # weight gradient, so the checkpoint stays exactly as frozen as `predict_preprocessed` leaves it.
    image = image.clone().detach().requires_grad_(True)
    side_index = torch.tensor([SIDE_TO_INDEX[side]], dtype=torch.long, device=device)
    level_index = torch.tensor([LEVEL_TO_INDEX[level]], dtype=torch.long, device=device)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_inputs, grad_outputs):
        gradients["value"] = grad_outputs[0]

    target_module = model.backbone.bn2
    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(image[None], side_index, level_index)
            if target_class is None:
                class_index = int(torch.argmax(logits, dim=1).item())
            else:
                if target_class not in CLASS_NAMES:
                    raise GradCamError("unknown_target_class")
                class_index = CLASS_NAMES.index(target_class)
            score = logits[0, class_index]
            score.backward()

        if "value" not in activations or "value" not in gradients:
            raise GradCamError("gradcam_hooks_did_not_fire")

        activation = activations["value"].detach()
        gradient = gradients["value"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam_max = float(cam.max())
        if cam_max > 0:
            cam = cam / cam_max
        resized = F.interpolate(cam, size=(classifier.config.image_size, classifier.config.image_size), mode="bilinear", align_corners=False)
        heatmap = resized[0, 0].cpu().numpy().astype(np.float32)
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)

    return GradCamResult(
        modelId=MODEL_ID,
        modelVersion=classifier._checkpoint_sha256 or classifier.config.expected_checkpoint_sha256,  # noqa: SLF001
        task=classifier.config.task,
        targetClass=CLASS_NAMES[class_index],
        layer=TARGET_LAYER,
        method=METHOD,
        heatmap=heatmap,
        generatedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        experimental=True,
        disclaimer=DISCLAIMER,
        layerDisclaimer=LAYER_DISCLAIMER,
    )


def explain_preprocessed(
    classifier: SubarticularFrozenClassifier,
    sample: np.ndarray,
    *,
    side: str,
    level: str,
    target_class: str | None = None,
) -> GradCamResult:
    return _compute_gradcam(classifier, sample, side=side, level=level, target_class=target_class)


def explain_from_roi(
    classifier: SubarticularFrozenClassifier,
    roi: SubarticularRoi,
    *,
    target_class: str | None = None,
) -> GradCamResult:
    stack, metadata = build_preprocessed_from_roi(roi, classifier.config)
    return _compute_gradcam(classifier, stack, side=metadata["side"], level=metadata["level"], target_class=target_class)
