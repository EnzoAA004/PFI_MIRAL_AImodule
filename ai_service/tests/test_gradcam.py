"""EXT-03: real, non-mocked sanity tests for gradcam.py against the actual frozen checkpoint.

Deliberately NOT using the `_TinyBackbone`/fake-`timm` fixture the existing
`test_subarticular_frozen_classifier.py` suite uses for its unit tests: Grad-CAM hooks
`backbone.bn2`, a real timm EfficientNet-B0 module that does not exist on the fake backbone, and
the whole point of this stage is a real local heatmap, never a mock. Skipped (not failed) when the
real checkpoint file is not present on this machine, so this suite stays runnable in environments
without the large, gitignored model artifact.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pfi_ai_service import gradcam
from pfi_ai_service.subarticular_frozen_classifier import (
    CLASS_NAMES,
    SubarticularFrozenClassifier,
    SubarticularFrozenClassifierConfig,
)

REAL_CHECKPOINT = Path(os.environ.get(
    "PFI_SUBARTICULAR_CHECKPOINT_PATH",
    r"C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVP_Juntos\PFI_MVPTest_Enzo_AImodule\models\subarticular\frozen_subarticular_checkpoint.pt",
))

pytestmark = [
    pytest.mark.real_checkpoint,
    pytest.mark.skipif(not REAL_CHECKPOINT.is_file(), reason="real frozen checkpoint not present on this machine"),
]


@pytest.fixture(scope="module")
def classifier() -> SubarticularFrozenClassifier:
    config = SubarticularFrozenClassifierConfig(checkpoint_path=REAL_CHECKPOINT, map_location="cpu")
    return SubarticularFrozenClassifier(config).load()


def _sample(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Same synthetic-input convention as the EXT-07E determinism check (documented, never a real
    # patient image, never an official validation/test case).
    return rng.integers(0, 256, size=(3, 224, 224)).astype(np.uint8)


def test_heatmap_is_not_empty_and_has_the_expected_resolution(classifier):
    result = gradcam.explain_preprocessed(classifier, _sample(1), side="left", level="L4-L5")
    assert result.heatmap.shape == (224, 224)
    assert float(result.heatmap.max()) > 0.0


def test_heatmap_values_are_normalized_between_0_and_1(classifier):
    result = gradcam.explain_preprocessed(classifier, _sample(2), side="left", level="L4-L5")
    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) <= 1.0 + 1e-6


def test_same_input_produces_a_reasonably_reproducible_map(classifier):
    sample = _sample(3)
    first = gradcam.explain_preprocessed(classifier, sample, side="left", level="L4-L5")
    second = gradcam.explain_preprocessed(classifier, sample, side="left", level="L4-L5")
    # Eval mode, no dropout active, no test-time augmentation, CPU: bit-for-bit reproducible.
    assert np.array_equal(first.heatmap, second.heatmap)


def test_different_target_class_can_produce_a_different_map(classifier):
    sample = _sample(4)
    maps = [gradcam.explain_preprocessed(classifier, sample, side="left", level="L4-L5", target_class=cls).heatmap for cls in CLASS_NAMES]
    # Not asserting all three differ (a saturated/degenerate case is possible), only that they are
    # not literally forced to be identical objects/values by construction.
    assert not (np.array_equal(maps[0], maps[1]) and np.array_equal(maps[1], maps[2]))


def test_metadata_carries_the_required_explainability_asset_fields(classifier):
    result = gradcam.explain_preprocessed(classifier, _sample(5), side="right", level="L5-S1")
    metadata = result.to_metadata()
    for field in ("modelId", "modelVersion", "task", "targetClass", "layer", "method", "generatedAt", "experimental", "disclaimer", "layerDisclaimer"):
        assert field in metadata
    assert metadata["method"] == "gradcam"
    assert metadata["experimental"] is True
    assert metadata["layer"] == "backbone.bn2"


def test_unknown_target_class_is_rejected_not_silently_ignored(classifier):
    with pytest.raises(gradcam.GradCamError):
        gradcam.explain_preprocessed(classifier, _sample(6), side="left", level="L4-L5", target_class="not_a_real_class")


def test_never_leaves_gradients_enabled_on_frozen_parameters(classifier):
    gradcam.explain_preprocessed(classifier, _sample(7), side="left", level="L4-L5")
    for parameter in classifier._model.parameters():  # noqa: SLF001 - test-only introspection
        assert parameter.requires_grad is False
        assert parameter.grad is None or bool((parameter.grad == 0).all())


def test_does_not_crash_on_repeated_calls_with_different_sides_and_levels(classifier):
    for side in ("left", "right"):
        for level in ("L1-L2", "L3-L4", "L5-S1"):
            result = gradcam.explain_preprocessed(classifier, _sample(8), side=side, level=level)
            assert result.heatmap.shape == (224, 224)
