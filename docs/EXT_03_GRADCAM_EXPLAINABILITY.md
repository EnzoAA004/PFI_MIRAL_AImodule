# EXT-03 — Classifier Explainability (Grad-CAM)

Experimental, never a diagnosis. This is the one extension that touches AI, and only additively:
no existing file was modified, no checkpoint changed, no retraining, no active/online learning.

Branches (HEAD verified before branching — never assumed from memory):

- AI: `extensions/ext-03-explainability-ai`, worktree `PFI_MVPTest_Enzo_AImodule_ext03`, base
  `f5b9380` (this session's own `enzo/p10-7-spider-degenerative-multitask` HEAD, confirmed clean
  via `git status` immediately before branching — never a historical head used without
  inspection).
- Backend: `extensions/ext-03-explainability-backend`, worktree
  `PFI_MVPTest_Enzo_Backend_ext03`, base `b097078` (EXT-04B5).
- Frontend: `extensions/ext-03-explainability-frontend`, worktree
  `PFI_MVPTest_Enzo_Frontend_ext03`, base `83b9209` (EXT-04B5).

## Model chosen and why

The frozen RSNA subarticular classifier (`ai_service/pfi_ai_service/subarticular_frozen_classifier.py`)
already uses an **EfficientNet-B0** backbone (via `timm`) — exactly the spec's preferred choice —
with a real, checkpoint-hash-verified, `eval()`-frozen model (`EXPECTED_CHECKPOINT_SHA256`). It
also has a real, local, gitignored checkpoint file already present on this machine
(`models/subarticular/frozen_subarticular_checkpoint.pt`, 48.6MB), a controlled, reproducible ROI
input path, and — critically — is **not** gated by the blocked P10.7 preprocessing-parity issue
(that gate applies to the disc-level degenerative-findings pipeline, a different classifier). No
other classifier in this codebase was as immediately real-inference-ready.

## `gradcam.py`

New module, additive only — imports from `subarticular_frozen_classifier.py` but never edits it.
Hooks `backbone.bn2` (the EfficientNet-B0 backbone's last spatial feature map, before global
average pooling) with a temporary forward + full-backward hook, always removed in a `finally`
block. Standard Grad-CAM: global-average-pool the gradients per channel, weight the activation map
by them, ReLU, normalize to `[0, 1]`, upsample to `224×224`.

**Frozen means frozen, still.** The classifier's parameters keep `requires_grad=False` (set once
at `load()`, untouched) — Grad-CAM instead makes the **input tensor** the graph's leaf
(`.requires_grad_(True)`), which is enough for autograd to produce activation gradients without
ever computing (or needing) a weight gradient. Verified by a dedicated test that no parameter ends
up with a non-zero gradient after explaining.

**Layer-scope honesty, built into the contract.** The model's final decision also depends on
`side_embedding`/`level_embedding`, concatenated *after* the pooled visual features — those have
no spatial location and cannot be visualized. Every `GradCamResult` carries a permanent
`layerDisclaimer` field saying exactly this, not just a one-off UI label.

### Explainability asset contract

`modelId, modelVersion (checkpoint sha256), task, targetClass, layer, method="gradcam", generatedAt,
experimental=true, disclaimer, layerDisclaimer` + the heatmap itself (kept as a numpy array
internally; the HTTP layer encodes it as a small base64 PNG, never inline as a raw array — see
below).

## Sanity tests (real, not mocked)

`ai_service/tests/test_gradcam.py` — **8/8 passing**, run against the actual frozen checkpoint
(skipped, not failed, on a machine without it — `@pytest.mark.real_checkpoint`, same convention
`test_subarticular_runtime_api.py` already established):

1. Heatmap is non-empty and has the expected `224×224` resolution.
2. Values are normalized to `[0, 1]`.
3. Same input → bit-for-bit identical map (eval mode, no dropout, no TTA, CPU — genuinely
   deterministic, verified with `np.array_equal`, not an approximate comparison).
4. A different target class can produce a different map (not asserted to always differ — a
   saturated/degenerate case is possible — only that all three are not forced identical).
5. The metadata carries every required explainability-asset field.
6. An unknown target class is rejected (`GradCamError`), never silently ignored.
7. No frozen parameter ends up with a nonzero gradient after explaining.
8. Repeated calls across every side/level combination never crash.

Full existing `ai_service` test suite re-run after these additions: **144 passed** (excluding 4
failures and 1 collection error that are pre-existing on the pristine, untouched worktree —
verified identically reproducible there, confirming EXT-03 introduced zero regressions; the 153
`PermissionError` "errors" reported alongside are Windows-specific `tmp_path` teardown noise,
also reproducible on the pristine worktree, unrelated to any code).

## AI service HTTP endpoint

`POST /degenerative-findings/subarticular/explain` — new route in `api.py`, mirroring
`/degenerative-findings/subarticular/predict` exactly (same ROI resolution via
`_registered_axial_input`, same structured-error response shape, same `humanReviewRequired`/
`notClinicalDiagnosis` flags). Returns the metadata fields plus `heatmapPng` (base64, ~a few KB at
`224×224` grayscale — never a "blob gigante", per instruction). `pillow` and `timm` were already
declared `ai_service` dependencies before this stage — no new dependency, no Dockerfile change
needed to run this in the real AI service image.

## Backend proxy

New `POST /api/ai/degenerative-findings/subarticular/explain` in
`AiDegenerativeFindingsController` — same allowlisted-fields validation as `/subarticular`
(refactored the shared "reject unknown fields" + "parse ROI fields" logic into two reusable
private methods so neither endpoint duplicates the other's validation), plus a `targetClass`
allowlist check (`normal_mild|moderate|severe`, rejected before ever calling the AI service if
invalid). `AiServiceClient.explainSubarticular` mirrors `predictSubarticular`'s exact
WebClient/error-mapping/metrics pattern. **17/17** controller tests passing (11 pre-existing +
6 new: success, optional `targetClass` forwarded, invalid `targetClass` rejected without calling
the AI service, unknown field rejected, `502` on a malformed upstream response, `503` when the
checkpoint isn't loaded). Full Backend regression: **995 run, 0 failures, 0 errors, 1 skipped**
(JDK 21).

## Frontend

`ActivationMapPanel.tsx` — a new "Mapa de activación asociado a la predicción" toggle (off by
default) next to the existing subarticular-ROI classification flow (`DegenerativeFindingsPanel`),
gated on the same ROI (`side`/`level`) a professional already marked. An opacity slider controls
the heatmap image's rendered opacity. Both of the Backend's own disclaimers are always rendered
alongside the image — never just the image alone. Never labeled "zona de la patología" or
"explicación causal". 6 new pure-function contract tests (endpoint path, body shape, optional
`targetClass` forwarding, response pass-through, no banned wording) — **6/6 passing**; full
`npm test`/`typecheck`/`lint`/`build` chain green.

## E2E — scope and honest limitation

The **real, non-mocked local AI inference** required as this stage's "prueba principal" was
proven directly against the actual frozen checkpoint (`test_gradcam.py`, 8/8, no mock). The
Backend↔AI-service wiring was proven with a mocked `AiServiceOperations` at the controller layer —
the same testing convention this codebase's own pre-existing `/subarticular/predict` endpoint
already uses (its own controller tests mock the same interface identically). A full three-tier
live Docker chain (real AI-service container with the checkpoint mounted → real Backend → real
Frontend, one connected request) was **not** executed this session — building and running the
real `ai_service` Docker image (installing `torch`+`timm`, mounting the 48MB gitignored
checkpoint) was judged out of scope for this stage's remaining time budget, and is named here
explicitly rather than fabricated. What *is* proven end-to-end is the exact same testing depth
this repository already accepts for the sibling `/predict` endpoint it mirrors.

## Frontend manual flow

Reading Room → Hallazgos IA tab → mark a subarticular ROI (existing flow, unchanged) → toggle
"Mapa de activación asociado a la predicción" → heatmap renders with the opacity slider and both
disclaimers.

## Limitations

1. No full three-service live Docker E2E was run this stage (see above) — the real-inference
   claim is backed by direct, non-mocked pytest against the actual checkpoint instead.
2. The heatmap explains only the visual (backbone) branch — the side/level embeddings are not
   represented, and every result says so explicitly (`layerDisclaimer`).
3. Grad-CAM++ was not implemented (explicitly optional in the spec, skipped to keep this stage's
   scope to the one required method).
4. Only the subarticular classifier was instrumented — no other classifier in this product got a
   Grad-CAM path this stage.
5. This is a research/explainability aid, never a diagnosis, never a severity signal, never a
   substitute for professional judgment.

**Status: `EXT_03_GRADCAM_EXPLAINABILITY_READY`** — real local Grad-CAM/Backend proxy/Frontend
toggle/regressions all backed by real evidence gathered this session, with the one named E2E
scope limitation above. READY_TO_COMMIT — no commit, no push, no merge performed on any of the
three repos.
