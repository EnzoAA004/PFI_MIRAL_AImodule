# Cloud Stage 5 — AI Durable Redeploy (2026-09-22/23)

Goal: deploy the current durable-asset-storage implementation
(`cloud-p0-durable-asset-persistence`, `87eae581c63bb24237761dcfe066905290d993bd`)
to the existing private Cloud Run AI service `pfi-ai-module`, keep it
private, prove durable GCS write/read, and prove rehydration after a
real restart. Backend and Frontend were not touched.

**Result: partial success.** Durable GCS writes are fully proven
end-to-end with real checksums. Rehydration -- reading a previously
written asset from a fresh container instance -- **fails**, and the
cause is a confirmed source-code gap, not a deployment, IAM, or
configuration problem.

## 1. Pre-build verification (Steps 1-3)

- Source state confirmed via `git log`/`git status`: branch
  `cloud-p0-durable-asset-persistence`, HEAD `87eae581c63b...`,
  clean tree -- matches the expected commit exactly.
- Durable-storage implementation confirmed present via direct source
  read: `ai_service/pfi_ai_service/object_storage.py`
  (`ObjectStorage`, `FilesystemObjectStorage`,
  `GoogleCloudStorageObjectStorage`, `get_object_storage()`) and
  `asset_registry.py` (`resolve_run_asset`, `rehydrate_run_asset`).
- Local test environment had a corrupted `torch` install
  (`_C.cp312-win_amd64.pyd` present but only 10,752 bytes -- far too
  small to be the real compiled extension). Root-caused and fixed with
  `pip install --force-reinstall --no-deps torch==2.12.1` (123 MB
  download, clean reinstall). Unrelated to durable-storage code, as
  anticipated.
- Full durable-storage test suite after the fix:

  ```
  ai_service/tests/test_object_storage.py
  ai_service/tests/test_asset_registry.py
  ai_service/tests/test_asset_serving.py
  ============ 43 passed, 129 warnings in 13.91s ============
  ```

  Matches the historical baseline of 43/43 exactly.

## 2. Live baseline before redeploy (Step 4) -- verified, not assumed

- **Service**: `pfi-ai-module`, single revision `pfi-ai-module-00001-m62`
  (created 2026-08-11, never redeployed since).
- **Image**: `...pfi-ai-module@sha256:c54475f2...`, tag `p10-9-99b71d9`.
- **Resources**: 4 vCPU / 2Gi, concurrency=1, timeout=300s, gen2,
  `startup-cpu-boost=true`, `maxScale=1`, no `minScale` (default 0).
- **Service account**: `pfi-ai-runtime@pfi-asplanatti-fabrello-v1.iam.gserviceaccount.com`.
- **Privacy**: `gcloud run services get-iam-policy` returned no
  bindings at all (`{"etag": "ACAB"}`) -- no `allUsers`/
  `allAuthenticatedUsers`, private/authenticated-only, confirmed.
- **Env vars**: only `PFI_P10_7_CHECKPOINT_PATH` and
  `PFI_P10_7_DEVICE=cpu` -- no `PFI_STORAGE_*` yet.
- **GCS mount**: read-only `gcsfuse` volume at `/mnt/pfi-artifacts` ->
  `pfi-rm-lumbar-artifacts-2026-ef` (model checkpoints).
- **IAM on `pfi-backend-durable-dev`**: `pfi-ai-runtime` has exactly
  `roles/storage.objectCreator` + `roles/storage.objectViewer`, no
  `storage.admin` -- already correctly scoped from Stage 2.
- **IAM on `pfi-rm-lumbar-artifacts-2026-ef`**: `pfi-ai-runtime` has
  only `roles/storage.objectViewer`, matching the read-only mount.

## 3. Build and push (Steps 5-6)

- Built from the clean, verified `87eae581c63b...` tree:
  `docker build -t
  us-central1-docker.pkg.dev/.../pfi-ai-module:ai-durable-87eae58 .`
- Tag: `ai-durable-87eae58` (immutable, no rebuild of existing tags,
  no deletion of `p10-9-99b71d9`).
- Digest: `sha256:44933f3467d0b67c843217368956eea1f099d151e71ef449c32088ff0a7e7de2`,
  856-byte manifest, ~2.17 GB image.
- Pushed and independently verified present in Artifact Registry via
  `gcloud artifacts docker images list` (not just trusted from the
  `docker push` output) -- matches, `p10-9-99b71d9` untouched.

## 4. Redeploy (Steps 7-10)

```
gcloud run deploy pfi-ai-module \
  --image=...@sha256:44933f3467d0... \
  --service-account=pfi-ai-runtime@... \
  --execution-environment=gen2 \
  --cpu=4 --memory=2Gi --concurrency=1 --timeout=300 \
  --min-instances=0 --max-instances=1 \
  --no-allow-unauthenticated --ingress=all \
  --set-env-vars=PFI_P10_7_CHECKPOINT_PATH=...,PFI_P10_7_DEVICE=cpu,
                 PFI_STORAGE_PROVIDER=gcs,
                 PFI_STORAGE_GCS_BUCKET=pfi-backend-durable-dev \
  --add-volume=name=cloud-storage-pav-goc,type=cloud-storage,
                bucket=pfi-rm-lumbar-artifacts-2026-ef,readonly=true \
  --add-volume-mount=volume=cloud-storage-pav-goc,mount-path=/mnt/pfi-artifacts
```

All prior resource settings preserved exactly (4 vCPU/2Gi,
concurrency=1, min=0/max=1, gen2, timeout=300s, service account,
ingress). The read-only checkpoint mount was re-specified identically
-- confirmed byte-for-byte the same `volumeMounts`/`volumes` block in
the resulting service spec as the pre-redeploy baseline.

Result: **`pfi-ai-module-00002-ml8`** deployed, `Ready=True`,
`ConfigurationsReady=True`, `RoutesReady=True`, serving 100% traffic.

## 5. Startup validation (Step 11) -- PASS

Logs for the new revision show a clean GCSFuse mount
(`File system has been successfully mounted`, bucket
`pfi-rm-lumbar-artifacts-2026-ef`) followed by a clean FastAPI/Uvicorn
startup (`Application startup complete`, `Uvicorn running on
0.0.0.0:8000`) and a successful startup TCP probe. Zero
`severity>=WARNING` entries in the startup window.

## 6. Authenticated / unauthenticated smoke (Step 12) -- PASS

- Unauthenticated `GET /health` and `GET /` -> **403** both times.
- Authenticated (`Authorization: Bearer $(gcloud auth print-identity-token)`)
  `GET /health` -> **200**, real service payload:
  `modelsConfigured: true`, `outputStorageConfigured: true`,
  `discMultitask` checkpoint `configured: true, artifactPresent: true`.

IAM invoker policy re-checked after deploy: still empty (no bindings)
-- confirmed the service stayed private through the redeploy.

## 7. Durable GCS write/read test (Step 13) -- PASS

Real (synthetic, non-PHI) inference run against the baked-in fixture
`ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy`
(confirmed present in the built image via `docker run ... ls`):

```
POST /pipeline/run
{"caseId": "CASE-STAGE5-DURABLE-TEST", "plane": "sagittal",
 "modelKey": "sagittal_spider",
 "inputPath": "/app/ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy",
 "metadata": {"inferenceMode": "real_baseline", "allowContractFallback": false,
              "traceId": "trace-stage5-durable-test"}}
-> 200, runId = 3b1c11af99f88238
```

All 8 assets reported `"storageProvider": "gcs"` with keys
`runs/3b1c11af99f88238/sagittal/{assetName}` (no PHI, patientId
de-identified as `PAT-DEIDENTIFIED`). Independently verified in GCS
(not just trusted from the API response):

```
gcloud storage ls gs://pfi-backend-durable-dev/runs/3b1c11af99f88238/sagittal/
-> all 8 objects present
gcloud storage objects describe gs://.../overlay.png
-> size=67919 (matches API-reported size exactly)
```

Fetched `overlay.png` via the authenticated app endpoint
(`GET /assets/{run_id}/sagittal/overlay.png`) on the same instance
that wrote it -> 200, SHA-256 of the fetched bytes matched the
API-reported checksum exactly
(`1ad7a4cece6cb1111b20341283f588076f8551a730e5c022ac502e42308db061`).
**Durable write and same-process read are fully proven.**

## 8. Rehydration test (Step 14) -- **FAIL -- core finding of this stage**

To prove rehydration (not just same-process caching), a fresh
container instance was forced by redeploying the identical image with
only a label change (`gcloud run services update
--update-labels=stage5-rehydration-test=true`) -> new revision
**`pfi-ai-module-00003-jlm`**, a genuinely new process with an empty
in-memory asset registry.

Fetching the same run's assets from this fresh instance:

```
GET /assets/3b1c11af99f88238/sagittal/mask.npy     -> 404
GET /assets/3b1c11af99f88238/sagittal/overlay.png  -> 404
```

Both objects were independently re-confirmed still present in GCS at
the time of the failed fetch (unchanged, undeleted).

**Root cause, confirmed by reading the source (not inferred):**

- `ObjectStorage.local_path()` (`object_storage.py:48-49`) returns
  `None` by default.
- `GoogleCloudStorageObjectStorage` (`object_storage.py:110-163`)
  never overrides `local_path()`. Its `open()` method even documents
  this explicitly: `"gcs objects must be materialized before local
  serving"` (`object_storage.py:157`).
- `asset_registry.load_asset_index_record()`
  (`asset_registry.py:323-352`) unconditionally calls
  `storage.local_path(object_key)` during rehydration and returns
  `None` (-> 404 upstream) whenever that comes back `None`
  (`asset_registry.py:340-341`).
- Searched the codebase for any GCS-materialization path used by the
  asset-serving flow (`materializ`, `download_to`,
  `download_as_filename`) -- the only matches
  (`gcs_release_materializer.py`, `model_materializer.py`) are an
  unrelated mechanism for downloading *model checkpoint releases*, not
  run assets, and are never called from `asset_registry.py`.

**This means: for the `gcs` storage provider, durable writes succeed,
but the asset-serving/rehydration read path can never succeed once an
asset leaves the in-memory cache -- deterministically, independent of
restart timing, IAM, or bucket configuration.** It is not specific to
this particular redeploy; any second process, any horizontal replica,
or any real restart would reproduce the same 404.

This was surfaced to the user mid-stage. Decision (confirmed
explicitly): keep the new revision live (it is healthy, private, and
costs nothing extra at `min=0`), document the gap honestly, and do
not roll back -- since rolling back to `pfi-ai-module-00001-m62` would
only remove the (working) durable-write capability without fixing
anything, and the new revision is not operationally unhealthy.

## 9. Scale-to-zero / cold-start cycle (Step 15) -- cost/latency evidence only

Given Step 8's finding is deterministic and code-level (not a
timing/warm-vs-cold artifact), a natural scale-to-zero -> cold-start ->
warm-request cycle was still run once, for cost and latency evidence
only (not to re-test rehydration, which is already conclusively
proven broken by source inspection and a live 404).

The service was left idle (no traffic) from `2026-09-23T00:26:10Z`
(end of the rehydration test) until the next authenticated request at
`2026-09-23T12:41:57Z` -- over 12 hours idle, comfortably past the
~15-minute scale-to-zero window observed in earlier stages. Evidence
pulled directly from Cloud Logging (not client-side timing, which was
unreliable due to a missing `bc` binary in the shell):

| Request | Timestamp | Latency | Status | Note |
|---|---|---|---|---|
| Cold | `12:41:57.908213Z` | **21.869s** | 200 | `Starting new instance. Reason: AUTOSCALING` -- genuine scale-from-zero, not a deploy-triggered start |
| Warm #1 | `12:42:20.352992Z` | **0.040s** | 200 | immediately after cold start |
| Warm #2 | `12:42:32.904678Z` | **0.042s** | 200 | confirms stable warm state |

Cold-start sequence: GCSFuse mount completed in ~0.23s
(`12:42:00.531`-`12:42:00.764`), then a ~18.5s gap before
`Application startup complete` (`12:42:19.313`) -- consistent with
Python/FastAPI/model-registry import cost on a fresh process, not a
GCS-storage-specific delay. This matches the ~20-21s cold-start range
observed for the Backend service in Stage 4I, confirming this is a
normal Cloud Run cold-start profile for this image size (~2.17 GB),
not a Stage-5-introduced regression.

## 10. Model / checkpoint safety (Steps 16-17)

- The deployed image and revision use the same
  `PFI_P10_7_CHECKPOINT_PATH` pointing at the same GCS-mounted
  checkpoint path as the pre-Stage-5 baseline -- no model or checkpoint
  change was made in this stage.
- No clinical-validation claims (67B2/67C/P10.7 accuracy/performance)
  are made here beyond the runtime evidence collected: the checkpoint
  loads (`artifactPresent: true`) and the health payload continues to
  report `humanReviewRequired: true, notClinicalDiagnosis: true` on
  every response, unchanged.

## 11. Rollback readiness (Step 19)

Not executed -- not needed per the decision in Section 8. `pfi-ai-module-00001-m62`
remains present and `Ready=True` (confirmed via `gcloud run revisions
list`), available as an immediate rollback target
(`gcloud run services update-traffic --to-revisions=pfi-ai-module-00001-m62=100`)
if ever required.

## 12. Cost observations (Step 18)

- `min=0, max=1` unchanged from baseline -- idle compute cost remains
  $0 between requests.
- Two redeploys (image + storage config, then label-only for the
  rehydration test) -- each a few seconds of build-time Cloud Run
  control-plane activity, no additional fixed cost.
- No new GCP resources created; `pfi-backend-durable-dev` bucket and
  its IAM bindings were already provisioned in Stage 2.
- Checkpoint bucket `pfi-rm-lumbar-artifacts-2026-ef` was never
  written to or deleted from during this stage -- read-only mount
  confirmed unchanged throughout.

## 13. Summary

> **Update 2026-09-23 (Stage 5B):** the rehydration gap documented below
> was root-caused and fixed in
> [`CLOUD_STAGE_5B_GCS_REHYDRATION_FIX_REPORT.md`](CLOUD_STAGE_5B_GCS_REHYDRATION_FIX_REPORT.md).
> Fresh-process rehydration now passes live with exact checksum matches
> (commit `c854f0e`, revision `pfi-ai-module-00006-ppg`).
> `READY_FOR_BACKEND_AI_AUTH_INTEGRATION` is now **YES** as of Stage 5B,
> superseding the NO recorded below.


| Check | Result |
|---|---|
| Source state matches expected commit | PASS |
| Durable-storage tests (43/43) | PASS |
| Live baseline recorded before change | PASS |
| Image built/pushed with immutable tag | PASS |
| Redeploy healthy (Ready/Configurations/Routes) | PASS |
| Service remains private | PASS |
| Unauthenticated rejected | PASS (403) |
| Authenticated invocation works | PASS (200) |
| Durable GCS write | PASS |
| Durable GCS read (same process) | PASS, checksum match |
| **Rehydration after real restart** | **FAIL -- source-code gap** |
| Checkpoint mount preserved | PASS, unchanged |
| Legacy revision available for rollback | PASS, untouched |
