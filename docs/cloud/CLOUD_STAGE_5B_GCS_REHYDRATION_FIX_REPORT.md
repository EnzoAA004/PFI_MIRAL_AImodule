# Cloud Stage 5B — GCS Asset Rehydration Fix (2026-09-23)

Goal: make GCS-backed AI run assets retrievable after a real process
restart / scale-to-zero, without changing the external API contract.
Fixes the blocker found in Stage 5
(`docs/cloud/CLOUD_STAGE_5_AI_DURABLE_REDEPLOY_REPORT.md`, section 8):
`REHYDRATION_AFTER_RESTART = NO`.

**Result: fixed and proven live.** A fresh process now rehydrates a
previously-written asset with an exact SHA-256 match, at the cost of a
one-time ~0.5-2.9s materialization per asset; a second read of the
same asset in the same process is ~4.5ms (local cache reuse). No
Backend, Frontend, Cloud SQL, or IAM changes were made.

## 1. Root cause (confirmed by source reading, Stage 5)

Two independent gaps, both required for rehydration to work at all:

1. `GoogleCloudStorageObjectStorage.local_path()` was never implemented
   (inherited the base class's `return None`), so `asset_registry`'s
   rehydration path always got `None` and 404'd, regardless of whether
   the blob still existed in GCS.
2. The asset index (the `object_key`/`sha256`/`size` lookup a fresh
   process needs to even know *what* to materialize) was written only
   to local ephemeral disk (`PFI_OUTPUT_DIR/asset_registry/...`)
   regardless of storage provider -- so even a correct blob download
   would have had no metadata to look up in a fresh container.

## 2. Fix design

### 2a. Real GCS materialization (`object_storage.py`)

`GoogleCloudStorageObjectStorage` now implements `local_path()` /
`materialize()`:

1. Resolve a deterministic local cache path under a dedicated,
   configurable cache root (default `<tempdir>/pfi-ai-gcs-cache`,
   override via `PFI_STORAGE_GCS_CACHE_ROOT`) -- separate from
   `PFI_OUTPUT_DIR` so it never collides with pipeline working files.
2. If already cached locally, return immediately (no GCS round trip).
3. Otherwise: check the blob exists, `reload()` for its declared size,
   download to a `.tmp-<uuid>` file, verify the downloaded byte count
   matches the declared size (fails closed with a clear
   `ObjectStorageError` on mismatch -- a first line of defense before
   the asset_registry layer's separate SHA-256 check), then atomically
   `rename()` into place.
4. `open()` now materializes-then-opens instead of unconditionally
   raising; `delete()` also evicts the local cache entry.

No public signed URLs, no bucket-mount dependency, no service-account
JSON, no in-memory-only bytes -- exactly the constraint given.

### 2b. Durable asset metadata, without IAM broadening (`asset_registry.py`)

First implementation attempt used one shared, repeatedly-overwritten
`_index.json` per run/plane. **That failed live** (see section 3): the
`pfi-ai-runtime` service account only has `roles/storage.objectCreator`
(create) + `roles/storage.objectViewer` (get/list) on the durable
bucket -- create-only does not cover overwriting an existing object,
and the shared index was overwritten once per asset (up to 8x per
run).

Final design: one **immutable metadata object per asset**,
`runs/{run_id}/{plane}/_meta/{asset_name}.json`, written exactly once
per `(run_id, plane, asset_name)` in normal operation -- fits
create-only permissions with zero IAM changes. `persist_asset_index()`
still writes the existing local index file unchanged (so the
filesystem provider and its file layout are 100% unaffected), and
additionally uploads just that one asset's metadata to its own key
when `storage_provider == "gcs"`.

Reading (`load_asset_index_record`): try the local index file first
(fast path, same as before); on a cache miss, materialize that single
asset's metadata key from GCS instead of a shared index, then backfill
the local index file so later reads in the same process skip GCS
entirely.

Listing all of a run's assets (`rehydrate_assets_for_run`, used by
`registered_assets_for_run`, which itself is called from
`real_inference_runtime.py` while building the `pipeline/run`
response): added `ObjectStorage.list_keys(prefix)`, implemented for
GCS via `bucket.list_blobs(prefix=...)` (list is covered by
`objectViewer`, already granted) so a fresh process can enumerate
`runs/{run}/{plane}/_meta/` instead of depending on a combined index.

## 3. Live deploy attempt #1 -- caught the overwrite-permission gap

Built/pushed/deployed `ai-durable-rehydrate-f1f1c4a` (source commit
`f1f1c4a`) with the shared-index design. `POST /pipeline/run` against
the live private service returned **500**:

```
google.api_core.exceptions.Forbidden: 403 POST
https://storage.googleapis.com/upload/storage/v1/b/pfi-backend-durable-dev/o
...
File ".../asset_registry.py", line 335, in persist_asset_index
    storage.put_bytes(payload_text.encode("utf-8"), index_key, "application/json")
```

This was full real evidence the fix's IAM assumption was wrong,
caught immediately rather than after a redeploy declared "done" --
consistent with this session's evidence-over-assumption practice. No
IAM broadening was used to work around it (not authorized); the index
design was corrected instead. This is source commit `f1f1c4a`,
superseded by `c854f0e` below -- kept in git history, not force-pushed
over.

## 4. Tests

### Focused (Section 7 of the spec)

- **A** (materializes absent object): `test_gcs_storage_materializes_absent_object_into_local_cache`
- **B** (second read reuses cache): `test_gcs_storage_second_read_reuses_local_cache_without_redownload`
- **C** (checksum mismatch rejected): `test_asset_registry_gcs_rehydration_fails_clearly_on_checksum_mismatch`
  (asset_registry layer) + `test_gcs_storage_materialize_rejects_size_mismatch`
  (object_storage layer, defense in depth)
- **D** (missing GCS object -> controlled not-found): `test_gcs_storage_materialize_returns_none_for_missing_object`,
  `test_asset_registry_gcs_rehydration_returns_404_when_object_missing_from_bucket`
- **E** (path traversal key rejected): `test_gcs_storage_materialize_rejects_traversal_key`
- **F** (fresh registry/process resolves same durable asset -- the core
  local proof): `test_asset_registry_rehydrates_gcs_backed_asset_in_fresh_process`,
  `test_asset_registry_rehydrates_multiple_gcs_backed_assets_in_fresh_process`

All against a lightweight fake GCS test double (`_FakeBlob`/`_FakeBucket`,
a plain dict backing store shared across independently-constructed
`GoogleCloudStorageObjectStorage` instances, each with its own cache
root -- simulating separate container processes sharing only the
bucket, never local disk). The fake does not enforce IAM, so a
dedicated test (`test_asset_registry_gcs_persists_one_immutable_meta_key_per_asset`)
explicitly asserts two assets in the same run get distinct GCS keys,
guarding against regressing back to the shared/overwritten design that
real IAM caught.

### Storage contract + full durable-storage + pipeline suites

```
ai_service/tests/test_object_storage.py ..... 20/20 (15 historical + 5 new)
ai_service/tests/test_asset_registry.py ..... 28/28 (18 historical + 10 new)
ai_service/tests/test_asset_serving.py ...... 6/6 (unchanged)
= 53/53 passed (43 historical baseline + 10 new)

+ test_input_id_pipeline.py, test_pipeline_contract.py,
  test_pipeline_strict_real_baseline_contract.py,
  test_pipeline_trace_endpoint.py, test_pipeline_traceability.py,
  test_real_baseline_pipeline.py: 12/12
= 65/65 total, zero regressions, filesystem provider untouched.
```

No unrelated environment failures this stage (the `torch` install fix
from Stage 5 carried over cleanly).

## 5. Commits

- `f1f1c4a` -- `fix: rehydrate GCS-backed AI assets` (materialization +
  shared-index design; superseded).
- `c854f0e` -- `fix: avoid GCS overwrite-permission requirement in
  durable asset index` (per-asset immutable metadata keys; the design
  actually deployed and validated below).

Both pushed to `cloud-p0-durable-asset-persistence`, no merge.

## 6. Image

- Final tag: **`ai-durable-rehydrate-c854f0e`**
- Digest: `sha256:14cdaff9efd8bd2ae601dfd2f02f56a3bbe1ab532056c803565d4c8fdba9a16d`
- Size: ~2.17 GB
- (Superseded attempt `ai-durable-rehydrate-f1f1c4a`,
  digest `sha256:2f1fdb3e5a7b27fb8bd7e2f9cf129a6796945eef3991e0287ab986de8141fe0a`,
  left in Artifact Registry, not deleted.)
- Both independently verified present via `gcloud artifacts docker
  images list`, not just trusted from `docker push` output.

## 7. Redeploy

Two live redeploys this stage, both preserving every prior setting
exactly (4 vCPU / 2Gi, concurrency=1, timeout=300s, gen2, `min=0,
max=1`, `pfi-ai-runtime` service account, private ingress, the
read-only `pfi-rm-lumbar-artifacts-2026-ef` checkpoint mount, and
`PFI_STORAGE_PROVIDER=gcs` / `PFI_STORAGE_GCS_BUCKET=pfi-backend-durable-dev`):

- `pfi-ai-module-00004-brs` (attempt #1, `f1f1c4a` image) -- healthy
  deploy, but `POST /pipeline/run` 500'd (section 3).
- `pfi-ai-module-00005-drq` (attempt #2, `c854f0e` image) -- healthy,
  `POST /pipeline/run` succeeded.
- `pfi-ai-module-00006-ppg` -- a third, label-only redeploy of the same
  `c854f0e` image, used purely to force a genuinely fresh process
  (empty in-memory registry) for the rehydration proof below.

All three `Ready=True` / `ConfigurationsReady=True` / `RoutesReady=True`.
`pfi-ai-module-00001-m62` (pre-Stage-5 baseline) remains present and
`Ready=True`, untouched, as an immediate rollback target.

## 8. Health / auth regression (Step 13)

- Unauthenticated `GET /health` -> **403** (both post-fix revisions).
- Authenticated `GET /health` -> **200**, `outputStorageConfigured: true`.
- `gcloud run services get-iam-policy` -> empty bindings (`{}`) both
  before and after every redeploy this stage -- confirmed private
  throughout.

## 9. New durable run (Step 14)

`POST /pipeline/run` on `pfi-ai-module-00005-drq` with the same
synthetic/non-PHI fixture used throughout this work
(`sagittal_sample_input.npy`):

```
runId = aba8d7d0e6dcdd6c
```

All 8 assets reported `"storageProvider": "gcs"` with real SHA-256
checksums; independently confirmed present via
`gcloud storage ls gs://pfi-backend-durable-dev/runs/aba8d7d0e6dcdd6c/sagittal/_meta/`
(8 `_meta/*.json` objects, one per asset, matching the per-asset-key
design).

## 10. Same-process read (Step 15)

`GET /assets/aba8d7d0e6dcdd6c/sagittal/overlay.png` on the writing
revision -> **200**, SHA-256 exact match
(`1ad7a4cece6cb1111b20341283f588076f8551a730e5c022ac502e42308db061`).

(`mask.npy` returned 403 on this same check -- expected, pre-existing,
unrelated behavior: raw `.npy` arrays are intentionally excluded from
`PUBLIC_BROWSER_ASSET_NAMES` and never served through this endpoint,
same as before this fix.)

## 11. Fresh-process proof (Step 16) -- the core gate

Forced a genuinely new revision (`pfi-ai-module-00006-ppg`, same
image, label-only change) -> new container, empty in-memory asset
registry, empty local `PFI_OUTPUT_DIR`, empty local GCS materialize
cache.

```
GET /assets/aba8d7d0e6dcdd6c/sagittal/overlay.png       -> 200
  sha256 = 1ad7a4cece6cb1111b20341283f588076f8551a730e5c022ac502e42308db061 (exact match)
GET /assets/aba8d7d0e6dcdd6c/sagittal/mask-preview.png  -> 200
  sha256 = ec2e28c90b0c001fab26a41eb5f1e9fd2a54dad51b1b2b29eacf8190e0c95058 (exact match)
```

**PASS.** HTTP 200, identical bytes, identical SHA-256, on an instance
that never wrote either asset.

## 12. Multi-asset rehydration (Step 17)

Both assets above are of different kinds (a segmentation overlay PNG
and a preview PNG derived from a different internal artifact) and both
rehydrated correctly -- not a single-asset coincidence.

## 13. Second read after materialization (Step 18)

Immediately re-fetched `overlay.png` on the same fresh revision:

| Request | Latency | Note |
|---|---|---|
| 1st (overlay.png, cold materialize) | 2.901s | GCS list + download + local write |
| 2nd (mask-preview.png, cold materialize) | 0.535s | different asset, still a first materialize |
| 3rd (overlay.png again) | **0.0045s** | local cache hit, no GCS round trip |

Confirms the cache-reuse design works exactly as intended: ~645x
faster on the warm path.

## 14. Log evidence (Step 19)

From Cloud Logging, `pfi-ai-module-00006-ppg`, no secrets:

```
GET /assets/aba8d7d0e6dcdd6c/sagittal/overlay.png       200  2.900966372s
GET /assets/aba8d7d0e6dcdd6c/sagittal/mask-preview.png  200  0.534750955s
GET /assets/aba8d7d0e6dcdd6c/sagittal/overlay.png       200  0.004522149s
```

## 15. Cold/warm latency (Step 20) -- kept separate from rehydration cost

- **AI service cold start** (container start to ready): not
  re-measured this stage -- Stage 5 already established ~20-21s for
  this image size/profile on this service, and this fix does not touch
  startup code, so that number is unaffected and not repeated here to
  avoid unnecessary scale-to-zero waits.
- **Asset first-rehydration latency** (this stage's new metric): 2.901s
  (first asset touched by a fresh process) / 0.535s (second asset,
  same process) -- the actual cost this fix adds/removes, distinct
  from container cold start.
- **Asset warm-read latency**: 0.0045s.

## 16. Failure regression (Step 21)

- Unknown `run_id` -> **404** (`GET /assets/nonexistent-run-id-zzz/...`).
- Checksum-mismatch and missing-blob-with-surviving-index cases:
  proven at the unit-test level against a controlled fake GCS backend
  (section 4, tests C/D) -- both return controlled errors
  (`AssetRegistryError` 500 for checksum mismatch, 404 for a
  deleted-but-indexed blob), never an unhandled 500.

## 17. Rollback status (Step 25)

Not executed -- not needed. `pfi-ai-module-00001-m62` remains present,
`Ready=True`, untouched, and available via
`gcloud run services update-traffic --to-revisions=pfi-ai-module-00001-m62=100`
if ever required.

## 18. Remaining Backend -> AI work

This fix closes the specific gap Stage 5 found. Stage 6 (Backend v2 ->
private AI authenticated integration) can now proceed on the premise
that AI-side durable assets survive AI-side restarts. It does not
change or depend on anything in the Backend repo, Cloud SQL, or
Frontend, and the `pfi-ai-runtime` <-> `pfi-backend-durable-dev` IAM
bindings are unchanged from Stage 2 throughout.

## 19. Final summary

| Check | Result |
|---|---|
| Root cause identified from source (not guessed) | PASS |
| Materialization implemented (download, verify, atomic write, cache) | PASS |
| Durable metadata implemented without IAM broadening | PASS (after one corrected design) |
| Focused tests (A-F) | PASS, 10 new tests |
| Full durable-storage + pipeline suite | 65/65 PASS |
| Code committed, pushed, no merge | PASS (`f1f1c4a`, `c854f0e`) |
| Image built, pushed, old images preserved | PASS |
| Redeploy healthy | PASS |
| Service remains private | PASS |
| Auth regression | PASS |
| New durable GCS write | PASS |
| Same-process read | PASS, checksum match |
| **Fresh-process rehydration (core gate)** | **PASS**, checksum match |
| Multi-asset rehydration | PASS |
| Cache reuse on second read | PASS, ~645x faster |
| Missing/corrupt-asset failure behavior | PASS, controlled errors only |
| Checkpoint mount / bucket untouched | PASS |
| Rollback target available | PASS, untouched |

**AI_REHYDRATION_FIXED = YES**
**READY_FOR_BACKEND_AI_AUTH_INTEGRATION = YES**
