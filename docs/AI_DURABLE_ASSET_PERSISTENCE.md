# AI service — durable asset persistence

Estado: implementado y testeado en `ai_service`. Alcance de este documento: solo
el AI Module (`PFI_MVPTest_Enzo_AImodule`). El Backend (`PFI_MVPTest_Enzo_Backend`)
queda fuera de este cambio: ya persiste el contenido binario de sus artefactos en
Postgres (`domain_run_asset_payloads`, ver `RunAssetContentStorage`), no en un mapa
en memoria, asi que no tenia el problema que este documento resuelve.

## Problema

`asset_registry.py` guardaba cada asset registrado (overlays, mascaras, previews,
mesh 3D) unicamente en un diccionario en memoria del proceso (`_ASSET_REGISTRY`).
Un restart del proceso Python (deploy, crash, reinicio de contenedor) perdia el
diccionario. Los `runId`/`assetName` devueltos previamente al cliente quedaban
apuntando a nada: `GET /assets/{runId}/{plane}/{assetName}` devolvia 404 aunque el
archivo siguiera en disco.

## Diseno

- **Contrato de storage** (`ai_service/pfi_ai_service/object_storage.py`): clase
  abstracta `ObjectStorage` con `put_file`, `put_bytes`, `open`, `exists`,
  `delete`, `local_path`. Dos providers:
  - `FilesystemObjectStorage`: raiz bajo `PFI_STORAGE_LOCAL_ROOT` (default
    `${PFI_OUTPUT_DIR}/object_storage`). Valida que la object key no escape la
    raiz (`is_relative_to`).
  - `GoogleCloudStorageObjectStorage`: usa `google.cloud.storage.Client()`
    (Application Default Credentials, sin JSON de service account). `open()` no
    esta soportado todavia (los objetos GCS no se materializan localmente); es
    una limitacion conocida, ver "Pendiente".
  - `get_object_storage()` lee `PFI_STORAGE_PROVIDER` (`filesystem` default o
    `gcs`) y **falla cerrado**: si el provider es `gcs` y falta
    `PFI_STORAGE_GCS_BUCKET`, o si el provider no es reconocido, levanta
    `ObjectStorageError` en vez de caer en filesystem en silencio.

- **Registry como cache, filesystem como fuente de verdad**
  (`ai_service/pfi_ai_service/asset_registry.py`):
  - Cada asset registrado (`register_run_assets`, `register_workspace_asset`) se
    persiste con `persist_asset_file` en el object storage bajo la key
    `runs/{runId}/{plane}/{assetName}`, y su metadata se escribe atomicamente
    (`*.json.tmp` + `replace`) en `${PFI_OUTPUT_DIR}/asset_registry/{runId}/{plane}.json`.
  - `_ASSET_REGISTRY` (el diccionario en memoria) sigue existiendo, pero ahora es
    un cache: `resolve_run_asset` primero lo consulta y, si no encuentra el
    asset, llama `rehydrate_run_asset`, que lee el JSON durable, valida el
    object key contra el storage y repuebla el cache.
  - Metadata durable minima por asset: `runId`, `plane`, `assetName`,
    `objectKey`, `contentType`, `sizeBytes`, `sha256`, `storageProvider`,
    `createdAt`.

- **Verificacion de integridad en rehidratacion**: al rehidratar desde el indice
  JSON, se recalcula el sha256 del archivo en disco y se compara contra el
  guardado. Si no coincide, se levanta `AssetRegistryError` (500,
  "checksum de asset no coincide") en vez de servir contenido corrupto o un 404
  enganoso.

- **Fallos explicitos, no silenciosos**:
  - Metadata JSON corrupta o ilegible -> `load_asset_index_record` devuelve
    `None` -> `resolve_run_asset` responde 404 ("asset no registrado").
  - Archivo durable borrado del disco pero metadata presente -> mismo camino,
    404.
  - Checksum no coincide -> 500 explicito ("checksum de asset no coincide"),
    nunca 404 silencioso ni contenido corrupto servido.
  - `PFI_STORAGE_PROVIDER=gcs` sin bucket configurado -> `ObjectStorageError` ->
    500 al intentar persistir o resolver, nunca fallback silencioso a
    filesystem.

## Compatibilidad

- IDs de `runId`, nombres de asset y endpoints (`GET /assets/{runId}/{plane}/{assetName}`)
  no cambiaron.
- La forma publica de la respuesta de `/pipeline/run` (`assets: {...}`) no
  cambio; los campos nuevos (`objectKey`, `storageProvider`, `contentType`,
  `sha256`) se agregan solo cuando el record tiene `object_key` (es decir,
  siempre que paso por el storage durable), y no reemplazan los campos previos
  (`runId`, `plane`, `assetName`, `size`).
- Los outputs de inferencia (mascaras, overlays, mesh) no cambian: el storage
  copia bytes tal cual, no los transforma.

## Tests

`ai_service/tests/test_object_storage.py` (contrato de storage: put/get/exists/
delete, rechazo de object keys invalidas, fail-closed de GCS mal configurado) y
`ai_service/tests/test_asset_registry.py` /
`ai_service/tests/test_asset_serving.py` (rehidratacion tras `clear_asset_registry()`
simulando restart, checksum coincide en el camino feliz, y los 4 casos de falla:
metadata corrupta, archivo durable faltante, checksum mismatch, provider mal
configurado).

## Pendiente / fuera de alcance de este cambio

- `GoogleCloudStorageObjectStorage.open()` no esta implementado: para servir un
  asset directamente desde GCS (sin copia local previa) falta materializarlo.
  Hoy el provider GCS esta preparado (put/exists/delete) pero no verificado
  end-to-end contra un bucket real — no se hicieron escrituras a GCP como parte
  de este cambio.
- No se corrio un test de restart real con Docker Compose (proceso AI
  reiniciado como contenedor separado) — la simulacion de restart en los tests
  (`clear_asset_registry()` + pedir el mismo asset) cubre la perdida del
  diccionario en memoria, que es la causa raiz, pero no un restart de proceso
  real.
- Backend (Postgres-backed storage) no se toco: ya es durable por diseno.
- IAM minimo para un bucket GCS de produccion no esta documentado todavia
  (pendiente si se decide desplegar el provider `gcs`).

## Corridas anteriores al indice (adopcion local)

El indice durable solo se escribe al registrar los assets de una corrida. Las corridas hechas antes (o por una version
anterior del servicio) tienen sus archivos en `{PFI_OUTPUT_DIR}/real_inference/{runId}/{plane}/` pero ningun registro,
asi que seguian respondiendo 404 tras un reinicio.

`resolve_run_asset` las recupera sin configuracion: si no hay registro en memoria ni indice, adopta el archivo de esa
carpeta (solo planos `sagittal`/`axial`, solo nombres permitidos, sin salir de `real_inference`) y lo manda por el mismo
camino que una corrida nueva (copia al storage, sha256, indice). Desde ahi sobrevive a otro reinicio sin depender del
origen. No se adopta cuando el indice existe pero esta corrupto, ni cuando el asset figura registrado sin su archivo
durable: eso sigue fallando de forma explicita. Un lock evita que los pedidos en paralelo de un visor copien el mismo
archivo a la vez o pierdan entradas del indice. El origen no se modifica.

Costo: cada asset adoptado queda copiado en `object_storage/` (en el stack local, ~35 MB para cuatro corridas).
Tests: `ai_service/tests/test_asset_legacy_adoption.py`.
