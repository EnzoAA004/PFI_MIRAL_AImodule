# 72 AI Product Orchestration

## Objetivo

La etapa 72 agrega una capa explicita de orquestacion de producto para que Spring Boot consuma un unico resultado estructurado del AI Module. FastAPI sigue siendo un servicio de IA/orquestacion, no el backend principal del producto.

El contrato nuevo es `pfi.ai-product-result.v1` y consume el release congelado `post-e50-ai-product-freeze-v1`.

## Arquitectura encontrada

- `api.py` centraliza endpoints publicos, sanitiza con `clean_for_json` y aplica `sanitize_public_payload`.
- `pipeline.py` conserva el pipeline visual/revisable por plano y ya soporta `inputId` registrado.
- `input_registry.py` resuelve inputs server-side o uploads y evita paths enviados por frontend cuando se usa `inputId`.
- `readiness.py` reporta contrato tecnico, artifacts y runtimes degenerativos.
- `disc_degenerative_product_runtime.py` existe como bridge P10.7 desde ROI segmentacion-derivado, pero el freeze 71 mantiene la inferencia automatica bloqueada por paridad productiva no habilitada.

## Endpoint

`POST /v1/product/analyze`

Request minimo:

```json
{
  "caseId": "CASE-123",
  "sagittalInputId": "inp_sagittal",
  "axialInputId": "inp_axial"
}
```

Tambien acepta:

```json
{
  "caseId": "CASE-123",
  "inputIds": {
    "sagittal": "inp_sagittal",
    "axial": "inp_axial"
  }
}
```

El modelo `ProductAnalyzeRequest` usa `extra=forbid`, por lo que no acepta `inputPath`, paths locales, pixeles embebidos ni campos DICOM/paciente innecesarios como parte del contrato.

## Flujo

1. Valida que exista al menos un `inputId` registrado.
2. Resuelve cada `inputId` contra `caseId` y plano.
3. Reutiliza el pipeline por plano en modo `contract`, sin activar inferencia patologica.
4. Consolida planos, mediciones descriptivas, warnings y estados de revision.
5. Construye `discLevels`.
6. Si no hay runtime productivo 67B2 materializado, devuelve `adapter_pending` y no fabrica nivel.
7. `abstained` queda reservado solamente para una abstencion real emitida por 67B2.
8. 67C se reporta siempre `validated=false`, `experimental=true`, `reviewRequired=true`; sin adapter materializado queda `adapter_pending`.
9. P10.7 queda `blocked_preprocessing_parity` y `automaticInferenceEnabled=false`.

Nota 73: se agregaron adapters runtime candidatos para 67B2/67C, pero sin evidencia permitida suficiente de paridad contra los SHAs congelados. Por eso el producto queda fail-closed como `unavailable`/`rejected`, no como `completed`, y `productEnabled=false`.

## Estados Globales

- `completed`: solamente cuando todas las capabilities requeridas solicitadas terminaron sin pendientes, bloqueos, abstenciones ni estados experimentales.
- `partial`: cuando hay resultado util de segmentacion/morfometria, pero alguna capability solicitada quedo `unavailable`, `adapter_pending`, `experimental`, `abstained` o `blocked`.
- `failed`: cuando no hay resultado util del analisis principal.

En 72/73, con 67B2/67C solicitados y adapters productivos pendientes o con paridad runtime no probada, el resultado esperado es `partial`.

## Capability Matrix

| Capability | Artifact | Runtime productivo | Validacion | Estado de producto |
| --- | --- | --- | --- | --- |
| Segmentacion sagital | available | available via pipeline | validated | enabled |
| Segmentacion axial v2 | available | available via pipeline | validated | enabled |
| Localizacion absoluta 67B2 | frozen SHA documentado, artifact productivo no disponible localmente | runtime candidato, productEnabled=false por paridad no probada | validated_with_abstention historico | unavailable/parity_not_proven, no abstention inventada |
| Pairing 67C | frozen SHA documentado, artifact productivo no disponible localmente | runtime candidato, productEnabled=false por paridad no probada | not validated | rejected/parity_not_proven, experimental review required |
| Morfometria sagital 68 | available | available via pipeline | technical QC PASS | descriptive only |
| Morfometria axial 68 | available | available via pipeline | technical QC PASS | raw classes preserved |
| P10.7 patologias | checkpoint verified | automatic runtime blocked | preprocessing parity not enabled for product | blocked_preprocessing_parity |

## Readiness

`GET /readiness` ahora incluye `productRelease`, con `releaseId`, `schemaVersion`, `orchestrationReady`, `automaticPathologyInference=false`, `humanReviewRequired=true` y matriz por capability distinguiendo `artifactAvailable`, `runtimeAvailable`, `productEnabled`, `validated`, `experimental`, `blocked`, `adapterPending`, `parityVerified` y blockers.

## Privacidad

El resultado de producto no expone paths internos, no devuelve `PatientName`, `PatientID` ni `patientId`, conserva `traceId`, pasa por `clean_for_json` y `sanitize_public_payload`, convierte `NaN`/`Inf` a `null` y nunca contiene diagnostico clinico.

## Ejemplos Response

### A. Adapter pending

```json
{
  "schemaVersion": "pfi.ai-product-result.v1",
  "runId": "product-...",
  "traceId": "trace-123",
  "caseId": "CASE-123",
  "release": {
    "releaseId": "post-e50-ai-product-freeze-v1"
  },
  "status": "partial",
  "planes": {
    "sagittal": {
      "status": "available",
      "runtimeAvailable": true,
      "inputId": "inp_sagittal",
      "segmentation": {
        "status": "completed",
        "validated": true,
        "reviewRequired": true,
        "maskClasses": ["vertebral_body", "disc", "spinal_canal"]
      },
      "morphometry": {
        "status": "available",
        "validated": true,
        "descriptiveOnly": true,
        "reviewRequired": true,
        "measurementCount": 3
      }
    },
    "axial": {
      "status": "available",
      "runtimeAvailable": true,
      "inputId": "inp_axial",
      "segmentation": {
        "status": "completed",
        "validated": true,
        "reviewRequired": true,
        "maskClasses": ["raw_0", "raw_50", "raw_100", "raw_150", "raw_200"]
      },
      "morphometry": {
        "status": "available",
        "validated": true,
        "descriptiveOnly": true,
        "reviewRequired": true,
        "measurementCount": 3
      }
    }
  },
  "discLevels": [
    {
      "level": null,
      "localization": {
        "status": "adapter_pending",
        "validated": true,
        "source": "67B2",
        "abstentionPreserved": false,
        "adapterPending": true
      },
      "axialPairing": {
        "status": "adapter_pending",
        "validated": false,
        "experimental": true,
        "reviewRequired": true,
        "adapterPending": true,
        "source": "67C"
      },
      "measurements": [],
      "degenerativeFindings": {
        "status": "blocked_preprocessing_parity",
        "automaticInferenceEnabled": false,
        "reviewRequired": true
      }
    }
  ],
  "warnings": ["67B2_localization_adapter_pending"],
  "review": {
    "required": true,
    "status": "pending",
    "professionalReviewRequired": true
  },
  "notClinicalDiagnosis": true,
  "autonomousDiagnosis": false,
  "humanReviewRequired": true
}
```

### B. 67B2 abstained real

```json
{
  "schemaVersion": "pfi.ai-product-result.v1",
  "release": {
    "releaseId": "post-e50-ai-product-freeze-v1"
  },
  "status": "partial",
  "discLevels": [
    {
      "level": null,
      "localization": {
        "status": "abstained",
        "validated": true,
        "source": "67B2",
        "abstentionPreserved": true,
        "adapterPending": false
      },
      "axialPairing": {
        "status": "adapter_pending",
        "validated": false,
        "experimental": true,
        "reviewRequired": true,
        "adapterPending": true,
        "source": "67C"
      },
      "degenerativeFindings": {
        "status": "blocked_preprocessing_parity",
        "automaticInferenceEnabled": false,
        "reviewRequired": true
      }
    }
  ],
  "review": {
    "required": true,
    "status": "pending"
  },
  "notClinicalDiagnosis": true,
  "autonomousDiagnosis": false,
  "humanReviewRequired": true
}
```

### C. Pipeline completed

Este estado aplica cuando las capabilities pendientes/bloqueadas no fueron solicitadas para esa corrida.

```json
{
  "schemaVersion": "pfi.ai-product-result.v1",
  "release": {
    "releaseId": "post-e50-ai-product-freeze-v1"
  },
  "status": "completed",
  "discLevels": [
    {
      "level": null,
      "localization": {
        "status": "not_requested",
        "validated": true,
        "source": "67B2",
        "abstentionPreserved": false,
        "adapterPending": false
      },
      "axialPairing": {
        "status": "not_requested",
        "validated": false,
        "experimental": true,
        "reviewRequired": true,
        "adapterPending": false,
        "source": "67C"
      },
      "degenerativeFindings": {
        "status": "blocked_preprocessing_parity",
        "automaticInferenceEnabled": false,
        "reviewRequired": true
      }
    }
  ],
  "review": {
    "required": true,
    "status": "pending"
  },
  "notClinicalDiagnosis": true,
  "autonomousDiagnosis": false,
  "humanReviewRequired": true
}
```

## Tests

Agregados en `ai_service/tests/test_product_orchestration.py` para releaseId, revision humana, autonomousDiagnosis=false, notClinicalDiagnosis=true, `partial` cuando falta adapter solicitado, abstencion real 67B2, adapter ausente 67B2 sin abstencion inventada, 67C no validado, P10.7 automatic inference false, clases axiales raw, payload sin paths/identificadores prohibidos, `NaN`/`Inf`, fallo fail-closed y readiness.

## Blockers para 73

- Materializar adapter productivo 67B2 desde artifacts congelados sin inventar algoritmo alternativo.
- Materializar adapter 67C como apoyo experimental revisable con trazabilidad de inputs y abstenciones.
- Definir si el bridge P10.7 de ROI segmentacion-derivado puede pasar a release productivo en una etapa futura; en 72 queda bloqueado por contrato.
- Completar validacion visual/end-to-end con estudios registrados reales de demo, sin usar datasets de validacion/internal_test/test oficial.
