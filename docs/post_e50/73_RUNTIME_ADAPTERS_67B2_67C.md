# 73 Runtime Adapters 67B2 / 67C

## Objetivo

La etapa 73 materializa interfaces runtime simples para 67B2 y 67C, pero no las habilita como resultado productivo validado porque no se encontraron artifacts permitidos suficientes para demostrar paridad contra el freeze.

Resultado final:

- `67B2_RUNTIME_PARITY_NOT_PROVEN`
- `67C_RUNTIME_PARITY_NOT_PROVEN`
- `productEnabled=false`
- `parityVerified=false`
- P10.7 sigue `blocked_preprocessing_parity`

## Fuentes Inspeccionadas

67B2:

- `notebooks/post_e50/67B_postE50_absolute_level_anchor_rsna.ipynb`
- `notebooks/post_e50/71_postE50_ai_product_freeze_v1.ipynb`
- `artifacts/post_e50/absolute_level_anchor/post_e50_67B_summary.json`
- `artifacts/post_e50/absolute_level_anchor/post_e50_67B_real_closing_evidence.json`
- `artifacts/post_e50/level_localization/disc_localizations.json`
- `artifacts/post_e50/level_localization/localization_summary.json`

67C:

- `notebooks/post_e50/66_postE50_multiseries_geometry_pairing.ipynb`
- `notebooks/post_e50/71_postE50_ai_product_freeze_v1.ipynb`
- `artifacts/post_e50/series_pairing/post_e50_pairing_summary.json`
- `artifacts/post_e50/series_pairing/post_e50_pairing_metrics_full.csv`

Los artifacts locales encontrados no prueban paridad con los SHAs congelados de 71 para un adapter productivo 67B2/67C. No se abrieron datasets de validation, internal_test ni official test.

## 67B2

Freeze obligatorio:

- status: `VALIDATED_WITH_ABSTENTION`
- model SHA: `bb9e9d812972305cbc3dc87b96b95f78b7c9de4afc8ac4e6f071b35ae6fffd2`
- locked predictions SHA: `baae269fe8f0e7bc7725a44ebeb1431600d7ce23d56b7dd82c05a0cbcc0405`
- summary SHA: `2b74c8a5b0d8ee562f6a774e136abf95225074d32c32c7646a5ca3f0fd6e8dcb`
- `automatic_disc_localization_validated=true`
- `abstention_supported=true`

Runtime agregado:

- `ai_service/pfi_ai_service/absolute_disc_localization_runtime.py`
- API: `localize_absolute_disc_levels(...)`

Semantica:

- `localized`: solo cuando existe prediccion 67B2 con paridad verificada y nivel lumbar permitido.
- `abstained`: solo cuando el payload 67B2 real indica abstencion.
- `unavailable`: cuando falta paridad, artifact, input o el nivel congelado es invalido.
- Nunca fabrica `level`.
- Nunca convierte adapter ausente o paridad no probada en `abstained`.

## 67C

Freeze obligatorio:

- method: `BASELINE_A_NEAREST_PLANE_PLUS_FOV`
- `require_inside_fov=true`
- `min_margin=0.26162020362301863`
- `max_plane_distance=23.818094078633493`
- summary SHA: `4568440b9a2637996a1fc5722de0609e81c95f37f17273b7bcf0b596d5dc32bd`
- artifact predictions SHA: `9aa3778f1cbb24e6308346566dc1b95cfbe3486185ffc065a1dc42bf120c3a07`
- `validated=false`
- `experimental=true`
- `reviewRequired=true`

Runtime agregado:

- `ai_service/pfi_ai_service/axial_level_pairing_runtime.py`
- API: `pair_disc_to_axial_slice(...)`

Algoritmo reconstruido:

1. Toma el centroide sagital del disco en coordenadas paciente.
2. Evalua slices axiales candidatos con geometria fisica DICOM.
3. Calcula normal por slice con `rowCosines x columnCosines`.
4. Calcula distancia punto-plano.
5. Proyecta a coordenadas fila/columna con `ImagePositionPatient`, orientacion y `PixelSpacing`.
6. Exige inside FOV y margen minimo.
7. Rechaza distancias mayores que el maximo congelado.
8. Elige el plano valido con menor distancia absoluta.

`SliceLocation` e `InstanceNumber` pueden salir como metadata/display, pero no controlan el pairing geometrico.

## Parity

Procedimiento requerido:

- comparar adapter runtime contra artifacts congelados permitidos de TRAIN/DEV;
- 67B2: mismo `localized/abstained`, mismo `level` y cero niveles fabricados en abstention;
- 67C: misma serie/slice, mismo reject/emit, `planeDistanceMm` dentro de tolerancia y `insideFov` identico;
- tolerancia numerica sugerida para distancia deterministica: `1e-6`.

Resultado:

- `67B2_RUNTIME_PARITY_NOT_PROVEN`
- `67C_RUNTIME_PARITY_NOT_PROVEN`

Motivo: no se encontraron en el repo artifacts permitidos suficientes que correspondan a los SHAs congelados y permitan comparar una muestra deterministica. Por contrato, los adapters quedan fail-closed y no habilitados para producto.

## Integracion Producto

`product_orchestration.py` ahora distingue:

- adapter runtime presente;
- artifact congelado disponible o no;
- `parityVerified`;
- `productEnabled`;
- validacion;
- experimentalidad;
- blocker.

Si 67B2 no tiene paridad verificada, localization devuelve:

```json
{
  "status": "unavailable",
  "source": "67B2",
  "validated": true,
  "abstentionPreserved": false,
  "level": null,
  "warnings": ["67B2_RUNTIME_PARITY_NOT_PROVEN"]
}
```

Si 67B2 localiza con paridad verificada pero 67C no tiene paridad, 67C devuelve:

```json
{
  "status": "rejected",
  "source": "67C",
  "validated": false,
  "experimental": true,
  "reviewRequired": true,
  "warnings": ["67C_RUNTIME_PARITY_NOT_PROVEN"]
}
```

El resultado global queda `partial` cuando existe segmentacion/morfometria util, pero 67B2 o 67C quedan `unavailable`, `rejected`, `abstained`, `experimental` o bloqueadas.

## Readiness Final

67B2:

- `artifactAvailable=false`
- `runtimeAvailable=true`
- `productEnabled=false`
- `validated=true`
- `abstentionSupported=true`
- `parityVerified=false`
- `status=parity_not_proven`
- `blocker=67B2_RUNTIME_PARITY_NOT_PROVEN`

67C:

- `artifactAvailable=false`
- `runtimeAvailable=true`
- `productEnabled=false`
- `validated=false`
- `experimental=true`
- `reviewRequired=true`
- `parityVerified=false`
- `status=parity_not_proven`
- `blocker=67C_RUNTIME_PARITY_NOT_PROVEN`

P10.7:

- `automaticInferenceEnabled=false`
- `status=blocked_preprocessing_parity`

## Tests

Agregados en `ai_service/tests/test_runtime_adapters_67b2_67c.py`:

- 67B2 localized valido
- 67B2 abstention preservada
- campos de gobernanza no usados como features de prediccion
- unavailable no es abstained
- no fallback a mejor nivel
- SHA congelado correcto
- 67C nearest physical plane
- FOV requerido
- max plane distance
- min margin
- `SliceLocation` no controla pairing
- `InstanceNumber` no controla pairing
- normal/orientacion por slice
- `validated=false`
- `experimental=true`
- `reviewRequired=true`
- sin paridad no hay pairing productivo
- readiness runtime no product-enabled
- orquestacion 67B2 localized permite intento 67C fail-closed
- orquestacion 67B2 abstained no fuerza 67C
- runtime unavailable produce global `partial`
- P10.7 sigue bloqueado
- no diagnostico autonomo

Suite especifica 72/73:

```text
35 passed
```

## Limitaciones

- No se habilito `productEnabled` para 67B2 ni 67C.
- No se entreno ni modifico ningun checkpoint.
- No se abrieron datasets prohibidos.
- La etapa 74 deberia materializar o ubicar artifacts permitidos congelados de TRAIN/DEV y ejecutar la prueba de paridad deterministica.
