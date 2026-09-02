# AGENTS.md — db_bad_clust (rama de investigación)

**`main` es la rama de investigación: detección no supervisada de anti-patrones de esquema
Oracle con embeddings de oraciones. BERT es el método.** El motor de reglas y el producto de
auditoría viven en la rama `rule-engine` (congelada en `b676caf`, en `origin`) y no deben
volver aquí.

## Setup

```bash
docker compose up -d                         # Oracle 23c, healthcheck ~30s
.venv/bin/python -m pip install -e ".[dev]"
```

Todas las dependencias ML son de base en esta rama: el reparto base/[research] existía para
proteger un camino de auditoría sin ML, y ese camino está en `rule-engine`. `oracledb` y
`pyyaml` siguen porque el notebook 01 y `scripts/build_embeddings.py` extraen de Oracle.

Usa `.venv/bin/python -m <cmd>` — los shebangs del venv están obsoletos (la carpeta se renombró).

## El experimento

```bash
# Todo se puntúa contra output/manual_labels.csv (243 columnas etiquetadas a mano)
.venv/bin/python -m db_bad_clust.cli experiment --sweep
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --ablation output/intermediate_02.pkl output/intermediate_docs.pkl output/intermediate_anchors.pkl
```

```bash
# Reconstruir los bloques desde Oracle (contenedor arriba)
.venv/bin/python scripts/apply_comments.py                     # una vez
.venv/bin/python scripts/build_embeddings.py --dry-run         # ver los documentos
.venv/bin/python scripts/build_embeddings.py --output output/intermediate_docs.pkl
.venv/bin/python scripts/build_embeddings.py --semantic anchors --output output/intermediate_anchors.pkl
.venv/bin/python scripts/build_embeddings.py --no-table-comment --output output/intermediate_docs_nocomment.pkl
```

Los artefactos: `intermediate_02.pkl` es el **baseline histórico** (embedding del nombre, sin
tipo ni comentarios) y no se sobrescribe nunca. Los demás los produce el script.

## Los resultados

Todos contra las 243 etiquetas manuales. `--without-giants` excluye `TABLA_BASE_DATOS` (40) y
`BACKUP_DATOS` (50), las dos únicas tablas que aportan la clase `giant_table`.

### Corpus completo (243 columnas), α = 1.00

| representación         |    ARI |    NMI |    AMI | Accuracy | F1-macro |  k |
| ---------------------- | -----: | -----: | -----: | -------: | -------: | -: |
| solo estructura (α=0)  | 0.5031 | 0.4245 | 0.3619 |   0.7119 |   0.2670 | 12 |
| nombre (viejo, 384d)   | 0.3014 | 0.4512 | 0.3863 |   0.7037 |   0.2163 | 14 |
| **documento (384d)**   | 0.3748 | 0.5484 | 0.4818 |   0.7984 |   0.4622 | 19 |
| anclas (12d)           | 0.1428 | 0.3229 | 0.2819 |   0.5761 |   0.1598 |  6 |

### Sin las dos gigantes (153 columnas), α = 1.00 — la evaluación que vale

| representación         |     ARI |    NMI |    AMI | Accuracy | F1-macro |  k |
| ---------------------- | ------: | -----: | -----: | -------: | -------: | -: |
| solo estructura (α=0)  |  0.0657 | 0.2432 | 0.1212 |   0.6078 |   0.2002 | 12 |
| nombre (viejo, 384d)   | -0.0538 | 0.2043 | 0.0747 |   0.5425 |   0.1346 | 12 |
| **documento (384d)**   |  0.0802 | 0.3788 | 0.2553 |   0.6797 |   0.3781 | 16 |
| anclas (12d)           |  0.0369 | 0.0762 | 0.0297 |   0.5359 |   0.1097 |  3 |

El documento gana las cinco métricas. El embedding del nombre da **ARI negativo**: peor que el
azar.

### Barrido de α con el documento (corpus completo, fusión corregida)

| α    |    ARI |    NMI |    AMI | Accuracy | F1-macro |
| ---- | -----: | -----: | -----: | -------: | -------: |
| 0.00 | 0.5031 | 0.4245 | 0.3619 |   0.7119 |   0.2670 |
| 0.25 | 0.3506 | 0.4862 | 0.4256 |   0.7284 |   0.2491 |
| 0.50 | 0.3539 | 0.4934 | 0.4219 |   0.7366 |   0.3806 |
| 0.75 | 0.3646 | 0.5238 | 0.4538 |   0.7737 |   0.4361 |
| 1.00 | 0.3748 | 0.5484 | 0.4818 |   0.7984 |   0.4622 |

AMI, accuracy y F1 crecen monótonamente con α. ARI no — ver el punto 2 de abajo sobre por qué
el 0.5031 de α=0 no es lo que parece.

## Invariantes críticos (aprendidos midiendo, no diseñando)

- **α no ponderaba nada.** `_zscore` normaliza por dimensión, así que la varianza total de un
  bloque es su número de dimensiones. Con α nominal 0.15 los embeddings se quedaban el **89%**
  de la varianza; con 0.30, el **98%**. El `alpha_sweep` histórico ("ARI 0.5008 → 0.30 al
  encender BERT") comparaba *sin embeddings* contra *embeddings y casi nada más*. Usa
  `normalize="block"` (por defecto); `"zscore"` solo para reproducir lo viejo.

- **El baseline estructural no individúa el corpus.** Con α=0, 243 columnas colapsan en **24
  vectores distintos** (97% duplicadas, mayor empate de 108). Dos columnas con el mismo vector
  no pueden recibir etiquetas distintas de ningún algoritmo. Su ARI tampoco es reproducible:
  **0.1877 en float32 contra 0.5031 en float64**. `load_dataset` fija float64 a propósito;
  `representation_degeneracy` y `precision_sensitivity` miden ambas cosas. El 0.5008 histórico
  se reproduce exactamente en `zscore`/float32 — era el ancho en que se calculó.

- **Lo que se le da a leer al encoder es el experimento entero.** `process()` devuelve
  "empleados: fecha nacimiento", sin el tipo, así que ningún encoder puede distinguir
  `FECHA_INGRESO DATE` de `FECHA_INGRESO VARCHAR2(20)` — que es exactamente
  `wrong_data_types`. `build_document()` escribe tipo y restricciones en palabras.

- **`build_document` escribe español: usa `TextPreprocessor.for_documents()`**, que aplica
  `ABBREVIATIONS_ES`. El diccionario por defecto expande al inglés, correcto para `process()` y
  equivocado dentro de una frase española; toca los dos tokens más frecuentes del esquema,
  `col` (53 columnas) e `id` (24).

- **Las anclas leen el *nombre*, nunca el documento** — el documento dice el tipo, que es la
  respuesta que la feature de discordancia está tratando de dar.

- **El mismatch de anclas es por compatibilidad, no por identidad.** La versión por identidad
  marcaba `NOMBRE_CLIENTE VARCHAR2` (concepto "persona" ≠ "texto_libre") y no separaba nada:
  `wrong_data_types` 0.3355 contra `clean` 0.2863. Con `CONCEPT_COMPATIBLE_CATEGORIES` los
  casos reales suben con el concepto correcto (EMAIL DATE → contacto 0.540, PRECIO_TOTAL
  VARCHAR2 → cantidad 0.305). Quedan 52 falsos positivos: `BACKUP_DATOS.COL_0xx`, de nombre
  vacío, caen en el ancla "binario"; **pesar por confianza no lo arregla** — esos nombres
  tienen mayor margen (0.1726) que los `wrong_data_types` (0.1204).

- **Reporta AMI, no NMI, entre filas con distinto k.** Las configuraciones producen entre 2 y
  19 clusters y NMI sube con k por sí solo.

- **`giant_table` es propiedad de la tabla, no de la columna** — 90 de 243 columnas, todas de
  `TABLA_BASE_DATOS` y `BACKUP_DATOS`. En el corpus completo, buena parte de la ganancia del
  documento viene del comentario de tabla, que filtra identidad de tabla; sin él, α=1.00 cae a
  ARI 0.2173 / F1 0.1664 (`--no-table-comment`). Por eso `--without-giants` es la evaluación
  honesta.

- **Accuracy y F1 de un clustering son cota superior** — el nombrado por voto mayoritario usa
  la verdad de terreno que el pipeline nunca vio. Dilo donde los cites.

- **Resultado negativo, conservado:** las anclas (12 dimensiones interpretables) pierden contra
  las 384 crudas como representación de agrupamiento. Se conservan por lo que sí hacen:
  explicar columna a columna.

- **El escalar de discordancia como feature aparte da un resultado MIXTO, no una mejora limpia.**
  Un sondeo supervisado (RF, mismas features, con el escalar reforzado x5 artificialmente) subía
  el techo de F1-macro de 0.4118 a 0.4255 — la señal existe. Pero metido en el pipeline de
  clustering real (`scripts/build_embeddings.py --mismatch`, que lo añade como segunda columna
  de `e_stat`) y aislado con su propio peso honesto (δ=0.5, sin `data_length` compitiendo por el
  mismo δ), sobre las 153 columnas sin gigantes:

  | métrica | documento solo | documento + mismatch (δ=0.5) |
  | ------- | --------------: | ----------------------------: |
  | ARI     |          0.0802 |                        0.0758 |
  | NMI/AMI |   0.3788/0.2553 |                 0.3709/0.2463 |
  | F1-macro|          0.3781 |                         0.4137 |

  F1-macro sube; ARI/NMI/AMI bajan un poco. F1 es cota superior por voto mayoritario; ARI/AMI
  son las métricas honestas sobre la partición. La lectura correcta: el escalar ayuda a *nombrar*
  el cluster que ya se formó (por eso sube F1), pero no cambia qué columnas quedan juntas (por
  eso ARI/AMI no mejoran). `--mismatch` queda como flag opcional (por defecto `False`) en vez de
  comportamiento por defecto, precisamente porque no es una mejora limpia.

- **Nunca sobrescribas `output/intermediate_02.pkl`.** Es el baseline histórico; un experimento
  que sobrescribe su propio baseline no se puede comprobar.

## Tests y lint

```bash
.venv/bin/python -m pytest tests/ -v      # 372 tests, sin BD y sin descargar el modelo
.venv/bin/python -m ruff check src tests scripts   # lint-clean
```

Los tests de `semantic_anchors` inyectan un embedder de prueba que coloca cada texto en un eje
unitario, así que los cosenos esperados son literales y no un recálculo de lo que hace el código.

## Oracle — no es reproducible al 100% desde el repo

- Las 23 tablas se crearon ad-hoc; no hay generador de DDL del esquema real.
- `generation/anti_patterns.py` solo guarda `TABLE_COMMENTS`/`COLUMN_COMMENTS`, aplicados a
  Oracle vía `scripts/apply_comments.py`. La generación del esquema en sí no está en este repo.
- Nombres de tabla sensibles a mayúsculas — siempre entre comillas dobles.
- Oracle permite una sola columna LONG por tabla (ORA-01754).

## Referencia

- `docs/research_extraction_preprocessing.md` — investigación de fuentes primarias sobre
  extracción de metadata Oracle y preprocesamiento de texto para embeddings.
- Los reportes sobre el motor de reglas (`00_PROJECT_STATUS_REPORT.md`,
  `rule_engine_and_future_ml_report.{md,html}`, `veredicto_reglas_vs_ml.html`) viven en la
  rama `rule-engine`, no aquí. `veredicto_reglas_vs_ml.html` concluía "BERT es dañino" — es
  justo el artefacto que el invariante 1 de esta rama explica.
