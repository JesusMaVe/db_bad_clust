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
# La representación de conflicto, todo elegido a ciegas, y su robustez ante el fraseo de las anclas
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --pickle output/intermediate_docs.pkl --conflict --robustness
# Intervalos de confianza al 95 % y diferencias pareadas (~0.8 s por remuestreo)
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --pickle output/intermediate_docs.pkl --bootstrap 500
# La rejilla completa de HDBSCAN con la celda elegida por validez
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --pickle output/intermediate_docs.pkl --stability
```

```bash
# Primera vez / volumen Oracle nuevo: bootstrap del esquema (ver sección Oracle abajo)
.venv/bin/python scripts/generate_schema_sql.py                # escribe sql_init/001_bad_schema.sql
docker compose up -d                                            # aplica el DDL en un volumen nuevo
.venv/bin/python scripts/verify_schema.py                       # confirma 23 tablas / 243 columnas

# Reconstruir los bloques desde Oracle (contenedor arriba)
.venv/bin/python scripts/apply_comments.py                     # una vez
.venv/bin/python scripts/build_embeddings.py --dry-run         # ver los documentos
.venv/bin/python scripts/build_embeddings.py --output output/intermediate_docs.pkl
.venv/bin/python scripts/build_embeddings.py --semantic anchors --output output/intermediate_anchors.pkl
.venv/bin/python scripts/build_embeddings.py --no-table-comment --output output/intermediate_docs_nocomment.pkl
# Sin Oracle: re-embebe usando el esquema guardado en un pickle anterior
.venv/bin/python scripts/build_embeddings.py --from-pickle output/intermediate_docs.pkl --output output/intermediate_docs.pkl

# intermediate_02.pkl (baseline "viejo", nombre-solo) via notebooks 01+02, headless:
#   jupyter nbconvert --to notebook --execute --output executed_01.ipynb notebooks/01_data_preparation.ipynb
#   jupyter nbconvert --to notebook --execute --output executed_02.ipynb notebooks/02_ml_embeddings.ipynb
# (requiere ipykernel, en los dev deps; el notebook 01 necesita el contenedor arriba)
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

- **Bloque de features agregadas por tabla (`features/table_aggregates.py`, peso `epsilon` en
  `FeatureBuilder`) — mejora real pero modesta, y con un riesgo de fuga documentado, por eso
  queda opt-in.** `giant_table`/`eav`/`polymorphic` son propiedades de tabla que ningún embedding
  por columna puede ver solo; `load_dataset` calcula 5 estadísticas puramente estructurales por
  tabla (log1p(nº columnas), % nullable, diversidad de tipos, % nombres genéricos, % con
  comentario de columna) y se las adjunta a cada una de sus columnas — sin nombre de tabla, sin
  reglas. Barrido de `epsilon` sobre el documento (α=1.00), sin gigantes: pico en
  `epsilon≈0.25-0.30` (ARI 0.1107→0.1158, y mejora en las cinco métricas), cae por debajo del
  baseline pasado `epsilon≈0.35`. **Riesgo medido, no solo teórico:** de las 23 tablas, el vector
  agregado de 5 dimensiones es distinto en 21 — casi tan identificador como el nombre de tabla en
  sí. Tranquilizador parcialmente: la mejora se mide en `--without-giants`, que ya excluye las dos
  tablas cuyo "atajo" de identidad es el que preocupa (`TABLA_BASE_DATOS`/`BACKUP_DATOS`), así que
  no viene de re-identificarlas a ellas — pero no se descarta que otra parte de la mejora sea el
  mismo atajo aplicado a otras tablas. `epsilon` se queda en 0.0 por defecto (ningún peso
  existente lo incluye, así que ningún experimento existente cambia); disponible para quien quiera
  seguir explorándolo vía `evaluate(..., {"epsilon": 0.25, ...})`. Ver
  `docs/research_improving_clustering.md`, candidato #4.

- **Un anti-patrón es un conflicto entre facetas, y una oración las promedia.** Sin gigantes, el
  clustering del documento tiene ARI **0.593 contra la tabla** de cada columna y 0.111 contra la
  etiqueta: 14 de 16 clusters son una sola tabla. Quitando la identidad de tabla (sin comentario,
  centrado por tabla, oración solo de columna) el ARI cae a −0.008 / 0.041 / 0.008. El bloque de
  conflicto (`ConflictBlock` en `features/semantic_anchors.py`, peso `zeta`) mide por separado qué
  espera el **nombre** (zero-shot contra 6 familias de almacenamiento) y qué declara el tipo, y
  resta. Elegido a ciegas sobre la rejilla de HDBSCAN: ARI 0.1276 / AMI 0.2181 / ARI~tbl 0.010,
  contra 0.0478 / 0.1590 / 0.381 del documento con el mismo protocolo. La mediana de su rejilla
  (0.1186) supera al máximo de la del documento (0.111). En el corpus completo el documento sigue
  ganando porque reconoce las tablas gigantes (ARI~tbl 0.884). `zeta` es 0.0 por defecto. Detalle
  completo: `docs/research_improving_clustering.md`, candidato #6.

- **El bloque de conflicto no se z-scorea por dimensión.** Las indicadoras escasas de la familia
  declarada reciben z-scores de ±6 y colapsan HDBSCAN a k≈5 y ARI 0. `scale_conflict_subblocks`
  escala por sub-bloque (diferencia 1.0, declarado 0.7, entropía 0.5). También lee el *nombre*
  desnudo, nunca el documento ni la tabla.

- **La expectativa del bloque de conflicto es dura, y Ward es el algoritmo principal** (candidato
  #7). La versión suave (softmax) se aplana con la temperatura o el fraseo, deja dominar al tipo
  declarado, y el criterio ciego elige k=2 (ARI -0.08). La dura es un one-hot sobre la familia más
  cercana, con el margen como confianza, y no colapsa con ningún fraseo. Votar o promediar entre
  fraseos o encoders reintroduce el colapso. Sin gigantes, fusionado, Ward con k por silueta:
  media de 3 fraseos ARI 0.1256 / AMI 0.2004 / ARI~tbl 0.002, con el fraseo por defecto 0.1693 /
  0.2464. El documento, igual de ciego, da 0.0505 / 0.2105 / 0.867. Bootstrap pareado (500
  submuestras del 80 %, pipeline ciego completo en cada una): **ARI +0.073 [IC 95 % +0.042,
  +0.106], gana el 100 %; AMI empata, [-0.053, +0.061].** En el corpus completo el documento es
  mejor en AMI por ~0.07 (el intervalo excluye el cero) y empata en ARI. Reporta la media sobre
  fraseos: el fraseo por defecto se eligió mirando el ARI.

- **Anclas hechas de nombres de ejemplo: resultado negativo.** Centroides de 3 conjuntos de nombres
  inventados empatan sin gigantes (ARI +0.016 [-0.008, +0.043]) y pierden en el corpus completo
  (ARI -0.079 [-0.128, -0.032]). El lector se vuelve más estable (acuerdo 0.77 contra 0.63) sin
  volverse más acertado. Se quedan las frases.

- **Lector por inferencia (NLI): resultado negativo.** Tres modelos XNLI (BERT español, MiniLMv2 y
  mDeBERTa multilingües) como lector del nombre dan ARI medio de -0.03 a 0.05 sin gigantes, contra
  0.126 del coseno. Sobre 18 nombres inequívocos aciertan de 8 a 11; el coseno, 18. Un nombre
  desnudo no es una premisa. El lector actual solo falla en nombres ambiguos de verdad, y ninguna
  palanca probada sobre el lector los resuelve.

- **Señal nueva para `inconsistent_naming`/`polymorphic`/`eav`/`reserved_words`: negativo, por
  granularidad.** Cinco señales de BERT sobre el nombre y sus hermanos no ayudan: con peso chico no
  mueven nada, y con peso grande agrupan esas clases pero rompen los clusters de conflicto (ARI
  0.126 → 0.04–0.06). Tres de las cuatro son propiedades de la tabla (`CONFIGURACION` 6/6,
  `TBL_DATOS` 9/9, `REPORTES` 4/6). Los conflictos de tipo cruzan tablas, y estas clases viven dentro
  de una. Un solo clustering plano de columnas no sirve a las dos. La vía que queda es un diseño en
  dos niveles (columnas y tablas).

- **Una diferencia se afirma solo con su intervalo pareado** (`--bootstrap N`). Ambos modos
  submuestrean sin reemplazo: con reemplazo, las columnas duplicadas inflan ARI y AMI.

- **Elige hiperparámetros a ciegas y reporta ARI~tbl.** `min_samples=3` (candidato #1) se eligió
  mirando el ARI contra las etiquetas. Con elección ciega el documento sin gigantes da ARI 0.0478,
  no 0.1107. `--stability` elige la celda por `relative_validity_` de HDBSCAN y publica la rejilla
  entera. `format_table` imprime ARI~tbl (ARI de la partición contra la tabla): un número alto ahí
  significa que el clustering reconoció tablas, no anti-patrones.

- **Nunca sobrescribas `output/intermediate_02.pkl`.** Es el baseline histórico; un experimento
  que sobrescribe su propio baseline no se puede comprobar.

## Tests y lint

```bash
.venv/bin/python -m pytest tests/ -v      # 520 tests, sin BD y sin descargar el modelo
.venv/bin/python -m ruff check src tests scripts   # lint-clean
```

Los tests de `semantic_anchors` inyectan un embedder de prueba que coloca cada texto en un eje
unitario, así que los cosenos esperados son literales y no un recálculo de lo que hace el código.

## Oracle — el esquema original no es recuperable, pero hay un bootstrap reconstruido

- Las 23 tablas originales se crearon ad-hoc; nunca hubo generador de DDL versionado, y ese
  esquema en sí (con sus PK/FK/longitudes reales) es irrecuperable — no vive en ningún branch.
- `scripts/generate_schema_sql.py` reconstruye un esquema desde cero a partir de
  `output/manual_labels.csv` (nombre y tipo exactos de las 243 columnas) y escribe
  `sql_init/001_bad_schema.sql`, que `docker-compose.yml` aplica automáticamente en el primer
  arranque de un volumen nuevo (`docker compose up -d` tras `docker compose down -v` si ya
  existía un volumen). PK/FK se infieren con una heurística documentada en el propio script
  (columnas `NUMBER` con forma de ID, excluyendo las etiquetadas `impossible_data`/
  `inconsistent_naming`); longitudes son fijas por tipo. Es fiel en nombres/tipos/semántica, no
  en constraints — no lo trates como el esquema original. `scripts/verify_schema.py` confirma
  que lo creado coincide exactamente (ambas direcciones) con `manual_labels.csv`.
- Sin `INSERT`s: el pipeline (`SchemaExtractor`) solo lee vistas de catálogo, nunca contenido de
  filas — las tablas quedan vacías a propósito.
- `generation/anti_patterns.py` solo guarda `TABLE_COMMENTS`/`COLUMN_COMMENTS`, aplicados a
  Oracle vía `scripts/apply_comments.py` una vez el esquema existe.
- Validado con el `HDBSCAN_MIN_SAMPLES` histórico (2): a α=1.00 las representaciones nombre-solo
  y anclas (que nunca leen tipo/constraints) reproducían la tabla histórica de este documento
  cifra por cifra, porque dependen solo de los nombres de tabla/columna — invariantes a la
  reconstrucción del esquema. El documento (que sí escribe tipo y constraints en la oración)
  quedaba cerca pero no idéntico (ARI 0.3850 vs 0.3748 en el corpus completo). Esa propiedad de
  reproducción exacta se rompió a propósito al subir `min_samples` a 3 (ver el bullet siguiente)
  — el cambio de hiperparámetro de clustering afecta a *todas* las representaciones, no solo a
  las estructurales, así que ya no hay una comparación cifra-por-cifra disponible como chequeo de
  cordura. Lo que sigue sosteniéndose es el hallazgo cualitativo: el documento le gana a
  estructura-sola y a nombre-solo en las cinco métricas, con o sin este cambio.
- **`HDBSCAN_MIN_SAMPLES` subió de 2 a 3** tras barrer `cluster_selection_method` (`eom`/`leaf`) ×
  `min_samples` (2/3/5) sobre la representación documento — ver
  `docs/research_improving_clustering.md`, candidato #1. `cluster_selection_method` no cambió
  nada en este corpus (eom y leaf dieron resultados idénticos); `min_samples=3` le ganó a 2 en
  las cinco métricas sobre la evaluación honesta (sin gigantes): ARI 0.0938→0.1107, NMI
  0.3754→0.3781, AMI 0.2464→0.2497, Accuracy 0.6732→0.6797, F1-macro 0.3676→0.4003. En el corpus
  completo el efecto es mixto por representación (mejora en documento y anclas, empeora en
  nombre-solo) — el criterio de selección fue la evaluación sin gigantes, la que AGENTS.md ya
  trata como la que vale. "Solo estructura" sigue siendo la fila más sensible a la reconstrucción
  del esquema (ver `representation_degeneracy` — colapso más severo que el original, 10 vectores
  distintos en vez de 24, por la política de longitud fija por tipo), independiente de este
  cambio de `min_samples`.
- Nombres de tabla sensibles a mayúsculas — siempre entre comillas dobles.
- Oracle permite una sola columna LONG por tabla (ORA-01754).

## Referencia

- `docs/research_extraction_preprocessing.md` — investigación de fuentes primarias sobre
  extracción de metadata Oracle y preprocesamiento de texto para embeddings.
- Los reportes sobre el motor de reglas (`00_PROJECT_STATUS_REPORT.md`,
  `rule_engine_and_future_ml_report.{md,html}`, `veredicto_reglas_vs_ml.html`) viven en la
  rama `rule-engine`, no aquí. `veredicto_reglas_vs_ml.html` concluía "BERT es dañino" — es
  justo el artefacto que el invariante 1 de esta rama explica.
