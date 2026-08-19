# De la teoria a la implementacion

> Documento complementario a `docs/context.md` — explica que se construyo realmente, las diferencias con el diseno teorico, y los resultados obtenidos.

---

## 1. Proposito

`docs/context.md` describe el diseno teorico y la hipotesis principal del proyecto (H1: BERT + DBSCAN es la combinacion mas prometedora). Este documento detalla lo que se implemento realmente, las decisiones que cambiaron respecto al diseno original, y los resultados experimentales concretos.

---

## 2. Que se implemento

### Fase 1 — Generacion de base de datos Oracle

- Oracle 23c en Docker con 22 tablas de esquema mal disenado
- Generacion de datos sucios (fechas como VARCHAR2, numeros como texto, booleanos inconsistentes)
- Anti-patrones catalogados en `anti_patterns.py` (7 categorias: date_as_text, number_as_text, boolean_as_text, no_primary_key, reserved_word, nullable_all, mixed_domain)
- Tabla DATOS_MAESTROS falla deliberadamente (LONG/RAW eliminados en Oracle 23c)

### Fase 2 — Pipeline de ML

Pipeline completo de 10 modulos:

| # | Modulo | Archivo | Proposito |
|---|---|---|---|
| 1 | SchemaExtractor | `schema_extractor.py` | Extrae metadatos de Oracle (columnas, tipos, constraints, indices) |
| 2 | TextPreprocessor | `text_preprocessor.py` | Limpia nombres: separa camelCase, expande abreviaturas, contexto de tabla |
| 3 | BERTEmbedder | `bert_embedder.py` | Embeddings 768D via `bert-base-multilingual-cased` |
| 4 | StructuralEncoder | `structural_encoder.py` | One-hot de tipos (12) + constraints binarias (5) + data_length |
| 5 | FeatureBuilder | `feature_builder.py` | Vector compuesto phi = alpha*BERT + beta*type + gamma*constraints + delta*stat |
| 6 | DimensionalityReducer | `dimensionality_reducer.py` | PCA / UMAP / SVD / t-SNE (786 -> N dims) |
| 7 | ClusterEngine | `cluster_engine.py` | KMeans / DBSCAN / HDBSCAN / Agglomerative / MeanShift |
| 8 | Evaluator | `evaluator.py` | Metricas internas (Silhouette, DB, CH) + pureza + composicion + deteccion de anomalias |
| 9 | Recommender | `recommender.py` | Recomendaciones por cluster (fechas como texto, numeros como texto, nulos, booleanos) |
| 10 | Visualizer | `visualizer.py` | Graficos 2D/3D, silhouette, heatmap, composicion |

### Fase 3 — Validacion de hipotesis

- Ground truth generado automaticamente desde `anti_patterns.py` (235 columnas etiquetadas)
- Validacion externa: ARI (Adjusted Rand Index) y NMI (Normalized Mutual Information)
- Deteccion de anomalias estructurales via distancia al centroide
- Comparacion sistematica: 4 metodos de reduccion x 5 algoritmos de clustering
- Grid search de 360 combinaciones de pesos (alpha, beta, gamma, delta)

---

## 3. Diferencias entre diseno teorico e implementacion

### Algoritmos propuestos vs reales

| Diseno (context.md) | Implementacion | Nota |
|---|---|---|
| K-Means | KMeans | Implementado |
| DBSCAN | DBSCAN | Implementado, con auto-eps via k-distance |
| Mean Shift | MeanShift | Implementado, pero falla (ARI ~0) |
| — | **HDBSCAN** | Nuevo, no estaba en el diseno |
| — | **Agglomerative** | Nuevo, no estaba en el diseno |
| — | **KMeans con UMAP** | Resulto ser el mejor metodo |

### Stack tecnologico

| Componente | Diseno (context.md) | Implementacion real |
|---|---|---|
| Embeddings | BERT multilingual | BERT multilingual (igual) |
| Clustering | scikit-learn | scikit-learn + hdbscan |
| Reduccion | umap-learn, PCA | umap-learn, PCA, SVD, t-SNE |
| Visualizacion | Matplotlib, Seaborn, Plotly | Matplotlib + Seabon (sin Plotly) |
| Conexion BD | SQLAlchemy | oracledb (driver nativo de Oracle) |
| Framework de pruebas | — | pytest (364 tests) |
| Linter | — | Ruff (0 errores) |
| Type hints | — | 100% cobertura en scripts/ |

### Metricas de validacion

| Diseno | Implementacion |
|---|---|
| Solo metricas internas (Silhouette, DB, CH) | Internas + externas (ARI, NMI) |
| Sin ground truth | Ground truth automatico para 235 columnas |
| Deteccion de anomalias solovia DBSCAN | Deteccion centroid-based para cualquier algoritmo |

### Flujo por defecto

El pseudocodigo en context.md muestra DBSCAN como default. En la implementacion:
- El algoritmo default es **KMeans** (no DBSCAN)
- La reduccion default es **PCA** (no UMAP)
- El CLI permite cambiar ambos via `--cluster-method` y `--method`

---

## 4. Hallazgos experimentales principales

### Hipotesis H1: REFUTADA

H1 decian: "BERT + DBSCAN produce clusters mas coherentes y detecta mas anomalias que K-Means o Mean Shift."

**Resultado:** UMAP + KMeans (ARI=0.4085) supera a UMAP + DBSCAN (ARI=0.3636). DBSCAN produce 103 puntos de ruido de 235, lo que degrada el ARI.

### Resultados comparativos

| Algoritmo | Reduccion | ARI | NMI | Clusters | Ruido |
|---|---|---|---|---|---|
| UMAP + KMeans | 5D | 0.4085 | 0.6109 | 5 | 0 |
| UMAP + DBSCAN | 20D | 0.3636 | 0.6328 | 6 | 103 |
| PCA + KMeans | 5D | 0.3517 | 0.5066 | 5 | 0 |
| UMAP + HDBSCAN | 20D | 0.3064 | 0.5696 | 4 | 0 |
| PCA + DBSCAN | 20D | 0.2475 | 0.4770 | 4 | 141 |
| MeanShift | 5D | -0.0037 | 0.3222 | 3 | 0 |

### Sub-hipotesis

| ID | Enunciado | Resultado | Veredicto |
|---|---|---|---|
| H1_a | DBSCAN ARI > KMeans ARI | KMeans 0.4085 > DBSCAN 0.3636 | REFUTADA |
| H1_d | UMAP > PCA | UMAP 0.4085 > PCA 0.3517 | CONFIRMADA |

### Ranking de pesos (grid search)

gamma (constraints) = 0.50 domina. Los constraints (PK, FK, Unique, Nullable, Index) son la senal mas discriminativa.

delta (data_length) = 0.00 consistentemente. La longitud del tipo de dato no correlaciona con anti-patrones.

alpha (BERT) = 0.35 optimo. BERT ayuda pero no es la senal principal.

beta (type) = 0.15 optimo. El tipo de dato ayuda moderadamente.

---

## 4-bis. Re-validacion con embeddings reales (agosto 2026, issue #1)

Tras el cambio a `paraphrase-multilingual-MiniLM-L12-v2` (mean pooling, 384D)
+ comentarios de columnas en el texto, se re-corrio la comparativa con un
protocolo documentado y reproducible: 6 configuraciones de pesos x {UMAP, PCA}
x {5D, 20D} x 5 algoritmos, ARI/NMI contra el ground truth completo
(incluye `clean`). Resultados en `output/phase3_revalidation.json`.

### Resultados (mejor config por algoritmo)

| Algoritmo | Pesos | Reduccion | ARI | NMI | Clusters | Ruido |
|---|---|---|---|---|---|---|
| **HDBSCAN** | a=0.00 b=0.35 g=0.45 d=0.20 | PCA 20D | **0.5815** | 0.4354 | 9 | 18 |
| MeanShift | a=0.00 b=0.35 g=0.45 d=0.20 | UMAP 5D | 0.4653 | 0.3935 | 4 | 0 |
| DBSCAN | a=0.00 b=0.35 g=0.45 d=0.20 | UMAP 5D | 0.4640 | 0.3855 | 5 | 19 |
| KMeans | a=0.00 b=0.35 g=0.45 d=0.20 | UMAP 5D | 0.4606 | 0.3896 | 5 | 0 |
| Agglomerative | a=0.00 b=0.35 g=0.45 d=0.20 | UMAP 5D | 0.4606 | 0.3896 | 5 | 0 |

**Nota sobre comparabilidad:** los numeros de la seccion 4 provienen del
grid search del orchestrator original (embeddings sinteticos, protocolo de
validacion no recuperable). Con el protocolo actual y los mismos pesos
"old-best" (a=0.35 b=0.15 g=0.50 d=0.00), UMAP+KMeans da ARI=0.2443 — el
0.4085 original no es reproducible con el protocolo documentado aqui.

### Hallazgos principales

1. **HDBSCAN es el nuevo mejor algoritmo** (ARI 0.5815). La familia de
   densidad se rehabilita parcialmente: HDBSCAN supera a KMeans sin el
   problema de ruido masivo de DBSCAN (18 puntos vs 103 del reporte original).

2. **H1_a sigue refutada para DBSCAN:** con embeddings reales y pesos
   old-best, KMeans (ARI=0.2443) > DBSCAN (ARI=0.1429, 74 puntos de ruido).

3. **El texto semantico NO mejora el ARI en este ground truth.** Todas las
   configuraciones ganadoras tienen alpha=0. Dos causas identificadas:
   - **El peso alpha esta roto dimensionalmente:** el bloque de texto aporta
     varianza total 384*alpha^2 (384 dims z-scoreadas), que ahoga a los ~18
     dims estructurales. Con alpha=0.05 el texto ya tiene MAS varianza total
     (0.96) que la estructura (0.88). Incluso normalizando por bloque
     (varianza texto = varianza estructura), el ARI con alpha>0 se queda en
     ~0.27 vs 0.58 con alpha=0.
   - **El ground truth es estructural por tabla** (giant_table=105,
     clean=81, wrong_data_types=22...): la semantica cruza tablas (todas las
     FECHA_* juntas) y contradice las clases del GT, que dependen del patron
     a nivel de tabla. El embedding semantico agrupa por significado, no por
     anti-patron.

4. **Implicacion de diseno:** el rol correcto del embedding semantico en
   este sistema es la deteccion de redundancia semantica (columnas con
   igual significado en tablas distintas), no la clasificacion de
   anti-patrones estructurales — para eso rinden las features estructurales
   y el Rule Engine.

---

## 4-ter. Consolidado post-mejoras (agosto 2026, issues #2-#5)

Resumen de todos los cambios de agosto 2026 sobre el pipeline y su efecto
medido. Detalle por issue en los commits referenciados.

### Pipeline de datos (issue de extraccion/embeddings, commit 5e10b3e)

| Componente | Antes | Despues |
|---|---|---|
| Embedding | mBERT + [CLS], 768D | paraphrase-multilingual-MiniLM-L12-v2, mean pooling + L2, 384D |
| Texto a embeber | nombre de columna preprocesado | + comentario de columna (76 comentarios overlay) |
| Extraccion | N+1 (~111 queries) | 9 queries bulk |
| Metadata | tipos + constraints | + comentarios, identity, virtual, indices con posicion |
| Lowercasing | todo a minusculas | solo tokens ALL-CAPS (modelo cased) |
| Random Forest (CV) | F1-macro 0.17 | F1-macro 0.34 |

Sanity check semantico del modelo nuevo: cos(salario, precio)=0.283 >
cos(salario, email)=0.179 — agrupa conceptos relacionados correctamente.
Justificacion con fuentes primarias: docs/research_extraction_preprocessing.md.

### Clasificacion (issues #2 y #3, commits 260c54e y 5d02348)

| Metrica | Antes | Despues | Causa del cambio |
|---|---|---|---|
| wrong_data_types F1 | 0.81 | **1.00** | keyword substring bug + GT under-labeling + prioridad de merge |
| self_contradictory F1 | 0.00 | 0.40 | colision de nombres duplicados + DATOS_MAESTROS en DB |
| Accuracy | 0.9149 | **0.9465** | acumulado |
| F1-macro | 0.7260 | **0.7938** | acumulado |
| Columnas | 235 | 243 | DATOS_MAESTROS alineada con el catalogo |

Los tres fixes de #3: (1) matching por token-boundary en keywords ('fec'
hacia match dentro de 'aFECtada'); (2) GT detecta wrong_data_types a nivel
columna en cualquier tabla, no solo tablas con flag; (3) la deteccion
column-level date/number_as_text ya no es enmascarada por la heuristica
table-level inconsistent_naming.

### Senales SchemaSpy (issue #4, commit 5ab68f3)

Tres detecciones de reporte (sin impacto en metricas de clasificacion):
missing_pk (23 tablas), implicit_fk (2 columnas: EMPLEADO_ID,
PROVEEDORES_ID), redundant_index (2 tablas: EMPLEADOS, ORDENES_COMPRA).
Metodos detect_* publicos fuera del chain de prioridad; el Recommender las
integra al reporte final.

---

## 5. Anti-patrones detectados

### Fechas como texto (11 columnas)
`FECHA_NACIMIENTO`, `FECHA_ALTA`, `FECHA_ORDEN`, `FECHA_VENTA`, `FECHA_EXPIRACION`, etc. almacenadas como VARCHAR2.

### Numeros como texto (9 columnas)
`SALARIO`, `CANTIDAD`, `PRECIO`, `MONTO`, `PRECIO_TOTAL` almacenados como VARCHAR2.

### Booleanos inconsistentes (4 formatos distintos)
- `ACTIVO` como CHAR(1) en algunas tablas, VARCHAR2 en otras
- `ES_ACTIVO` como CLOB
- `ESTADO` como NUMBER (0/1)
- `FLAG_S_N`, `FLAG_Y_N`, `FLAG_1_0`, `FLAG_T_F` en TODO_EN_UNO

### 100% nullable
Las 235 columnas permiten NULL. No hay campos obligatorios.

### TODO_EN_UNO (dominio mezclado)
16 columnas con nombres como `FECHA_O_DIRECCION`, `CANT_O_PRECIO`, `ESTADO_O_ACTIVO`. Cada columna almacena dos conceptos distintos.

---

## 6. Arquitectura del codigo

```
db_bad_clust/
+-- scripts/                    # Codigo fuente (Phase 1 + Phase 2/3)
|   +-- orchestrator.py         # Entrypoint Phase 1 (generar Oracle)
|   +-- phase2_orchestrator.py  # Entrypoint Phase 2/3 (ML pipeline)
|   +-- anti_patterns.py        # Definiciones de tablas mal disenadas
|   +-- cluster_engine.py       # 5 algoritmos de clustering
|   +-- dimensionality_reducer.py # 4 metodos de reduccion
|   +-- evaluator.py            # Metricas + validacion + anomalias
|   +-- ground_truth.py         # Ground truth automatico
|   +-- exceptions.py           # Jerarquia de excepciones (9 clases)
|   +-- ...                     # Modulos restantes
+-- tests/                      # 364 tests (sin DB, con mocks)
+-- dashboard/                  # Dashboard web HTML
+-- output/                     # Resultados de ejecuciones (gitignored)
+-- docs/
|   +-- context.md              # Diseno teorico original
|   +-- implementation.md       # Este documento
|   +-- phase2/                 # Documentacion por modulo
|   +-- phase3/                 # Diseno de validacion + reporte final
+-- README.md                   # Instrucciones de uso
```

---

## 7. Como navegar los resultados

1. Ejecutar el pipeline: `cd scripts && python3 phase2_orchestrator.py --method umap --cluster-method kmeans --validate`
2. Ver resultados en `output/run_<timestamp>/` (reportes .txt + graficos .png)
3. Explorar dashboard: abrir `dashboard/index.html` en navegador
4. Leer reporte completo: `docs/phase3/02_final_report.md`

---

## 8. Stack real utilizado

| Componente | Tecnologia | Version |
|---|---|---|
| Base de datos | Oracle 23c (Docker) | 23.3.0 |
| Embeddings | Hugging Face transformers | 4.50+ |
| Clustering | scikit-learn + hdbscan | 1.6+ / 0.8+ |
| Reduccion | umap-learn, scikit-learn | 0.5+ |
| Visualizacion | Matplotlib + Seaborn | 3.9+ / 0.13+ |
| Tests | pytest | 8+ |
| Linter | Ruff | 0.11+ |
| Python | CPython | 3.14 |
| Driver Oracle | oracledb | 2+ |
| CLI | argparse | standard library |

---

*Documento generado en Mayo 2026, actualizado en agosto 2026 (secciones 4-bis y 4-ter). Proyecto: 23 tablas, 243 columnas, 5 algoritmos, 4 reducciones, 458 tests, accuracy 0.9465 / F1-macro 0.7938 (Rule Engine), ARI maximo 0.5815 (PCA-20D + HDBSCAN).*
