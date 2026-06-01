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

*Documento generado en Mayo 2026. Proyecto completo: 22 tablas, 235 columnas, 5 algoritmos, 4 reducciones, 364 tests, ARI maximo 0.4085.*
