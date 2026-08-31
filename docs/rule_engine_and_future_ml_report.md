# Motor de Reglas Determinista y Roadmap ML Futuro

> **Fecha:** 27 de Agosto, 2026
> **Estado actual:** Motor de reglas production-ready, ML en infraestructura muerta
> **Resultados vigentes:** Accuracy 0.893 / F1-macro 0.847 (ground truth manual, 243 columnas)

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Motor de Reglas Determinista — Arquitectura](#2-motor-de-reglas-determinista--arquitectura)
3. [Reglas de Columna (8 activas)](#3-reglas-de-columna-8-activas)
4. [Reglas de Tabla (4 activas)](#4-reglas-de-tabla-4-activas)
5. [Lógica de Merge y Prioridades](#5-lógica-de-merge-y-prioridades)
6. [Resultados de Evaluación](#6-resultados-de-evaluación)
7. [SBERT — Estado Actual vs Potencial](#7-sbert--estado-actual-vs-potencial)
8. [GNN — Propuesta Futura](#8-gnn--propuesta-futura)
9. [Roadmap de Integración](#9-roadmap-de-integración)
10. [Decisiones Técnicas Clave](#10-decisiones-técnicas-clave)

---

## 1. Resumen Ejecutivo

El sistema de clasificación actual es un **motor de reglas determinista** que detecta 10 categorías de anti-patrones en esquemas de bases de datos Oracle. No utiliza machine learning para la clasificación final: los embeddings BERT (SBERT) se calculan pero no se consumen, y el Random Forest/One-Class SVM fueron descartados por resultados inaceptables.

El motor de reglas alcanza **accuracy 89.3%** y **F1-macro 84.7%** contra ground truth manual (evaluación honesta, no circular), usando exclusivamente metadata de Oracle: tipos de dato, constraints, nombres de columnas, y estructura del esquema.

Dos líneas de mejora futura existen:
- **SBERT mejorado** — actualmente ya usa un modelo sentence-transformers pero no consume los embeddings para clasificación
- **GNN (Graph Neural Networks)** — cero código actual, requiere nueva dependencia (`torch_geometric`) y arquitectura completa

---

## 2. Motor de Reglas Determinista — Arquitectura

### Componentes

```
rule_engine.py (1611 lineas)
+-- ColumnRuleEngine          <- 8 reglas de columna + 4 report-level
+-- TableRuleEngine           <- 4 reglas de tabla + 6 report-level
+-- classify()                <- Funcion unificada de merge
```

### Flujo de Datos

```
Oracle 23c (docker)
    |
SchemaExtractor (bulk queries)
    |
DatabaseSchema { tables: [TableMetadata] }
    |
+-------------------------------------+
|  ColumnRuleEngine.classify()        |  -> 8 reglas por columna
|  TableRuleEngine.classify()         |  -> 4 reglas por tabla
|  classify(schema)                   |  -> merge + propagate + clean default
+-------------------------------------+
    |
[ClassificationResult] <- 243 resultados
```

### Entrada del Motor

El motor recibe un objeto `DatabaseSchema` con la siguiente estructura (extraído de `schema_extractor.py`):

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `table.name` | `str` | Nombre de la tabla Oracle |
| `col.name` | `str` | Nombre de la columna |
| `col.data_type` | `str` | Tipo Oracle (`VARCHAR2`, `NUMBER`, `DATE`, etc.) |
| `col.nullable` | `bool` | `Y` o `N` |
| `col.is_primary_key` | `bool` | Pertenece a PK |
| `col.is_foreign_key` | `bool` | Es FK |
| `col.fk_references_table` | `str \| None` | Tabla referenciada |
| `col.fk_references_column` | `str \| None` | Columna referenciada |
| `col.data_length` | `int` | Tamaño declarado en bytes |
| `col.avg_length` | `int` | Tamaño promedio real |
| `col.num_distinct` | `int \| None` | Valores distintos |
| `col.num_nulls` | `int \| None` | Valores nulos |
| `table.columns` | `list` | Lista de columnas de la tabla |

**Lo que NO ingresa al motor:** embeddings BERT (384D), vector compuesto phi, features estructurales codificadas, reduccion dimensional. El motor trabaja con metadata pura de Oracle.

---

## 3. Reglas de Columna (8 activas)

### Regla 1: `wrong_date_as_text` (confianza: 0.90)

**Condición:** Columna con nombre sugiriendo fecha pero tipo `VARCHAR2` o `CHAR`.

**Keywords (lowercase, token-boundary):**
```
fecha, fec, fch, fh, date, time, timestamp, dia, mes, anio, ano,
hor, hora, created, updated, modified, start, end, begin, nacimiento,
alta, baja, expiracion, generacion, actualizacion, cambio, orden
```

**Excepciones (no detecta como fecha):**
```
direccion, fabrica, beneficio, especificacion, fabricacion,
localizacion, documentacion, reubicacion
```

**Ejemplo real:** `CLIENTES_DIRECCIONES.FECHA_REGISTRO` -> tipo `VARCHAR2` -> detectado como `wrong_date_as_text`

### Regla 2: `wrong_number_as_text` (confianza: 0.85)

**Condición:** Columna con nombre sugiriendo número pero tipo `VARCHAR2` o `CHAR`.

**Keywords:**
```
precio, costo, salario, monto, total, amount, price, salary, cost,
balance, saldo, deuda, pago, revenue, income, cantidad, quantity,
porcentaje, percentage, tasa, rate, impuesto, tax, descuento,
discount, comision, fee, capital, inversion, budget, presupuesto
```

**Excepciones:**
```
costumbre, totalizar, contacto, percentil
```

### Regla 3: `self_contradictory` (confianza: 0.80)

**Condición:** Columna booleana almacenada como `VARCHAR2` o `CHAR`.

**Keywords (word-boundary):**
```
flag, active, activo, enabled, disabled, es_activo, is_active,
is_enabled
```

**Patrones de prefijo:**
```
es_*, is_*, has_*, tiene_*, puede_*, allow_*, permit_*
```

### Regla 4: `reserved_words` (confianza: 0.95)

**Condición:** Nombre de columna es exactamente una palabra reservada SQL.

**91 palabras reservadas:** `SELECT`, `FROM`, `WHERE`, `NULL`, `TIMESTAMP`, `CREATE`, `DROP`, `TABLE`, `INDEX`, etc.

**Match:** Comparación exacta (case-sensitive). No es substring.

### Regla 5: `self_referencing` (confianza: 0.95/0.70)

**Condición:** FK que referencia la misma tabla donde está.

**Nombres típicos:** `parent_id`, `manager_id`, `ref_*`, `superior_id`

**Confianza:** 0.95 si el nombre sugiere auto-referencia, 0.70 si solo se detecta por FK pattern.

### Regla 6: `impossible_data` (confianza: 0.95)

**Condición:** Columna en PK que permite NULL, o FK que referencia columna inexistente.

**Limitación documentada (issue #6):** En Oracle real, las FK constraints requieren parent keys únicos sin duplicados/nulls. El extractor correcto no encuentra esta situación en el dataset actual (solo existe en el benchmark sintético).

### Regla 7: `polymorphic` (confianza: 0.75)

**Condición:** Columna con nombre de discriminador de tipo o patrón `_O_`.

**Nombres típicos:** `type`, `tipo`, `tipo_dato`, `tipo_registro`, `discriminador`, `class`, `category`

**Patrón regex:** `_O_` (columnas como `DESCRIPCION_O_FACTURAS`, `IMPORTE_O_PAGOS`)

**Excepción de merge:** `polymorphic` gana sobre `giant_table` (issue #7: columnas `_O_` en TODO_EN_UNO).

### Regla 8: `wrong_data_types` (confianza: 0.85, tabla-level)

**Condición:** Tabla con >20% de columnas `VARCHAR2` cuyos nombres contienen keywords de fecha/número.

**Trigger:** Solo se activa para columnas donde la regla de columna detecto `number_as_text` o `date_as_text` -> el merge las renombra a `wrong_data_types`.

### Reglas Report-Level (no en la cadena de classify)

Estas reglas NO participan en la clasificación. Solo generan reportes:

| Regla | Detecta | Razón de exclusión |
|-------|---------|-------------------|
| `detect_missing_pk` | Tablas sin PK | Enmascararía todas las detecciones (23 tablas sin PK) |
| `detect_redundant_indexes` | Índices redundantes | Nivel de índice, no de columna |
| `detect_implicit_fks` | FKs implícitas | Nivel de relación, no de columna |
| `detect_obsolete_types` | LONG/LONG RAW | Tipos obsoletos, no anti-pattern de diseño |
| `detect_oversized_varchars` | VARCHAR2 sobre-declarados | Requiere acceso a datos reales |
| `detect_fk_without_index` | FKs sin índice | Rendimiento, no diseño |
| `detect_stale_statistics` | Estadísticas stale | Optimizer, no diseño |
| `detect_disabled_constraints` | Constraints deshabilitadas | Estado, no diseño |
| `detect_partition_candidates` | Tablas grandes con fechas | Sugerencia de particionado |

---

## 4. Reglas de Tabla (4 activas)

### Regla T1: `giant_table` (umbral: 15 columnas)

**Condición:** Tabla con más de 15 columnas.

**Propagación:** Todas las columnas de la tabla heredan el label `giant_table`.

**Excepción:** Columnas con detección `polymorphic` preservan su label (issue #7).

### Regla T2: `eav_pattern` (regex)

**Condición:** Tabla con al menos una columna `entity_id` + una columna tipo `attribute` + una columna tipo `value`.

**Patrones:**
```
entity_id, attribute_name, attribute_value, value, attr, name_value
```

### Regla T3: `inconsistent_naming` (detección por sets)

**Condición:** Mezcla de nombres en español e inglés, o predominio de nombres genéricos.

**Genéricos:** `id`, `name`, `value`, `description`, `code`, `type`, `status`, `date`, `flag`

**Excepción de merge:** Columnas con `date_as_text` o `number_as_text` ganan sobre `inconsistent_naming` (issue #3).

### Regla T4: `wrong_data_types` (tabla-level)

**Condición:** Tabla donde >20% de columnas VARCHAR2 tienen keywords de fecha/número en el nombre.

---

## 5. Lógica de Merge y Prioridades

La función `classify()` (línea 1473) implementa un merge con prioridades:

```
Para cada columna de cada tabla:
  1. ¿Hay label de tabla?
     SI -> Es "inconsistent_naming" y la columna tiene date_as_text/number_as_text?
            SI -> usar label de columna (mas especifico)
            NO -> Es "giant_table" y la columna tiene polymorphic?
                   SI -> usar polymorphic (mas especifico)
                   NO -> propagar label de tabla
     NO -> Hay label de columna?
            SI -> usar label de columna
            NO -> clean (default)
```

### Tabla de Prioridades

| Prioridad | Label de Columna | Label de Tabla | Resultado |
|-----------|-----------------|----------------|-----------|
| 1 (máxima) | `polymorphic` | `giant_table` | **polymorphic** |
| 2 | `date_as_text` | `inconsistent_naming` | **date_as_text** |
| 3 | `number_as_text` | `inconsistent_naming` | **number_as_text** |
| 4 | Cualquier otro | Cualquier otro | **label de tabla** (propaga) |

### Propagación de `wrong_data_types`

Cuando una tabla tiene label `wrong_data_types`, las columnas con `number_as_text` o `date_as_text` se renombran a `wrong_data_types` (unificación de vocabulario). Otras detecciones de columna (como `bad_boolean`) se preservan.

---

## 6. Resultados de Evaluación

### Ground Truth Manual (evaluación honesta)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | **0.8930** |
| **F1-macro** | **0.8467** |
| **F1-weighted** | **0.8857** |
| Columnas clasificadas | 243 |
| Anti-patterns detectados | 178 (73.3%) |
| Ground truth | Manual (no circular) |

### Classification Report (243 columnas)

| Clase | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| clean | 0.94 | 0.78 | 0.85 | 78 |
| eav | 1.00 | 1.00 | 1.00 | 6 |
| giant_table | 0.89 | 1.00 | 0.94 | 90 |
| impossible_data | 0.00 | 0.00 | 0.00 | 4 |
| inconsistent_naming | 0.63 | 0.81 | 0.71 | 21 |
| polymorphic | 1.00 | 1.00 | 1.00 | 7 |
| reserved_words | 1.00 | 1.00 | 1.00 | 4 |
| self_contradictory | 1.00 | 1.00 | 1.00 | 5 |
| self_referencing | 1.00 | 1.00 | 1.00 | 1 |
| wrong_data_types | 0.96 | 0.96 | 0.96 | 27 |

### Ground Truth Rule-Engine-Aligned (circular, para referencia)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.9877 |
| F1-macro | 0.9057 |
| Coverage | 74.5% (175/235) |

**Nota:** Este resultado es circular (el ground truth viene del mismo rule engine). Se reporta solo para comparación histórica.

### Por Qué Funciona el Motor de Reglas

1. **Los anti-patrones son estructurales, no semánticos.** Un `VARCHAR2` que debería ser `DATE` se detecta por el tipo de dato + keyword en el nombre, no por embeddings semánticos.

2. **La metadata de Oracle es rica y suficiente.** PK, FK, Unique, tipos de dato, nullable, data_length — toda la señal que las reglas necesitan viene del data dictionary.

3. **Las reglas son interpretables.** Cada detección genera `explanation` + `fix` concreto (ej: "Migrar a DATE: FECHA_ALTA"). En auditorías de BD se necesita saber **por qué**.

4. **Determinismo = reproducibilidad.** Mismas reglas + mismo esquema = mismos resultados. Sin varianza de modelo, sin semilla aleatoria.

---

## 7. SBERT — Estado Actual vs Potencial

### Estado Actual

El archivo `bert_embedder.py` (220 líneas) ya implementa SBERT:

```python
# Línea 74
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

**Qué hace bien:**
- Usa el modelo correcto (MiniLM-L12-v2, entrenado para semantic similarity)
- Mean pooling con attention mask (correcto)
- L2 normalization (correcto)
- 384 dimensiones de salida
- Lazy loading + cache en memoria
- Device auto-detection (CUDA/MPS/CPU)

**Qué hace manualmente (podría simplificarse):**
- Mean pooling implementado a mano (líneas 184-191) en vez de usar `SentenceTransformer.encode()`
- Tokenización manual con `AutoTokenizer` + `AutoModel` en vez de la API de sentence-transformers

### Lo Que No Se Consume

El notebook `03_classification.ipynb` carga el pickle de notebook 02 (que contiene los embeddings) pero **solo usa `schema` y `column_index`**:

```python
# notebook 03, línea 30-36
schema = data["schema"]
phi = data["phi"]           # <- vector compuesto (incluye BERT)
column_index = data["column_index"]
# ...
rule_results = classify(schema=schema)  # <- solo pasa schema, no phi
```

Los embeddings BERT están en `phi` pero `classify()` nunca los recibe. Son infraestructura muerta.

### Mejora Propuesta: SBERT como Feature del Rule Engine

**Opción A: Embeddings como señal de breaking tie**

Agregar los embeddings BERT como señal de desempate cuando el rule engine tiene confianza baja:

```python
# Concepto
if result.confidence < 0.85:
    # Usar similitud coseno con embeddings de columnas conocidas
    # para reforzar o debilitar la clasificación
    embedding_similarity = cosine_sim(phi[col_idx], known_patterns[label])
    result.confidence *= (1 + embedding_similarity * 0.1)
```

**Opción B: Clasificador de fallback para columnas "clean"**

Usar un clasificador simple (KNN o SVM lineal) sobre los embeddings para las 60 columnas que el rule engine marca como `clean` pero que podrían tener anti-patrones sutiles:

```python
# Concepto
clean_columns = [r for r in results if r.predicted_label == "clean"]
if clean_columns:
    embeddings = phi[clean_column_indices]
    ml_predictions = fallback_clf.predict(embeddings)
    # Refinar clean -> possible_anti_pattern
```

**Opción C: Usar SentenceTransformer class (limpieza de código)**

Reemplazar el mean pooling manual por la API de sentence-transformers:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts, normalize_embeddings=True)
```

**Ventaja:** ~50 líneas menos de código, misma funcionalidad.

### Impacto Esperado de SBERT

| Métrica | Actual (sin embeddings) | Con SBERT como feature |
|---------|------------------------|----------------------|
| Accuracy | 0.893 | ~0.91-0.93 (estimado) |
| F1-macro | 0.847 | ~0.86-0.88 (estimado) |
| Cobertura | 73.3% | ~78-82% (estimado) |

**Nota:** Estas son estimaciones conservadoras. El beneficio principal sería en la "brecha limpia" (60 columnas sin detectar) y en `inconsistent_naming` (precision baja: 0.63).

---

## 8. GNN — Propuesta Futura

### Estado Actual

- **Cero código** en el repositorio
- **Cero dependencias** (`torch_geometric` no está en `requirements.txt`)
- **Una mención** en `docs/00_PROJECT_STATUS_REPORT.md:873`: "Graph Neural Networks sobre relaciones FK" (prioridad media)

### Por Qué GNN Para Este Problema

Los anti-patrones en bases de datos no son solo propiedades individuales de columnas — son **relaciones entre entidades**:

- `wrong_data_types` -> relacion entre nombre y tipo de dato
- `eav_pattern` -> relacion entre 3 columnas de la misma tabla
- `giant_table` -> relacion entre columnas de una tabla
- `self_referencing` -> relacion FK de una tabla consigo misma
- `inconsistent_naming` -> relacion entre nombres de columnas en la misma tabla

Un GNN puede capturar estas relaciones de forma natural, mientras que el rule engine actual las evalúa con heurísticas manuales.

### Arquitectura Propuesta

#### Grafo de Schema

```
Nodos:
  - Tabla (23 nodos)
    features: [n_columns, has_pk, has_fk, avg_data_length, ...]
  - Columna (243 nodos)
    features: [data_type_one_hot(12), nullable, is_pk, is_fk, 
               data_length, embedding(384), keyword_flags(10)]

Aristas:
  - TABLE -> COLUMN (pertenencia)
  - COLUMN -> COLUMN (misma tabla)
  - COLUMN -> COLUMN (FK reference)
  - COLUMN -> COLUMN (nombre similar, cosine > 0.8)
```

#### Modelo: GraphSAGE o GCN

```
Input: Grafo de schema con features en nodos
  |
GCN Layer 1 (64 output) + ReLU + Dropout(0.3)
  |
GCN Layer 2 (32 output) + ReLU + Dropout(0.3)
  |
Readout: Global Mean Pooling
  |
Classifier: Linear(32 -> 10) + Softmax
  |
Output: Probabilidad por cada anti-pattern label
```

#### Dependencias Necesarias

```
torch_geometric>=2.4.0
torch>=2.0.0  (ya existe)
networkx>=3.0  (para construcción del grafo)
```

### Ventajas del GNN vs Rule Engine

| Aspecto | Rule Engine | GNN |
|---------|-------------|-----|
| Interpretabilidad | Alta (cada regla explica) | Baja (black box) |
| Mantenimiento | Manual (agregar keywords) | Automático (aprende de datos) |
| Generalización | Limitada a reglas conocidas | Puede descubrir nuevos patrones |
| Requiere training | No | Sí (necesita ground truth etiquetado) |
| Robustez a ruido | Alta (determinista) | Media (depende del training data) |
| Captura de relaciones | Heurísticas manuales | Aprendida del grafo |

### Limitaciones para Este Proyecto

1. **Dataset demasiado pequeño.** 243 columnas, 23 tablas — insuficiente para entrenar un GNN robusto.
2. **Ground truth limitado.** Solo 10 categorías, 7 con <10 muestras.
3. **Overfitting casi garantizado.** Un GNN con 200+ parámetros sobre 243 muestras memorizará el training set.
4. **Costo de infraestructura.** `torch_geometric` agrega ~200MB de dependencias.

### Escenario Donde GNN Sí Tiene Sentido

Si el proyecto escala a:
- **+1000 tablas** (enterprise database)
- **Ground truth acumulado** de auditorías manuales
- **Múltiples SGBDs** (MySQL, PostgreSQL, Oracle)
- **Evolución temporal** (detectar degradación de esquemas)

En ese escenario, el GNN podría aprender patrones que el rule engine no puede codificar manualmente.

---

## 9. Roadmap de Integración

### Fase 1: SBERT como Feature (1-2 semanas)

**Objetivo:** Usar los embeddings BERT como señal adicional en el rule engine.

**Pasos:**
1. Refactorizar `bert_embedder.py` para usar `SentenceTransformer` class (reducción de código)
2. Modificar `classify()` para recibir `phi` como parámetro opcional
3. Agregar lógica de desempate: cuando confidence < 0.85, usar similitud coseno con embeddings de columnas conocidas
4. Evaluar impacto en accuracy/F1 con ground truth manual

**Resultado esperado:** Accuracy ~0.91-0.93, cobertura ~78-82%.

### Fase 2: GNN como Clasificador Paralelo (2-3 meses)

**Objetivo:** Entrenar un GNN como segundo opinion parallel al rule engine.

**Pasos:**
1. Agregar `torch_geometric` y `networkx` a `requirements.txt`
2. Implementar `schema_graph.py` — construir grafo de schema desde `DatabaseSchema`
3. Implementar `gnn_classifier.py` — GCN/GraphSAGE para clasificación
4. Agregar notebook `05_gnn_exploration.ipynb` — experimentación
5. Evaluar ARI/F1 del GNN vs rule engine

**Resultado esperado:** GNN como validación cruzada, no como reemplazo.

### Fase 3: Ensemble (1 mes)

**Objetivo:** Combinar rule engine + GNN + SBERT para resultado final.

**Arquitectura:**
```
Columna X
  +-- Rule Engine -> label + confidence
  +-- GNN -> probability distribution
  +-- SBERT similarity -> reference match
  |
Ensemble Aggregator (weighted voting)
  |
Final label + confidence
```

**Pesos del ensemble:**
- Rule Engine: 0.5 (interpretable, high confidence)
- GNN: 0.3 (captura relaciones)
- SBERT: 0.2 (señal semántica)

---

## 10. Decisiones Técnicas Clave

| Decisión | Alternativa Descartada | Razón |
|----------|----------------------|-------|
| Reglas deterministas sobre ML | One-Class SVM / Random Forest | F1=0.17, dataset demasiado pequeño, clases desbalanceadas |
| Token-boundary matching (`_kw_match`) | Substring matching (`kw in name`) | Substring causaba falsos positivos ('fec' dentro de 'afectada') |
| Modelos report-level separados de classify() | Incluir en classify() | Enmascararían todas las detecciones (23 tablas sin PK) |
| Ground truth manual sobre rule-engine-aligned | Solo rule-engine-aligned | Evaluación circular daba accuracy artificial de 98.77% |
| Sentence-transformers MiniLM-L12 sobre bert-base-multilingual | BERT crudo + [CLS] | SBERT paper: CLS=29.19 vs SBERT=74.89 Spearman on STS |
| Merge con prioridades (column > table) | Promedio o voting | date_as_text es más específico que inconsistent_naming |
| `polymorphic` > `giant_table` en merge | giant_table propaga a todos | Issue #7: columnas `_O_` en TODO_EN_UNO son polymorphic, no giant |

---

## Referencias

- **SBERT Paper:** arXiv:1908.10084 — Reimers & Gurevych, 2019
- **MiniLM-L12-v2:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384D, 50+ idiomas)
- **torch_geometric:** PyTorch Geometric — libraries for deep learning on irregular data
- **Oracle Data Dictionary:** ALL_TAB_COLUMNS, ALL_CONSTRAINTS, ALL_CONS_COLUMNS, ALL_INDEXES
- **Issues #1-#7:** GitHub issues documentando cada mejora con root-cause analysis
