# DEVELOPER.md — db_bad_clust (ml_bad_db_trainer)

## Índice

1. [Entorno de desarrollo](#1-entorno-de-desarrollo)
2. [Hyperparameter tuning](#2-hyperparameter-tuning)
3. [Estándares de calidad de código](#3-estándares-de-calidad-de-código)
4. [Pruebas](#4-pruebas)
5. [Linteo con Ruff](#5-linteo-con-ruff)
6. [Type hints](#6-type-hints)
7. [Manejo de excepciones](#7-manejo-de-excepciones)
8. [Arquitectura del proyecto](#8-arquitectura-del-proyecto)
9. [Estado actual del código](#9-estado-actual-del-código)

---

## 1. Entorno de desarrollo

### Prerequisitos

- Python 3.11+
- Docker (para Oracle 23c)
- `pip` / `venv`

### Setup inicial

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecutar pipeline Phase 2

```bash
cd scripts
python phase2_orchestrator.py              # con BERT
python phase2_orchestrator.py --skip-bert  # sin BERT
```

Todos los comandos deben ejecutarse desde `scripts/` porque `OracleConnector` usa paths relativos a `config.yaml`.

### Phase 2 orchestrator CLI

```
--skip-bert         Skip BERT (use synthetic random embeddings)
--method            PCA | umap | svd | tsne (default: pca)
--n-components      Target dimensions (default: auto, 80% variance)
--n-clusters        Number of clusters (default: auto via silhouette)
--cluster-method    kmeans | dbscan | agglomerative (default: kmeans)
--alpha             BERT semantic weight (default: 0.40)
--beta              Data type weight (default: 0.30)
--gamma             Constraint weight (default: 0.25)
--delta             Statistical feature weight (default: 0.05)
--tune              Auto-tune hyperparameters (grid search)
--output-dir        Output directory (default: ../output/run_&lt;timestamp&gt;)
```

---

## 2. Hyperparameter tuning

### Qué hace `--tune`

Sin `--tune`, el pipeline usa **pesos fijos** y ejecuta una sola vez: phi → PCA → KMeans → reporte.

Con `--tune`, hace un **grid search** de 360 combinaciones para encontrar la mejor config automáticamente:

```
alpha ∈ [0.25, 0.30, 0.35]         # 3 valores
beta  ∈ [0.15, 0.20, 0.25, 0.30]   # 4 valores
gamma = 1.0 - alpha - beta - delta  # derivado
delta ∈ [0.00, 0.03, 0.05, 0.10, 0.15]  # 5 valores
n_components ∈ [5, 10]             # 2 valores
n_clusters   ∈ [3, 5, 7]           # 3 valores
```

Para cada combinación: recalcula phi con los pesos, PCA, KMeans, evalúa silhouette. Al final selecciona la mejor.

### Impacto real

| Métrica | Sin `--tune` (defaults) | Con `--tune` (mejor encontrado) |
|---|---|---|
| Silhouette | ~0.34 | **0.4614** |
| Davies-Bouldin | ~1.76 | **0.976** |
| Calinski-Harabasz | ~57 | **148.9** |
| Pureza | ~82% | **82.77%** |

### Mejor combo conocido

```
alpha=0.35  beta=0.15  gamma=0.50  delta=0.00
n_components=5  n_clusters=5
```

### Tiempo

En una MacBook Air M3, las 360 iteraciones toman ~2s adicionales sobre la ejecución base de ~10s.

### Recomendación de uso

```bash
# Rápido, resultados decentes
python phase2_orchestrator.py

# Mejor calidad, ~2s extra
python phase2_orchestrator.py --tune

# Tune + output personalizado
python phase2_orchestrator.py --tune --output-dir ../output/mi_experimento
```

---

## 3. Estándares de calidad de código

### Resumen

| Estándar | Herramienta | Estado |
|---|---|---|
| Formateo | Ruff (formatter) | Línea máxima 100 cols, comillas dobles |
| Linting | Ruff (linter) | 0 errores |
| Type hints | mypy-strict (ideal) / manual | 100% cobertura |
| Tests | pytest | 297 tests, ~100% cobertura de módulos Phase 2 |
| Excepciones | Jerarquía propia (`exceptions.py`) | Sin `except Exception` genéricos |
| Docstrings | Inglés, Google style | Todos traducidos |
| Logs | Inglés, con logging estándar | Todos traducidos |

### Reglas de Ruff activas

Definidas en `pyproject.toml`:

- **E** (pycodestyle errors)
- **F** (pyflakes)
- **I** (isort — import sorting)
- **N** (pep8-naming)
- **W** (pycodestyle warnings)
- **UP** (pyupgrade — sintaxis moderna)
- **RUF** (ruff-specific)

Excepciones permitidas:
- `E501` — line-length manejado por formatter
- `N802/N803/N806` — nombres de funciones CamelCase tolerados (DDL/DML)

### Uso diario

```bash
# Lintear todo
ruff check scripts/

# Auto-fix
ruff check scripts/ --fix

# Formatear
ruff format scripts/

# Tests
python3 -m pytest tests/ -v

# Tests con cobertura
python3 -m pytest tests/ --cov=scripts/ --cov-report=term-missing
```

---

## 4. Pruebas

### Framework

- **pytest** 8.0+ como test runner
- **pytest-mock** para mocking de Oracle (evitar dependencia de DB real)
- **Configuración**: en `pyproject.toml`, sección `[tool.pytest.ini_options]`

### Estructura

```
tests/
  __init__.py
  conftest.py         # Fixtures compartidos (ColumnMetadata, embeddings, etc.)
  test_exceptions.py
  test_text_preprocessor.py
  test_structural_encoder.py
  test_feature_builder.py
  test_cluster_engine.py
  test_dimensionality_reducer.py
  test_evaluator.py
  test_recommender.py
  test_visualizer.py     # (opcional, requiere matplotlib)
```

### Mock de Oracle

Las pruebas no requieren Oracle corriendo. Se usa `unittest.mock` para simular `oracledb.Connection` y `Cursor` en los tests de `SchemaExtractor` y `OracleConnector`.

### Ejecutar tests

```bash
# Todos los tests
python3 -m pytest tests/ -v

# Test específico
python3 -m pytest tests/test_evaluator.py -v -k "test_silhouette"

# Con cobertura
python3 -m pytest tests/ --cov=scripts/ --cov-report=term-missing

# Sin warnings de sklearn
python3 -m pytest tests/ -v -W ignore::RuntimeWarning
```

---

## 5. Linteo con Ruff

### Historial de limpieza

Durante la auditoría de calidad de código se corrigieron **263 errores** de Ruff:

| Categoría | Regla | Descripción | Errores | Fix |
|---|---|---|---|---|
| Pyupgrade | UP006/UP035 | `Dict`/`List` → `dict`/`list` | ~100 | `--fix` automático |
| Pyupgrade | UP045 | `Optional[X]` → `X \| None` | ~30 | `--fix` automático |
| Isort | I001 | Import sorting | ~50 | `--fix` automático |
| Ruff-specific | RUF001/RUF002 | Caracteres griegos en strings | 17 | Manual → `alpha`/`beta`/`gamma`/`delta` |
| Pyflakes | E741 | Variable `l` ambigua | 1 | Manual → `letter` |
| Pyflakes | F841 | Variables asignadas no usadas | 2 | Manual — eliminadas |
| Pyflakes | F601 | Key duplicado en dict | 1 | Manual — `"min": "minimum"` unificado |
| Pyflakes | F401 | Import sin usar | 1 | Manual — `import torch as _` con `# noqa` |

**Estado actual: 0 errores, 0 warnings.**

### Recomendaciones

- Correr `ruff check scripts/` antes de cada commit
- Usar `ruff check scripts/ --fix` para auto-correcciones
- Si una regla debe ignorarse deliberadamente, usar `# noqa: <RULE>` con comentario explicativo
- NO agregar reglas completas al ignore de `pyproject.toml` sin discusión

---

## 6. Type hints

### Cobertura

**100%** en los 16 scripts del proyecto (3,969 líneas de código).

### Convenciones

```python
from __future__ import annotations   # Siempre al inicio

# En lugar de:
from typing import List, Dict, Optional
def foo(x: List[str]) -> Optional[Dict[str, int]]: ...

# Usar sintaxis moderna (Python 3.11+):
def foo(x: list[str]) -> dict[str, int] | None: ...
```

### Patrones aprobados

```python
# Dataclasses con type hints
@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    nullable: bool
    data_length: int | None = None
    is_primary_key: bool = False

# Funciones con tipo de retorno explícito
def encode_all(self, columns: list[ColumnMetadata]) -> dict[str, np.ndarray]: ...

# Type aliases para tipos complejos
type Vector = np.ndarray
type ColumnBatch = list[ColumnMetadata]
```

### Excepciones

- `np.ndarray` se usa como tipo (no hay tipo genérico para arrays numpy)
- Los callbacks internos pueden omitir tipos si son triviales

---

## 7. Manejo de excepciones

### Jerarquía

Todas las excepciones del proyecto viven en `scripts/exceptions.py`:

```
BadDBError (base)
├── DatabaseError
│   ├── ConnectionError      # Error de conexión a Oracle
│   └── QueryError           # Error en consulta SQL
├── SchemaError              # Error de extracción de metadatos
├── EmbeddingError           # Error en generación de embeddings
├── ClusteringError          # Error en reducción/clustering
├── ConfigError              # Error de configuración (YAML, args)
├── RecommendationError      # Error en generación de recomendaciones
├── VisualizationError       # Error en generación de gráficos
└── DimensionalityError      # Error en reducción dimensional
```

### Reglas

1. **Nunca** usar `except Exception` genérico — capturar la excepción específica de la jerarquía
2. Usar `raise ... from cause` para encadenar causas
3. Las excepciones deben tener mensajes informativos en inglés
4. Al capturar errores externos (Oracle, sklearn), envolverlos en la excepción de dominio correspondiente

```python
# Bien
try:
    cursor.execute(sql)
except oracledb.DatabaseError as e:
    raise QueryError(f"Failed to execute query: {sql[:50]}...") from e

# Mal
try:
    cursor.execute(sql)
except Exception as e:
    print(f"Error: {e}")
```

---

## 8. Arquitectura del proyecto

### Estructura de directorios

```
.
├── AGENTS.md              # Documentación para agentes de IA
├── DEVELOPER.md           # Este archivo
├── pyproject.toml          # Configuración de Ruff + pytest
├── requirements.txt        # Dependencias Python
├── config.yaml            # Configuración de Oracle
├── docker-compose.yml     # Oracle 23c container
├── run_all.sh             # Script one-shot
├── scripts/               # Código principal
│   ├── orchestrator.py         # Phase 1 entry
│   ├── anti_patterns.py        # Dataclasses + generators
│   ├── ddl_generator.py        # CREATE/DROP TABLE
│   ├── dml_generator.py        # INSERT statements
│   ├── db_connector.py         # OracleConnector
│   ├── exceptions.py           # Jerarquía de excepciones
│   ├── schema_extractor.py     # Metadata extraction
│   ├── text_preprocessor.py    # NLP preprocessing
│   ├── bert_embedder.py        # BERT embeddings
│   ├── structural_encoder.py   # Type + constraint encoding
│   ├── feature_builder.py      # Feature vector construction
│   ├── dimensionality_reducer.py # PCA/UMAP/SVD/t-SNE
│   ├── cluster_engine.py       # KMeans/DBSCAN/Agglomerative
│   ├── evaluator.py            # Metrics + evaluation
│   ├── recommender.py          # Fix recommendations
│   ├── visualizer.py           # Plot generation
│   └── phase2_orchestrator.py  # Phase 2 pipeline
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_exceptions.py
│   ├── test_text_preprocessor.py
│   ├── test_structural_encoder.py
│   ├── test_feature_builder.py
│   ├── test_cluster_engine.py
│   ├── test_dimensionality_reducer.py
│   ├── test_evaluator.py
│   └── test_recommender.py
├── docs/phase2/           # Documentación de módulos Phase 2
│   ├── 00_anti_patterns_catalog.md
│   ├── 01_schema_extractor.md
│   ├── ...
│   └── 10_visualizer.md
└── sql_init/              # Init SQL para Oracle
```

---

## 9. Estado actual del código

### Métricas

| Métrica | Valor |
|---|---|
| Scripts Python | 17 |
| Líneas de código | ~3,969 |
| Cobertura type hints | 100% |
| Tests | 297 |
| Errores Ruff | 0 |
| Excepciones genéricas (`except Exception`) | 0 |
| Idioma código | Inglés |
| Idioma docs | Inglés (código), Español (docs/phase2/) |

### Pipeline Phase 2 — Mejor resultado conocido

| Parámetro | Valor |
|---|---|
| alpha (BERT) | 0.35 |
| beta (data type) | 0.15 |
| gamma (constraints) | 0.50 |
| delta (statistical) | 0.00 |
| n_components | 5 |
| n_clusters | 5 |
| **Silhouette** | **0.4614** |
| Davies-Bouldin | 0.976 |
| Calinski-Harabasz | 148.9 |
| Pureza | 82.77% |

### Feature `data_length`

El feature estadístico `data_length` (δ) **no mejora** el clustering. En todas las corridas de tuning, δ=0.00 produce los mejores resultados. Se mantiene implementado por completitud pero desactivado por defecto.

---

## Convenciones de commit

- Usar inglés para mensajes de commit
- Prefijos sugeridos: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Mantener commits atómicos (un cambio por commit)

---

*Última actualización: mayo 2026*
