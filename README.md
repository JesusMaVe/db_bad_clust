# db_bad_clust

Detección no supervisada de anti-patrones de diseño en esquemas Oracle, con embeddings de BERT y
clustering.

## Enfoque

Un anti-patrón como `wrong_data_types` es un conflicto entre lo que el nombre de una columna
promete y lo que su tipo declara: `FECHA_INGRESO` guardada como `VARCHAR2`. El pipeline usa un
encoder de oraciones multilingüe (`paraphrase-multilingual-MiniLM-L12-v2`) para leer qué tipo de
dato sugiere el nombre, lo contrasta con el tipo declarado y agrupa las columnas con Ward.

Ningún paso ve las etiquetas: los hiperparámetros se eligen con criterios internos. Las 243
columnas etiquetadas a mano en `output/manual_labels.csv` solo se usan para puntuar el resultado.

## Resultados

Evaluación sobre 153 columnas, sin las dos tablas gigantes, cuya etiqueta es propiedad de la tabla
y no de la columna. Diferencias medidas con un bootstrap pareado de 500 submuestras.

| representación             |   ARI |   AMI | ARI contra la tabla |
| -------------------------- | ----: | ----: | ------------------: |
| descripción de la columna  | 0.051 | 0.211 |               0.867 |
| conflicto nombre–tipo      | 0.123 | 0.197 |               0.003 |

- La representación de conflicto mejora el ARI en +0.072, con un intervalo de confianza del 95 %
  de +0.040 a +0.104. En AMI empatan.
- La descripción de la columna agrupa sobre todo por tabla. El conflicto agrupa anti-patrones que
  cruzan tablas.
- En el corpus completo, la descripción es mejor en AMI porque reconoce las dos tablas gigantes.

El detalle, los resultados negativos y la metodología están en
[`docs/research_improving_clustering.md`](docs/research_improving_clustering.md).

## Uso

Requiere Python 3.10+ y Docker para la base Oracle de prueba. Contra otra base Oracle solo se leen
metadatos del catálogo (vistas `ALL_*`), nunca filas; el usuario conectado necesita algún privilegio
sobre las tablas del esquema.

```bash
.venv/bin/python -m pip install -e ".[dev]"

# Evaluar, sin base de datos
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --pickle output/intermediate_docs.pkl --conflict --robustness

# Intervalos de confianza (varios minutos)
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --pickle output/intermediate_docs.pkl --bootstrap 500

# Analizar otra base: configurar la conexión en config.yaml y elegir el esquema
.venv/bin/python scripts/build_embeddings.py --owner ESQUEMA --output output/otra_base.pkl

# Reconstruir las representaciones del corpus de prueba
.venv/bin/python scripts/generate_schema_sql.py   # solo con un volumen nuevo
docker compose up -d
.venv/bin/python scripts/verify_schema.py
.venv/bin/python scripts/apply_comments.py
.venv/bin/python scripts/build_embeddings.py --output output/intermediate_docs.pkl

# Tests y lint
.venv/bin/python -m pytest tests/
.venv/bin/python -m ruff check src tests scripts
```

## Estructura

```
src/db_bad_clust/
├── data/         extracción de metadatos Oracle
├── features/     representaciones: documento, conflicto, señales de nombre
├── clustering/   reducción, clustering y nivel de tabla
└── evaluation/   puntuación, experimentos y bootstrap
scripts/          construcción de embeddings y esquema de prueba
```

La rama `rule-engine` contiene un motor de reglas y la herramienta de auditoría, fuera del alcance
de esta rama. [`AGENTS.md`](AGENTS.md) documenta los invariantes y decisiones del proyecto.

## Limitaciones

- Evaluado sobre un solo corpus sintético de 23 tablas y 243 columnas, en español. No se ha
  probado contra una base Oracle real, y los grupos que produce no traen nombre: hay que revisarlos.
- Clases desbalanceadas: `giant_table` es el 37 % de las columnas y `self_referencing` tiene una.
- Los anti-patrones propios de la tabla (`eav`, nombres inconsistentes en toda una tabla) no se
  detectan de forma demostrable. El diseño en dos niveles (`--two-level`) es experimental.
