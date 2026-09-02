# db_bad_clust — Detección de anti-patrones en esquemas Oracle

Audita una base de datos Oracle y reporta problemas de diseño (tipos de dato equivocados,
palabras reservadas, EAV, llaves polimórficas, tablas gigantes, ...), con el SQL de corrección
sugerido para cada uno.

> **Estás en la rama `rule-engine`: el producto.** Contiene el motor de reglas, el comando
> `audit`, el generador de SQL correctivo y los 472 tests. Es la rama estable y no cambia.
>
> La rama **`main`** es ahora otra cosa: la investigación con BERT, sin motor de reglas. Ahí se
> corrigieron dos errores que afectan al veredicto que este README enlaza más abajo —
> el peso α de la fusión nunca ponderó nada (con α nominal 0.15 los embeddings se quedaban el
> 89% del espacio), y el baseline estructural colapsa 243 columnas en 24 vectores distintos,
> con un ARI que va de 0.1877 a 0.5031 según se calcule en float32 o float64. La comparación
> A vs B de este documento sigue siendo válida *para las configuraciones que probó*, pero la
> conclusión "BERT es dañino" no se sostiene con la fusión corregida.

El proyecto tiene **dos ramas** que hacen lo mismo por caminos distintos:

- **`rule-engine` (esta) — motor de reglas.** Heurísticas escritas a mano. Es la herramienta.
- **`main` — clustering con BERT.** Agrupa sin etiquetas. Es la investigación.

El [veredicto medido](docs/veredicto_reglas_vs_ml.html) explica la comparación original, con
ejemplos; léelo junto al aviso de arriba.

---

## 1. Requisitos

- **Docker** — para el contenedor de Oracle 23c
- **Python 3.10+**
- ~2 GB libres si vas a correr la Rama B (descarga torch + el modelo BERT)

## 2. Instalación desde cero

```bash
git clone https://github.com/JesusMaVe/db_bad_clust.git
cd db_bad_clust

python3 -m venv .venv                            # crea el entorno
.venv/bin/python -m pip install -e ".[dev]"      # instala el paquete + todo
```

Levanta Oracle (tarda ~30s en quedar `healthy`):

```bash
docker compose up -d
docker compose ps                                # espera a que diga (healthy)
```

**Verifica que quedó bien** — si estos tres comandos funcionan, estás listo:

```bash
.venv/bin/python -c "import db_bad_clust; print('paquete OK')"
.venv/bin/python -m pytest tests/ -q             # debe decir: 472 passed
.venv/bin/db-bad-clust --help
```

### Variantes de instalación

| Comando | Qué instala | Cuándo |
| --- | --- | --- |
| `pip install -e .` | solo `oracledb` + `pyyaml` | solo quieres auditar (ligero, sin torch) |
| `pip install -e ".[research]"` | + torch, transformers, sklearn, umap, hdbscan | vas a correr los notebooks o `compare` |
| `pip install -e ".[dev]"` | todo lo anterior + pytest, ruff | vas a desarrollar |

La instalación base **no** descarga torch. Eso es a propósito: la auditoría no usa nada de ML.

---

## 3. Auditar un esquema

Es el uso principal. Necesita Oracle levantado.

```bash
.venv/bin/db-bad-clust audit
```

Imprime los hallazgos agrupados por tipo de problema. Para guardar resultados:

```bash
.venv/bin/db-bad-clust audit --sql output/fix.sql --json output/audit.json
```

| Opción | Qué hace |
| --- | --- |
| `--sql RUTA` | escribe el SQL de corrección (revísalo antes de ejecutarlo — lo peligroso va comentado) |
| `--json RUTA` | escribe los hallazgos en JSON, para procesarlos con otra herramienta |
| `--config RUTA` | usa otro archivo de conexión (por defecto `config.yaml`) |

Salida esperada contra la BD del repo:

```
Schema: 23 tables, 243 columns

🔴 [WRONG_DATA_TYPES] Found in 31 columns:
   Columns: FECHA, FECHA, FECHA_ACTUALIZACION, ... (+21 more)
   ✅ Action: Review and fix;
...
```

## 4. Comparar la Rama A contra la Rama B

No necesita Oracle: lee resultados ya guardados.

```bash
.venv/bin/db-bad-clust compare --sweep
```

Mide las dos ramas con las mismas cuatro varas y muestra en qué columnas discrepan. Escribe
el detalle columna por columna en `output/head_to_head.csv`.

| Opción | Qué agrega |
| --- | --- |
| `--sweep` | cuánto aporta BERT, subiendo su peso de 0 a 0.70 |
| `--baselines` | los intentos de ML que perdieron: Random Forest y los cinco algoritmos de clustering |

Requiere el extra `[research]` (usa sklearn y hdbscan).

## 5. Reproducir la investigación (notebooks)

Los notebooks son el **registro de investigación**, no el producto. Córrelos en orden; cada
uno deja un pickle en `output/` que el siguiente lee.

```bash
.venv/bin/jupyter lab notebooks/
```

| # | Notebook | Qué hace | ¿Oracle? |
| --- | --- | --- | --- |
| 01 | `01_data_preparation.ipynb` | extracción + preprocesamiento + encoding estructural | sí |
| 02 | `02_ml_embeddings.ipynb` | embeddings + vector compuesto + reducción | no (lee pickle) |

Los notebooks 03 y 04 se eliminaron: `audit` y `compare` hacen todo lo que hacían. **No borres
01 y 02** — son los únicos que generan `output/intermediate_02.pkl`, que es lo que lee
`compare`. Sin ellos la evidencia deja de poder reproducirse.

Sin abrir Jupyter:

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/02_ml_embeddings.ipynb
```

**Para ir rápido:** pon `SKIP_BERT = True` en el notebook 02 y usa embeddings sintéticos
(evita descargar ~470 MB). Con BERT real tardan varios minutos.

### Ground truth manual

Las métricas honestas salen de etiquetas puestas a mano, no del propio motor:

```bash
.venv/bin/python -m db_bad_clust.generation.label_export   # genera output/manual_labels.csv
```

Llena la columna `label` (vocabulario en `generation/ground_truth.MANUAL_LABEL_VOCABULARY`).
`compare` las usa automáticamente. Sin ellas, la única referencia disponible es el catálogo
del propio motor de reglas, lo que vuelve la evaluación **circular** — sirve para detectar
regresiones, no para reportar exactitud.

## 6. Tests y lint

```bash
.venv/bin/python -m pytest tests/ -q                    # 472 tests, no necesita Oracle
.venv/bin/python -m pytest tests/test_rule_engine.py -q # un solo archivo
.venv/bin/python -m ruff check src tests scripts        # 11 errores preexistentes (RUF012)
```

Los 11 errores de ruff son conocidos y anteriores; código nuevo debe salir limpio.

---

## 7. Problemas comunes

**`ModuleNotFoundError: No module named 'db_bad_clust'`**

El editable install quedó viejo (típico si moviste o renombraste la carpeta del proyecto).
Reinstala:

```bash
.venv/bin/python -m pip install -e . --no-deps
```

Como salida de emergencia, `PYTHONPATH=src .venv/bin/python -m db_bad_clust.cli ...` funciona
sin reinstalar.

**`db-bad-clust: command not found`**

El script se crea al instalar el paquete. Corre `pip install -e .` de nuevo, o usa la forma
equivalente `.venv/bin/python -m db_bad_clust.cli`.

**La auditoría no conecta a Oracle**

Revisa que el contenedor esté `healthy` (no solo `Up`) — tarda ~30s desde el arranque:

```bash
docker compose ps
```

**`ModuleNotFoundError: No module named 'hdbscan'` (o sklearn, torch...)**

Estás en la instalación base. Los notebooks y `compare` necesitan el extra:

```bash
.venv/bin/python -m pip install -e ".[research]"
```

**Los notebooks leen datos viejos**

Se encadenan por archivos pickle en `output/`. Si el 01 falla a medias, el 02 lee el pickle
anterior sin avisar. Ante resultados raros, corre los dos en orden desde cero.

---

## 8. Resultados

Las dos ramas medidas con las mismas cuatro varas, contra 243 columnas etiquetadas a mano:

| Rama | ARI | NMI | Exactitud | F1-macro |
| --- | ---: | ---: | ---: | ---: |
| **A — motor de reglas** | **0.7667** | **0.8058** | **0.8930** | **0.8467** |
| B — clustering ML | 0.5008 | 0.4199 | 0.7078 | 0.2639 |

El motor de reglas gana las cuatro, **incluidas ARI y NMI que son las métricas propias del
clustering**. Y las cifras del clustering son un techo, no su resultado real: para calcularle
exactitud hay que ponerle nombre a sus grupos usando las respuestas correctas, información que
el motor de reglas nunca recibe.

Reprodúcelo con `db-bad-clust compare`. El [reporte completo](docs/veredicto_reglas_vs_ml.html)
lo explica con ejemplos.

> **Ojo con la métrica circular.** Verás por ahí `exactitud 0.9877 / F1 0.9057`. Ese número sale
> de comparar el motor contra un ground truth generado *por el propio motor*. No es exactitud
> real; sirve para detectar regresiones. La cifra honesta es **0.8930 / 0.8467**.

---

## 9. Estructura

```
src/db_bad_clust/
├── cli.py         # audit + compare — la superficie del producto
├── data/          # db_connector, schema_extractor — Oracle entra, dataclasses salen
├── rules/         # rule_engine (Rama A), recommender, recommendation_reporter
├── features/      # text_preprocessor, structural_encoder, bert_embedder, feature_builder
├── clustering/    # dimensionality_reducer, cluster_engine        ─┐ Rama B
├── evaluation/    # metrics, validation, anomaly, head_to_head     ─┘ (investigación)
└── generation/    # anti_patterns (catálogo), schema_adapter, ddl_generator,
                   # ground_truth, label_export
scripts/           # apply_comments.py — script suelto, no lo importa el paquete
notebooks/         # 01-04, el registro de investigación
tests/             # 472 tests, todos con mocks — no necesitan BD
docs/              # reportes y research
```

## 10. Documentación

| Archivo | Qué contiene |
| --- | --- |
| [`docs/veredicto_reglas_vs_ml.html`](docs/veredicto_reglas_vs_ml.html) | por qué la Rama A le gana a la B, con ejemplos |
| `docs/00_PROJECT_STATUS_REPORT.md` | reporte de estado fase por fase |
| `docs/rule_engine_and_future_ml_report.md` | diseño del motor de reglas |
| `docs/research_extraction_preprocessing.md` | investigación de buenas prácticas con fuentes |
| `AGENTS.md` | invariantes que hay que conocer **antes** de tocar `rule_engine.py` o `generation/` |

Si vas a modificar el motor de reglas, lee `AGENTS.md` primero: varios comportamientos que
parecen bugs son deliberados y están documentados con su causa raíz.
